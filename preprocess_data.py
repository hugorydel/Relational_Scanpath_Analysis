#!/usr/bin/env python3
"""
Synthetic Visual Genome (SVG) Relational Dataset Creator
Filters SVG images for gaze-based relational memory experiments

Refactored with:
- Memorability caching
- SVG dataset caching
- Category-based predicate weighting
- Config-driven parameters
"""

# Fix OpenMP conflict BEFORE importing other libraries
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Use non-interactive matplotlib backend to avoid Qt threading issues
import matplotlib

matplotlib.use("Agg")

import json
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset
from PIL import Image
from pycocotools import mask as mask_utils
from resmem import ResMem, transformer
from tqdm import tqdm

import config


class MemorabilityCache:
    """Manages caching of memorability scores."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.modified = False

    def _load_cache(self) -> Dict[str, float]:
        """Load existing cache from disk."""
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}

    def get(self, image_id: str) -> Optional[float]:
        """Get cached memorability score."""
        return self.cache.get(image_id)

    def set(self, image_id: str, score: float):
        """Store memorability score in cache."""
        self.cache[image_id] = float(score)
        self.modified = True

    def save(self):
        """Save cache to disk if modified."""
        if self.modified:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=2)
            print(f"Saved memorability cache with {len(self.cache)} entries")


class SVGDatasetCache:
    """Manages caching of preprocessed SVG dataset."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path

    def exists(self) -> bool:
        """Check if cache exists."""
        return self.cache_path.exists()

    def load(self) -> List[Dict]:
        """Load preprocessed dataset from cache."""
        print(f"Loading SVG dataset from cache: {self.cache_path}")
        with open(self.cache_path, "rb") as f:
            dataset = pickle.load(f)
        print(f"Loaded {len(dataset)} scene graphs from cache")
        return dataset

    def save(self, dataset: List[Dict]):
        """Save preprocessed dataset to cache."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved SVG dataset cache with {len(dataset)} entries")


class SVGRelationalDataset:
    """Filter and process SVG for relational gaze experiments."""

    def __init__(self):
        """Initialize the SVG dataset creator using config parameters."""
        self.svg_root = Path(config.SVG_ROOT)
        self.vg_image_root = Path(config.VG_IMAGE_ROOT)
        self.output_dir = Path(config.OUTPUT_DIR)
        self.target_size = config.TARGET_SIZE
        self.source_filter = config.SOURCE_FILTER

        # Filtering thresholds from config
        self.min_memorability = config.MIN_MEMORABILITY
        self.min_mask_area_percent = config.MIN_MASK_AREA_PERCENT
        self.min_objects = config.MIN_OBJECTS
        self.max_objects = config.MAX_OBJECTS
        self.min_relations = config.MIN_RELATIONS
        self.min_coverage_percent = config.MIN_COVERAGE_PERCENT

        # Predicate weights from config
        self.predicate_weights = config.PREDICATE_WEIGHTS

        # Create output directories
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        print(f"Created fresh output directory: {self.output_dir}")

        print("\n" + "=" * 60)
        print("IMPORTANT: SVG Dataset Schema Notes")
        print("=" * 60)
        print("SVG uses the following structure:")
        print(
            "  - image_id: filename (e.g., '1.jpg' for VG, 'ADE_frame_*.jpg' for ADE)"
        )
        print("  - Metadata is FLATTENED: 'metadata/height', 'metadata/width'")
        print("  - scene_graph: JSON string with objects & relations")
        print("  - regions: segmentation masks and bboxes")
        print("  - relations format: [[subj_idx, obj_idx, predicate], ...]")
        print("  - Contains MULTIPLE datasets: VG, ADE20K, etc.")
        print("  - VG filtering: images with numeric filenames (1.jpg, 234.jpg)")
        print("=" * 60 + "\n")

        # Initialize caches
        self.memorability_cache = MemorabilityCache(
            Path(config.MEMORABILITY_CACHE_PATH)
        )
        self.svg_cache = SVGDatasetCache(Path(config.SVG_CACHE_PATH))

        # Load memorability predictor
        print("Loading ResMem predictor...")
        self.resmem = ResMem()
        print("ResMem predictor loaded")

        # Color palette for visualization
        np.random.seed(config.RANDOM_SEED)
        self.colors = self._generate_colors(100)

    def _generate_colors(self, n: int) -> np.ndarray:
        """Generate distinguishable colors for visualization."""
        colors = sns.color_palette("husl", n)
        return np.array(colors)

    def _load_svg_metadata(self) -> List[Dict]:
        """
        Load SVG scene graphs from cache or HuggingFace.
        FIXED: Now correctly parses actual SVG HuggingFace schema.

        Returns:
            List of preprocessed scene graph dictionaries
        """
        # Try to load from cache first
        if self.svg_cache.exists():
            return self.svg_cache.load()

        # Load from HuggingFace
        print("Loading SVG dataset from HuggingFace...")
        dataset = load_dataset("jamepark3922/svg", split="train")

        # PRE-FILTER to Visual Genome images only (numeric filenames)
        print("Pre-filtering to Visual Genome images only...")

        def is_vg_image(example):
            img_id = example.get("image_id", "")
            if not img_id:
                return False
            filename_without_ext = img_id.replace(".jpg", "")
            return filename_without_ext.isdigit()

        dataset = dataset.filter(is_vg_image)
        print(f"After VG filtering: {len(dataset)} images")

        # Limit for debugging if MAX_IMAGES is set
        if config.MAX_IMAGES and config.MAX_IMAGES < len(dataset):
            dataset = dataset.select(range(config.MAX_IMAGES))
            print(f"Limited to {config.MAX_IMAGES} images for debugging")

        # Preprocess into our format
        scene_graphs = []
        skip_reasons = {
            "no_image_id": 0,
            "no_dimensions": 0,
            "image_not_found": 0,
            "json_decode_error": 0,
            "no_objects_after_parse": 0,
            "other_error": 0,
        }

        # Debug: Print first row structure
        if len(dataset) > 0:
            first_row = dataset[0]
            print(f"\nDEBUG - First VG row:")
            print(f"  image_id: {first_row.get('image_id')}")
            print(f"  Available keys: {list(first_row.keys())}")
            print(f"  metadata/height: {first_row.get('metadata/height')}")
            print(f"  metadata/width: {first_row.get('metadata/width')}")
            print(f"  regions count: {len(first_row.get('regions', []))}")
            sg = first_row.get("scene_graph", "")
            print(f"  scene_graph length: {len(sg)} chars")
            print(f"  scene_graph preview: {sg[:200]}...")

            # Debug path construction
            test_img_id = first_row.get("image_id", "")
            test_path = self.vg_image_root / test_img_id
            print(f"\nDEBUG - Path construction:")
            print(f"  VG_IMAGE_ROOT: {self.vg_image_root}")
            print(f"  image_id: {test_img_id}")
            print(f"  Constructed path: {test_path}")
            print(f"  Path exists: {test_path.exists()}")

            # List first 10 files in VG_IMAGE_ROOT
            if self.vg_image_root.exists():
                files = sorted(list(self.vg_image_root.glob("*.jpg")))[:10]
                print(f"  First 10 files in VG_IMAGE_ROOT: {[f.name for f in files]}")
            else:
                print(f"  WARNING: VG_IMAGE_ROOT does not exist!")

            # Debug regions structure
            if len(first_row.get("regions", [])) > 0:
                first_region = first_row["regions"][0]
                print(f"\nDEBUG - First region structure:")
                print(f"  Type: {type(first_region)}")
                if isinstance(first_region, dict):
                    print(f"  Keys: {list(first_region.keys())}")
                    print(f"  bbox: {first_region.get('bbox')}")
                    print(f"  Full region: {first_region}")
                else:
                    print(f"  Value: {first_region}")

        for row in tqdm(dataset, desc="Preprocessing SVG dataset"):
            try:
                # Get image_id
                image_id = row.get("image_id", "")
                if not image_id:
                    skip_reasons["no_image_id"] += 1
                    continue

                # FIXED: Try multiple ways to get dimensions
                # After filtering, flattened columns might not work
                img_width = row.get("metadata/width")
                img_height = row.get("metadata/height")

                # If None, try getting from image file directly
                if img_width is None or img_height is None:
                    img_path = self.vg_image_root / image_id
                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img_height, img_width = img.shape[:2]
                        else:
                            skip_reasons["no_dimensions"] += 1
                            continue
                    else:
                        skip_reasons["image_not_found"] += 1
                        continue

                if img_width == 0 or img_height == 0:
                    skip_reasons["no_dimensions"] += 1
                    continue

                # Check if image exists (if we haven't already)
                img_path = self.vg_image_root / image_id
                if not img_path.exists():
                    skip_reasons["image_not_found"] += 1
                    continue

                # Parse scene_graph JSON STRING
                raw_scene_graph = row.get("scene_graph", "{}")
                if isinstance(raw_scene_graph, str):
                    try:
                        scene_graph_json = json.loads(raw_scene_graph)
                    except json.JSONDecodeError as e:
                        skip_reasons["json_decode_error"] += 1
                        continue
                else:
                    scene_graph_json = raw_scene_graph

                # Get regions (already a list of dicts)
                regions = row.get("regions", [])
                if not isinstance(regions, list):
                    regions = []

                # Build our scene graph structure
                scene_graph = {
                    "image_id": image_id,
                    "image": image_id,
                    "source": "visual_genome",
                    "width": img_width,
                    "height": img_height,
                    "objects": [],
                    "relationships": [],
                }

                # Parse objects from scene_graph JSON
                # CRITICAL FIX: SVG objects are STRINGS, not dicts!
                # Format: {"objects": ["description1", "description2", ...]}
                sg_objects = scene_graph_json.get("objects", [])

                for idx, obj_description in enumerate(sg_objects):
                    # obj_description is a STRING like "green clock on tall pole"
                    # Create object entry
                    obj_entry = {
                        "object_id": idx,  # Use index as ID
                        "name": (
                            obj_description
                            if isinstance(obj_description, str)
                            else str(obj_description)
                        ),
                    }

                    # Get bbox and segmentation from regions (regions[idx] corresponds to objects[idx])
                    if idx < len(regions):
                        region = regions[idx]
                        # FIXED: Ensure region is a dict
                        if not isinstance(region, dict):
                            continue

                        bbox = region.get("bbox", [])
                        if isinstance(bbox, list) and len(bbox) == 4:
                            obj_entry["bbox"] = bbox
                        else:
                            # Try alternative bbox field names
                            if "x" in region and "y" in region:
                                x = region.get("x", 0)
                                y = region.get("y", 0)
                                w = region.get("width", region.get("w", 0))
                                h = region.get("height", region.get("h", 0))
                                if w > 0 and h > 0:
                                    obj_entry["bbox"] = [x, y, w, h]

                        # Store segmentation mask for visualization
                        if "segmentation" in region:
                            obj_entry["segmentation"] = region["segmentation"]

                    # Skip objects without bbox
                    if "bbox" not in obj_entry:
                        continue

                    scene_graph["objects"].append(obj_entry)

                # Parse relations from scene_graph JSON
                sg_relations = scene_graph_json.get("relations", [])

                for rel in sg_relations:
                    if isinstance(rel, list) and len(rel) >= 3:
                        subj_idx, obj_idx, predicate = rel[0], rel[1], rel[2]

                        # Map indices to object_ids
                        if subj_idx < len(scene_graph["objects"]) and obj_idx < len(
                            scene_graph["objects"]
                        ):
                            relationship = {
                                "subject_id": scene_graph["objects"][subj_idx][
                                    "object_id"
                                ],
                                "object_id": scene_graph["objects"][obj_idx][
                                    "object_id"
                                ],
                                "predicate": (
                                    predicate
                                    if isinstance(predicate, str)
                                    else str(predicate)
                                ),
                                "predicate_category": self._infer_predicate_category(
                                    predicate
                                    if isinstance(predicate, str)
                                    else str(predicate)
                                ),
                            }
                            scene_graph["relationships"].append(relationship)

                # Only add if we have objects
                if len(scene_graph["objects"]) > 0:
                    scene_graphs.append(scene_graph)
                else:
                    skip_reasons["no_objects_after_parse"] += 1

            except Exception as e:
                skip_reasons["other_error"] += 1
                # Print first 5 errors with details
                if skip_reasons["other_error"] <= 5:
                    print(
                        f"\nError #{skip_reasons['other_error']} - image_id: {image_id}"
                    )
                    print(f"  Exception: {type(e).__name__}: {e}")
                    import traceback

                    traceback.print_exc()
                continue

        total_skipped = sum(skip_reasons.values())
        print(
            f"Successfully preprocessed {len(scene_graphs)} scene graphs (skipped {total_skipped})"
        )
        print(f"Skip reasons breakdown:")
        for reason, count in skip_reasons.items():
            if count > 0:
                print(f"  - {reason}: {count}")

        # Save to cache
        if len(scene_graphs) > 0:
            self.svg_cache.save(scene_graphs)

        return scene_graphs

    def _infer_predicate_category(self, predicate: str) -> str:
        """
        Infer predicate category from predicate text.
        Used when predicate_category is not provided in dataset.

        Args:
            predicate: Predicate string (e.g., "holding", "on", "near")

        Returns:
            Category string: "interactional", "functional", "social", "emotional", or "spatial"
        """
        pred_lower = predicate.lower()

        # Interaction predicates
        interaction_words = {
            "holding",
            "carrying",
            "wearing",
            "reading",
            "eating",
            "drinking",
            "sitting on",
            "standing on",
            "riding",
            "using",
            "playing",
            "looking at",
            "touching",
            "hugging",
            "kissing",
            "petting",
            "cutting",
            "cooking",
            "writing",
            "typing",
            "painting",
        }
        if pred_lower in interaction_words:
            return "interactional"

        # Functional predicates
        functional_words = {
            "has",
            "belongs to",
            "part of",
            "attached to",
            "connected to",
            "made of",
            "contains",
            "holds",
            "supports",
        }
        if pred_lower in functional_words:
            return "functional"

        # Social/emotional predicates
        social_words = {
            "with",
            "talking to",
            "watching",
            "smiling at",
            "laughing with",
            "next to person",
            "beside person",
            "group",
            "family",
            "friends",
        }
        if pred_lower in social_words or "person" in pred_lower:
            return "social"

        # Default to spatial
        return "spatial"

    def get_memorability(self, image_path: Path, image_id: str) -> float:
        """
        Get memorability score with caching.

        Args:
            image_path: Path to image
            image_id: Unique image identifier for cache key

        Returns:
            Memorability score (0-1)
        """
        # Check cache first
        cached_score = self.memorability_cache.get(image_id)
        if cached_score is not None:
            return cached_score

        # Compute with ResMem
        try:
            # Load image with PIL (ResMem expects PIL Image)
            img = Image.open(image_path).convert("RGB")

            # Apply transformer (preprocessing)
            image_x = transformer(img)

            # Reshape to batch format and run inference
            # Shape: (1, 3, 227, 227)
            with torch.no_grad():
                prediction = self.resmem(image_x.view(-1, 3, 227, 227))

            # FIXED: Handle ResMem output properly
            # Output might be multi-dimensional, take mean like in examples
            if prediction.dim() > 1:
                score = prediction.view(-1).mean().item()
            else:
                score = prediction.item()

            # Ensure score is in [0, 1] range
            score = max(0.0, min(1.0, score))

        except Exception as e:
            print(f"Warning: ResMem prediction failed for {image_id}: {e}")
            import traceback

            traceback.print_exc()
            return 0.0

        # Cache the result
        self.memorability_cache.set(image_id, score)

        return float(score)

    def _calculate_mask_area(self, obj: Dict, img_width: int, img_height: int) -> float:
        """Calculate object area as percentage of image."""
        img_area = img_width * img_height

        if "polygon" in obj:
            polygon = np.array(obj["polygon"]).reshape(-1, 2)
            x = polygon[:, 0]
            y = polygon[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            return (area / img_area) * 100

        elif "bbox" in obj:
            x, y, w, h = obj["bbox"]
            area = w * h
            return (area / img_area) * 100

        return 0.0

    def _compute_relational_graph(
        self, valid_objects: List[Dict], relations: List[Dict]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute weighted relational graph using predicate categories.
        FIXED: Now builds adjacency only for valid_objects to avoid index mismatch.

        Args:
            valid_objects: Filtered list of objects (only those passing area threshold)
            relations: List of relationships from scene graph

        Returns:
            adjacency_matrix: NxN weighted adjacency matrix for valid objects
            object_id_map: Mapping from object IDs to indices in valid_objects
        """
        n_objects = len(valid_objects)
        adjacency = np.zeros((n_objects, n_objects))

        # Create object ID to index mapping FOR VALID OBJECTS ONLY
        object_id_map = {obj["object_id"]: idx for idx, obj in enumerate(valid_objects)}
        valid_object_ids = set(object_id_map.keys())

        # Process relations using predicate_category
        for rel in relations:
            subj_id = rel.get("subject_id")
            obj_id = rel.get("object_id")
            predicate_category = rel.get("predicate_category", "spatial")

            # FIXED: Only include relations between valid objects
            if subj_id not in valid_object_ids or obj_id not in valid_object_ids:
                continue

            subj_idx = object_id_map[subj_id]
            obj_idx = object_id_map[obj_id]

            # Get weight from config based on category
            weight = self.predicate_weights.get(predicate_category, 0.3)

            # Add edge (accumulate if multiple relations)
            adjacency[subj_idx, obj_idx] += weight
            # Make undirected
            adjacency[obj_idx, subj_idx] += weight

        # Normalize to [0, 1]
        max_weight = adjacency.max()
        if max_weight > 0:
            adjacency = adjacency / max_weight

        return adjacency, object_id_map

    def _calculate_coverage(
        self, objects: List[Dict], img_width: int, img_height: int
    ) -> float:
        """Calculate coverage of image by objects."""
        coverage_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        for obj in objects:
            if "polygon" in obj:
                polygon = np.array(obj["polygon"]).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(coverage_mask, [polygon], 1)
            elif "bbox" in obj:
                x, y, w, h = obj["bbox"]
                coverage_mask[int(y) : int(y + h), int(x) : int(x + w)] = 1

        covered_pixels = np.sum(coverage_mask > 0)
        total_pixels = img_width * img_height
        coverage_percent = (covered_pixels / total_pixels) * 100

        return coverage_percent

    def filter_images(self) -> List[Dict]:
        """
        Filter SVG images based on config criteria.

        Returns:
            List of selected image metadata with scene graphs
        """
        print("\nFiltering SVG images...")
        print(f"Criteria:")
        print(f"  - Source: {self.source_filter}")
        print(f"  - Memorability: >= {self.min_memorability}")
        print(
            f"  - Objects: {self.min_objects}-{self.max_objects} (>= {self.min_mask_area_percent}% area)"
        )
        print(f"  - Relations: >= {self.min_relations}")
        print(f"  - Coverage: >= {self.min_coverage_percent}%")

        scene_graphs = self._load_svg_metadata()
        candidate_images = []

        stats = {
            "total": len(scene_graphs),
            "wrong_source": 0,
            "low_memorability": 0,
            "too_few_objects": 0,
            "too_many_objects": 0,
            "too_few_relations": 0,
            "low_coverage": 0,
            "missing_image": 0,
        }

        for scene_graph in tqdm(scene_graphs, desc="Scanning images"):
            # FIXED: Source filter - SVG is already VG-derived, so skip this check
            # if self.source_filter is set to empty string or if source matches
            if self.source_filter:
                source = scene_graph.get("source", "").lower()
                if source and self.source_filter.lower() not in source:
                    stats["wrong_source"] += 1
                    continue

            # Get image path
            img_filename = scene_graph.get("image", "")
            img_id = scene_graph.get("image_id", img_filename)
            img_path = self.vg_image_root / img_filename

            if not img_path.exists():
                stats["missing_image"] += 1
                continue

            # Check memorability (with caching)
            memorability = self.get_memorability(img_path, img_id)
            if memorability < self.min_memorability:
                stats["low_memorability"] += 1
                continue

            # Get image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                stats["missing_image"] += 1
                continue
            img_height, img_width = img.shape[:2]

            # Filter objects by mask area
            objects = scene_graph.get("objects", [])
            valid_objects = []

            for obj in objects:
                area_percent = self._calculate_mask_area(obj, img_width, img_height)
                if area_percent >= self.min_mask_area_percent:
                    obj["area_percent"] = area_percent
                    valid_objects.append(obj)

            # Check object count
            n_objects = len(valid_objects)
            if n_objects < self.min_objects:
                stats["too_few_objects"] += 1
                continue
            if n_objects > self.max_objects:
                stats["too_many_objects"] += 1
                continue

            # Check relation count
            relations = scene_graph.get("relationships", [])
            if len(relations) < self.min_relations:
                stats["too_few_relations"] += 1
                continue

            # Check coverage
            coverage = self._calculate_coverage(valid_objects, img_width, img_height)
            if coverage < self.min_coverage_percent:
                stats["low_coverage"] += 1
                continue

            # FIXED: Compute relational graph for valid_objects only
            adjacency, object_id_map = self._compute_relational_graph(
                valid_objects, relations
            )

            # Passed all filters
            candidate_images.append(
                {
                    "scene_graph": scene_graph,
                    "image_path": img_path,
                    "img_width": img_width,
                    "img_height": img_height,
                    "objects": valid_objects,
                    "relations": relations,
                    "adjacency_matrix": adjacency,
                    "object_id_map": object_id_map,
                    "memorability": memorability,
                    "coverage_percent": coverage,
                    "n_objects": n_objects,
                    "n_relations": len(relations),
                }
            )

        # Save memorability cache
        self.memorability_cache.save()

        # Print statistics
        print(f"\nFiltering Statistics:")
        print(f"  Total images: {stats['total']}")
        print(f"  ✗ Wrong source: {stats['wrong_source']}")
        print(f"  ✗ Low memorability: {stats['low_memorability']}")
        print(f"  ✗ Too few objects: {stats['too_few_objects']}")
        print(f"  ✗ Too many objects: {stats['too_many_objects']}")
        print(f"  ✗ Too few relations: {stats['too_few_relations']}")
        print(f"  ✗ Low coverage: {stats['low_coverage']}")
        print(f"  ✗ Missing image: {stats['missing_image']}")
        print(f"  ✓ Selected: {len(candidate_images)}")

        return candidate_images

    def _letterbox_image(
        self, img: np.ndarray, target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox image while maintaining aspect ratio."""
        h, w = img.shape[:2]
        target_w, target_h = target_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        letterboxed = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        letterboxed[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return letterboxed, scale, (x_offset, y_offset)

    def _transform_objects(
        self, objects: List[Dict], scale: float, offset: Tuple[int, int]
    ) -> List[Dict]:
        """Transform object coordinates for letterboxed image."""
        x_off, y_off = offset
        transformed = []

        for obj in objects:
            obj_new = obj.copy()

            if "polygon" in obj_new:
                polygon = np.array(obj_new["polygon"]).reshape(-1, 2)
                polygon = polygon * scale + [x_off, y_off]
                obj_new["polygon"] = polygon.flatten().tolist()

            if "bbox" in obj_new:
                x, y, w, h = obj_new["bbox"]
                obj_new["bbox"] = [
                    x * scale + x_off,
                    y * scale + y_off,
                    w * scale,
                    h * scale,
                ]

            transformed.append(obj_new)

        return transformed

    def visualize_image(self, img_data: Dict) -> None:
        """Create visualization showing objects and relational graph."""
        scene_graph = img_data["scene_graph"]
        img_id = scene_graph.get("image_id", "unknown")

        img = cv2.imread(str(img_data["image_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        fig.suptitle(
            f"Image {img_id}\n"
            f"Objects: {img_data['n_objects']} | Relations: {img_data['n_relations']} | "
            f"Coverage: {img_data['coverage_percent']:.1f}% | "
            f"Memorability: {img_data['memorability']:.2f}",
            fontsize=14,
            fontweight="bold",
        )

        # Plot 1: Original with objects
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        self._draw_objects(ax1, img_data["objects"], "Objects & Labels")

        # Plot 2: Relational graph overlay
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img)
        self._draw_relational_graph(
            ax2,
            img_data["objects"],
            img_data["adjacency_matrix"],
            "Relational Graph (edge thickness = strength)",
        )

        # Plot 3: Letterboxed
        letterboxed, scale, offset = self._letterbox_image(img, self.target_size)
        transformed_objects = self._transform_objects(
            img_data["objects"], scale, offset
        )

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(letterboxed)
        self._draw_objects(
            ax3,
            transformed_objects,
            f"Letterboxed to {self.target_size[0]}×{self.target_size[1]}",
        )

        # Plot 4: Adjacency matrix
        ax4 = fig.add_subplot(gs[1, 1])
        self._draw_adjacency_matrix(
            ax4, img_data["adjacency_matrix"], "Adjacency Matrix (Relation Strengths)"
        )

        plt.tight_layout()

        output_path = self.output_dir / "visualizations" / f"{img_id}_viz.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _draw_objects(self, ax: plt.Axes, objects: List[Dict], title: str) -> None:
        """Draw objects with labels."""
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

        for idx, obj in enumerate(objects):
            color = self.colors[idx % len(self.colors)]

            if "polygon" in obj:
                polygon = np.array(obj["polygon"]).reshape(-1, 2)
                patch = mpatches.Polygon(
                    polygon,
                    closed=True,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=(*color, 0.3),
                )
                ax.add_patch(patch)

                centroid = polygon.mean(axis=0)
                label = obj.get("name", f"obj_{idx}")
                ax.text(
                    centroid[0],
                    centroid[1],
                    label,
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    color="white",
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

            elif "bbox" in obj:
                x, y, w, h = obj["bbox"]
                rect = mpatches.Rectangle(
                    (x, y),
                    w,
                    h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=(*color, 0.3),
                )
                ax.add_patch(rect)

                label = obj.get("name", f"obj_{idx}")
                ax.text(
                    x + w / 2,
                    y + h / 2,
                    label,
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    color="white",
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

    def _draw_relational_graph(
        self, ax: plt.Axes, objects: List[Dict], adjacency: np.ndarray, title: str
    ) -> None:
        """Draw relational graph as overlay."""
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

        # Get centroids
        centroids = []
        for obj in objects:
            if "polygon" in obj:
                polygon = np.array(obj["polygon"]).reshape(-1, 2)
                centroid = polygon.mean(axis=0)
            elif "bbox" in obj:
                x, y, w, h = obj["bbox"]
                centroid = np.array([x + w / 2, y + h / 2])
            else:
                centroid = np.array([0, 0])
            centroids.append(centroid)

        centroids = np.array(centroids)

        # Draw edges
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                weight = adjacency[i, j]
                if weight > 0:
                    linewidth = 0.5 + weight * 3
                    alpha = 0.3 + weight * 0.5

                    ax.plot(
                        [centroids[i, 0], centroids[j, 0]],
                        [centroids[i, 1], centroids[j, 1]],
                        color="red",
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=1,
                    )

        # Draw nodes
        for idx, centroid in enumerate(centroids):
            ax.plot(
                centroid[0],
                centroid[1],
                "o",
                color=self.colors[idx % len(self.colors)],
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=2,
                zorder=2,
            )

    def _draw_adjacency_matrix(
        self, ax: plt.Axes, adjacency: np.ndarray, title: str
    ) -> None:
        """Draw adjacency matrix heatmap."""
        ax.set_title(title, fontsize=12, fontweight="bold")
        im = ax.imshow(adjacency, cmap="hot", vmin=0, vmax=1)
        ax.set_xlabel("Object Index")
        ax.set_ylabel("Object Index")
        plt.colorbar(im, ax=ax, label="Relation Strength")

    def _draw_edge_statistics(
        self, ax: plt.Axes, adjacency: np.ndarray, relations: List[Dict]
    ) -> None:
        """Draw edge weight distribution and relation type statistics."""
        ax.set_title("Relational Statistics", fontsize=12, fontweight="bold")

        # Get non-zero edge weights
        edge_weights = adjacency[adjacency > 0]

        # Count relation types
        category_counts = Counter(rel["predicate_category"] for rel in relations)

        # Create text summary
        stats_text = []
        stats_text.append(f"Total Relations: {len(relations)}")
        stats_text.append(f"Unique Edges: {len(edge_weights)}")
        stats_text.append(f"Avg Edge Strength: {edge_weights.mean():.3f}")
        stats_text.append(f"Max Edge Strength: {edge_weights.max():.3f}")
        stats_text.append("")
        stats_text.append("Relation Types:")

        for category, count in category_counts.most_common():
            weight = self.predicate_weights.get(category, 0.0)
            stats_text.append(f"  {category}: {count} (w={weight})")

        # Display as text
        ax.text(
            0.05,
            0.95,
            "\n".join(stats_text),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Add a small histogram of edge weights
        from matplotlib.patches import Rectangle

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Histogram in bottom half
        if len(edge_weights) > 0:
            bins = np.linspace(0, 1, 11)
            hist, bin_edges = np.histogram(edge_weights, bins=bins)
            max_count = hist.max() if hist.max() > 0 else 1

            for i, (count, left_edge) in enumerate(zip(hist, bin_edges[:-1])):
                width = bin_edges[1] - bin_edges[0]
                height = (count / max_count) * 0.3
                rect = Rectangle(
                    (left_edge, 0.05),
                    width,
                    height,
                    facecolor="steelblue",
                    edgecolor="black",
                    alpha=0.7,
                )
                ax.add_patch(rect)

            ax.text(
                0.5,
                0.02,
                "Edge Strength Distribution",
                transform=ax.transAxes,
                ha="center",
                fontsize=9,
            )

        ax.set_xticks([])
        ax.set_yticks([])

    def create_summary_statistics(self, selected_images: List[Dict]) -> None:
        """Create summary statistics visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Dataset Summary: {len(selected_images)} Images",
            fontsize=16,
            fontweight="bold",
        )

        # Object count
        ax = axes[0, 0]
        counts = [img["n_objects"] for img in selected_images]
        ax.hist(counts, bins=20, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Number of Objects")
        ax.set_ylabel("Frequency")
        ax.set_title("Object Count Distribution")
        ax.axvline(
            np.mean(counts),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(counts):.1f}",
        )
        ax.legend()

        # Relations count
        ax = axes[0, 1]
        rel_counts = [img["n_relations"] for img in selected_images]
        ax.hist(rel_counts, bins=20, edgecolor="black", alpha=0.7, color="green")
        ax.set_xlabel("Number of Relations")
        ax.set_ylabel("Frequency")
        ax.set_title("Relations Count Distribution")
        ax.axvline(
            np.mean(rel_counts),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(rel_counts):.1f}",
        )
        ax.legend()

        # Coverage
        ax = axes[0, 2]
        coverages = [img["coverage_percent"] for img in selected_images]
        ax.hist(coverages, bins=20, edgecolor="black", alpha=0.7, color="orange")
        ax.set_xlabel("Coverage (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Coverage Distribution")
        ax.axvline(
            np.mean(coverages),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(coverages):.1f}%",
        )
        ax.legend()

        # Memorability
        ax = axes[1, 0]
        mems = [img["memorability"] for img in selected_images]
        ax.hist(mems, bins=20, edgecolor="black", alpha=0.7, color="purple")
        ax.set_xlabel("Memorability Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Memorability Distribution")
        ax.axvline(
            self.min_memorability,
            color="red",
            linestyle="--",
            label=f"Threshold: {self.min_memorability}",
        )
        ax.legend()

        # Relational density
        ax = axes[1, 1]
        ratios = [img["n_relations"] / img["n_objects"] for img in selected_images]
        ax.hist(ratios, bins=20, edgecolor="black", alpha=0.7, color="cyan")
        ax.set_xlabel("Relations per Object")
        ax.set_ylabel("Frequency")
        ax.set_title("Relational Density")
        ax.axvline(
            np.mean(ratios),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(ratios):.1f}",
        )
        ax.legend()

        # Edge weights
        ax = axes[1, 2]
        all_weights = []
        for img in selected_images:
            adj = img["adjacency_matrix"]
            weights = adj[adj > 0]
            all_weights.extend(weights)

        ax.hist(all_weights, bins=30, edgecolor="black", alpha=0.7, color="brown")
        ax.set_xlabel("Edge Weight")
        ax.set_ylabel("Frequency")
        ax.set_title("Relational Edge Strength Distribution")
        ax.axvline(
            np.mean(all_weights),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(all_weights):.2f}",
        )
        ax.legend()

        plt.tight_layout()

        output_path = self.output_dir / "summary_statistics.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\nSaved summary statistics: {output_path}")

    def process_dataset(self) -> None:
        """Main processing pipeline."""
        # Filter images
        selected_images = self.filter_images()

        if len(selected_images) == 0:
            print("No images met the criteria!")
            return

        # Create summary statistics
        print("\nGenerating summary statistics...")
        self.create_summary_statistics(selected_images)

        # Visualize sample images
        n_samples = min(config.VISUALIZE_SAMPLES, len(selected_images))
        print(f"\nGenerating {n_samples} example visualizations...")
        sample_indices = np.linspace(0, len(selected_images) - 1, n_samples, dtype=int)

        for idx in tqdm(sample_indices, desc="Creating visualizations"):
            self.visualize_image(selected_images[idx])

        # Process and save all images
        print("\nProcessing all selected images...")
        dataset_metadata = []

        for img_data in tqdm(selected_images, desc="Processing images"):
            scene_graph = img_data["scene_graph"]
            img_id = scene_graph.get("image_id", "unknown")

            # Load image
            img = cv2.imread(str(img_data["image_path"]))

            # Letterbox
            letterboxed, scale, offset = self._letterbox_image(img, self.target_size)

            # Transform objects
            transformed_objects = self._transform_objects(
                img_data["objects"], scale, offset
            )

            # Save letterboxed image
            output_img_path = self.output_dir / "images" / f"{img_id}.jpg"
            cv2.imwrite(str(output_img_path), letterboxed)

            # Prepare metadata
            metadata = {
                "image_id": img_id,
                "file_name": f"{img_id}.jpg",
                "original_file": scene_graph.get("image", ""),
                "width": self.target_size[0],
                "height": self.target_size[1],
                "original_width": img_data["img_width"],
                "original_height": img_data["img_height"],
                "scale": scale,
                "offset": offset,
                "memorability": img_data["memorability"],
                "coverage_percent": img_data["coverage_percent"],
                "n_objects": img_data["n_objects"],
                "n_relations": img_data["n_relations"],
                "objects": transformed_objects,
                "relations": img_data["relations"],
                "adjacency_matrix": img_data["adjacency_matrix"].tolist(),
                "scene_graph": scene_graph,
            }
            dataset_metadata.append(metadata)

        # Save dataset metadata
        metadata_path = self.output_dir / "annotations" / "dataset.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "info": {
                        "description": "SVG Relational Stimuli for Gaze-Based Memory Experiments",
                        "date_created": "2025-12-05",
                        "source_filter": self.source_filter,
                        "min_memorability": self.min_memorability,
                        "min_mask_area_percent": self.min_mask_area_percent,
                        "min_objects": self.min_objects,
                        "max_objects": self.max_objects,
                        "min_relations": self.min_relations,
                        "min_coverage_percent": self.min_coverage_percent,
                        "target_size": self.target_size,
                        "predicate_weights": self.predicate_weights,
                        "num_images": len(dataset_metadata),
                    },
                    "images": dataset_metadata,
                },
                f,
                indent=2,
            )

        print(f"\n✓ Processing complete!")
        print(f"  Images saved: {self.output_dir / 'images'}")
        print(f"  Visualizations: {self.output_dir / 'visualizations'}")
        print(f"  Metadata: {metadata_path}")
        print(f"\nKey features:")
        print(f"  ✓ Source: {self.source_filter}")
        print(f"  ✓ {self.min_objects}-{self.max_objects} objects per image")
        print(f"  ✓ Minimum {self.min_relations} relations per image")
        print(f"  ✓ Memorability >= {self.min_memorability}")
        print(f"  ✓ Category-based predicate weighting")
        print(f"  ✓ Memorability caching enabled")
        print(f"  ✓ Ready for gaze-relational memory analysis")


def main():
    """Main entry point."""
    creator = SVGRelationalDataset()
    creator.process_dataset()


if __name__ == "__main__":
    main()
