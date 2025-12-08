#!/usr/bin/env python3
"""
SVG Relational Dataset Creator - Memory-Optimized with Lazy Loading
Filters SVG images for gaze-based relational memory experiments
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from resmem import ResMem
from tqdm import tqdm

import config
from caching import MemorabilityCache, PrecomputedStatsCache
from scene_graph_loader import SceneGraphLoader
from utils import compute_relational_graph
from visualization import ImageVisualizer


class SVGRelationalDataset:
    """Filter and process SVG for relational gaze experiments."""

    def __init__(self):
        """Initialize dataset creator."""
        self.vg_image_root = Path(config.VG_IMAGE_ROOT)
        self.output_dir = Path(config.OUTPUT_DIR)
        self.target_size = config.TARGET_SIZE

        # Filtering thresholds
        self.min_memorability = config.MIN_MEMORABILITY
        self.min_mask_area_percent = config.MIN_MASK_AREA_PERCENT
        self.min_objects = config.MIN_OBJECTS
        self.max_objects = config.MAX_OBJECTS
        self.min_relations = config.MIN_RELATIONS
        self.min_coverage_percent = config.MIN_COVERAGE_PERCENT
        self.min_interactional_relations = config.MIN_INTERACTIONAL_RELATIONS
        self.predicate_weights = config.PREDICATE_WEIGHTS

        # Setup output directories
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)

        # Initialize caches
        self.memorability_cache = MemorabilityCache(
            Path(config.MEMORABILITY_CACHE_PATH)
        )
        self.precomputed_stats_cache = PrecomputedStatsCache(
            Path(config.PRECOMPUTED_STATS_CACHE_PATH)
        )

        # Initialize scene graph loader
        self.scene_graph_loader = SceneGraphLoader(self.vg_image_root)

        # Initialize visualizer
        self.visualizer = ImageVisualizer(
            output_dir=self.output_dir,
            target_size=self.target_size,
            predicate_weights=self.predicate_weights,
        )

        # Load ResMem
        print("Loading ResMem...")
        self.resmem = ResMem(pretrained=True)
        self.resmem.eval()
        print("✓ ResMem loaded\n")

    def filter_images(self) -> List[Dict]:
        """
        Filter images using lazy loading for memory efficiency.
        Phase 1: Filter using cached stats only
        Phase 2: Load scene graphs for passing images only
        """
        print("=" * 60)
        print("FILTERING CRITERIA")
        print("=" * 60)
        print(f"  Memorability: ≥ {self.min_memorability}")
        print(f"  Objects: {self.min_objects}-{self.max_objects}")
        print(f"  Relations: ≥ {self.min_relations}")
        print(f"  Coverage: ≥ {self.min_coverage_percent}%")
        print(f"  Interactional rels: ≥ {self.min_interactional_relations}")
        print("=" * 60 + "\n")

        stats = {
            "total": 0,
            "rejected": 0,
            "passed": 0,
        }

        # Phase 1: Filter using cache only
        print("[1/2] Filtering using cached stats...")
        passing_image_ids = []
        cache_items = list(self.precomputed_stats_cache.cache.items())
        stats["total"] = len(cache_items)

        for cache_key, cached_stats in tqdm(
            cache_items, desc="Filtering", mininterval=1.0
        ):
            img_id = cache_key.rsplit("_", 1)[0]

            # Apply filters
            if not self._passes_filters(cached_stats):
                stats["rejected"] += 1
                continue

            passing_image_ids.append((img_id, cached_stats))
            stats["passed"] += 1

        print(f"✓ {stats['passed']}/{stats['total']} images passed filters\n")

        # Phase 2: Load scene graphs for passing images
        print("[2/2] Loading scene graphs...")
        scene_graph_lookup = self.scene_graph_loader.build_scene_graph_lookup(
            passing_image_ids
        )

        # Build final candidates
        candidate_images = []
        for img_id, cached_stats in tqdm(
            passing_image_ids, desc="Building candidates", mininterval=1.0
        ):
            scene_graph = scene_graph_lookup.get(img_id)
            if scene_graph is None:
                continue

            img_path = self.vg_image_root / scene_graph.get("image", img_id)
            if not img_path.exists():
                continue

            adjacency, object_id_map = compute_relational_graph(
                cached_stats["valid_objects"],
                cached_stats["relations"],
                self.predicate_weights,
            )

            candidate_images.append(
                {
                    "scene_graph": scene_graph,
                    "image_path": img_path,
                    "img_width": cached_stats["img_width"],
                    "img_height": cached_stats["img_height"],
                    "objects": cached_stats["valid_objects"],
                    "relations": cached_stats["relations"],
                    "adjacency_matrix": adjacency,
                    "object_id_map": object_id_map,
                    "memorability": cached_stats["memorability"],
                    "coverage_percent": cached_stats["coverage_percent"],
                    "n_objects": cached_stats["n_objects"],
                    "n_relations": cached_stats["n_relations"],
                }
            )

        print(f"✓ {len(candidate_images)} final candidates\n")
        return candidate_images

    def _passes_filters(self, stats: Dict) -> bool:
        """Check if image passes all filter criteria."""

        relations = stats.get("relations", [])
        n_interactional = sum(
            1 for rel in relations if rel.get("predicate_category") == "interactional"
        )

        return (
            self.min_objects <= stats.get("n_objects", 0) <= self.max_objects
            and stats.get("n_relations", 0) >= self.min_relations
            and stats.get("coverage_percent", 0) >= self.min_coverage_percent
            and stats.get("memorability", 0) >= self.min_memorability
            and n_interactional >= self.min_interactional_relations
        )

    def process_dataset(self) -> None:
        """Main processing pipeline."""
        # Filter images
        selected_images = self.filter_images()

        if len(selected_images) == 0:
            print("No images met the criteria!")
            return

        # Create summary statistics
        print("Generating summary statistics...")
        self.visualizer.create_summary_statistics(selected_images)

        # Visualize samples
        n_samples = min(config.VISUALIZE_SAMPLES, len(selected_images))
        sample_indices = np.linspace(0, len(selected_images) - 1, n_samples, dtype=int)

        print(f"Creating {n_samples} visualizations...")
        for idx in tqdm(sample_indices, desc="Visualizing", mininterval=1.0):
            self.visualizer.visualize_image(selected_images[idx])

        # Process all images
        print("\nProcessing final images...")
        dataset_metadata = []

        for img_data in tqdm(selected_images, desc="Processing", mininterval=1.0):
            scene_graph = img_data["scene_graph"]
            img_id = scene_graph.get("image_id", "unknown")

            # Load and letterbox image
            img = cv2.imread(str(img_data["image_path"]))
            letterboxed, scale, offset = self._letterbox_image(img, self.target_size)
            transformed_objects = self._transform_objects(
                img_data["objects"], scale, offset
            )

            # Save image
            output_img_path = self.output_dir / "images" / f"{img_id}.jpg"
            cv2.imwrite(str(output_img_path), letterboxed)

            # Build metadata
            metadata = {
                "image_id": img_id,
                "file_name": f"{img_id}.jpg",
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

        # Save metadata
        metadata_path = self.output_dir / "annotations" / "dataset.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "info": {
                        "description": "SVG Relational Stimuli Dataset",
                        "num_images": len(dataset_metadata),
                        "min_memorability": self.min_memorability,
                        "min_objects": self.min_objects,
                        "max_objects": self.max_objects,
                        "min_relations": self.min_relations,
                        "min_coverage_percent": self.min_coverage_percent,
                    },
                    "images": dataset_metadata,
                },
                f,
                indent=2,
            )

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"✓ {len(selected_images)} images processed")
        print(f"✓ Images: {self.output_dir / 'images'}")
        print(f"✓ Visualizations: {self.output_dir / 'visualizations'}")
        print(f"✓ Metadata: {metadata_path}")
        print("=" * 60)

    def _letterbox_image(self, img: np.ndarray, target_size: tuple) -> tuple:
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
        self, objects: List[Dict], scale: float, offset: tuple
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


def main():
    """Main entry point."""
    creator = SVGRelationalDataset()
    creator.process_dataset()


if __name__ == "__main__":
    main()
