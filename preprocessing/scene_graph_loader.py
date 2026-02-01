"""
Scene graph loading utilities for SVG dataset
Uses predicate→category mapping from SVG-Relations dataset
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
from datasets import load_dataset
from pycocotools import mask as mask_utils
from tqdm import tqdm


class SceneGraphLoader:
    """Handles loading scene graphs from HuggingFace SVG dataset."""

    def __init__(self, vg_image_root: Path, predicate_map_path: Path = None):
        self.vg_image_root = vg_image_root

        # Load predicate→category mapping
        if predicate_map_path is None:
            predicate_map_path = Path("data/predicate_to_category.json")

        if predicate_map_path.exists():
            with open(predicate_map_path, "r") as f:
                self.predicate_map = json.load(f)
            print(f"✓ Loaded {len(self.predicate_map)} predicate mappings")
        else:
            print(
                f"⚠️  Warning: {predicate_map_path} not found, using default 'spatial' category"
            )
            self.predicate_map = {}

    def load_scene_graphs_for_images(self, image_ids: List[str]) -> Dict[str, Dict]:
        """
        Load scene graphs for specific images only.
        Memory-efficient: Only loads needed scene graphs.

        Args:
            image_ids: List of image IDs (e.g., ["2345678.jpg", "1.jpg"])

        Returns:
            Dict mapping img_id to scene_graph
        """
        needed_ids = set(image_ids)

        print(f"Loading {len(needed_ids)} scene graphs from HuggingFace...")

        # Load and filter dataset
        dataset = load_dataset("jamepark3922/svg", split="train")
        dataset = dataset.filter(self._is_vg_image)

        # Extract only needed scene graphs
        scene_graphs = {}
        for row in tqdm(dataset, desc="Loading scene graphs", mininterval=1.0):
            img_id = row.get("image_id", "")

            if img_id in needed_ids:
                scene_graph = self._parse_scene_graph_row(row)
                if scene_graph:
                    scene_graphs[img_id] = scene_graph

                # Early exit when all found
                if len(scene_graphs) >= len(needed_ids):
                    break

        if len(scene_graphs) < len(needed_ids):
            missing = len(needed_ids) - len(scene_graphs)
            print(
                f"⚠️  Warning: Found {len(scene_graphs)}/{len(needed_ids)} scene graphs ({missing} missing)"
            )

        return scene_graphs

    def _is_vg_image(self, example: Dict) -> bool:
        """Check if image is from Visual Genome (numeric filename)."""
        img_id = example.get("image_id", "")
        if not img_id:
            return False
        filename_without_ext = img_id.replace(".jpg", "")
        return filename_without_ext.isdigit()

    def _get_predicate_category(self, predicate: str) -> str:
        """
        Get category for a predicate using SVG-Relations mapping.

        Args:
            predicate: Predicate string (e.g., "wearing", "on", "near")

        Returns:
            Category from SVG-Relations taxonomy (defaults to 'spatial')
        """
        return self.predicate_map.get(predicate, "spatial")

    def _parse_scene_graph_row(self, row: Dict) -> Optional[Dict]:
        """Parse a HuggingFace row into scene graph format."""
        try:
            image_id = row.get("image_id", "")
            if not image_id:
                return None

            # Get dimensions
            img_width = row.get("metadata/width")
            img_height = row.get("metadata/height")

            if img_width is None or img_height is None:
                img_path = self.vg_image_root / image_id
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                    else:
                        return None
                else:
                    return None

            if img_width == 0 or img_height == 0:
                return None

            # Parse scene_graph JSON
            raw_scene_graph = row.get("scene_graph", "{}")
            if isinstance(raw_scene_graph, str):
                try:
                    scene_graph_json = json.loads(raw_scene_graph)
                except json.JSONDecodeError:
                    return None
            else:
                scene_graph_json = raw_scene_graph

            regions = row.get("regions", [])
            if not isinstance(regions, list):
                regions = []

            # Build scene graph structure
            scene_graph = {
                "image_id": image_id,
                "image": image_id,
                "source": "visual_genome",
                "width": img_width,
                "height": img_height,
                "objects": [],
                "relationships": [],
            }

            # Parse objects
            sg_objects = scene_graph_json.get("objects", [])

            for idx, obj_description in enumerate(sg_objects):
                obj_entry = {
                    "object_id": idx,
                    "name": (
                        obj_description
                        if isinstance(obj_description, str)
                        else str(obj_description)
                    ),
                    "img_width": img_width,  # Add dimensions for area calculation
                    "img_height": img_height,
                }

                # Get bbox and segmentation from regions
                if idx < len(regions):
                    region = regions[idx]
                    if not isinstance(region, dict):
                        continue

                    bbox = region.get("bbox", [])
                    if isinstance(bbox, list) and len(bbox) == 4:
                        obj_entry["bbox"] = bbox
                    else:
                        if "x" in region and "y" in region:
                            x = region.get("x", 0)
                            y = region.get("y", 0)
                            w = region.get("width", region.get("w", 0))
                            h = region.get("height", region.get("h", 0))
                            if w > 0 and h > 0:
                                obj_entry["bbox"] = [x, y, w, h]

                    # Convert segmentation RLE to polygon
                    if "segmentation" in region:
                        seg = region["segmentation"]
                        if isinstance(seg, dict) and "counts" in seg:
                            try:
                                mask = mask_utils.decode(seg)
                                contours, _ = cv2.findContours(
                                    mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE,
                                )
                                if len(contours) > 0:
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    polygon = largest_contour.squeeze()
                                    if len(polygon.shape) == 2 and len(polygon) > 2:
                                        obj_entry["polygon"] = (
                                            polygon.flatten().tolist()
                                        )
                            except Exception:
                                pass

                if "bbox" not in obj_entry:
                    continue

                scene_graph["objects"].append(obj_entry)

            # Parse relations using predicate mapping
            sg_relations = scene_graph_json.get("relations", [])

            for rel in sg_relations:
                if isinstance(rel, list) and len(rel) >= 3:
                    subj_idx, obj_idx, predicate = rel[0], rel[1], rel[2]

                    if subj_idx < len(scene_graph["objects"]) and obj_idx < len(
                        scene_graph["objects"]
                    ):
                        predicate_str = (
                            predicate if isinstance(predicate, str) else str(predicate)
                        )

                        relationship = {
                            "subject_id": scene_graph["objects"][subj_idx]["object_id"],
                            "object_id": scene_graph["objects"][obj_idx]["object_id"],
                            "predicate": predicate_str,
                            "predicate_category": self._get_predicate_category(
                                predicate_str
                            ),
                        }
                        scene_graph["relationships"].append(relationship)

            if len(scene_graph["objects"]) > 0:
                return scene_graph

            return None

        except Exception as e:
            # Silently skip corrupted entries
            return None
