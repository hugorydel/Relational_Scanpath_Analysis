"""
Scene graph loading utilities for SVG dataset
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from datasets import load_dataset
from pycocotools import mask as mask_utils
from tqdm import tqdm

from utils import infer_predicate_category


class SceneGraphLoader:
    """Handles loading scene graphs from HuggingFace SVG dataset."""

    def __init__(self, vg_image_root: Path):
        self.vg_image_root = vg_image_root

    def build_scene_graph_lookup(
        self, passing_image_ids: List[Tuple[str, Dict]]
    ) -> Dict[str, Dict]:
        """
        Build lookup for scene graphs of ONLY passing images.
        Memory-efficient: Only loads needed scene graphs.

        Args:
            passing_image_ids: List of (img_id, cached_stats) tuples

        Returns:
            Dict mapping img_id to scene_graph
        """
        needed_ids = set(img_id for img_id, _ in passing_image_ids)

        print(f"\nLoading {len(needed_ids)} scene graphs from HuggingFace...")

        # Load and filter dataset
        dataset = load_dataset("jamepark3922/svg", split="train")
        dataset = dataset.filter(self._is_vg_image)

        # Extract only needed scene graphs
        lookup = {}
        for row in tqdm(dataset, desc="Loading scene graphs", mininterval=1.0):
            img_id = row.get("image_id", "")

            if img_id in needed_ids:
                scene_graph = self._parse_scene_graph_row(row)
                if scene_graph:
                    lookup[img_id] = scene_graph

                # Early exit when all found
                if len(lookup) >= len(needed_ids):
                    break

        if len(lookup) < len(needed_ids):
            print(f"Warning: Found {len(lookup)}/{len(needed_ids)} scene graphs")

        return lookup

    def _is_vg_image(self, example: Dict) -> bool:
        """Check if image is from Visual Genome (numeric filename)."""
        img_id = example.get("image_id", "")
        if not img_id:
            return False
        filename_without_ext = img_id.replace(".jpg", "")
        return filename_without_ext.isdigit()

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

            # Parse relations
            sg_relations = scene_graph_json.get("relations", [])

            for rel in sg_relations:
                if isinstance(rel, list) and len(rel) >= 3:
                    subj_idx, obj_idx, predicate = rel[0], rel[1], rel[2]

                    if subj_idx < len(scene_graph["objects"]) and obj_idx < len(
                        scene_graph["objects"]
                    ):
                        relationship = {
                            "subject_id": scene_graph["objects"][subj_idx]["object_id"],
                            "object_id": scene_graph["objects"][obj_idx]["object_id"],
                            "predicate": (
                                predicate
                                if isinstance(predicate, str)
                                else str(predicate)
                            ),
                            "predicate_category": infer_predicate_category(
                                predicate
                                if isinstance(predicate, str)
                                else str(predicate)
                            ),
                        }
                        scene_graph["relationships"].append(relationship)

            if len(scene_graph["objects"]) > 0:
                return scene_graph

            return None

        except Exception:
            return None
