#!/usr/bin/env python3
"""
SVG Relational Dataset Creator - Memory-Optimized with Reference Data
Filters SVG images for gaze-based relational memory experiments

Uses pre-computed reference data (precomputed_stats.json) for efficient filtering.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

import config
from preprocessing.output_visualization import ImageVisualizer
from preprocessing.preprocess_functions import (
    compute_relational_graph,
    ensure_jpg,
    validate_paths,
)
from preprocessing.reference_data_loader import ReferenceDataLoader
from preprocessing.scene_graph_loader import SceneGraphLoader


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
        self.predicate_weights = config.PREDICATE_WEIGHTS

        # Setup output directories
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)

        # Load reference data
        print("Loading reference data...")
        self.reference_data = ReferenceDataLoader(Path(config.PRECOMPUTED_STATS_PATH))

        # Initialize scene graph loader
        self.scene_graph_loader = SceneGraphLoader(self.vg_image_root)

        # Initialize visualizer
        self.visualizer = ImageVisualizer(
            output_dir=self.output_dir,
            target_size=self.target_size,
            predicate_weights=self.predicate_weights,
        )

    def filter_images(self) -> List[Dict]:
        """
        Two-phase filtering using lazy loading for memory efficiency.

        Phase 1: Filter using pre-computed statistics only
        Phase 2: Load scene graphs for passing images only
        """
        print("=" * 60)
        print("FILTERING CRITERIA")
        print("=" * 60)
        print(f"  Memorability: ≥ {self.min_memorability}")
        print(f"  Objects: {self.min_objects}-{self.max_objects}")
        print(f"  Relations: ≥ {self.min_relations}")
        print(f"  Coverage: ≥ {self.min_coverage_percent}%")
        print("=" * 60 + "\n")

        stats = {
            "total": len(self.reference_data),
            "rejected_phase1": 0,
            "passed_phase1": 0,
            "rejected_phase2": 0,
            "passed_phase2": 0,
        }

        # PHASE 1: Filter using pre-computed statistics only
        print("[Phase 1/2] Filtering using pre-computed statistics...")
        passing_image_ids = []

        for img_id, ref_stats in tqdm(
            self.reference_data.items(), desc="Phase 1", mininterval=1.0
        ):
            # Validate entry has all required fields
            if not self.reference_data.validate_entry(img_id, ref_stats):
                stats["rejected_phase1"] += 1
                continue

            # Apply filter thresholds
            if not self._passes_phase1_filters(ref_stats):
                stats["rejected_phase1"] += 1
                continue

            passing_image_ids.append(img_id)
            stats["passed_phase1"] += 1

        print(
            f"✓ Phase 1: {stats['passed_phase1']}/{len(self.reference_data)} images passed\n"
        )

        if stats["passed_phase1"] == 0:
            print("No images passed Phase 1 filtering!")
            return []

        # PHASE 2: Load scene graphs and build final candidates
        print("[Phase 2/2] Loading scene graphs for passing images...")
        scene_graphs = self.scene_graph_loader.load_scene_graphs_for_images(
            passing_image_ids
        )

        candidate_images = []

        for img_id in tqdm(passing_image_ids, desc="Phase 2", mininterval=1.0):
            ref_stats = self.reference_data.get(img_id)
            scene_graph = scene_graphs.get(img_id)

            if scene_graph is None:
                stats["rejected_phase2"] += 1
                continue

            # Verify image file exists
            img_path = self.vg_image_root / scene_graph.get("image", img_id)
            if not img_path.exists():
                stats["rejected_phase2"] += 1
                continue

            # Filter objects by mask area (on-the-fly from scene graph)
            valid_objects = self._filter_objects_by_area(scene_graph["objects"])

            # Filter relations to only those between valid objects
            valid_object_ids = set(obj["object_id"] for obj in valid_objects)
            filtered_relations = [
                rel
                for rel in scene_graph["relationships"]
                if rel["subject_id"] in valid_object_ids
                and rel["object_id"] in valid_object_ids
            ]

            # Compute relational graph
            adjacency, object_id_map = compute_relational_graph(
                valid_objects, filtered_relations, self.predicate_weights
            )

            # Build candidate data structure
            candidate_images.append(
                {
                    "scene_graph": scene_graph,
                    "image_path": img_path,
                    "img_width": ref_stats["img_width"],
                    "img_height": ref_stats["img_height"],
                    "objects": valid_objects,
                    "relations": filtered_relations,
                    "adjacency_matrix": adjacency,
                    "object_id_map": object_id_map,
                    "memorability": ref_stats["memorability"],
                    "coverage_percent": ref_stats["coverage_percent"],
                    "n_objects": ref_stats["n_objects"],
                    "n_relations": ref_stats["n_relations"],
                }
            )
            stats["passed_phase2"] += 1

        print(f"✓ Phase 2: {stats['passed_phase2']} final candidates\n")

        # Print filtering summary
        print("=" * 60)
        print("FILTERING SUMMARY")
        print("=" * 60)
        print(f"Total images in reference data: {len(self.reference_data)}")
        print(f"Passed Phase 1 (statistics): {stats['passed_phase1']}")
        print(f"Passed Phase 2 (scene graphs): {stats['passed_phase2']}")
        print(f"Final candidates: {len(candidate_images)}")
        print("=" * 60 + "\n")

        return candidate_images

    def _passes_phase1_filters(self, stats: Dict) -> bool:
        """
        Check if image passes Phase 1 filter criteria using pre-computed stats.

        All required data is in the reference data, so this is fast.
        """
        return (
            self.min_objects <= stats["n_objects"] <= self.max_objects
            and stats["n_relations"] >= self.min_relations
            and stats["coverage_percent"] >= self.min_coverage_percent
            and stats["memorability"] >= self.min_memorability
        )

    def _filter_objects_by_area(self, objects: List[Dict]) -> List[Dict]:
        """
        Filter objects by minimum mask area.

        This recreates the filtering that was done when building reference data,
        ensuring consistency with the pre-computed counts.
        """
        valid_objects = []

        for obj in objects:
            # Must have bbox
            if "bbox" not in obj:
                continue

            bbox = obj["bbox"]
            if len(bbox) != 4:
                continue

            x, y, w, h = bbox
            area_percent = (
                (w * h) / (obj.get("img_width", 800) * obj.get("img_height", 600)) * 100
            )

            # Apply area threshold
            if area_percent >= self.min_mask_area_percent:
                valid_objects.append(obj)

        return valid_objects

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
            file_name = ensure_jpg(img_id)

            # Load and letterbox image
            img = cv2.imread(str(img_data["image_path"]))
            letterboxed, scale, offset = self._letterbox_image(img, self.target_size)
            transformed_objects = self._transform_objects(
                img_data["objects"], scale, offset
            )

            # Save image
            output_img_path = self.output_dir / "images" / file_name
            cv2.imwrite(str(output_img_path), letterboxed)

            # Build metadata
            metadata = {
                "image_id": img_id,
                "file_name": file_name,
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
    # Validate paths before expensive initialization
    print("=" * 60)
    print("SVG RELATIONAL DATASET CREATOR")
    print("=" * 60)
    print("Validating paths...\n")
    validate_paths(required_for_processing=True, required_for_diversity=False)
    print("✓ Path validation passed\n")

    # Proceed with dataset creation
    creator = SVGRelationalDataset()
    creator.process_dataset()


if __name__ == "__main__":
    main()
