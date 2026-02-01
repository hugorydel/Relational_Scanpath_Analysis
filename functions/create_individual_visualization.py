#!/usr/bin/env python3
"""
create_individual_visualizations.py

Standalone script to create individual visualization components for each image.
Each image gets its own subfolder with separate files:
  - raw_image.jpg: Original image
  - segmentations_no_labels.png: Object masks/polygons overlayed (no labels)
  - relational_graph_overlay.png: Weighted relational graph overlayed
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

import config
from utils.misc import ensure_jpg


class IndividualVisualizer:
    """Creates separate visualization files for each image component."""

    def __init__(self, output_dir: Path, predicate_weights: Dict):
        self.output_dir = output_dir
        self.predicate_weights = predicate_weights

        # Generate color palette
        np.random.seed(42)
        self.colors = np.array(sns.color_palette("husl", 100))

    def create_visualizations(
        self, img_data: Dict, img_folder: Path, processed_images_dir: Path
    ) -> None:
        """
        Create all individual visualization files for an image.

        Args:
            img_data: Image metadata dictionary
            img_folder: Path to image's subfolder
            processed_images_dir: Path to processed images directory
        """
        img_id = img_data["image_id"]

        # Load from processed images (already letterboxed/transformed)
        processed_img_path = processed_images_dir / ensure_jpg(img_id)
        if not processed_img_path.exists():
            raise FileNotFoundError(f"Processed image not found: {processed_img_path}")

        img = cv2.imread(str(processed_img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert adjacency matrix from list to numpy array
        adjacency = np.array(img_data["adjacency_matrix"])

        # 1. Save raw image
        self._save_raw_image(img, img_folder)

        # 2. Save segmentations without labels
        self._save_segmentations_no_labels(img_rgb, img_data["objects"], img_folder)

        # 3. Save relational graph overlay
        self._save_relational_graph_overlay(
            img_rgb, img_data["objects"], adjacency, img_folder
        )

        # 4. Save relational graph only (no image background)
        self._save_relational_graph_only(
            img_rgb.shape, img_data["objects"], adjacency, img_folder
        )

    def _save_raw_image(self, img: np.ndarray, img_folder: Path) -> None:
        """Save the raw original image."""
        output_path = img_folder / "raw_image.jpg"
        cv2.imwrite(str(output_path), img)

    def _save_segmentations_no_labels(
        self, img: np.ndarray, objects: List[Dict], img_folder: Path
    ) -> None:
        """Save image with object segmentations overlayed (no labels)."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=150)
        ax.imshow(img)
        ax.axis("off")

        # Draw each object
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

        output_path = img_folder / "segmentations_no_labels.png"
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close()

    def _save_relational_graph_overlay(
        self,
        img: np.ndarray,
        objects: List[Dict],
        adjacency: np.ndarray,
        img_folder: Path,
    ) -> None:
        """Save image with weighted relational graph overlayed."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=150)
        ax.imshow(img)
        ax.axis("off")

        # Get centroids for each object
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

        # Draw edges (relations)
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                weight = adjacency[i, j]
                if weight > 0:
                    ax.plot(
                        [centroids[i, 0], centroids[j, 0]],
                        [centroids[i, 1], centroids[j, 1]],
                        color="red",
                        linewidth=0.5 + weight * 3,
                        alpha=0.3 + weight * 0.5,
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

        output_path = img_folder / "relational_graph_overlay.png"
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close()

    def _save_relational_graph_only(
        self,
        img_shape: tuple,
        objects: List[Dict],
        adjacency: np.ndarray,
        img_folder: Path,
    ) -> None:
        """Save relational graph on empty background (no image)."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=150)

        # Create empty background (same size as image)
        h, w = img_shape[:2]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # Inverted y-axis to match image coordinates
        ax.axis("off")
        ax.set_aspect("equal")

        # Get centroids for each object
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

        # Draw edges (relations)
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                weight = adjacency[i, j]
                if weight > 0:
                    ax.plot(
                        [centroids[i, 0], centroids[j, 0]],
                        [centroids[i, 1], centroids[j, 1]],
                        color="red",
                        linewidth=0.5 + weight * 3,
                        alpha=0.3 + weight * 0.5,
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

        output_path = img_folder / "relational_graph_only.png"
        plt.savefig(
            output_path, bbox_inches="tight", pad_inches=0, dpi=150, transparent=True
        )
        plt.close()


def load_dataset(dataset_path: Path) -> List[Dict]:
    """Load processed dataset."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data["images"]


def main():
    """Main entry point."""
    print("=" * 60)
    print("CREATING INDIVIDUAL VISUALIZATIONS")
    print("=" * 60)

    # Load dataset
    dataset_path = Path(config.OUTPUT_DIR) / "annotations" / "dataset.json"
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found!")
        print("Please run preprocess_data.py first.")
        return

    print(f"\nLoading dataset from {dataset_path}...")
    images = load_dataset(dataset_path)
    print(f"✓ Loaded {len(images)} images\n")

    # Get processed images directory
    processed_images_dir = Path(config.OUTPUT_DIR) / "images"
    if not processed_images_dir.exists():
        print(f"Error: {processed_images_dir} not found!")
        print("Please run preprocess_data.py first.")
        return

    # Create output directory
    output_dir = Path(config.OUTPUT_DIR) / "individual_visualizations"
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created output directory: {output_dir}\n")

    # Initialize visualizer
    visualizer = IndividualVisualizer(
        output_dir=output_dir, predicate_weights=config.PREDICATE_WEIGHTS
    )

    # Process only the same 20 samples that were visualized in main pipeline
    n_samples = min(config.VISUALIZE_SAMPLES, len(images))
    sample_indices = np.linspace(0, len(images) - 1, n_samples, dtype=int)

    print(f"Creating individual visualizations for {n_samples} sample images...")
    for idx in tqdm(sample_indices, desc="Processing", mininterval=1.0):
        img_data = images[idx]
        img_id = img_data["image_id"]

        # Create subfolder for this image
        img_folder = output_dir / img_id
        img_folder.mkdir(exist_ok=True)

        # Create all visualization components
        try:
            visualizer.create_visualizations(img_data, img_folder, processed_images_dir)
        except Exception as e:
            print(f"\nWarning: Failed to process image {img_id}: {e}")
            continue

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"✓ Processed {n_samples} sample images (same as main visualizations)")
    print(f"✓ Output directory: {output_dir}")
    print("\nEach image folder contains:")
    print("  - raw_image.jpg")
    print("  - segmentations_no_labels.png")
    print("  - relational_graph_overlay.png")
    print("  - relational_graph_only.png (graph on transparent background)")
    print("=" * 60)


if __name__ == "__main__":
    main()
