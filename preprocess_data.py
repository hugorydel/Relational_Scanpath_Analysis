#!/usr/bin/env python3
"""
Synthetic Visual Genome (SVG) Relational Dataset Creator
Filters SVG images for gaze-based relational memory experiments
"""

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Try to import memorability predictor
import torch
from datasets import load_dataset

# You'll need to install/implement ResMem or ViTMem
from resmem import ResMemPredictor
from tqdm import tqdm

ds = load_dataset("jamepark3922/svg", split="train")


class SVGRelationalDataset:
    """Filter and process SVG for relational gaze experiments."""

    def __init__(
        self,
        svg_root: str,
        output_dir: str = "./svg_relational_stimuli",
        # Filtering thresholds
        min_memorability: float = 0.5,
        min_mask_area_percent: float = 0.3,
        min_objects: int = 10,
        max_objects: int = 30,
        min_relations: int = 10,
        min_coverage_percent: float = 70.0,
        # Relational weights
        interaction_predicates: List[str] = None,
        spatial_predicates: List[str] = None,
        interaction_weight: float = 1.0,
        spatial_weight: float = 0.3,
        # Image parameters
        target_size: Tuple[int, int] = (1024, 768),
        source_filter: str = "visual_genome",  # Filter to VG-sourced images
    ):
        """
        Initialize the SVG dataset creator.

        Args:
            svg_root: Path to SVG dataset root
            output_dir: Where to save processed images and annotations
            min_memorability: Minimum memorability score (0-1)
            min_mask_area_percent: Minimum mask area as % of image
            min_objects: Minimum number of objects after filtering tiny masks
            max_objects: Maximum number of objects
            min_relations: Minimum number of relations in scene graph
            min_coverage_percent: Minimum % of image covered by AOIs
            interaction_predicates: List of interaction predicate names (higher weight)
            spatial_predicates: List of spatial predicate names (lower weight)
            interaction_weight: Weight for interaction predicates (0-1)
            spatial_weight: Weight for spatial predicates (0-1)
            target_size: Fixed canvas size (width, height) for letterboxing
            source_filter: Filter to specific source dataset (e.g., "visual_genome")
        """
        self.svg_root = Path(svg_root)
        self.output_dir = Path(output_dir)
        self.min_memorability = min_memorability
        self.min_mask_area_percent = min_mask_area_percent
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.min_relations = min_relations
        self.min_coverage_percent = min_coverage_percent
        self.target_size = target_size
        self.source_filter = source_filter

        # Default interaction predicates (higher semantic weight)
        if interaction_predicates is None:
            interaction_predicates = [
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
                "has",
                "belongs to",
                "part of",
                "attached to",
                "connected to",
            ]

        # Default spatial predicates (lower semantic weight)
        if spatial_predicates is None:
            spatial_predicates = [
                "left of",
                "right of",
                "above",
                "below",
                "in front of",
                "behind",
                "in",
                "inside",
                "on",
                "under",
                "between",
                "around",
                "outside",
                "near",
                "far from",
                "next to",
                "beside",
                "by",
                "near to",
                "close to",
                "touching",
                "overlapping",
                "against",
                "covering",
                "intersecting",
                "on top of",
                "over",
                "top of",
                "on top",
                "over the top of",
                "underneath",
                "beneath",
                "with",
                "at",
                "inside of",
                "towards",
                "away from",
                "into",
                "out of",
                "across",
                "along",
                "through",
                "past",
                "beyond",
                "across from",
                "opposite",
                "at the top of",
                "bottom of",
            ]

        self.interaction_predicates = set(p.lower() for p in interaction_predicates)
        self.spatial_predicates = set(p.lower() for p in spatial_predicates)
        self.interaction_weight = interaction_weight
        self.spatial_weight = spatial_weight

        # Create output directories
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        print(f"Created fresh output directory: {self.output_dir}")

        # Load memorability predictor
        self.memorability_predictor = None
        print("Loading memorability predictor...")
        self.memorability_predictor = ResMemPredictor()
        print("Memorability predictor loaded")

        # Color palette for visualization
        self.colors = self._generate_colors(100)  # Enough for most scenes

    def _generate_colors(self, n: int) -> np.ndarray:
        """Generate distinguishable colors for visualization."""
        np.random.seed(42)
        colors = sns.color_palette("husl", n)
        return np.array(colors)

    def _load_svg_metadata(self) -> List[Dict]:
        """
        Load SVG scene graphs and metadata.

        Returns:
            List of scene graph dictionaries
        """
        # SVG typically has scene_graphs.json or similar
        scene_graph_path = self.svg_root / "scene_graphs.json"

        if not scene_graph_path.exists():
            raise FileNotFoundError(f"Scene graphs not found at {scene_graph_path}")

        print(f"Loading SVG scene graphs from {scene_graph_path}...")
        with open(scene_graph_path, "r") as f:
            scene_graphs = json.load(f)

        print(f"Loaded {len(scene_graphs)} scene graphs")
        return scene_graphs

    def _predict_memorability(self, img_path: Path) -> float:
        """
        Predict memorability score for an image.

        Args:
            img_path: Path to image

        Returns:
            Memorability score (0-1)
        """
        # Load and preprocess image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run predictor
        score = self.memorability_predictor.predict(img)

        return float(score)

    def _calculate_mask_area(
        self, mask: Dict, img_width: int, img_height: int
    ) -> float:
        """
        Calculate mask area as percentage of image.

        Args:
            mask: Mask dictionary (could be polygon, RLE, or bbox)
            img_width: Image width
            img_height: Image height

        Returns:
            Area as percentage of image
        """
        img_area = img_width * img_height

        if "polygon" in mask:
            # Polygon format: list of [x, y] points
            polygon = np.array(mask["polygon"]).reshape(-1, 2)
            # Calculate area using shoelace formula
            x = polygon[:, 0]
            y = polygon[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            return (area / img_area) * 100

        elif "bbox" in mask:
            # Bounding box: [x, y, w, h]
            x, y, w, h = mask["bbox"]
            area = w * h
            return (area / img_area) * 100

        elif "area" in mask:
            # Direct area field
            return (mask["area"] / img_area) * 100

        return 0.0

    def _compute_relational_graph(self, scene_graph: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Compute weighted relational graph from scene graph.

        Args:
            scene_graph: SVG scene graph dictionary

        Returns:
            adjacency_matrix: NxN weighted adjacency matrix
            object_id_map: Mapping from object indices to object IDs
        """
        objects = scene_graph.get("objects", [])
        relations = scene_graph.get("relationships", [])

        n_objects = len(objects)
        adjacency = np.zeros((n_objects, n_objects))

        # Create object ID to index mapping
        object_id_map = {obj["object_id"]: idx for idx, obj in enumerate(objects)}

        # Process relations
        for rel in relations:
            subj_id = rel.get("subject_id")
            obj_id = rel.get("object_id")
            predicate = rel.get("predicate", "").lower()

            # Check if both objects exist
            if subj_id not in object_id_map or obj_id not in object_id_map:
                continue

            subj_idx = object_id_map[subj_id]
            obj_idx = object_id_map[obj_id]

            # Determine weight based on predicate type
            if predicate in self.interaction_predicates:
                weight = self.interaction_weight
            elif predicate in self.spatial_predicates:
                weight = self.spatial_weight
            else:
                # Unknown predicate - give it medium weight
                weight = (self.interaction_weight + self.spatial_weight) / 2

            # Add edge (weighted by predicate type)
            # If multiple relations exist, sum them up
            adjacency[subj_idx, obj_idx] += weight
            # Make symmetric (undirected graph)
            adjacency[obj_idx, subj_idx] += weight

        # Normalize to [0, 1] per edge
        max_weight = adjacency.max()
        if max_weight > 0:
            adjacency = adjacency / max_weight

        return adjacency, object_id_map

    def _calculate_coverage(
        self, objects: List[Dict], img_width: int, img_height: int
    ) -> float:
        """
        Calculate coverage of image by objects.

        Args:
            objects: List of object dictionaries with masks
            img_width: Image width
            img_height: Image height

        Returns:
            Coverage percentage
        """
        # Create binary mask
        coverage_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        for obj in objects:
            if "polygon" in obj:
                # Draw polygon
                polygon = np.array(obj["polygon"]).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(coverage_mask, [polygon], 1)
            elif "bbox" in obj:
                # Draw bbox
                x, y, w, h = obj["bbox"]
                coverage_mask[int(y) : int(y + h), int(x) : int(x + w)] = 1

        # Calculate coverage
        covered_pixels = np.sum(coverage_mask > 0)
        total_pixels = img_width * img_height
        coverage_percent = (covered_pixels / total_pixels) * 100

        return coverage_percent

    def filter_images(self) -> List[Dict]:
        """
        Filter SVG images based on criteria.

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
            # Check source
            source = scene_graph.get("source", "").lower()
            if self.source_filter and self.source_filter.lower() not in source:
                stats["wrong_source"] += 1
                continue

            # Get image path
            img_filename = scene_graph.get("image", "")
            img_path = self.svg_root / "images" / img_filename

            if not img_path.exists():
                stats["missing_image"] += 1
                continue

            # Check memorability
            memorability = self._predict_memorability(img_path)
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

            # Compute relational graph
            adjacency, object_id_map = self._compute_relational_graph(scene_graph)

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

        # Print filtering statistics
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
        """
        Letterbox image to target size while maintaining aspect ratio.

        Returns:
            letterboxed_img: Padded image
            scale: Scale factor applied
            offset: (x_offset, y_offset) for coordinate transformation
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create canvas
        letterboxed = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        # Center
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

            # Transform polygon if present
            if "polygon" in obj_new:
                polygon = np.array(obj_new["polygon"]).reshape(-1, 2)
                polygon = polygon * scale + [x_off, y_off]
                obj_new["polygon"] = polygon.flatten().tolist()

            # Transform bbox if present
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
        """
        Create visualization showing objects and relational graph.

        Args:
            img_data: Image metadata from filter_images()
        """
        scene_graph = img_data["scene_graph"]
        img_id = scene_graph.get("image_id", "unknown")

        # Load image
        img = cv2.imread(str(img_data["image_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Title
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

        # Save
        output_path = self.output_dir / "visualizations" / f"{img_id}_viz.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _draw_objects(self, ax: plt.Axes, objects: List[Dict], title: str) -> None:
        """Draw objects with labels."""
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

        for idx, obj in enumerate(objects):
            color = self.colors[idx % len(self.colors)]

            # Draw polygon or bbox
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

                # Label at centroid
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

                # Label at center
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
        """Draw relational graph as overlay on image."""
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

        # Get object centroids
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
                    # Line thickness proportional to weight
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

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Relation Strength")

    def create_summary_statistics(self, selected_images: List[Dict]) -> None:
        """Create summary statistics visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Dataset Summary: {len(selected_images)} Images",
            fontsize=16,
            fontweight="bold",
        )

        # Object count distribution
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

        # Relations count distribution
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

        # Coverage distribution
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

        # Memorability distribution
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

        # Relations/Objects ratio
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

        # Edge weight distribution
        ax = axes[1, 2]
        all_weights = []
        for img in selected_images:
            adj = img["adjacency_matrix"]
            weights = adj[adj > 0]  # Non-zero weights
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

        # Save
        output_path = self.output_dir / "summary_statistics.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\nSaved summary statistics: {output_path}")

    def process_dataset(self, visualize_samples: int = 20) -> None:
        """
        Main processing pipeline.

        Args:
            visualize_samples: Number of example visualizations to create
        """
        # Filter images
        selected_images = self.filter_images()

        if len(selected_images) == 0:
            print("No images met the criteria!")
            return

        # Create summary statistics
        print("\nGenerating summary statistics...")
        self.create_summary_statistics(selected_images)

        # Visualize sample images
        print(f"\nGenerating {visualize_samples} example visualizations...")
        sample_indices = np.linspace(
            0,
            len(selected_images) - 1,
            min(visualize_samples, len(selected_images)),
            dtype=int,
        )

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
                        "date_created": "2025-12-04",
                        "source_filter": self.source_filter,
                        "min_memorability": self.min_memorability,
                        "min_mask_area_percent": self.min_mask_area_percent,
                        "min_objects": self.min_objects,
                        "max_objects": self.max_objects,
                        "min_relations": self.min_relations,
                        "min_coverage_percent": self.min_coverage_percent,
                        "target_size": self.target_size,
                        "interaction_weight": self.interaction_weight,
                        "spatial_weight": self.spatial_weight,
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
        print(
            f"  ✓ Weighted relational graphs (interaction={self.interaction_weight}, spatial={self.spatial_weight})"
        )
        print(f"  ✓ Ready for gaze-relational memory analysis")


def main():
    """Main entry point."""

    # Configuration
    SVG_ROOT = "./svg_dataset"  # Adjust to your SVG dataset path
    OUTPUT_DIR = "./svg_relational_stimuli"

    # Initialize dataset creator
    creator = SVGRelationalDataset(
        svg_root=SVG_ROOT,
        output_dir=OUTPUT_DIR,
        min_memorability=0.5,
        min_mask_area_percent=0.3,
        min_objects=10,
        max_objects=30,
        min_relations=10,
        min_coverage_percent=70.0,
        interaction_weight=1.0,
        spatial_weight=0.3,
        target_size=(1024, 768),
        source_filter="visual_genome",
    )

    # Process dataset
    creator.process_dataset(visualize_samples=20)


if __name__ == "__main__":
    main()
