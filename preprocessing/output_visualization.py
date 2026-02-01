"""
Visualization utilities for SVG relational dataset
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class ImageVisualizer:
    """Handles visualization of images and statistics."""

    def __init__(self, output_dir: Path, target_size: tuple, predicate_weights: Dict):
        self.output_dir = output_dir
        self.target_size = target_size
        self.predicate_weights = predicate_weights

        # Generate color palette
        np.random.seed(42)
        self.colors = np.array(sns.color_palette("husl", 100))

    def visualize_image(self, img_data: Dict) -> None:
        """Create visualization showing objects and relational graph."""
        scene_graph = img_data["scene_graph"]
        img_id = scene_graph.get("image_id", "unknown")

        img = cv2.imread(str(img_data["image_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        fig.suptitle(
            f"Image {img_id} | Objects: {img_data['n_objects']} | "
            f"Relations: {img_data['n_relations']} | "
            f"Coverage: {img_data['coverage_percent']:.1f}% | "
            f"Memorability: {img_data['memorability']:.2f}",
            fontsize=14,
            fontweight="bold",
        )

        # Plot 1: Objects with labels
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        self._draw_objects(ax1, img_data["objects"], "Objects & Labels")

        # Plot 2: Relational graph overlay
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img)
        self._draw_relational_graph(
            ax2, img_data["objects"], img_data["adjacency_matrix"]
        )

        # Plot 3: Letterboxed view
        letterboxed = self._create_letterbox_preview(img)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(letterboxed)
        ax3.set_title(
            f"Letterboxed ({self.target_size[0]}×{self.target_size[1]})",
            fontsize=12,
            fontweight="bold",
        )
        ax3.axis("off")

        # Plot 4: Edge statistics
        ax4 = fig.add_subplot(gs[1, 1])
        self._draw_edge_statistics(
            ax4, img_data["adjacency_matrix"], img_data["relations"]
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
            elif "bbox" in obj:
                x, y, w, h = obj["bbox"]
                rect = mpatches.Rectangle(
                    (x, y), w, h, linewidth=2, edgecolor=color, facecolor=(*color, 0.3)
                )
                ax.add_patch(rect)
                centroid = np.array([x + w / 2, y + h / 2])
            else:
                continue

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

    def _draw_relational_graph(
        self, ax: plt.Axes, objects: List[Dict], adjacency: np.ndarray
    ) -> None:
        """Draw relational graph overlay."""
        ax.set_title("Relational Graph", fontsize=12, fontweight="bold")
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

    def _draw_edge_statistics(
        self, ax: plt.Axes, adjacency: np.ndarray, relations: List[Dict]
    ) -> None:
        """Draw edge weight distribution and relation statistics."""
        ax.set_title("Relational Statistics", fontsize=12, fontweight="bold")

        edge_weights = adjacency[adjacency > 0]
        category_counts = Counter(rel["predicate_category"] for rel in relations)

        stats_text = [
            f"Total Relations: {len(relations)}",
            f"Unique Edges: {len(edge_weights)}",
            f"Avg Strength: {edge_weights.mean():.3f}",
            f"Max Strength: {edge_weights.max():.3f}",
            "",
            "Relation Types:",
        ]

        for category, count in category_counts.most_common():
            weight = self.predicate_weights.get(category, 0.0)
            stats_text.append(f"  {category}: {count} (w={weight})")

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

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

    def _create_letterbox_preview(self, img: np.ndarray) -> np.ndarray:
        """Create letterboxed preview of image."""
        h, w = img.shape[:2]
        target_w, target_h = self.target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        letterboxed = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        letterboxed[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return letterboxed

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
        ax.set_title("Relations Distribution")
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
        ax.set_title("Edge Strength Distribution")
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
