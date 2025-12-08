#!/usr/bin/env python3
"""
compare_embeddings.py - Compare text vs image embeddings for diversity selection
Helps you decide which embedding type best suits your needs.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
from diversity_selection import DiversitySelector


def compare_embedding_types(images_data, n_compare=50):
    """
    Compare text and image embeddings on sample of images.

    Args:
        images_data: List of image data dictionaries
        n_compare: Number of images to compare (for speed)
    """
    print("=" * 60)
    print("EMBEDDING TYPE COMPARISON")
    print("=" * 60)
    print(f"Comparing on {n_compare} sample images\n")

    # Sample images if needed
    if len(images_data) > n_compare:
        np.random.seed(config.RANDOM_SEED)
        indices = np.random.choice(len(images_data), n_compare, replace=False)
        sample_data = [images_data[i] for i in indices]
    else:
        sample_data = images_data

    # Initialize selector
    selector = DiversitySelector(embedding_cache_path=Path(config.EMBEDDING_CACHE_PATH))

    # Compute both types of embeddings
    print("Computing text embeddings...")
    text_embeddings = selector.compute_embeddings(
        sample_data, embedding_type="text", batch_size=config.EMBEDDING_BATCH_SIZE
    )

    print("\nComputing image embeddings...")
    image_embeddings = selector.compute_embeddings(
        sample_data, embedding_type="image", batch_size=config.EMBEDDING_BATCH_SIZE
    )

    # Normalize
    text_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    image_norm = image_embeddings / np.linalg.norm(
        image_embeddings, axis=1, keepdims=True
    )

    # Compute pairwise similarities
    n = len(sample_data)
    mask = ~np.eye(n, dtype=bool)

    text_sim_matrix = text_norm @ text_norm.T
    image_sim_matrix = image_norm @ image_norm.T

    text_similarities = text_sim_matrix[mask]
    image_similarities = image_sim_matrix[mask]

    # Print statistics
    print("\n" + "=" * 60)
    print("SIMILARITY STATISTICS")
    print("=" * 60)

    print("\nText Embeddings (Semantic):")
    print(f"  Mean similarity: {text_similarities.mean():.3f}")
    print(f"  Std similarity:  {text_similarities.std():.3f}")
    print(f"  Min similarity:  {text_similarities.min():.3f}")
    print(f"  Max similarity:  {text_similarities.max():.3f}")

    print("\nImage Embeddings (Visual):")
    print(f"  Mean similarity: {image_similarities.mean():.3f}")
    print(f"  Std similarity:  {image_similarities.std():.3f}")
    print(f"  Min similarity:  {image_similarities.min():.3f}")
    print(f"  Max similarity:  {image_similarities.max():.3f}")

    # Correlation between text and image similarities
    correlation = np.corrcoef(text_similarities, image_similarities)[0, 1]
    print(f"\nCorrelation between text and image similarities: {correlation:.3f}")
    print(f"  (Low correlation = embeddings capture different aspects of diversity)\n")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Text similarity distribution
    axes[0].hist(text_similarities, bins=30, alpha=0.7, color="blue", edgecolor="black")
    axes[0].axvline(text_similarities.mean(), color="red", linestyle="--", label="Mean")
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Text Embeddings\n(Semantic Diversity)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Image similarity distribution
    axes[1].hist(
        image_similarities, bins=30, alpha=0.7, color="green", edgecolor="black"
    )
    axes[1].axvline(
        image_similarities.mean(), color="red", linestyle="--", label="Mean"
    )
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Image Embeddings\n(Visual Diversity)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Correlation scatter
    # Sample points to avoid overcrowding
    n_scatter = min(5000, len(text_similarities))
    scatter_indices = np.random.choice(len(text_similarities), n_scatter, replace=False)

    axes[2].scatter(
        text_similarities[scatter_indices],
        image_similarities[scatter_indices],
        alpha=0.3,
        s=10,
    )
    axes[2].plot([0, 1], [0, 1], "r--", alpha=0.5, label="y=x")
    axes[2].set_xlabel("Text Similarity")
    axes[2].set_ylabel("Image Similarity")
    axes[2].set_title(f"Correlation: {correlation:.3f}\n(Lower = more complementary)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(config.OUTPUT_DIR) / "embedding_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved comparison plot to {output_path}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if text_similarities.mean() < image_similarities.mean():
        print("✓ Text embeddings show MORE diversity (lower mean similarity)")
        print("  → Better for ensuring semantically different scenes")
    else:
        print("✓ Image embeddings show MORE diversity (lower mean similarity)")
        print("  → Better for ensuring visually different appearances")

    if correlation < 0.5:
        print(
            f"\n✓ Low correlation ({correlation:.3f}) suggests embeddings capture "
            "different aspects"
        )
        print("  → Text focuses on WHAT (semantic content)")
        print("  → Image focuses on HOW (visual appearance)")
        print("  → Could use BOTH for multi-dimensional diversity (advanced)")
    else:
        print(
            f"\n✓ High correlation ({correlation:.3f}) suggests embeddings are similar"
        )
        print("  → Either type will work similarly for your dataset")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("\nFor your relational gaze experiment:")
    print("  1. Text embeddings are RECOMMENDED")
    print("     - Focuses on semantic/relational differences")
    print("     - Aligns with your research question")
    print("     - Less affected by lighting/style variations")
    print("\n  2. Image embeddings as alternative if text similarity too high")
    print("     - Use if mean similarity > 0.6")
    print("     - Ensures visual variety")
    print("=" * 60 + "\n")


def main():
    """Run embedding comparison."""
    # Load filtered images
    dataset_path = Path(config.OUTPUT_DIR) / "annotations" / "dataset.json"
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found!")
        print("Please run preprocess_data.py first.")
        return

    with open(dataset_path, "r") as f:
        data = json.load(f)
    filtered_images = data["images"]

    # Prepare image data
    vg_image_root = Path(config.VG_IMAGE_ROOT)
    images_data = []

    for img_meta in filtered_images:
        img_path = vg_image_root / img_meta["scene_graph"]["image"]
        images_data.append(
            {
                "scene_graph": img_meta["scene_graph"],
                "image_path": img_path,
                "metadata": img_meta,
            }
        )

    # Run comparison
    compare_embedding_types(images_data, n_compare=50)


if __name__ == "__main__":
    main()
