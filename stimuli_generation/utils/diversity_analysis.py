#!/usr/bin/env python3
"""
diversity_analysis.py - Analyze semantic diversity of manually selected stimuli

Simple analysis tool that computes pairwise similarity statistics for images
marked as "selected" in manual annotations. Does NOT perform any filtering
or selection - purely analytical.

Usage:
    python utils/diversity_analysis.py

Reads from:
    - stimuli/annotations/manual_annotations.json (images with status="selected")
    - data/scored_selection/annotations/selected_dataset.json (for story text)

Outputs to data/embedding/:
    - similarity_analysis.json: Per-image similarity details and encoding interference analysis
    - diversity_metrics.json: Similarity statistics
    - story_embeddings.npy: Embedding vectors
    - embedding_space_tsne.png: t-SNE visualization
    - embedding_space_pca.png: PCA visualization
    - similarity_matrix.npy: Full pairwise similarity matrix

Encoding Interference Analysis:
    Measures "neighborhood density" in semantic space - how crowded the area around
    each image is. Images in crowded neighborhoods have higher potential for memory
    encoding interference, while isolated images should encode more distinctly.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_manual_annotations(annotations_path: Path) -> Dict:
    """Load manual annotations and filter to selected images."""
    print(f"Loading manual annotations from {annotations_path}...")

    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Manual annotations not found: {annotations_path}\n"
            f"Please run manual_image_annotation.py first."
        )

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    # Filter to selected images only
    selected = {
        img_id: data
        for img_id, data in annotations.items()
        if data.get("status") == "selected"
    }

    print(f"✓ Loaded annotations: {len(annotations)} total, {len(selected)} selected")

    return selected


def load_image_stories(dataset_path: Path, selected_ids: List[str]) -> Dict[str, str]:
    """Load story text for selected images."""
    print(f"\nLoading image stories from {dataset_path}...")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Build mapping of image_id -> story
    stories = {}
    for img in dataset.get("images", []):
        img_id = str(img.get("image_id"))
        if img_id in selected_ids:
            stories[img_id] = img.get("story", "")

    print(f"✓ Found stories for {len(stories)}/{len(selected_ids)} selected images")

    # Warn about missing stories
    missing = set(selected_ids) - set(stories.keys())
    if missing:
        print(
            f"  Warning: {len(missing)} images missing stories: {sorted(missing)[:5]}..."
        )

    return stories


def compute_embeddings(stories: Dict[str, str], model_name: str) -> tuple:
    """
    Compute text embeddings from stories.

    Returns:
        image_ids: List of image IDs (in order)
        embeddings: (N, D) numpy array of embeddings
    """
    print(f"\nComputing embeddings using {model_name}...")

    # Sort by image_id for consistency
    image_ids = sorted(stories.keys())
    story_texts = [stories[img_id] for img_id in image_ids]

    # Check for empty stories
    empty_count = sum(1 for s in story_texts if not s)
    if empty_count > 0:
        print(f"  Warning: {empty_count} empty stories (will embed as empty string)")

    # Load model and compute embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        story_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )

    print(f"✓ Computed embeddings: shape {embeddings.shape}")

    return image_ids, embeddings


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    print("\nComputing pairwise similarities...")

    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity_matrix = embeddings_norm @ embeddings_norm.T

    print(f"✓ Computed similarity matrix: shape {similarity_matrix.shape}")

    return similarity_matrix


def compute_diversity_metrics(similarity_matrix: np.ndarray) -> Dict:
    """Compute diversity statistics from similarity matrix."""
    print("\nComputing diversity metrics...")

    n = len(similarity_matrix)

    # Extract pairwise similarities (excluding diagonal)
    mask = ~np.eye(n, dtype=bool)
    pairwise_similarities = similarity_matrix[mask]

    # Basic statistics
    metrics = {
        "n_images": int(n),
        "mean_similarity": float(pairwise_similarities.mean()),
        "std_similarity": float(pairwise_similarities.std()),
        "min_similarity": float(pairwise_similarities.min()),
        "max_similarity": float(pairwise_similarities.max()),
        "median_similarity": float(np.median(pairwise_similarities)),
        "q25_similarity": float(np.percentile(pairwise_similarities, 25)),
        "q75_similarity": float(np.percentile(pairwise_similarities, 75)),
    }

    # Similarity distribution
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(pairwise_similarities, bins=bins)
    metrics["similarity_distribution"] = {
        f"[{bins[i]:.1f}, {bins[i+1]:.1f})": int(count) for i, count in enumerate(hist)
    }

    # High similarity percentages
    metrics["pct_above_0.5"] = float(
        (pairwise_similarities > 0.5).sum() / len(pairwise_similarities) * 100
    )
    metrics["pct_above_0.6"] = float(
        (pairwise_similarities > 0.6).sum() / len(pairwise_similarities) * 100
    )
    metrics["pct_above_0.7"] = float(
        (pairwise_similarities > 0.7).sum() / len(pairwise_similarities) * 100
    )
    metrics["pct_above_0.8"] = float(
        (pairwise_similarities > 0.8).sum() / len(pairwise_similarities) * 100
    )

    print("✓ Computed diversity metrics")

    return metrics


def analyze_image_similarities(
    similarity_matrix: np.ndarray,
    image_ids: List[str],
    top_k: int = 5,
    similarity_threshold: float = 0.5,
) -> Dict:
    """
    Analyze which specific images are most similar to each other.

    Args:
        similarity_matrix: (N, N) pairwise similarity matrix
        image_ids: List of image IDs
        top_k: Number of top similar images to report per image
        similarity_threshold: Threshold for flagging high similarity

    Returns:
        Dictionary with detailed similarity analysis
    """
    print("\nAnalyzing image-specific similarities...")

    n = len(image_ids)
    analysis = {}

    # For each image, find its most similar neighbors
    image_similarities = {}
    for i, img_id in enumerate(image_ids):
        # Get similarities to all other images (excluding self)
        sims = similarity_matrix[i].copy()
        sims[i] = -1  # Exclude self

        # Get top K most similar
        top_indices = np.argsort(sims)[::-1][:top_k]
        top_similar = [
            {"image_id": image_ids[idx], "similarity": float(sims[idx])}
            for idx in top_indices
        ]

        # Average similarity to all others
        avg_sim = float(similarity_matrix[i, [j for j in range(n) if j != i]].mean())

        # Count high similarity neighbors
        high_sim_count = int((sims > similarity_threshold).sum())

        image_similarities[img_id] = {
            "top_similar_images": top_similar,
            "avg_similarity_to_others": avg_sim,
            "num_highly_similar": high_sim_count,  # Above threshold
        }

    analysis["per_image_similarities"] = image_similarities

    # Find most similar pairs overall
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(
                {
                    "image_1": image_ids[i],
                    "image_2": image_ids[j],
                    "similarity": float(similarity_matrix[i, j]),
                }
            )

    # Sort by similarity
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    analysis["most_similar_pairs"] = pairs[:20]  # Top 20 pairs

    # Encoding interference analysis: neighborhood density
    # Measure how "crowded" the semantic space is around each image
    k_neighbors = min(10, n - 1)  # Use top 10 neighbors (or all if less than 10)

    interference_scores = []
    for i, img_id in enumerate(image_ids):
        # Get similarities to all other images (excluding self)
        sims = similarity_matrix[i].copy()
        sims[i] = -1  # Exclude self

        # Get top K nearest neighbors
        top_k_sims = np.sort(sims)[::-1][:k_neighbors]

        # Cumulative similarity to K nearest neighbors (higher = more crowded)
        cumulative_neighbor_sim = float(top_k_sims.sum())

        # Average similarity to K nearest neighbors
        avg_neighbor_sim = float(top_k_sims.mean())

        # Count neighbors in similarity bands (for context)
        neighbors_0_3_to_0_4 = int(((top_k_sims >= 0.3) & (top_k_sims < 0.4)).sum())
        neighbors_0_4_to_0_5 = int(((top_k_sims >= 0.4) & (top_k_sims < 0.5)).sum())
        neighbors_0_5_plus = int((top_k_sims >= 0.5).sum())

        interference_scores.append(
            {
                "image_id": img_id,
                "cumulative_neighbor_similarity": cumulative_neighbor_sim,
                "avg_neighbor_similarity": avg_neighbor_sim,
                "k_neighbors_used": k_neighbors,
                "neighbors_0.3_0.4": neighbors_0_3_to_0_4,
                "neighbors_0.4_0.5": neighbors_0_4_to_0_5,
                "neighbors_0.5_plus": neighbors_0_5_plus,
                "top_neighbor_ids": [
                    image_ids[idx] for idx in np.argsort(sims)[::-1][:5]
                ],  # Top 5 for reference
            }
        )

    # Sort by cumulative similarity (highest interference potential first)
    interference_scores.sort(
        key=lambda x: x["cumulative_neighbor_similarity"], reverse=True
    )
    analysis["encoding_interference_ranking"] = interference_scores

    # Also provide top 10 highest and lowest interference
    analysis["highest_interference_potential"] = interference_scores[:10]
    analysis["lowest_interference_potential"] = interference_scores[-10:][
        ::-1
    ]  # Reverse to show lowest first

    print(f"✓ Analyzed similarities for {n} images")

    return analysis


def visualize_embedding_space(
    embeddings: np.ndarray, image_ids: List[str], output_dir: Path, method: str = "tsne"
):
    """Create 2D visualization of embedding space."""
    print(f"\nGenerating {method.upper()} visualization...")

    # Reduce to 2D
    if method == "tsne":
        reducer = TSNE(
            n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1)
        )
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)

    coords_2d = reducer.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=range(len(embeddings)),
        cmap="viridis",
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidths=1.5,
    )

    # Add labels for first few points
    for i in range(min(15, len(image_ids))):
        ax.annotate(
            image_ids[i],
            (coords_2d[i, 0], coords_2d[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
        )

    ax.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
    ax.set_ylabel(f"{method.upper()} Component 2", fontsize=12)
    ax.set_title(
        f"Embedding Space ({method.upper()})\n"
        f"{len(embeddings)} manually selected images",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label="Image Index")
    plt.tight_layout()

    output_path = output_dir / f"embedding_space_{method}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved {method.upper()} visualization to {output_path}")


def save_results(
    embeddings: np.ndarray,
    similarity_matrix: np.ndarray,
    metrics: Dict,
    similarity_analysis: Dict,
    image_ids: List[str],
    output_dir: Path,
):
    """Save all analysis results."""
    print("\nSaving results...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(output_dir / "story_embeddings.npy", embeddings)
    print(f"  ✓ story_embeddings.npy")

    # Save similarity matrix
    np.save(output_dir / "similarity_matrix.npy", similarity_matrix)
    print(f"  ✓ similarity_matrix.npy")

    # Save metrics
    metrics["image_ids"] = image_ids  # Include for reference
    with open(output_dir / "diversity_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ diversity_metrics.json")

    # Save similarity analysis
    with open(output_dir / "similarity_analysis.json", "w") as f:
        json.dump(similarity_analysis, f, indent=2)
    print(f"  ✓ similarity_analysis.json")


def print_report(metrics: Dict):
    """Print formatted diversity report."""
    print("\n" + "=" * 70)
    print("DIVERSITY ANALYSIS REPORT")
    print("=" * 70)
    print(f"Number of images: {metrics['n_images']}")

    print(f"\nPairwise Similarity Statistics:")
    print(f"  Mean:   {metrics['mean_similarity']:.4f}")
    print(f"  Std:    {metrics['std_similarity']:.4f}")
    print(f"  Min:    {metrics['min_similarity']:.4f}")
    print(f"  Q25:    {metrics['q25_similarity']:.4f}")
    print(f"  Median: {metrics['median_similarity']:.4f}")
    print(f"  Q75:    {metrics['q75_similarity']:.4f}")
    print(f"  Max:    {metrics['max_similarity']:.4f}")

    print(f"\nSimilarity Distribution:")
    total = sum(metrics["similarity_distribution"].values())
    for range_str, count in metrics["similarity_distribution"].items():
        pct = count / total * 100
        print(f"  {range_str}: {count:4d} ({pct:5.1f}%)")

    print(f"\nHigh Similarity Pairs:")
    print(f"  > 0.5: {metrics['pct_above_0.5']:5.1f}%")
    print(f"  > 0.6: {metrics['pct_above_0.6']:5.1f}%")
    print(f"  > 0.7: {metrics['pct_above_0.7']:5.1f}%")
    print(f"  > 0.8: {metrics['pct_above_0.8']:5.1f}%")

    print("\nInterpretation:")
    mean_sim = metrics["mean_similarity"]
    if mean_sim < 0.4:
        print("  → HIGH DIVERSITY: Images are semantically distinct")
    elif mean_sim < 0.6:
        print("  → MODERATE DIVERSITY: Some semantic overlap")
    else:
        print("  → LOW DIVERSITY: Images share many semantic similarities")

    print("=" * 70)


def print_similarity_analysis(analysis: Dict):
    """Print formatted similarity analysis report."""
    print("\n" + "=" * 70)
    print("IMAGE SIMILARITY ANALYSIS")
    print("=" * 70)

    # Most similar pairs
    print("\nTop 10 Most Similar Image Pairs:")
    print("-" * 70)
    for i, pair in enumerate(analysis["most_similar_pairs"][:10], 1):
        print(
            f"{i:2d}. {pair['image_1']} ↔ {pair['image_2']}: {pair['similarity']:.4f}"
        )

    # Encoding interference: images in crowded neighborhoods
    print("\n" + "=" * 70)
    print("ENCODING INTERFERENCE POTENTIAL")
    print("(Images in crowded semantic neighborhoods)")
    print("=" * 70)
    print("\nHighest Interference (most similar neighbors):")
    print("-" * 70)
    for i, img in enumerate(analysis["highest_interference_potential"], 1):
        print(f"{i:2d}. Image {img['image_id']}")
        print(
            f"    Cumulative similarity to {img['k_neighbors_used']} nearest neighbors: {img['cumulative_neighbor_similarity']:.4f}"
        )
        print(f"    Average neighbor similarity: {img['avg_neighbor_similarity']:.4f}")
        neighbors_dist = f"[0.3-0.4]: {img['neighbors_0.3_0.4']}, [0.4-0.5]: {img['neighbors_0.4_0.5']}, [0.5+]: {img['neighbors_0.5_plus']}"
        print(f"    Neighbor distribution: {neighbors_dist}")

    # Lowest interference: isolated images
    print("\n" + "=" * 70)
    print("Lowest Interference (most isolated in semantic space):")
    print("-" * 70)
    for i, img in enumerate(analysis["lowest_interference_potential"], 1):
        print(f"{i:2d}. Image {img['image_id']}")
        print(
            f"    Cumulative similarity to {img['k_neighbors_used']} nearest neighbors: {img['cumulative_neighbor_similarity']:.4f}"
        )
        print(f"    Average neighbor similarity: {img['avg_neighbor_similarity']:.4f}")

    print("=" * 70)
    print("\nInterpretation:")
    print(
        "  • High cumulative similarity = crowded neighborhood = high encoding interference"
    )
    print("  • Low cumulative similarity = isolated image = low encoding interference")
    print("\nNote: Full per-image details saved to similarity_analysis.json")
    print("      Includes top 5 most similar neighbors for each image")
    print("=" * 70)


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("DIVERSITY ANALYSIS - MANUALLY SELECTED STIMULI")
    print("=" * 70)

    # Configuration
    ANNOTATIONS_PATH = Path("data/stimuli/annotations/manual_annotations.json")
    DATASET_PATH = Path("data/scored_selection/annotations/selected_dataset.json")
    OUTPUT_DIR = Path("data/embedding")
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    print(f"Input: {ANNOTATIONS_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70 + "\n")

    # Load manual annotations
    selected_annotations = load_manual_annotations(ANNOTATIONS_PATH)

    if len(selected_annotations) == 0:
        print("\nNo selected images found! Please mark images as 'selected' first.")
        return

    selected_ids = list(selected_annotations.keys())

    # Load stories
    stories = load_image_stories(DATASET_PATH, selected_ids)

    if len(stories) == 0:
        print("\nNo stories found for selected images!")
        return

    # Compute embeddings
    image_ids, embeddings = compute_embeddings(stories, MODEL_NAME)

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)

    # Compute metrics
    metrics = compute_diversity_metrics(similarity_matrix)

    # Analyze image-specific similarities
    similarity_analysis = analyze_image_similarities(
        similarity_matrix, image_ids, top_k=5, similarity_threshold=0.5
    )

    # Save results
    save_results(
        embeddings,
        similarity_matrix,
        metrics,
        similarity_analysis,
        image_ids,
        OUTPUT_DIR,
    )

    # Generate visualizations
    visualize_embedding_space(embeddings, image_ids, OUTPUT_DIR, method="tsne")
    visualize_embedding_space(embeddings, image_ids, OUTPUT_DIR, method="pca")

    # Print reports
    print_report(metrics)
    print_similarity_analysis(similarity_analysis)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"✓ Analyzed {len(image_ids)} manually selected images")
    print(f"✓ Results saved to: {OUTPUT_DIR}")
    print(f"✓ Diversity metrics: {OUTPUT_DIR / 'diversity_metrics.json'}")
    print(f"✓ Similarity analysis: {OUTPUT_DIR / 'similarity_analysis.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
