#!/usr/bin/env python3
"""
select_scored_stimuli.py - Select diverse, high-scoring subset from OpenAI-scored images

Selects final stimuli from OpenAI-scored images by balancing:
1. Quality: Prioritizes highest-scoring images
2. Diversity: Ensures selected images are semantically distinct (via story embeddings)

Usage:
    python select_diverse_scored_stimuli.py

Configuration:
    Edit config.py to adjust:
    - SCORED_IMAGES_PATH: Path to results.jsonl
    - N_FINAL_SCORED_IMAGES: Target number of images
    - SCORED_SIMILARITY_THRESHOLD: Diversity threshold (0-1)
    - TEXT_EMBEDDING_MODEL: SentenceTransformer model
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from embedding.diversity_selection import DiversitySelector
from tqdm import tqdm

import config


def load_scored_images(jsonl_path: Path) -> List[Dict]:
    """
    Load OpenAI-scored images from JSONL file.

    Returns:
        List of dicts with image_id, story, CIC, SEP, DYN, QLT, score, eligible
    """
    print(f"Loading scored images from {jsonl_path}...")

    results = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # Calculate scores using centralized function
    try:
        from config import calculate_image_score
    except ImportError:
        print("Warning: Could not import scoring function, using fallback")

        def calculate_image_score(cic, sep, dyn, qlt):
            cic = int(cic) if cic is not None else 0
            sep = int(sep) if sep is not None else 0
            dyn = int(dyn) if dyn is not None else 0
            qlt = int(qlt) if qlt is not None else 0
            eligible = 1 if (cic >= 2 and sep >= 1 and dyn >= 1 and qlt >= 1) else 0
            score = eligible * (cic * 2.5 + sep + dyn * 1.5 + qlt)
            return {"eligible": eligible, "score": score}

    for r in results:
        scoring = calculate_image_score(
            r.get("CIC", 0), r.get("SEP", 0), r.get("DYN", 0), r.get("QLT", 0)
        )
        r["eligible"] = scoring["eligible"]
        r["score"] = scoring["score"]

    print(f"✓ Loaded {len(results)} scored images")

    return results


def filter_eligible_images(scored_images: List[Dict]) -> List[Dict]:
    """Filter to only eligible images."""
    eligible = [img for img in scored_images if img.get("eligible", 0) == 1]

    print(f"\nFiltering to eligible images...")
    print(f"  Total images: {len(scored_images)}")
    print(f"  Eligible: {len(eligible)} ({100*len(eligible)/len(scored_images):.1f}%)")
    print(f"  Non-eligible: {len(scored_images) - len(eligible)}")

    if len(eligible) == 0:
        raise ValueError(
            "No eligible images found! Check scoring criteria in config.py"
        )

    return eligible


def compute_story_embeddings(
    eligible_images: List[Dict], selector: DiversitySelector
) -> np.ndarray:
    """
    Compute text embeddings from image stories.

    Args:
        eligible_images: List of eligible image dicts with 'story' field
        selector: DiversitySelector instance

    Returns:
        embeddings: (N, D) numpy array
    """
    print("\nComputing story embeddings...")

    stories = [img.get("story", "") for img in eligible_images]

    # Check for missing stories
    missing_count = sum(1 for s in stories if not s)
    if missing_count > 0:
        print(
            f"  Warning: {missing_count} images missing stories (will use empty string)"
        )

    # Use SentenceTransformer from selector
    embeddings = selector.text_model.encode(
        stories, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )

    print(f"✓ Computed embeddings: shape {embeddings.shape}")

    return embeddings


def create_output_directory(base_dir: Path = None) -> Path:
    """Create output directory for selected images."""
    if base_dir is None:
        base_dir = Path("./data")

    output_dir = base_dir / "diverse_scored_selection"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "annotations").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)

    return output_dir


def save_selected_images(
    selected_indices: np.ndarray,
    eligible_images: List[Dict],
    embeddings: np.ndarray,
    diversity_metrics: Dict,
    output_dir: Path,
    image_source_dir: Path,
):
    """
    Save selected images and metadata.

    Args:
        selected_indices: Indices of selected images
        eligible_images: Full list of eligible images
        embeddings: Story embeddings
        diversity_metrics: Diversity statistics
        output_dir: Where to save outputs
        image_source_dir: Where to find source images
    """
    selected_images = [eligible_images[idx] for idx in selected_indices]
    selected_embeddings = embeddings[selected_indices]

    # Copy images
    print("\nCopying selected images...")
    copied = 0
    missing = 0

    for img in tqdm(selected_images, desc="Copying images"):
        img_id = img["image_id"]
        src_path = image_source_dir / f"{img_id}.jpg"
        dst_path = output_dir / "images" / f"{img_id}.jpg"

        if src_path.exists():
            shutil.copy(src_path, dst_path)
            copied += 1
        else:
            missing += 1
            print(f"  Warning: Missing image {img_id}.jpg")

    print(f"✓ Copied {copied} images")
    if missing > 0:
        print(f"  Warning: {missing} images not found")

    # Save metadata
    metadata = {
        "info": {
            "description": "Diverse High-Scoring Stimuli Dataset",
            "num_images": len(selected_images),
            "selection_method": "score_weighted_greedy",
            "diversity_threshold": config.SCORED_SIMILARITY_THRESHOLD,
            "text_embedding_model": config.TEXT_EMBEDDING_MODEL,
            "diversity_metrics": diversity_metrics,
            "scoring_criteria": {
                "CIC_threshold": config.CIC_THRESHOLD,
                "SEP_threshold": config.SEP_THRESHOLD,
                "DYN_threshold": config.DYN_THRESHOLD,
                "QLT_threshold": config.QLT_THRESHOLD,
                "CIC_weight": config.CIC_WEIGHT,
                "SEP_weight": config.SEP_WEIGHT,
                "DYN_weight": config.DYN_WEIGHT,
                "QLT_weight": config.QLT_WEIGHT,
            },
        },
        "images": selected_images,
    }

    metadata_path = output_dir / "annotations" / "selected_dataset.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata to {metadata_path}")

    # Save embeddings for future use
    embeddings_path = output_dir / "annotations" / "story_embeddings.npy"
    np.save(embeddings_path, selected_embeddings)
    print(f"✓ Saved embeddings to {embeddings_path}")


def print_selection_summary(
    selected_indices: np.ndarray,
    eligible_images: List[Dict],
):
    """Print summary statistics of selected images."""
    selected_images = [eligible_images[idx] for idx in selected_indices]

    scores = [img["score"] for img in selected_images]
    cic_vals = [img.get("CIC", 0) for img in selected_images]
    sep_vals = [img.get("SEP", 0) for img in selected_images]
    dyn_vals = [img.get("DYN", 0) for img in selected_images]
    qlt_vals = [img.get("QLT", 0) for img in selected_images]

    print("\n" + "=" * 70)
    print("SELECTION SUMMARY")
    print("=" * 70)
    print(f"Selected: {len(selected_images)} images")
    print(f"\nScore statistics:")
    print(f"  Range: [{min(scores):.2f}, {max(scores):.2f}]")
    print(f"  Mean: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"  Median: {np.median(scores):.2f}")
    print(f"\nDimension distributions:")
    print(f"  CIC: {np.bincount(cic_vals)}")
    print(f"  SEP: {np.bincount(sep_vals)}")
    print(f"  DYN: {np.bincount(dyn_vals)}")
    print(f"  QLT: {np.bincount(qlt_vals)}")
    print("=" * 70)


def main():
    """Main diversity selection pipeline for scored images."""
    print("=" * 70)
    print("DIVERSE SCORED STIMULI SELECTION")
    print("=" * 70)
    print(f"  Target images: {config.N_FINAL_SCORED_IMAGES}")
    print(f"  Similarity threshold: {config.SCORED_SIMILARITY_THRESHOLD}")
    print(f"  Text model: {config.TEXT_EMBEDDING_MODEL}")
    print("=" * 70 + "\n")

    # Validate paths
    scored_images_path = Path(config.SCORED_IMAGES_PATH)
    if not scored_images_path.exists():
        raise FileNotFoundError(f"Scored images not found: {scored_images_path}")

    # Load and filter images
    scored_images = load_scored_images(scored_images_path)
    eligible_images = filter_eligible_images(scored_images)

    if len(eligible_images) <= config.N_FINAL_SCORED_IMAGES:
        print(
            f"\nWarning: Only {len(eligible_images)} eligible images available, "
            f"requested {config.N_FINAL_SCORED_IMAGES}"
        )
        print("All eligible images will be included in final set.")
        # Just copy all images - no selection needed
        return

    # Initialize selector
    print("\nInitializing diversity selector...")
    selector = DiversitySelector(
        embedding_cache_path=Path(config.TEXT_EMBEDDING_CACHE_PATH),
    )

    # Compute story embeddings
    embeddings = compute_story_embeddings(eligible_images, selector)

    # Extract scores
    scores = np.array([img["score"] for img in eligible_images])

    # Select diverse, high-scoring subset
    selected_indices = selector.select_diverse_score_weighted(
        embeddings=embeddings,
        scores=scores,
        n_select=config.N_FINAL_SCORED_IMAGES,
        similarity_threshold=config.SCORED_SIMILARITY_THRESHOLD,
        random_state=config.RANDOM_SEED,
    )

    # Compute diversity metrics
    diversity_metrics = selector.compute_diversity_metrics(embeddings, selected_indices)

    print("\n" + "=" * 70)
    print("DIVERSITY METRICS")
    print("=" * 70)
    for key, value in diversity_metrics.items():
        print(
            f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}"
        )
    print("=" * 70)

    # Create output directory
    output_dir = create_output_directory()

    # Visualize embedding space
    print("\nGenerating visualizations...")
    labels = [img["image_id"] for img in eligible_images]

    selector.visualize_embedding_space(
        embeddings,
        selected_indices,
        output_path=output_dir / "visualizations" / "embedding_space_tsne.png",
        labels=labels,
        method="tsne",
    )

    selector.visualize_embedding_space(
        embeddings,
        selected_indices,
        output_path=output_dir / "visualizations" / "embedding_space_pca.png",
        labels=labels,
        method="pca",
    )

    # Save selected images
    image_source_dir = Path(config.OUTPUT_DIR) / "images"
    if not image_source_dir.exists():
        print(f"\nWarning: Image source directory not found: {image_source_dir}")
        print("Metadata will be saved but images won't be copied.")
        image_source_dir = Path(".")  # Dummy path

    save_selected_images(
        selected_indices,
        eligible_images,
        embeddings,
        diversity_metrics,
        output_dir,
        image_source_dir,
    )

    # Print summary
    print_selection_summary(selected_indices, eligible_images)

    # Final output
    print("\n" + "=" * 70)
    print("SELECTION COMPLETE")
    print("=" * 70)
    print(f"✓ Selected {len(selected_indices)} diverse, high-scoring images")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Images: {output_dir / 'images'}")
    print(f"✓ Metadata: {output_dir / 'annotations' / 'selected_dataset.json'}")
    print(f"✓ Visualizations: {output_dir / 'visualizations'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
