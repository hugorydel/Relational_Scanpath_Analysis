#!/usr/bin/env python3
"""
select_scored_stimuli.py - Select high-scoring subset from OpenAI-scored images

Selects final stimuli from OpenAI-scored images based purely on score.
No diversity filtering at this stage - that's done as post-hoc analysis.

Usage:
    python select_scored_stimuli.py

Configuration:
    Edit config.py to adjust:
    - SCORED_IMAGES_PATH: Path to results.jsonl
    - N_FINAL_SCORED_IMAGES: Target number of images
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

import config
from config import calculate_image_score


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


def select_top_by_score(eligible_images: List[Dict], n_select: int) -> List[Dict]:
    """
    Select top N images by score.

    Args:
        eligible_images: List of eligible image dicts
        n_select: Number of images to select

    Returns:
        selected_images: List of selected image dicts
    """
    print(f"\nSelecting top {n_select} images by score...")

    # Sort by score (descending)
    sorted_images = sorted(eligible_images, key=lambda x: x["score"], reverse=True)

    # Take top N
    selected_images = sorted_images[:n_select]

    print(f"✓ Selected {len(selected_images)} images")

    return selected_images


def create_output_directory(base_dir: Path = None) -> Path:
    """Create output directory for selected images."""
    if base_dir is None:
        base_dir = Path("./data")

    output_dir = base_dir / "scored_selection"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "annotations").mkdir(exist_ok=True)

    return output_dir


def save_selected_images(
    selected_images: List[Dict],
    output_dir: Path,
    image_source_dir: Path,
):
    """
    Save selected images and metadata.

    Args:
        selected_images: List of selected image dicts
        output_dir: Where to save outputs
        image_source_dir: Where to find source images
    """
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
            "description": "High-Scoring Stimuli Dataset (Score-only Selection)",
            "num_images": len(selected_images),
            "selection_method": "top_by_score",
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


def print_selection_summary(selected_images: List[Dict]):
    """Print summary statistics of selected images."""
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
    """Main selection pipeline for scored images."""
    print("=" * 70)
    print("HIGH-SCORING STIMULI SELECTION")
    print("=" * 70)
    print(f"  Target images: {config.N_FINAL_SCORED_IMAGES}")
    print(f"  Selection method: Top N by score (no diversity filtering)")
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
        selected_images = eligible_images
    else:
        # Select top N by score
        selected_images = select_top_by_score(
            eligible_images, config.N_FINAL_SCORED_IMAGES
        )

    # Create output directory
    output_dir = create_output_directory()

    # Save selected images
    image_source_dir = Path(config.OUTPUT_DIR) / "images"
    if not image_source_dir.exists():
        print(f"\nWarning: Image source directory not found: {image_source_dir}")
        print("Metadata will be saved but images won't be copied.")
        image_source_dir = Path(".")  # Dummy path

    save_selected_images(selected_images, output_dir, image_source_dir)

    # Print summary
    print_selection_summary(selected_images)

    # Final output
    print("\n" + "=" * 70)
    print("SELECTION COMPLETE")
    print("=" * 70)
    print(f"✓ Selected {len(selected_images)} high-scoring images")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Images: {output_dir / 'images'}")
    print(f"✓ Metadata: {output_dir / 'annotations' / 'selected_dataset.json'}")
    print("\nNext steps:")
    print("  1. Run manual_image_annotation.py for manual review")
    print("  2. Run diversity_analysis.py for diversity metrics")
    print("=" * 70)


if __name__ == "__main__":
    main()
