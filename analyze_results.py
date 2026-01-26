#!/usr/bin/env python3
"""
analyze_results.py - Analyze image scoring results

Quick analysis and filtering of scored images from results.jsonl

Usage:
    python analyze_results.py results.jsonl [--image-dir PATH]

Requirements:
    pip install numpy

Outputs:
    - Correlation matrix between CIC, SEP, and CHN
    - Top 10 high-scoring images by FinalScore
    - Copies top 10 images to ./top_images/ folder (if --image-dir provided)
    - Saves top 10 results to ./top_images/top_images_results.json
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_results(jsonl_path: str) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def calculate_scores(results: List[Dict]) -> List[Dict]:
    """
    Calculate eligibility and final score for each image.

    Eligible = 1 if CIC ≥ 2 AND SEP ≥ 1, otherwise 0
    FinalScore = Eligible × (2·CIC + SEP + CHN)

    Returns:
        Results with added 'eligible' and 'final_score' fields
    """
    for r in results:
        cic = r["CIC"]
        sep = r["SEP"]
        chn = r["CHN"]

        # Calculate eligibility
        eligible = 1 if (cic >= 2 and sep >= 1) else 0

        # Calculate final score
        final_score = eligible * (2 * cic + sep + chn)

        r["eligible"] = eligible
        r["final_score"] = final_score

    return results


def calculate_correlation_matrix(results: List[Dict]) -> Dict:
    """
    Calculate correlation matrix between CIC, SEP, and CHN.

    Returns:
        Dictionary with correlation coefficients
    """
    # Extract scores
    cic_scores = np.array([r["CIC"] for r in results])
    sep_scores = np.array([r["SEP"] for r in results])
    chn_scores = np.array([r["CHN"] for r in results])

    # Calculate correlations
    corr_cic_sep = np.corrcoef(cic_scores, sep_scores)[0, 1]
    corr_cic_chn = np.corrcoef(cic_scores, chn_scores)[0, 1]
    corr_sep_chn = np.corrcoef(sep_scores, chn_scores)[0, 1]

    return {"CIC_SEP": corr_cic_sep, "CIC_CHN": corr_cic_chn, "SEP_CHN": corr_sep_chn}


def print_correlation_matrix(correlations: Dict) -> None:
    """Print correlation matrix."""
    print("=" * 70)
    print("CORRELATION MATRIX (Pearson's r)")
    print("=" * 70)
    print("          CIC       SEP       CHN")
    print("-" * 70)
    print(
        f"CIC     1.000     {correlations['CIC_SEP']:6.3f}    {correlations['CIC_CHN']:6.3f}"
    )
    print(
        f"SEP     {correlations['CIC_SEP']:6.3f}    1.000     {correlations['SEP_CHN']:6.3f}"
    )
    print(
        f"CHN     {correlations['CIC_CHN']:6.3f}    {correlations['SEP_CHN']:6.3f}    1.000"
    )
    print("=" * 70)


def print_top_images(results: List[Dict], n: int = 10) -> List[Dict]:
    """
    Print top N images by final score.

    Returns:
        List of top N results
    """
    # Sort by final score (descending)
    sorted_results = sorted(results, key=lambda r: r["final_score"], reverse=True)

    # Get top N
    top_n = sorted_results[:n]

    print("\n" + "=" * 70)
    print(f"TOP {n} HIGH-SCORING IMAGES")
    print("=" * 70)
    print("Eligible = 1 if CIC ≥ 2 AND SEP ≥ 1, otherwise 0")
    print("FinalScore = Eligible × (2·CIC + SEP + CHN)")
    print("=" * 70)
    print(
        f"{'Image ID':<15} {'CIC':<5} {'SEP':<5} {'CHN':<5} {'Eligible':<10} {'FinalScore':<12}"
    )
    print("-" * 70)

    for r in top_n:
        print(
            f"{r['image_id']:<15} {r['CIC']:<5} {r['SEP']:<5} {r['CHN']:<5} "
            f"{r['eligible']:<10} {r['final_score']:<12}"
        )

    print("=" * 70)

    # Summary statistics
    total_eligible = sum(1 for r in results if r["eligible"] == 1)
    total_images = len(results)
    pct_eligible = 100 * total_eligible / total_images if total_images > 0 else 0

    print(f"\nTotal images: {total_images}")
    print(f"Eligible images: {total_eligible} ({pct_eligible:.1f}%)")
    print(
        f"Non-eligible images: {total_images - total_eligible} ({100 - pct_eligible:.1f}%)"
    )

    return top_n


def copy_top_images(top_results: List[Dict], image_dir: Path, output_dir: Path) -> None:
    """
    Copy top images to output directory and save their results.

    Args:
        top_results: List of top N results
        image_dir: Source directory containing images
        output_dir: Output directory for top images
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("COPYING TOP IMAGES")
    print("=" * 70)

    copied = 0
    missing = 0

    for r in top_results:
        image_id = r["image_id"]
        src_path = image_dir / f"{image_id}.jpg"
        dst_path = output_dir / f"{image_id}.jpg"

        if src_path.exists():
            shutil.copy(src_path, dst_path)
            copied += 1
            print(f"✓ Copied {image_id}.jpg")
        else:
            missing += 1
            print(f"✗ Missing: {image_id}.jpg")

    # Save results JSON
    results_path = output_dir / "top_images_results.json"
    with open(results_path, "w") as f:
        json.dump(top_results, f, indent=2)

    print("=" * 70)
    print(f"✓ Copied {copied} images to {output_dir}")
    if missing > 0:
        print(f"✗ {missing} images not found in source directory")
    print(f"✓ Saved results to {results_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze image scoring results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="./data/scored_images/results.jsonl",
        help="Path to results.jsonl file",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./data/processed/images",
        help="Directory containing source images (default: ./data/processed/images)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./top_images",
        help="Output directory for top images (default: ./top_images)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top images to copy (default: 20)",
    )

    args = parser.parse_args()

    # Validate results file
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: File not found: {results_path}")
        sys.exit(1)

    # Load results
    print(f"Loading results from {results_path}...\n")
    results = load_results(str(results_path))

    if not results:
        print("No results found!")
        sys.exit(1)

    # Calculate scores
    results = calculate_scores(results)

    # Print correlation matrix
    correlations = calculate_correlation_matrix(results)
    print_correlation_matrix(correlations)

    # Print top N images and get the list
    top_results = print_top_images(results, n=args.top_n)

    # Copy top images to output directory
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    if image_dir.exists():
        copy_top_images(top_results, image_dir, output_dir)
    else:
        print(f"\n⚠️  Image directory not found: {image_dir}")
        print(f"Skipping image copy. Use --image-dir to specify correct path.")
        print(f"Top {args.top_n} results displayed above.")


if __name__ == "__main__":
    main()
