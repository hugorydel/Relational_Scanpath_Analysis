#!/usr/bin/env python3
"""
analyze_results.py - Analyze image scoring results

Quick analysis and filtering of scored images from results.jsonl

Usage:
    python analyze_results.py --results_file results.jsonl [--image-dir PATH]

Requirements:
    pip install numpy

Outputs:
    - Correlation matrix between CIC, SEP, DYN, QLT
    - Top N high-scoring images by Score
    - Copies top N images to ./top_images/ folder (if --image-dir exists)
    - Saves top N results to ./top_images/top_images_results.json
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
    Calculate eligibility and score for each image.

    Eligible = 1 if CIC ≥ 2 AND SEP ≥ 1 AND QLT ≥ 1, otherwise 0
    Score = Eligible × (2·CIC + SEP + DYN + QLT)

    Returns:
        Results with added 'eligible' and 'score' fields
    """
    for r in results:
        # Gracefully handle missing keys (treat as 0)
        cic = int(r.get("CIC", 0))
        sep = int(r.get("SEP", 0))
        dyn = int(r.get("DYN", 0))
        qlt = int(r.get("QLT", 0))

        eligible = 1 if (cic >= 2 and sep >= 1 and dyn >= 1 and qlt >= 1) else 0
        score = eligible * (cic * 2.5 + sep + dyn * 1.5 + qlt)

        r["eligible"] = eligible
        r["score"] = score
        # Backward-compatible alias
        r["final_score"] = score

    return results


def calculate_correlation_matrix(results: List[Dict]) -> Dict:
    """
    Calculate correlation matrix between CIC, SEP, DYN, QLT.

    Returns:
        Dictionary with:
          - labels: list[str]
          - matrix: 2D numpy array (Pearson's r)
    """
    labels = ["CIC", "SEP", "DYN", "QLT"]

    data = np.array(
        [
            [
                int(r.get("CIC", 0)),
                int(r.get("SEP", 0)),
                int(r.get("DYN", 0)),
                int(r.get("QLT", 0)),
            ]
            for r in results
        ],
        dtype=float,
    )

    # If there's only one row, correlation is undefined; return identity for safe printing.
    if data.shape[0] < 2:
        matrix = np.eye(len(labels), dtype=float)
    else:
        matrix = np.corrcoef(data, rowvar=False)

        # Replace NaNs (e.g., constant columns) with 0 off-diagonal, 1 on diagonal.
        if np.isnan(matrix).any():
            fixed = np.nan_to_num(matrix, nan=0.0)
            np.fill_diagonal(fixed, 1.0)
            matrix = fixed

    return {"labels": labels, "matrix": matrix}


def print_correlation_matrix(correlations: Dict) -> None:
    """Print correlation matrix."""
    labels = correlations["labels"]
    mat = correlations["matrix"]

    print("=" * 70)
    print("CORRELATION MATRIX (Pearson's r)")
    print("=" * 70)

    header = " " * 10 + "".join([f"{lab:>9}" for lab in labels])
    print(header)
    print("-" * 70)

    for i, row_lab in enumerate(labels):
        row_vals = "".join([f"{mat[i, j]:9.3f}" for j in range(len(labels))])
        print(f"{row_lab:<10}{row_vals}")

    print("=" * 70)


def print_top_images(results: List[Dict], n: int = 20) -> List[Dict]:
    """
    Print top N images by score.

    Returns:
        List of top N results
    """
    # Sort by score (descending)
    sorted_results = sorted(results, key=lambda r: r.get("score", 0), reverse=True)

    # Get top N
    top_n = sorted_results[:n]

    print("\n" + "=" * 70)
    print(f"TOP {n} HIGH-SCORING IMAGES")
    print("=" * 70)
    print("Eligible = 1 if CIC ≥ 2 AND SEP ≥ 1 AND QLT ≥ 1, otherwise 0")
    print("Score = Eligible × (2·CIC + SEP + DYN + QLT)")
    print("=" * 70)
    print(
        f"{'Image ID':<15} {'CIC':<5} {'SEP':<5} {'DYN':<5} {'QLT':<5} {'Eligible':<10} {'Score':<8}"
    )
    print("-" * 70)

    for r in top_n:
        print(
            f"{r.get('image_id',''):<15} {int(r.get('CIC',0)):<5} {int(r.get('SEP',0)):<5} "
            f"{int(r.get('DYN',0)):<5} {int(r.get('QLT',0)):<5} {int(r.get('eligible',0)):<10} {int(r.get('score',0)):<8}"
        )

    print("=" * 70)

    # Summary statistics
    total_eligible = sum(1 for r in results if int(r.get("eligible", 0)) == 1)
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
