#!/usr/bin/env python3
"""
Compare scoring results between two OpenAI models.

Usage:
    python compare_model_results.py --gpt52 results.jsonl --gpt5mini results_gpt5mini.jsonl

Outputs:
    - Correlation analysis between models
    - Eligibility agreement statistics
    - Exact match proportions
    - Top 20 comparison
    - comparative_results.jsonl (gpt-5-mini's top 20)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_results(jsonl_path: Path) -> Dict[str, Dict]:
    """Load results from JSONL file, indexed by image_id."""
    results = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                results[result["image_id"]] = result
    return results


def calculate_eligibility(result: Dict) -> bool:
    """Calculate if an image is eligible (CIC >= 2 AND SEP >= 1 AND QLT >= 1)."""
    cic = int(result.get("CIC", 0))
    sep = int(result.get("SEP", 0))
    qlt = int(result.get("QLT", 0))
    return cic >= 2 and sep >= 1 and qlt >= 1


def calculate_score(result: Dict) -> int:
    """Calculate eligibility score."""
    cic = int(result.get("CIC", 0))
    sep = int(result.get("SEP", 0))
    dyn = int(result.get("DYN", 0))
    qlt = int(result.get("QLT", 0))

    eligible = 1 if calculate_eligibility(result) else 0
    return eligible * (2 * cic + sep + dyn + qlt)


def match_results(gpt52_results: Dict, gpt5mini_results: Dict) -> List[str]:
    """Find image IDs present in both result sets."""
    gpt52_ids = set(gpt52_results.keys())
    gpt5mini_ids = set(gpt5mini_results.keys())
    matched_ids = gpt52_ids & gpt5mini_ids
    return sorted(list(matched_ids), key=lambda x: int(x) if x.isdigit() else 0)


def calculate_correlations(
    matched_ids: List[str], gpt52_results: Dict, gpt5mini_results: Dict
) -> Dict:
    """Calculate Pearson correlations for each dimension."""
    dimensions = ["CIC", "SEP", "DYN", "QLT"]
    correlations = {}

    for dim in dimensions:
        gpt52_values = np.array(
            [int(gpt52_results[img_id].get(dim, 0)) for img_id in matched_ids]
        )
        gpt5mini_values = np.array(
            [int(gpt5mini_results[img_id].get(dim, 0)) for img_id in matched_ids]
        )

        if len(gpt52_values) > 1:
            corr = np.corrcoef(gpt52_values, gpt5mini_values)[0, 1]
            correlations[dim] = corr if not np.isnan(corr) else 0.0
        else:
            correlations[dim] = 0.0

    return correlations


def calculate_agreement_stats(
    matched_ids: List[str], gpt52_results: Dict, gpt5mini_results: Dict
) -> Dict:
    """Calculate agreement statistics between models."""

    # Eligibility agreement
    both_eligible = 0
    both_ineligible = 0
    gpt52_only = 0
    gpt5mini_only = 0

    # Exact matches
    exact_matches = 0
    cic_matches = 0
    sep_matches = 0
    dyn_matches = 0
    qlt_matches = 0

    for img_id in matched_ids:
        gpt52 = gpt52_results[img_id]
        gpt5mini = gpt5mini_results[img_id]

        gpt52_eligible = calculate_eligibility(gpt52)
        gpt5mini_eligible = calculate_eligibility(gpt5mini)

        # Eligibility agreement
        if gpt52_eligible and gpt5mini_eligible:
            both_eligible += 1
        elif not gpt52_eligible and not gpt5mini_eligible:
            both_ineligible += 1
        elif gpt52_eligible:
            gpt52_only += 1
        else:
            gpt5mini_only += 1

        # Exact matches per dimension
        if gpt52.get("CIC") == gpt5mini.get("CIC"):
            cic_matches += 1
        if gpt52.get("SEP") == gpt5mini.get("SEP"):
            sep_matches += 1
        if gpt52.get("DYN") == gpt5mini.get("DYN"):
            dyn_matches += 1
        if gpt52.get("QLT") == gpt5mini.get("QLT"):
            qlt_matches += 1

        # Complete exact match (all 4 dimensions)
        if (
            gpt52.get("CIC") == gpt5mini.get("CIC")
            and gpt52.get("SEP") == gpt5mini.get("SEP")
            and gpt52.get("DYN") == gpt5mini.get("DYN")
            and gpt52.get("QLT") == gpt5mini.get("QLT")
        ):
            exact_matches += 1

    total = len(matched_ids)

    return {
        "total_images": total,
        "eligibility": {
            "both_eligible": both_eligible,
            "both_ineligible": both_ineligible,
            "gpt52_only": gpt52_only,
            "gpt5mini_only": gpt5mini_only,
            "agreement_rate": (
                (both_eligible + both_ineligible) / total if total > 0 else 0
            ),
        },
        "exact_matches": {
            "complete_match": exact_matches,
            "complete_match_rate": exact_matches / total if total > 0 else 0,
            "CIC_match": cic_matches,
            "CIC_match_rate": cic_matches / total if total > 0 else 0,
            "SEP_match": sep_matches,
            "SEP_match_rate": sep_matches / total if total > 0 else 0,
            "DYN_match": dyn_matches,
            "DYN_match_rate": dyn_matches / total if total > 0 else 0,
            "QLT_match": qlt_matches,
            "QLT_match_rate": qlt_matches / total if total > 0 else 0,
        },
    }


def get_top_n(results: Dict, n: int = 20) -> List[Tuple[str, Dict, int]]:
    """Get top N images by score, filtered to score > 0."""
    scored = []
    for img_id, result in results.items():
        score = calculate_score(result)
        if score > 0:  # Only include images with positive scores
            scored.append((img_id, result, score))

    # Sort by score descending, then by image_id for tie-breaking
    scored.sort(key=lambda x: (-x[2], int(x[0]) if x[0].isdigit() else 0))
    return scored[:n]


def print_report(
    correlations: Dict, agreement: Dict, gpt52_top20: List, gpt5mini_top20: List
):
    """Print comprehensive comparison report."""

    print("=" * 80)
    print("MODEL COMPARISON REPORT: GPT-5.2 vs GPT-5-MINI")
    print("=" * 80)

    # Correlations
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS (Pearson's r)")
    print("=" * 80)
    for dim, corr in correlations.items():
        print(f"{dim:>4}: {corr:6.3f}")

    # Eligibility agreement
    print("\n" + "=" * 80)
    print("ELIGIBILITY AGREEMENT")
    print("=" * 80)
    print(f"Total images compared: {agreement['total_images']}")
    print(f"\nAgreement breakdown:")
    print(
        f"  Both eligible:     {agreement['eligibility']['both_eligible']:4d} ({100*agreement['eligibility']['both_eligible']/agreement['total_images']:5.1f}%)"
    )
    print(
        f"  Both ineligible:   {agreement['eligibility']['both_ineligible']:4d} ({100*agreement['eligibility']['both_ineligible']/agreement['total_images']:5.1f}%)"
    )
    print(
        f"  GPT-5.2 only:      {agreement['eligibility']['gpt52_only']:4d} ({100*agreement['eligibility']['gpt52_only']/agreement['total_images']:5.1f}%)"
    )
    print(
        f"  GPT-5-mini only:   {agreement['eligibility']['gpt5mini_only']:4d} ({100*agreement['eligibility']['gpt5mini_only']/agreement['total_images']:5.1f}%)"
    )
    print(
        f"\nOverall agreement rate: {100*agreement['eligibility']['agreement_rate']:.1f}%"
    )

    # Exact matches
    print("\n" + "=" * 80)
    print("EXACT MATCH ANALYSIS")
    print("=" * 80)
    print(f"Complete matches (all 4 dimensions identical):")
    print(f"  Count: {agreement['exact_matches']['complete_match']}")
    print(f"  Rate:  {100*agreement['exact_matches']['complete_match_rate']:.1f}%")
    print(f"\nPer-dimension match rates:")
    print(
        f"  CIC: {agreement['exact_matches']['CIC_match']:4d} ({100*agreement['exact_matches']['CIC_match_rate']:5.1f}%)"
    )
    print(
        f"  SEP: {agreement['exact_matches']['SEP_match']:4d} ({100*agreement['exact_matches']['SEP_match_rate']:5.1f}%)"
    )
    print(
        f"  DYN: {agreement['exact_matches']['DYN_match']:4d} ({100*agreement['exact_matches']['DYN_match_rate']:5.1f}%)"
    )
    print(
        f"  QLT: {agreement['exact_matches']['QLT_match']:4d} ({100*agreement['exact_matches']['QLT_match_rate']:5.1f}%)"
    )

    # Top 20 comparison
    print("\n" + "=" * 80)
    print("TOP 20 COMPARISON")
    print("=" * 80)

    gpt52_top20_ids = {img_id for img_id, _, _ in gpt52_top20}
    gpt5mini_top20_ids = {img_id for img_id, _, _ in gpt5mini_top20}
    overlap = gpt52_top20_ids & gpt5mini_top20_ids

    print(f"GPT-5.2 top 20:      {len(gpt52_top20)} images")
    print(f"GPT-5-mini top 20:   {len(gpt5mini_top20)} images")
    print(f"Overlap:             {len(overlap)} images ({100*len(overlap)/20:.1f}%)")
    print(f"GPT-5.2 unique:      {len(gpt52_top20_ids - overlap)} images")
    print(f"GPT-5-mini unique:   {len(gpt5mini_top20_ids - overlap)} images")

    # Show top 20 side by side
    print("\n" + "-" * 80)
    print(
        f"{'Rank':<5} {'GPT-5.2':<15} {'Score':<6} {'GPT-5-mini':<15} {'Score':<6} {'Match':<5}"
    )
    print("-" * 80)

    for i in range(max(len(gpt52_top20), len(gpt5mini_top20))):
        rank = i + 1

        if i < len(gpt52_top20):
            gpt52_id, _, gpt52_score = gpt52_top20[i]
        else:
            gpt52_id, gpt52_score = "-", "-"

        if i < len(gpt5mini_top20):
            gpt5mini_id, _, gpt5mini_score = gpt5mini_top20[i]
        else:
            gpt5mini_id, gpt5mini_score = "-", "-"

        match = "✓" if (gpt52_id != "-" and gpt52_id == gpt5mini_id) else ""

        print(
            f"{rank:<5} {str(gpt52_id):<15} {str(gpt52_score):<6} {str(gpt5mini_id):<15} {str(gpt5mini_score):<6} {match:<5}"
        )

    print("=" * 80)


def save_comparative_results(gpt5mini_top20: List, output_path: Path):
    """Save GPT-5-mini's top 20 to comparative_results.jsonl."""
    with open(output_path, "w") as f:
        for img_id, result, score in gpt5mini_top20:
            # Add score to result
            result_with_score = result.copy()
            result_with_score["score"] = score
            result_with_score["eligible"] = 1 if score > 0 else 0
            f.write(json.dumps(result_with_score) + "\n")

    print(f"\n✓ Saved GPT-5-mini top {len(gpt5mini_top20)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare scoring results between GPT-5.2 and GPT-5-mini"
    )
    parser.add_argument(
        "--gpt52", type=str, required=True, help="GPT-5.2 results JSONL file"
    )
    parser.add_argument(
        "--gpt5mini", type=str, required=True, help="GPT-5-mini results JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparative_results.jsonl",
        help="Output file for GPT-5-mini top 20 (default: comparative_results.jsonl)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top images to compare (default: 20)",
    )

    args = parser.parse_args()

    # Validate inputs
    gpt52_path = Path(args.gpt52)
    gpt5mini_path = Path(args.gpt5mini)

    if not gpt52_path.exists():
        print(f"Error: File not found: {gpt52_path}")
        return 1

    if not gpt5mini_path.exists():
        print(f"Error: File not found: {gpt5mini_path}")
        return 1

    # Load results
    print(f"Loading GPT-5.2 results from {gpt52_path}...")
    gpt52_results = load_results(gpt52_path)
    print(f"✓ Loaded {len(gpt52_results)} GPT-5.2 results")

    print(f"Loading GPT-5-mini results from {gpt5mini_path}...")
    gpt5mini_results = load_results(gpt5mini_path)
    print(f"✓ Loaded {len(gpt5mini_results)} GPT-5-mini results")

    # Match images
    matched_ids = match_results(gpt52_results, gpt5mini_results)
    print(f"✓ Matched {len(matched_ids)} images present in both result sets\n")

    if len(matched_ids) == 0:
        print("Error: No matching images found between the two result sets!")
        return 1

    # Calculate statistics
    correlations = calculate_correlations(matched_ids, gpt52_results, gpt5mini_results)
    agreement = calculate_agreement_stats(matched_ids, gpt52_results, gpt5mini_results)

    # Get top N for each model (from matched images only)
    gpt52_matched = {img_id: gpt52_results[img_id] for img_id in matched_ids}
    gpt5mini_matched = {img_id: gpt5mini_results[img_id] for img_id in matched_ids}

    gpt52_top20 = get_top_n(gpt52_matched, n=args.top_n)
    gpt5mini_top20 = get_top_n(gpt5mini_matched, n=args.top_n)

    # Print report
    print_report(correlations, agreement, gpt52_top20, gpt5mini_top20)

    # Save comparative results
    output_path = Path(args.output)
    save_comparative_results(gpt5mini_top20, output_path)

    return 0


if __name__ == "__main__":
    exit(main())
