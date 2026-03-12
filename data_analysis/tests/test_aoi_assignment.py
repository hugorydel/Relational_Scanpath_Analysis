"""
test_aoi_assignment.py
===========
Quick diagnostic for AOI assignment on a single participant.

Usage:
    python test_aoi_assignment.py
    python test_aoi_assignment.py --subject Encode-Decode_Experiment-1-1
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from pipeline.misc import setup_logging
from pipeline.module_3.aoi import run_aoi_assignment

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        default="Encode-Decode_Experiment-1-1",
        help="Subject ID stem (default: Encode-Decode_Experiment-1-1)",
    )
    args = parser.parse_args()

    subject_id = args.subject
    fixations_path = config.OUTPUT_EYETRACKING_DIR / f"{subject_id}_fixations.csv"
    output_path = config.OUTPUT_FEATURES_DIR / f"{subject_id}_fixations_aoi.csv"

    # Run assignment
    enriched = run_aoi_assignment(
        subject_id=subject_id,
        fixations_path=fixations_path,
        output_path=output_path,
    )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------
    import numpy as np
    import pandas as pd

    print("\n" + "=" * 60)
    print("AOI ASSIGNMENT DIAGNOSTICS")
    print("=" * 60)

    n = len(enriched)
    print(f"\nTotal fixations : {n}")

    # Assignment method breakdown by phase
    print("\n--- Assignment method by phase ---")
    breakdown = (
        enriched.groupby(["Phase", "AssignmentMethod"]).size().unstack(fill_value=0)
    )
    # Ensure all columns present
    for col in ["polygon", "proximity", "none"]:
        if col not in breakdown.columns:
            breakdown[col] = 0
    breakdown["total"] = breakdown.sum(axis=1)
    for col in ["polygon", "proximity", "none"]:
        breakdown[f"{col}_pct"] = (breakdown[col] / breakdown["total"] * 100).round(1)
    print(
        breakdown[
            [
                "polygon",
                "polygon_pct",
                "proximity",
                "proximity_pct",
                "none",
                "none_pct",
                "total",
            ]
        ].to_string()
    )

    # Proximity distance distribution (for proximity hits only)
    prox = enriched[enriched["AssignmentMethod"] == "proximity"]["ProximityDist_px"]
    if len(prox) > 0:
        print(f"\n--- Proximity fallback distances (n={len(prox)}) ---")
        print(f"  Min    : {prox.min():.1f} px")
        print(f"  Median : {prox.median():.1f} px")
        print(f"  Mean   : {prox.mean():.1f} px")
        print(f"  Max    : {prox.max():.1f} px")

    # Saliency sanity check
    sal = enriched["SalienceAtFixation"].dropna()
    print(f"\n--- Saliency at fixation (n={len(sal)}) ---")
    print(f"  Min    : {sal.min():.6f}")
    print(f"  Median : {sal.median():.6f}")
    print(f"  Mean   : {sal.mean():.6f}")
    print(f"  Max    : {sal.max():.6f}")

    # Per-image miss rates (AOI=none)
    print("\n--- Per-image miss rates (AssignmentMethod == 'none') ---")
    per_image = (
        enriched.groupby("StimID", group_keys=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n_fix": len(g),
                    "n_none": (g["AssignmentMethod"] == "none").sum(),
                    "pct_none": round(
                        100 * (g["AssignmentMethod"] == "none").mean(), 1
                    ),
                }
            ),
            include_groups=False,
        )
        .sort_values("pct_none", ascending=False)
    )
    print(per_image.head(10).to_string())

    # Sample rows
    print("\n--- Sample output rows (first 5) ---")
    cols = [
        "SubjectID",
        "StimID",
        "Phase",
        "ImgX",
        "ImgY",
        "ObjectID",
        "ObjectName",
        "AssignmentMethod",
        "ProximityDist_px",
        "SalienceAtFixation",
    ]
    print(enriched[cols].head().to_string(index=False))

    print(f"\nOutput written → {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
