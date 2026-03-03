"""
analyse_enc1_enc2_shift.py
==========================
Per-participant analysis of the encoding viewingNumber 1 → 2 shift in
relational scanpath alignment (SVG z-scores).

Tests whether participants show higher relational structure in their
second encoding viewing than their first — consistent with top-down
relational processing emerging after initial gist extraction.

Metrics tested:
    svg_z_inter  — interactional edges only (primary)
    svg_z_all    — all relation types (comparison)

Statistical test:
    Paired Wilcoxon signed-rank test (non-parametric; robust to non-normality
    in small per-participant samples). Falls back to reporting descriptives
    only if too few valid pairs exist.

Usage:
    # Single participant
    python analyse_enc1_enc2_shift.py --subject Encode-Decode_Experiment-1-1

    # All participants (runs per-participant, no group aggregation)
    python analyse_enc1_enc2_shift.py
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from pipeline.misc import get_subject_ids, setup_logging

logger = logging.getLogger(__name__)

# Minimum valid (non-NaN, non-low_n) pairs required to run the test
MIN_PAIRS = 5


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyse_shift(subject_id: str) -> dict | None:
    """
    Load trial_features.csv for one participant and test the enc1 -> enc2
    SVG z-score shift.

    Returns a results dict, or None if the file is missing.
    """
    features_path = config.OUTPUT_FEATURES_DIR / f"{subject_id}_trial_features.csv"
    if not features_path.exists():
        logger.error(
            f"  [{subject_id}] trial_features.csv not found -- run Module 3 first."
        )
        return None

    df = pd.read_csv(features_path, dtype={"StimID": str})

    enc = df[df["Phase"] == "encoding"].copy()
    enc1 = enc[enc["ViewingNumber"] == 1.0].set_index("StimID")
    enc2 = enc[enc["ViewingNumber"] == 2.0].set_index("StimID")

    # Align on shared StimIDs
    shared = enc1.index.intersection(enc2.index)
    enc1 = enc1.loc[shared]
    enc2 = enc2.loc[shared]

    results = {"subject_id": subject_id, "n_images": len(shared)}

    for metric in ["svg_z_inter", "svg_z_all"]:
        v1 = enc1[metric]
        v2 = enc2[metric]

        # Drop pairs where either viewing is NaN or low_n
        low_n1 = enc1["low_n"].fillna(True)
        low_n2 = enc2["low_n"].fillna(True)
        valid = v1.notna() & v2.notna() & ~low_n1 & ~low_n2

        d1 = v1[valid].values
        d2 = v2[valid].values
        diff = d2 - d1  # positive = enc2 > enc1

        n_valid = int(valid.sum())
        n_dropped = int((~valid).sum())
        mean_enc1 = float(np.mean(d1)) if n_valid > 0 else np.nan
        mean_enc2 = float(np.mean(d2)) if n_valid > 0 else np.nan
        mean_diff = float(np.mean(diff)) if n_valid > 0 else np.nan
        std_diff = float(np.std(diff, ddof=1)) if n_valid > 1 else np.nan

        # Wilcoxon signed-rank test
        stat, p = np.nan, np.nan
        test_note = ""
        if n_valid >= MIN_PAIRS:
            nonzero = diff[diff != 0]
            if len(nonzero) >= 1:
                try:
                    stat, p = stats.wilcoxon(diff, alternative="greater")
                    test_note = "Wilcoxon signed-rank (one-tailed: enc2 > enc1)"
                except Exception as e:
                    test_note = f"Test failed: {e}"
            else:
                test_note = "All differences zero -- test not applicable"
        else:
            test_note = (
                f"Too few valid pairs (n={n_valid} < {MIN_PAIRS}) -- descriptives only"
            )

        results[metric] = {
            "n_valid": n_valid,
            "n_dropped": n_dropped,
            "mean_enc1": mean_enc1,
            "mean_enc2": mean_enc2,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "stat": stat,
            "p": p,
            "test_note": test_note,
            "diffs": diff.tolist(),
        }

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: dict):
    sid = results["subject_id"]
    n = results["n_images"]

    print(f"\n{'='*65}")
    print(f"ENC1 -> ENC2 SVG ALIGNMENT SHIFT  |  {sid}")
    print(f"{'='*65}")
    print(f"Images: {n}")

    for metric in ["svg_z_inter", "svg_z_all"]:
        r = results[metric]
        label = (
            "SVG z (interactional)"
            if metric == "svg_z_inter"
            else "SVG z (all relations)"
        )

        print(f"\n--- {label} ---")
        print(f"  Valid pairs   : {r['n_valid']}  (dropped: {r['n_dropped']})")

        if r["n_valid"] == 0:
            print("  No valid data.")
            continue

        print(f"  Mean enc1     : {r['mean_enc1']:+.3f}")
        print(f"  Mean enc2     : {r['mean_enc2']:+.3f}")
        print(f"  Mean diff     : {r['mean_diff']:+.3f}  (enc2 - enc1)")
        print(f"  SD diff       : {r['std_diff']:.3f}")

        n_pos = sum(d > 0 for d in r["diffs"])
        n_neg = sum(d < 0 for d in r["diffs"])
        n_zer = sum(d == 0 for d in r["diffs"])
        print(f"  Direction     : {n_pos} images up  {n_neg} images down  {n_zer} ties")

        if not np.isnan(r["stat"]):
            sig = "  *" if r["p"] < 0.05 else ""
            print(f"  Wilcoxon W    : {r['stat']:.1f}")
            print(f"  p (one-tail)  : {r['p']:.4f}{sig}")
        print(f"  Note          : {r['test_note']}")

    print(f"\n{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    setup_logging(level=config.LOG_LEVEL)

    parser = argparse.ArgumentParser(
        description="Per-participant enc1 -> enc2 SVG alignment shift analysis."
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Single subject ID (default: all discovered subjects)",
    )
    args = parser.parse_args()

    if args.subject:
        subject_ids = [args.subject]
    else:
        subject_ids = get_subject_ids()
        logger.info(f"Discovered {len(subject_ids)} subjects.")

    for subject_id in subject_ids:
        results = analyse_shift(subject_id)
        if results:
            print_report(results)


if __name__ == "__main__":
    main()
