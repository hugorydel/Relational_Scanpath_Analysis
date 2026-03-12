"""
analyse_enc1_enc2_shift.py
==========================
Group-level analysis of the first → second encoding trial shift in
relational scanpath alignment (SVG z-scores).

With the current experiment design, each image is shown twice at encoding
in separate trials, each with a different relational question. This script
tests whether SVG alignment differs between the first and second encoding
trial per image — the two trials are distinguished by TrialIndex (lower
TrialIndex = shown first, higher = shown second).

Note: because the two encoding trials present different questions, any
shift is confounded with question content and should be interpreted
cautiously.

Group-level test (two-stage):
    Stage 1 — per participant: compute mean(enc_second - enc_first) SVG diff
              across all valid image pairs. This gives one value per participant.
    Stage 2 — across participants: one-sample Wilcoxon signed-rank test
              of those mean diffs against 0.
    This respects the nested structure (images within participants) and avoids
    pseudo-replication from pooling raw image-level diffs.

Per-participant descriptives are also printed for inspection.

Metric tested:
    svg_z  — core edges (interactional + spatial + functional)

Source data:
    Loads {SubjectID}_fixations_aoi.csv from output/features/ and computes
    SVG alignment per encoding trial (by TrialIndex) on the fly, since
    module 3 merges both encoding trials per image into one sequence.

Usage:
    # All participants (default)
    python analyse_enc1_enc2_shift.py

    # Single participant (descriptives only — group test needs ≥2)
    python analyse_enc1_enc2_shift.py --subject Encode-Decode_Experiment-1-1

    # Custom random seed
    python analyse_enc1_enc2_shift.py --seed 99
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
from pipeline.module_3.metrics import build_object_sequence, svg_alignment
from pipeline.module_3.scene_graph import build_graph_index

logger = logging.getLogger(__name__)

# Minimum valid (non-NaN, non-low_n) image pairs per participant
MIN_PAIRS_PER_SUBJECT = 5
# Minimum participants required to run the group-level test
MIN_PARTICIPANTS = 3


# ---------------------------------------------------------------------------
# Per-trial SVG computation
# ---------------------------------------------------------------------------


def _compute_svg_per_trial(
    fixations_aoi: pd.DataFrame,
    graph_index: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Compute SVG alignment scores separately for every (StimID, TrialIndex)
    combination in the encoding phase.

    Returns a DataFrame with columns:
        StimID, TrialIndex, svg_z, low_n
    """
    enc = fixations_aoi[fixations_aoi["Phase"] == "encoding"].copy()
    records = []

    for (stim_id, trial_idx), group in enc.groupby(["StimID", "TrialIndex"], sort=True):
        stim_id = str(stim_id)
        seq = build_object_sequence(group)

        edges = graph_index["all"].get(stim_id, set())
        svg = svg_alignment(
            seq, edges, n_permutations=config.SVG_N_PERMUTATIONS, rng=rng
        )

        records.append(
            {
                "StimID": stim_id,
                "TrialIndex": int(trial_idx),
                "svg_z": svg["svg_z"],
                "low_n": svg["low_n"],
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Per-participant analysis
# ---------------------------------------------------------------------------


def analyse_shift_subject(
    subject_id: str, graph_index: dict, rng: np.random.Generator
) -> dict | None:
    """
    Load fixations_aoi.csv for one participant, compute per-trial SVG scores,
    and compute the enc_first → enc_second diff for each image.

    Returns a dict with per-image diffs and per-participant summary stats,
    or None if the file is missing.
    """
    aoi_path = config.OUTPUT_FEATURES_DIR / f"{subject_id}_fixations_aoi.csv"
    if not aoi_path.exists():
        logger.error(
            f"  [{subject_id}] fixations_aoi.csv not found — run Module 3 first."
        )
        return None

    fixations_aoi = pd.read_csv(aoi_path, dtype={"StimID": str})
    trial_svg = _compute_svg_per_trial(fixations_aoi, graph_index, rng)

    # Rank the two TrialIndex values per StimID: rank 1 = first shown
    trial_svg = trial_svg.sort_values(["StimID", "TrialIndex"])
    trial_svg["enc_order"] = trial_svg.groupby("StimID").cumcount() + 1  # 1 or 2

    enc_first = trial_svg[trial_svg["enc_order"] == 1].set_index("StimID")
    enc_second = trial_svg[trial_svg["enc_order"] == 2].set_index("StimID")

    shared = enc_first.index.intersection(enc_second.index)
    enc_first = enc_first.loc[shared]
    enc_second = enc_second.loc[shared]

    result = {"subject_id": subject_id, "n_images": len(shared)}

    for metric in ["svg_z"]:
        v1 = enc_first[metric]
        v2 = enc_second[metric]
        low_n1 = enc_first["low_n"].fillna(True)
        low_n2 = enc_second["low_n"].fillna(True)
        valid = v1.notna() & v2.notna() & ~low_n1 & ~low_n2

        d1 = v1[valid].values
        d2 = v2[valid].values
        diff = d2 - d1

        n_valid = int(valid.sum())
        result[metric] = {
            "n_valid": n_valid,
            "n_dropped": int((~valid).sum()),
            "mean_enc_first": float(np.mean(d1)) if n_valid > 0 else np.nan,
            "mean_enc_second": float(np.mean(d2)) if n_valid > 0 else np.nan,
            "mean_diff": float(np.mean(diff)) if n_valid > 0 else np.nan,
            "std_diff": float(np.std(diff, ddof=1)) if n_valid > 1 else np.nan,
            "n_pos": int((diff > 0).sum()),
            "n_neg": int((diff < 0).sum()),
            "n_tie": int((diff == 0).sum()),
        }

    return result


# ---------------------------------------------------------------------------
# Group-level test
# ---------------------------------------------------------------------------


def run_group_test(all_results: list[dict]) -> dict:
    """
    Stage 2: one mean diff per participant → one-sample test against 0.

    Uses a one-sample Wilcoxon signed-rank test when n_participants >= 3,
    otherwise reports descriptives only.
    """
    group = {}

    for metric in ["svg_z"]:
        # Collect mean diffs from participants with enough valid pairs
        mean_diffs = []
        for r in all_results:
            m = r[metric]
            if m["n_valid"] >= MIN_PAIRS_PER_SUBJECT and not np.isnan(m["mean_diff"]):
                mean_diffs.append(m["mean_diff"])

        n_contrib = len(mean_diffs)
        arr = np.array(mean_diffs)

        stat, p = np.nan, np.nan
        test_note = ""

        if n_contrib >= MIN_PARTICIPANTS:
            nonzero = arr[arr != 0]
            if len(nonzero) >= 1:
                try:
                    stat, p = stats.wilcoxon(arr, alternative="greater")
                    test_note = "One-sample Wilcoxon (one-tailed: mean diff > 0)"
                except Exception as e:
                    test_note = f"Test failed: {e}"
            else:
                test_note = "All participant mean diffs are zero — test not applicable"
        else:
            test_note = (
                f"Only {n_contrib}/{len(all_results)} participants contributed "
                f"(need ≥{MIN_PARTICIPANTS} with ≥{MIN_PAIRS_PER_SUBJECT} valid pairs) — descriptives only"
            )

        group[metric] = {
            "n_contributing": n_contrib,
            "mean_diffs": arr.tolist(),
            "grand_mean_diff": float(np.mean(arr)) if n_contrib > 0 else np.nan,
            "grand_sd_diff": float(np.std(arr, ddof=1)) if n_contrib > 1 else np.nan,
            "stat": stat,
            "p": p,
            "test_note": test_note,
        }

    return group


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_per_subject_report(result: dict):
    sid = result["subject_id"]
    print(f"\n  {sid}  (n_images={result['n_images']})")
    for metric in ["svg_z"]:
        r = result[metric]
        if r["n_valid"] == 0:
            print(f"    [svg]  no valid pairs")
            continue
        print(
            f"    [svg]  n_valid={r['n_valid']}  "
            f"enc_first={r['mean_enc_first']:+.3f}  "
            f"enc_second={r['mean_enc_second']:+.3f}  "
            f"mean_diff={r['mean_diff']:+.3f}  "
            f"({r['n_pos']}↑ {r['n_neg']}↓ {r['n_tie']}=)"
        )


def print_group_report(group: dict, n_total: int):
    print(f"\n{'='*65}")
    print(f"GROUP-LEVEL ENC_FIRST → ENC_SECOND SHIFT  ({n_total} participants)")
    print(f"{'='*65}")
    print("(Test: one-sample Wilcoxon on per-participant mean diffs vs 0)\n")

    for metric in ["svg_z"]:
        g = group[metric]
        label = "SVG z (core: inter+spatial+func)"
        print(f"--- {label} ---")
        print(f"  Contributing participants: {g['n_contributing']}")

        if g["n_contributing"] == 0:
            print("  No data.")
            continue

        per_p = "  " + "  ".join(f"{d:+.3f}" for d in g["mean_diffs"])
        print(f"  Per-participant mean diffs:{per_p}")
        print(f"  Grand mean diff : {g['grand_mean_diff']:+.3f}")
        print(
            f"  Grand SD diff   : {g['grand_sd_diff']:.3f}"
            if not np.isnan(g["grand_sd_diff"])
            else "  Grand SD diff   : —"
        )

        if not np.isnan(g["stat"]):
            sig = "  *" if g["p"] < 0.05 else ""
            print(f"  Wilcoxon W      : {g['stat']:.1f}")
            print(f"  p (one-tail)    : {g['p']:.4f}{sig}")
        print(f"  Note            : {g['test_note']}")
        print()

    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    setup_logging(level=config.LOG_LEVEL)

    parser = argparse.ArgumentParser(
        description="Group-level enc_first → enc_second SVG alignment shift analysis."
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Single subject ID (descriptives only; group test skipped)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for SVG permutation null (default: 42)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    subject_ids = [args.subject] if args.subject else get_subject_ids()
    if not args.subject:
        logger.info(f"Discovered {len(subject_ids)} subjects.")

    # Load graph index once
    graph_index = build_graph_index()

    all_results = []
    print(f"\n{'='*65}")
    print("PER-PARTICIPANT DESCRIPTIVES")
    print(f"{'='*65}")

    for subject_id in subject_ids:
        logger.info(f"  [{subject_id}] Computing per-trial SVG scores ...")
        result = analyse_shift_subject(subject_id, graph_index, rng)
        if result:
            all_results.append(result)
            print_per_subject_report(result)

    if len(all_results) >= 2:
        group = run_group_test(all_results)
        print_group_report(group, len(all_results))
    else:
        print("\n(Need ≥2 participants for group report.)")


if __name__ == "__main__":
    main()
