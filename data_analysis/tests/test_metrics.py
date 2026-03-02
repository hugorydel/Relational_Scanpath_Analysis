"""
tests/test_metrics.py
=====================
Diagnostic tests for Steps 4–6: object sequence construction,
SVG alignment, and symbolic LCS / Kendall's tau.

Runs two kinds of checks:
  1. Unit tests — known inputs with expected outputs
  2. Live tests — real fixations_aoi.csv + scene graph for participant 1

Usage:
    python tests/test_metrics.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import numpy as np
import pandas as pd
from pipeline.misc import setup_logging
from pipeline.utils.metrics import (
    build_object_sequence,
    kendall_tau_shared,
    svg_alignment,
    symbolic_lcs,
)
from pipeline.utils.scene_graph import build_graph_index

setup_logging(level="INFO")
logger = logging.getLogger(__name__)

SUBJECT_ID = "Encode-Decode_Experiment-1-1"
PASS = "  PASS"
FAIL = "  FAIL"


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_build_object_sequence():
    print("\n--- test_build_object_sequence ---")

    # Basic consecutive collapse
    df = pd.DataFrame(
        {
            "FixStart_ms": [0, 1, 2, 3, 4, 5, 6],
            "ObjectID": [1, 1, 2, 3, 3, 2, 1],
        }
    )
    result = build_object_sequence(df)
    assert result == [1, 2, 3, 2, 1], f"Expected [1,2,3,2,1], got {result}"
    print(f"{PASS}  collapse consecutive duplicates: {result}")

    # NaN rows excluded
    df2 = pd.DataFrame(
        {
            "FixStart_ms": [0, 1, 2, 3],
            "ObjectID": [1, np.nan, 2, np.nan],
        }
    )
    result2 = build_object_sequence(df2)
    assert result2 == [1, 2], f"Expected [1,2], got {result2}"
    print(f"{PASS}  NaN exclusion: {result2}")

    # All NaN
    df3 = pd.DataFrame(
        {
            "FixStart_ms": [0, 1],
            "ObjectID": [np.nan, np.nan],
        }
    )
    result3 = build_object_sequence(df3)
    assert result3 == [], f"Expected [], got {result3}"
    print(f"{PASS}  all NaN → empty: {result3}")

    # Single fixation
    df4 = pd.DataFrame({"FixStart_ms": [0], "ObjectID": [5]})
    result4 = build_object_sequence(df4)
    assert result4 == [5], f"Expected [5], got {result4}"
    print(f"{PASS}  single fixation: {result4}")


def test_svg_alignment():
    print("\n--- test_svg_alignment ---")
    rng = np.random.default_rng(42)

    # Perfect relational sequence — all transitions are edges
    edges_full = {frozenset({1, 2}), frozenset({2, 3}), frozenset({3, 4})}
    seq_perfect = [1, 2, 3, 4]
    res = svg_alignment(seq_perfect, edges_full, n_permutations=500, rng=rng)
    assert res["svg_obs"] == 1.0, f"Expected obs=1.0, got {res['svg_obs']}"
    assert res["n_transitions"] == 3
    assert res["n_relational"] == 3
    assert res["svg_z"] > 0, f"Expected positive z, got {res['svg_z']}"
    print(f"{PASS}  perfect relational: obs={res['svg_obs']:.2f}  z={res['svg_z']:.2f}")

    # No relational transitions
    edges_none = set()
    res2 = svg_alignment([1, 2, 3, 4], edges_none, n_permutations=500, rng=rng)
    assert res2["svg_obs"] == 0.0
    assert res2["n_relational"] == 0
    print(f"{PASS}  no relations: obs={res2['svg_obs']:.2f}  z={res2['svg_z']}")

    # Empty sequence
    res3 = svg_alignment([], edges_full, n_permutations=100, rng=rng)
    assert np.isnan(res3["svg_z"])
    assert res3["n_transitions"] == 0
    print(f"{PASS}  empty sequence: n_transitions={res3['n_transitions']}  z=NaN ✓")

    # Single element
    res4 = svg_alignment([1], edges_full, n_permutations=100, rng=rng)
    assert np.isnan(res4["svg_z"])
    print(f"{PASS}  single element: z=NaN ✓")

    # low_n flag
    res5 = svg_alignment([1, 2], edges_full, n_permutations=100, rng=rng)
    assert res5["low_n"] is True  # 1 transition < MIN_VALID_TRANSITIONS=2
    print(f"{PASS}  low_n flag: 1 transition → low_n={res5['low_n']}")


def test_symbolic_lcs():
    print("\n--- test_symbolic_lcs ---")

    # Identical sequences
    res = symbolic_lcs([1, 2, 3, 4], [1, 2, 3, 4])
    assert res["lcs_score"] == 1.0, f"Expected 1.0, got {res['lcs_score']}"
    print(f"{PASS}  identical sequences: lcs={res['lcs_score']:.2f}")

    # No overlap
    res2 = symbolic_lcs([1, 2, 3], [4, 5, 6])
    assert res2["lcs_score"] == 0.0, f"Expected 0.0, got {res2['lcs_score']}"
    print(f"{PASS}  no overlap: lcs={res2['lcs_score']:.2f}")

    # Partial overlap — [1,3] is the LCS of [1,2,3] and [4,1,5,3]
    res3 = symbolic_lcs([1, 2, 3], [4, 1, 5, 3])
    assert res3["lcs_length"] == 2, f"Expected lcs_length=2, got {res3['lcs_length']}"
    assert res3["lcs_score"] == 2 / 3, f"Expected 2/3, got {res3['lcs_score']}"
    print(
        f"{PASS}  partial overlap: lcs_length={res3['lcs_length']}  score={res3['lcs_score']:.3f}"
    )

    # Repeats collapsed before comparison
    # [1,2,1,3] → first-occurrence [1,2,3], [1,3] → lcs=2, min_len=2, score=1.0
    res4 = symbolic_lcs([1, 2, 1, 3], [1, 3])
    assert res4["lcs_score"] == 1.0, f"Expected 1.0, got {res4['lcs_score']}"
    print(f"{PASS}  repeats collapsed: lcs={res4['lcs_score']:.2f}")

    # Empty sequences
    res5 = symbolic_lcs([], [1, 2, 3])
    assert np.isnan(res5["lcs_score"])
    print(f"{PASS}  empty sequence → NaN ✓")


def test_kendall_tau_shared():
    print("\n--- test_kendall_tau_shared ---")

    # Perfect agreement on shared objects
    res = kendall_tau_shared([1, 2, 3, 4], [1, 2, 3, 4])
    assert res["tau"] == 1.0, f"Expected tau=1.0, got {res['tau']}"
    assert res["n_shared"] == 4
    print(
        f"{PASS}  perfect agreement: tau={res['tau']:.2f}  n_shared={res['n_shared']}"
    )

    # Perfect reversal
    res2 = kendall_tau_shared([1, 2, 3, 4], [4, 3, 2, 1])
    assert res2["tau"] == -1.0, f"Expected tau=-1.0, got {res2['tau']}"
    print(f"{PASS}  perfect reversal: tau={res2['tau']:.2f}")

    # Disjoint sets
    res3 = kendall_tau_shared([1, 2, 3], [4, 5, 6])
    assert np.isnan(res3["tau"])
    assert res3["n_shared"] == 0
    print(f"{PASS}  disjoint → NaN ✓")

    # Only 1 shared object — tau undefined
    res4 = kendall_tau_shared([1, 2, 3], [1, 4, 5])
    assert np.isnan(res4["tau"])
    assert res4["n_shared"] == 1
    print(f"{PASS}  1 shared object → NaN ✓")

    # Partial overlap
    res5 = kendall_tau_shared([1, 2, 3, 4], [3, 1, 4, 2])
    assert res5["n_shared"] == 4
    print(
        f"{PASS}  partial ordering: tau={res5['tau']:.3f}  n_shared={res5['n_shared']}"
    )


# ---------------------------------------------------------------------------
# Live test on real data
# ---------------------------------------------------------------------------


def test_live():
    print("\n--- live test on real participant data ---")

    aoi_path = config.OUTPUT_FEATURES_DIR / f"{SUBJECT_ID}_fixations_aoi.csv"
    if not aoi_path.exists():
        print(f"  SKIP — {aoi_path.name} not found. Run test_aoi.py first.")
        return

    df = pd.read_csv(aoi_path, dtype={"StimID": str})
    graph_index = build_graph_index()
    rng = np.random.default_rng(42)

    print(f"  Loaded {len(df)} fixations for {SUBJECT_ID}")

    # Compute metrics for every trial
    records = []

    for (stim_id, phase, vn), group in df.groupby(
        ["StimID", "Phase", "ViewingNumber"], dropna=False
    ):
        seq = build_object_sequence(group)

        edges_all = graph_index["all"].get(stim_id, set())
        edges_inter = graph_index["interactional"].get(stim_id, set())

        svg_all = svg_alignment(seq, edges_all, n_permutations=200, rng=rng)
        svg_inter = svg_alignment(seq, edges_inter, n_permutations=200, rng=rng)

        records.append(
            {
                "StimID": stim_id,
                "Phase": phase,
                "ViewingNumber": vn,
                "seq_length": len(seq),
                "n_transitions": svg_all["n_transitions"],
                "n_relational": svg_all["n_relational"],
                "svg_obs_all": (
                    round(svg_all["svg_obs"], 3)
                    if not np.isnan(svg_all["svg_obs"] or np.nan)
                    else np.nan
                ),
                "svg_z_all": (
                    round(svg_all["svg_z"], 3)
                    if not np.isnan(svg_all["svg_z"] or np.nan)
                    else np.nan
                ),
                "svg_z_inter": (
                    round(svg_inter["svg_z"], 3)
                    if not np.isnan(svg_inter["svg_z"] or np.nan)
                    else np.nan
                ),
                "low_n": svg_all["low_n"],
            }
        )

    results_df = pd.DataFrame(records)

    print(f"\n  Trials processed: {len(results_df)}")
    print(f"  low_n trials    : {results_df['low_n'].sum()}")

    for phase in ["encoding", "decoding"]:
        sub = results_df[results_df["Phase"] == phase]
        print(f"\n  {phase.upper()} (n={len(sub)} trials)")
        print(
            f"    seq_length    : mean={sub['seq_length'].mean():.1f}  min={sub['seq_length'].min()}  max={sub['seq_length'].max()}"
        )
        print(
            f"    svg_obs_all   : mean={sub['svg_obs_all'].mean():.3f}  std={sub['svg_obs_all'].std():.3f}"
        )
        print(
            f"    svg_z_all     : mean={sub['svg_z_all'].mean():.3f}  std={sub['svg_z_all'].std():.3f}"
        )
        print(
            f"    svg_z_inter   : mean={sub['svg_z_inter'].mean():.3f}  std={sub['svg_z_inter'].std():.3f}"
        )

    # LCS and Kendall between encoding 2 and decoding
    print("\n  LCS / Kendall (Encoding 2 vs Decoding):")
    lcs_scores = []
    tau_scores = []

    stim_ids = results_df["StimID"].unique()
    for stim_id in stim_ids:
        enc2_fix = df[
            (df["StimID"] == stim_id)
            & (df["Phase"] == "encoding")
            & (df["ViewingNumber"] == 2.0)
        ]
        dec_fix = df[(df["StimID"] == stim_id) & (df["Phase"] == "decoding")]

        seq_enc2 = build_object_sequence(enc2_fix)
        seq_dec = build_object_sequence(dec_fix)

        if not seq_enc2 or not seq_dec:
            continue

        lcs = symbolic_lcs(seq_enc2, seq_dec)
        tau = kendall_tau_shared(seq_enc2, seq_dec)

        lcs_scores.append(lcs["lcs_score"])
        if not np.isnan(tau["tau"]):
            tau_scores.append(tau["tau"])

    lcs_arr = np.array(lcs_scores)
    tau_arr = np.array(tau_scores)
    print(
        f"    LCS  : mean={lcs_arr.mean():.3f}  std={lcs_arr.std():.3f}  "
        f"min={lcs_arr.min():.3f}  max={lcs_arr.max():.3f}  n={len(lcs_arr)}"
    )
    print(
        f"    Tau  : mean={tau_arr.mean():.3f}  std={tau_arr.std():.3f}  "
        f"min={tau_arr.min():.3f}  max={tau_arr.max():.3f}  n={len(tau_arr)}"
    )

    # Sample trial detail
    print("\n  Sample trial details (first 5 encoding trials):")
    sample = results_df[results_df["Phase"] == "encoding"].head(5)[
        [
            "StimID",
            "ViewingNumber",
            "seq_length",
            "n_transitions",
            "n_relational",
            "svg_obs_all",
            "svg_z_all",
            "svg_z_inter",
            "low_n",
        ]
    ]
    print(sample.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("METRICS UNIT TESTS")
    print("=" * 60)

    try:
        test_build_object_sequence()
        test_svg_alignment()
        test_symbolic_lcs()
        test_kendall_tau_shared()
        print("\nAll unit tests passed.")
    except AssertionError as e:
        print(f"\n  ASSERTION ERROR: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("LIVE DATA TEST")
    print("=" * 60)
    test_live()
