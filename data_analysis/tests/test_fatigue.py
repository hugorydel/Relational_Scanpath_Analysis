"""
tests/test_fatigue.py
=====================
Diagnostic: memory score as a function of decoding trial position (1–30).

Purpose
-------
Detect fatigue or proactive-interference effects — systematic declines in
recall quality across the session.

Output
------
Console:
  OLS trend β and p per DV (relational correct, objects correct).

Figure (output/analysis/fatigue_effects.png):
  1×2 panels — relational correct and objects correct.
  Group mean ± 1 SD band across trial positions, OLS trend line annotated.

Usage
-----
    python tests/test_fatigue.py
    python tests/test_fatigue.py --features path/to/trial_features_all.csv
    python tests/test_fatigue.py --scores   path/to/memory_scores.csv
    python tests/test_fatigue.py --output   path/to/output/analysis
    python tests/test_fatigue.py --no-plot
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import config

    _DEFAULT_FEATURES = config.OUTPUT_FEATURES_DIR / "trial_features_all.csv"
    _DEFAULT_SCORES = config.MEMORY_SCORES_FILE
    _DEFAULT_OUTPUT = config.OUTPUT_DIR / "analysis"
except Exception:
    _DEFAULT_FEATURES = _PROJECT_ROOT / "output" / "features" / "trial_features_all.csv"
    _DEFAULT_SCORES = _PROJECT_ROOT / "output" / "data_scoring" / "memory_scores.csv"
    _DEFAULT_OUTPUT = _PROJECT_ROOT / "output" / "analysis"

DVS = [
    ("n_relational_correct", "Relational correct\n(action + spatial)", "#084594"),
    ("n_objects_correct", "Objects correct\n(identity + attribute)", "#a63603"),
]

DERIVED = {
    "n_relational_correct": ["n_action_relation_correct", "n_spatial_relation_correct"],
    "n_objects_correct": ["n_object_identity_correct", "n_object_attribute_correct"],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load(features_path: Path, scores_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores not found: {scores_path}")

    features = pd.read_csv(features_path, dtype={"StimID": str, "SubjectID": str})
    dec = features[features["Phase"] == "decoding"].copy()

    scores = pd.read_csv(scores_path, dtype={"StimID": str, "SubjectID": str})

    for col, addends in DERIVED.items():
        if col not in scores.columns:
            present = [c for c in addends if c in scores.columns]
            if present:
                scores[col] = scores[present].sum(axis=1)

    merged = dec.merge(scores, on=["SubjectID", "StimID"], how="inner")

    if "wrong_image" in merged.columns:
        before = len(merged)
        merged = merged[merged["wrong_image"] != 1].copy()
        print(f"  Excluded {before - len(merged)} wrong-image trials.")

    merged = merged.dropna(subset=["dec_trial_index"])
    merged["dec_trial_index"] = merged["dec_trial_index"].astype(int)
    merged = merged.sort_values(["SubjectID", "dec_trial_index"]).reset_index(drop=True)

    print(
        f"  {len(merged)} scored decoding trials, "
        f"{merged['SubjectID'].nunique()} participants, "
        f"{merged['dec_trial_index'].nunique()} trial positions."
    )
    return merged


# ---------------------------------------------------------------------------
# OLS trend
# ---------------------------------------------------------------------------


def _ols(positions: np.ndarray, values: np.ndarray):
    mask = ~np.isnan(values)
    if mask.sum() < 4:
        return np.nan, np.nan, np.nan
    slope, intercept, _, p, _ = stats.linregress(positions[mask], values[mask])
    return slope, intercept, p


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(df: pd.DataFrame) -> None:
    positions = np.array(sorted(df["dec_trial_index"].unique()), dtype=float)

    print("\n" + "=" * 52)
    print("FATIGUE CHECK — OLS trend across trial positions")
    print("=" * 52)
    print(f"  {'DV':<40} {'beta':>8} {'p':>8}  sig")
    print("-" * 52)

    for col, label, _ in DVS:
        if col not in df.columns:
            continue
        means = (
            df.groupby("dec_trial_index")[col]
            .mean()
            .reindex(sorted(df["dec_trial_index"].unique()))
            .values
        )
        slope, _, p = _ols(positions, means)
        if np.isnan(slope):
            print(f"  {label.replace(chr(10),' '):<40}  insufficient data")
            continue
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {label.replace(chr(10),' '):<40} {slope:>+8.4f} {p:>8.4f}  {sig}")

    print("=" * 52)
    print("  beta = change in mean score per trial position")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(df: pd.DataFrame, output_path: Path) -> None:
    dv_cols = [col for col, _, _ in DVS if col in df.columns]
    fig, axes = plt.subplots(
        1, len(dv_cols), figsize=(5.5 * len(dv_cols), 4.2), squeeze=False
    )

    all_positions = np.arange(1, 31)

    for ax, (col, label, colour) in zip(axes[0], DVS):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        grp = df.groupby("dec_trial_index")[col]
        means = grp.mean().reindex(all_positions).values
        sds = grp.std().reindex(all_positions).values
        pos = all_positions.astype(float)

        ax.plot(
            pos, means, color=colour, linewidth=2.2, marker="o", markersize=4, zorder=4
        )
        ax.fill_between(pos, means - sds, means + sds, color=colour, alpha=0.15)

        slope, intercept, p = _ols(pos, means)
        if not np.isnan(slope):
            trend = intercept + slope * pos
            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            ax.plot(pos, trend, color="black", linewidth=1.4, linestyle="--", zorder=5)
            ax.annotate(
                f"beta = {slope:+.4f},  p = {p:.3f}  {sig}",
                xy=(0.05, 0.06),
                xycoords="axes fraction",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

        ax.set_title(label.replace("\n", " "), fontsize=10, fontweight="bold", pad=7)
        ax.set_xlabel("Decoding trial position", fontsize=9)
        ax.set_ylabel("Mean correct (+/-1 SD)", fontsize=9)
        ax.set_xlim(0.5, 30.5)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "Fatigue / proactive-interference check\n"
        "Group mean +/-1 SD across decoding trial positions  (dashed = OLS trend)",
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved -> {output_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fatigue / proactive-interference check for memory scores."
    )
    parser.add_argument("--features", default=str(_DEFAULT_FEATURES))
    parser.add_argument("--scores", default=str(_DEFAULT_SCORES))
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Fatigue check: memory score by trial position")
    print("=" * 60)

    df = _load(Path(args.features), Path(args.scores))
    _print_summary(df)

    if not args.no_plot:
        _plot(df, Path(args.output) / "fatigue_effects.png")


if __name__ == "__main__":
    main()
