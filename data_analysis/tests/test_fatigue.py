"""
tests/test_fatigue.py
=====================
Standalone diagnostic: average memory score as a function of decoding
presentation order (trial position 1–30 within each participant's free-recall
phase).

Purpose
-------
Detect fatigue or proactive-interference effects — systematic declines (or
non-linear dips) in recall quality across the session.

What it produces
----------------
1. Console table: per-position mean ± SD for each score column, collapsed
   across participants.
2. Figure saved to output/analysis/fatigue_effects.png:
     Row 1 — Primary measures: n_relational_correct, n_objects_correct
     Row 2 — All five content types (correct counts only)
   Each panel shows:
     - Thin lines per participant (coloured by subject)
     - Thick black line = cross-participant mean
     - Shaded band = ±1 SD across participants at each position
     - OLS trend line (dashed red) with β and p annotated

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

# ---------------------------------------------------------------------------
# Paths — resolve relative to project root (two levels up from tests/)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Score columns
# ---------------------------------------------------------------------------
CONTENT_TYPES = [
    "object_identity",
    "object_attribute",
    "action_relation",
    "spatial_relation",
    "scene_gist",
]
CONTENT_CORRECT = [f"n_{ct}_correct" for ct in CONTENT_TYPES]

DERIVED = {
    "n_relational_correct": ["n_action_relation_correct", "n_spatial_relation_correct"],
    "n_objects_correct": ["n_object_identity_correct", "n_object_attribute_correct"],
}

PRIMARY = ["n_relational_correct", "n_objects_correct"]
ALL_SCORE_COLS = PRIMARY + CONTENT_CORRECT

# Colours per participant (cycles if >7)
_SUBJ_COLOURS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load(features_path: Path, scores_path: Path) -> pd.DataFrame:
    """
    Merge trial_features (decoding rows only) with memory_scores, keyed on
    SubjectID × StimID.  Returns a DataFrame with dec_trial_index and all
    score columns.
    """
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path}")

    features = pd.read_csv(features_path, dtype={"StimID": str, "SubjectID": str})
    dec = features[features["Phase"] == "decoding"].copy()

    scores = pd.read_csv(scores_path, dtype={"StimID": str, "SubjectID": str})

    # Derive aggregate columns if using the 20-column schema
    for derived_col, addends in DERIVED.items():
        if derived_col not in scores.columns:
            present = [c for c in addends if c in scores.columns]
            if present:
                scores[derived_col] = scores[present].sum(axis=1)

    merged = dec.merge(scores, on=["SubjectID", "StimID"], how="inner")

    missing = [c for c in ALL_SCORE_COLS if c not in merged.columns]
    if missing:
        print(f"[WARNING] Score columns not found and will be skipped: {missing}")

    available = [c for c in ALL_SCORE_COLS if c in merged.columns]
    keep = ["SubjectID", "StimID", "dec_trial_index"] + available
    merged = merged[[c for c in keep if c in merged.columns]].copy()
    merged = merged.dropna(subset=["dec_trial_index"])
    merged["dec_trial_index"] = merged["dec_trial_index"].astype(int)
    merged = merged.sort_values(["SubjectID", "dec_trial_index"]).reset_index(drop=True)

    print(
        f"\nLoaded {len(merged)} scored decoding trials from "
        f"{merged['SubjectID'].nunique()} participant(s), "
        f"{merged['dec_trial_index'].nunique()} unique trial positions."
    )
    return merged


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_table(df: pd.DataFrame, score_cols: list[str]) -> None:
    """Print per-position mean ± SD collapsed across participants."""
    grouped = df.groupby("dec_trial_index")[score_cols]
    means = grouped.mean()
    sds = grouped.std()

    print("\n" + "=" * 70)
    print("FATIGUE CHECK — mean (SD) per decoding trial position")
    print("=" * 70)
    header = f"{'Pos':>4}" + "".join(f"  {c[:20]:>22}" for c in score_cols)
    print(header)
    print("-" * len(header))
    for pos in sorted(df["dec_trial_index"].unique()):
        if pos not in means.index:
            continue
        row = f"{pos:>4}"
        for c in score_cols:
            m = means.loc[pos, c] if c in means.columns else float("nan")
            s = sds.loc[pos, c] if c in sds.columns else float("nan")
            row += f"  {m:>8.2f} ({s:>5.2f})"
        print(row)
    print("=" * 70)


# ---------------------------------------------------------------------------
# OLS trend helper
# ---------------------------------------------------------------------------


def _ols_trend(positions: np.ndarray, values: np.ndarray):
    """Return (slope, intercept, p_value) from simple OLS."""
    mask = ~np.isnan(values)
    if mask.sum() < 4:
        return np.nan, np.nan, np.nan
    slope, intercept, r, p, _ = stats.linregress(positions[mask], values[mask])
    return slope, intercept, p


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot(df: pd.DataFrame, score_cols: list[str], output_path: Path) -> None:
    primary_cols = [c for c in PRIMARY if c in score_cols]
    content_cols = [c for c in CONTENT_CORRECT if c in score_cols]

    n_rows = (1 if not primary_cols else 1) + (1 if content_cols else 0)
    n_cols_top = max(len(primary_cols), 1)
    n_cols_bot = max(len(content_cols), 1)
    n_cols = max(n_cols_top, n_cols_bot)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )

    subjects = sorted(df["SubjectID"].unique())
    colour_map = {
        s: _SUBJ_COLOURS[i % len(_SUBJ_COLOURS)] for i, s in enumerate(subjects)
    }

    all_positions = np.arange(1, 31)

    def _draw(ax, col, title):
        if col not in df.columns:
            ax.set_visible(False)
            return

        # Per-participant lines
        for subj in subjects:
            sdf = df[df["SubjectID"] == subj].sort_values("dec_trial_index")
            ax.plot(
                sdf["dec_trial_index"],
                sdf[col],
                color=colour_map[subj],
                alpha=0.45,
                linewidth=1.2,
                marker="o",
                markersize=3,
                label=subj.split("_")[-1],
            )

        # Cross-participant mean ± SD
        grp = df.groupby("dec_trial_index")[col]
        means = grp.mean().reindex(all_positions)
        sds = grp.std().reindex(all_positions)
        pos_arr = np.array(means.index)
        m_arr = means.values
        s_arr = sds.values

        ax.plot(
            pos_arr,
            m_arr,
            color="black",
            linewidth=2.2,
            marker="o",
            markersize=4,
            zorder=5,
            label="Mean",
        )
        ax.fill_between(
            pos_arr, m_arr - s_arr, m_arr + s_arr, color="black", alpha=0.10
        )

        # OLS trend on the mean
        slope, intercept, p = _ols_trend(pos_arr, m_arr)
        if not np.isnan(slope):
            trend = intercept + slope * pos_arr
            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            ax.plot(
                pos_arr, trend, color="#d62728", linewidth=1.5, linestyle="--", zorder=6
            )
            ax.annotate(
                f"β={slope:+.3f}, p={p:.3f} {sig}",
                xy=(0.97, 0.06),
                xycoords="axes fraction",
                ha="right",
                fontsize=7.5,
                color="#d62728",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Decoding trial position", fontsize=8)
        ax.set_ylabel("Correct count", fontsize=8)
        ax.set_xlim(0.5, 30.5)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7.5)

    # Row 0: primary measures
    col_labels = {
        "n_relational_correct": "Relational correct",
        "n_objects_correct": "Objects correct",
    }
    for i, col in enumerate(primary_cols):
        _draw(axes[0][i], col, col_labels.get(col, col))
    for i in range(len(primary_cols), n_cols):
        axes[0][i].set_visible(False)

    # Add legend to first primary panel
    if primary_cols and primary_cols[0] in df.columns:
        handles, labels = axes[0][0].get_legend_handles_labels()
        axes[0][0].legend(
            handles, labels, fontsize=6.5, loc="upper right", framealpha=0.7, ncol=1
        )

    # Row 1: content types
    if n_rows > 1:
        ct_labels = {
            "n_object_identity_correct": "Object identity",
            "n_object_attribute_correct": "Object attribute",
            "n_action_relation_correct": "Action relation",
            "n_spatial_relation_correct": "Spatial relation",
            "n_scene_gist_correct": "Scene gist",
        }
        for i, col in enumerate(content_cols):
            _draw(axes[1][i], col, ct_labels.get(col, col))
        for i in range(len(content_cols), n_cols):
            axes[1][i].set_visible(False)

    fig.suptitle(
        "Fatigue / proactive-interference check\n"
        "Memory score by decoding trial position (dashed = OLS trend)",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fatigue / proactive-interference check for memory scores."
    )
    parser.add_argument(
        "--features",
        default=str(_DEFAULT_FEATURES),
        help="Path to trial_features_all.csv",
    )
    parser.add_argument(
        "--scores", default=str(_DEFAULT_SCORES), help="Path to memory_scores.csv"
    )
    parser.add_argument(
        "--output", default=str(_DEFAULT_OUTPUT), help="Output directory for the figure"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    df = _load(Path(args.features), Path(args.scores))

    available_score_cols = [c for c in ALL_SCORE_COLS if c in df.columns]
    _print_table(df, available_score_cols)

    if not args.no_plot:
        _plot(df, available_score_cols, Path(args.output) / "fatigue_effects.png")


if __name__ == "__main__":
    main()
