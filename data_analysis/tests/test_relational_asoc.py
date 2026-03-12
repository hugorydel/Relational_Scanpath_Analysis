"""
tests/test_relational_svg_assoc.py
===================================
Participant-level figure: proportion relational correct vs mean decoding SVG.

Y-axis (shared):
    Each participant's total relational correct
    (n_action_relation_correct + n_spatial_relation_correct, summed across all
    scored items) divided by the highest participant's total — so the top
    scorer sits at 1.0.

X-axis:
    Left panel  — mean svg_z_inter across that participant's decoding trials
                  (non-null trials only)
    Right panel — mean svg_z_all  across that participant's decoding trials

One data point per participant. Pearson r + p annotated on each panel.

Output: output/analysis/relational_svg_assoc.png

Usage
-----
    python tests/test_relational_svg_assoc.py
    python tests/test_relational_svg_assoc.py --features path/to/trial_features_all.csv
    python tests/test_relational_svg_assoc.py --scores   path/to/memory_scores.csv
    python tests/test_relational_svg_assoc.py --output   path/to/output/analysis
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
# Paths
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


# Short display labels — strip the long experiment prefix
def _label(subject_id: str) -> str:
    parts = subject_id.split("-")
    # Keep last two segments e.g. "1-1", "2-1"
    if len(parts) >= 2:
        return f"P{parts[-2]}-{parts[-1]}"
    return subject_id


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _build(features_path: Path, scores_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path}")

    # ── Memory scores ────────────────────────────────────────────────────
    scores = pd.read_csv(scores_path, dtype={"StimID": str, "SubjectID": str})

    # Derive relational correct from fine-grained columns if present
    if "n_relational_correct" not in scores.columns:
        addends = ["n_action_relation_correct", "n_spatial_relation_correct"]
        present = [c for c in addends if c in scores.columns]
        if not present:
            raise ValueError(
                "Cannot find n_relational_correct or its component columns "
                "in memory_scores.csv."
            )
        scores["n_relational_correct"] = scores[present].sum(axis=1)

    # Per-participant relational total → proportion of highest
    rel_totals = (
        scores.groupby("SubjectID")["n_relational_correct"]
        .sum()
        .reset_index()
        .rename(columns={"n_relational_correct": "relational_total"})
    )
    max_total = rel_totals["relational_total"].max()
    rel_totals["relational_prop"] = rel_totals["relational_total"] / max_total

    print("\nPer-participant relational totals:")
    print(rel_totals.to_string(index=False))
    print(f"  Highest total: {max_total:.0f}  (= 1.0)")

    # ── Decoding SVG scores ──────────────────────────────────────────────
    features = pd.read_csv(features_path, dtype={"StimID": str, "SubjectID": str})
    dec = features[features["Phase"] == "decoding"].copy()

    svg_means = (
        dec.groupby("SubjectID")[["svg_z_inter", "svg_z_all"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "svg_z_inter": "mean_svg_inter",
                "svg_z_all": "mean_svg_all",
            }
        )
    )

    # Count non-null trials used per metric per participant
    svg_n = (
        dec.groupby("SubjectID")[["svg_z_inter", "svg_z_all"]]
        .apply(lambda g: g.notna().sum())
        .reset_index()
        .rename(
            columns={
                "svg_z_inter": "n_inter",
                "svg_z_all": "n_all",
            }
        )
    )

    print("\nPer-participant mean decoding SVG scores:")
    print(svg_means.merge(svg_n, on="SubjectID").to_string(index=False))

    # ── Merge ────────────────────────────────────────────────────────────
    merged = rel_totals.merge(svg_means, on="SubjectID", how="inner")
    merged["label"] = merged["SubjectID"].apply(_label)
    merged = merged.sort_values("SubjectID").reset_index(drop=True)

    print(f"\nFinal analysis table ({len(merged)} participants):")
    print(
        merged[
            ["label", "relational_prop", "mean_svg_inter", "mean_svg_all"]
        ].to_string(index=False)
    )

    return merged


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


def _panel(
    ax,
    x: pd.Series,
    y: pd.Series,
    labels: pd.Series,
    xlabel: str,
    colours: list,
    show_ylabel: bool,
) -> None:
    """Draw one scatter panel with regression line and r annotation."""

    # Drop rows where x is NaN
    mask = x.notna() & y.notna()
    xv, yv, lv = x[mask].values, y[mask].values, labels[mask].values
    n = len(xv)

    # Scatter
    for i, (xi, yi, li) in enumerate(zip(xv, yv, lv)):
        ax.scatter(
            xi,
            yi,
            color=colours[i % len(colours)],
            s=120,
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.annotate(
            li,
            (xi, yi),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=8.5,
            color=colours[i % len(colours)],
            fontweight="bold",
        )

    # Regression line
    if n >= 3:
        slope, intercept, r, p, _ = stats.linregress(xv, yv)
        x_line = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(
            x_line,
            intercept + slope * x_line,
            color="#444444",
            linewidth=1.5,
            linestyle="--",
            zorder=4,
        )
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.annotate(
            f"r = {r:+.2f},  p = {p:.3f} {sig}\n(n = {n} participants)",
            xy=(0.05, 0.93),
            xycoords="axes fraction",
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
    else:
        ax.annotate(
            f"n = {n} (insufficient for regression)",
            xy=(0.05, 0.93),
            xycoords="axes fraction",
            fontsize=8,
            va="top",
            color="grey",
        )

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=10)
    if show_ylabel:
        ax.set_ylabel(
            "Relational correct\n(proportion of highest participant)",
            fontsize=10,
        )
    ax.set_ylim(-0.05, 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)


def _plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

    colours = [_COLOURS[i % len(_COLOURS)] for i in range(len(df))]

    _panel(
        axes[0],
        df["mean_svg_inter"],
        df["relational_prop"],
        df["label"],
        "Mean decoding SVG z-score (interactional edges)",
        colours,
        show_ylabel=True,
    )

    _panel(
        axes[1],
        df["mean_svg_all"],
        df["relational_prop"],
        df["label"],
        "Mean decoding SVG z-score (all edges)",
        colours,
        show_ylabel=False,
    )

    axes[0].set_title("Interactional SVG", fontsize=11, fontweight="bold", pad=8)
    axes[1].set_title("All-edges SVG", fontsize=11, fontweight="bold", pad=8)

    fig.suptitle(
        "Participant-level relational memory vs mean decoding scanpath-graph alignment",
        fontsize=11,
        y=1.02,
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
        description="Participant-level relational memory proportion vs mean decoding SVG."
    )
    parser.add_argument("--features", default=str(_DEFAULT_FEATURES))
    parser.add_argument("--scores", default=str(_DEFAULT_SCORES))
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    args = parser.parse_args()

    df = _build(Path(args.features), Path(args.scores))
    _plot(df, Path(args.output) / "relational_svg_assoc.png")


if __name__ == "__main__":
    main()
