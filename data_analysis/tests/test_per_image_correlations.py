"""
test_per_image_correlations.py
===============================
Per-image (within-image, between-participant) correlations between
encoding SVG and memory recall proportions.

For each StimID, computes Pearson r between svg_z_enc_within and each
proportion DV across the N participants who viewed that image.
Also runs svg_z_enc (uncentred) for comparison.

This is a diagnostic complement to the LMM:
  - If the per-image rs are consistently positive, the weak pooled β
    reflects noisy averaging across images, not an absent effect.
  - If they're scattered around zero, the LMM result may reflect a
    small number of influential images.

Outputs
-------
  output/analysis/per_image_corr.csv     — r, p, n per image × DV × predictor
  output/analysis/per_image_corr.png     — distribution of r values + summary

Usage
-----
    python test_per_image_correlations.py
    python test_per_image_correlations.py --input path/to/analysis_enc.csv
    python test_per_image_correlations.py --output-dir path/to/output/analysis
    python test_per_image_correlations.py --min-n 8
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent  # tests/
_ROOT = _HERE.parent  # data_analysis/ (project root)
sys.path.insert(0, str(_ROOT))
try:
    import config

    DEFAULT_INPUT = config.OUTPUT_DIR / "analysis" / "analysis_enc.csv"
    DEFAULT_OUTPUT = config.OUTPUT_DIR / "analysis"
except Exception:
    DEFAULT_INPUT = _ROOT / "output" / "analysis" / "analysis_enc.csv"
    DEFAULT_OUTPUT = _ROOT / "output" / "analysis"

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Predictors to test — (column_name, display_label)
PREDICTORS = [
    ("svg_z_enc_within", "SVG within-image"),
]

# DVs to test — (column_name, display_label, colour)
DVS = [
    ("prop_total", "Total recall", "#2166ac"),
    ("prop_relations", "Relational recall", "#084594"),
    ("prop_objects", "Object recall", "#a63603"),
]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_per_image_corrs(
    df: pd.DataFrame,
    min_n: int,
) -> pd.DataFrame:
    """
    For each StimID × predictor × DV combination, compute Pearson r.

    Returns a tidy DataFrame with columns:
        StimID, predictor, dv, r, p, n, sig
    """
    records = []
    stim_ids = sorted(df["StimID"].unique())

    for stim_id in stim_ids:
        sub = df[df["StimID"] == stim_id].copy()

        for pred_col, pred_label in PREDICTORS:
            if pred_col not in sub.columns:
                continue

            for dv_col, dv_label, _ in DVS:
                if dv_col not in sub.columns:
                    continue

                valid = sub[[pred_col, dv_col]].dropna()
                n = len(valid)

                if n < min_n:
                    logger.debug(
                        f"  [{stim_id}] {pred_col}→{dv_col}: n={n} < {min_n} — skipping"
                    )
                    continue

                r, p = stats.pearsonr(valid[pred_col], valid[dv_col])
                sig = (
                    "***"
                    if p < 0.001
                    else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                )
                records.append(
                    {
                        "StimID": stim_id,
                        "predictor": pred_label,
                        "dv": dv_label,
                        "r": round(r, 4),
                        "p": round(p, 4),
                        "n": n,
                        "sig": sig,
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------


def summarise_corrs(corr_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each predictor × DV, compute:
      mean_r, median_r, sd_r, pct_positive, pct_sig (p<.05),
      one-sample t-test of rs against 0.
    """
    rows = []
    for (pred, dv), grp in corr_df.groupby(["predictor", "dv"]):
        rs = grp["r"].values
        n_images = len(rs)
        t, p_ttest = stats.ttest_1samp(rs, 0)
        rows.append(
            {
                "predictor": pred,
                "dv": dv,
                "n_images": n_images,
                "mean_r": round(rs.mean(), 4),
                "median_r": round(np.median(rs), 4),
                "sd_r": round(rs.std(), 4),
                "pct_positive": round(100 * (rs > 0).mean(), 1),
                "pct_sig_05": round(100 * (grp["p"] < 0.05).mean(), 1),
                "t_vs_zero": round(t, 3),
                "p_vs_zero": round(p_ttest, 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_per_image_corrs(
    corr_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Two-row grid:
      Row 1: within-image predictor — strip + box per DV
      Row 2: uncentred predictor    — strip + box per DV
    """
    predictors = [
        p for p, _ in PREDICTORS if p in [c for c in corr_df["predictor"].unique()]
    ]
    n_pred = len([lab for _, lab in PREDICTORS if lab in corr_df["predictor"].unique()])
    n_dv = len(DVS)

    fig, axes = plt.subplots(
        n_pred,
        n_dv,
        figsize=(4.5 * n_dv, 3.8 * n_pred),
        sharey=False,
    )
    if n_pred == 1:
        axes = axes[np.newaxis, :]

    for row_i, (pred_col, pred_label) in enumerate(PREDICTORS):
        if pred_label not in corr_df["predictor"].unique():
            continue
        pred_df = corr_df[corr_df["predictor"] == pred_label]

        for col_j, (dv_col, dv_label, colour) in enumerate(DVS):
            ax = axes[row_i, col_j]
            dv_df = pred_df[pred_df["dv"] == dv_label]

            if dv_df.empty:
                ax.set_visible(False)
                continue

            rs = dv_df["r"].values

            # Jittered strip
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(rs))
            ax.scatter(
                np.ones(len(rs)) + jitter,
                rs,
                color=colour,
                alpha=0.55,
                s=28,
                linewidths=0,
                zorder=3,
            )

            # Boxplot (no fliers — already shown as strip)
            bp = ax.boxplot(
                rs,
                positions=[1],
                widths=0.25,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="white", linewidth=2),
                boxprops=dict(facecolor=colour, alpha=0.35),
                whiskerprops=dict(color=colour, alpha=0.6),
                capprops=dict(color=colour, alpha=0.6),
            )

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

            # Annotation: summary stats
            summ = summary_df[
                (summary_df["predictor"] == pred_label) & (summary_df["dv"] == dv_label)
            ]
            if not summ.empty:
                s = summ.iloc[0]
                sig = (
                    "***"
                    if s["p_vs_zero"] < 0.001
                    else (
                        "**"
                        if s["p_vs_zero"] < 0.01
                        else "*" if s["p_vs_zero"] < 0.05 else "ns"
                    )
                )
                ax.annotate(
                    f"mean r = {s['mean_r']:+.3f}  {sig}\n"
                    f"{s['pct_positive']:.0f}% positive  |  "
                    f"{s['pct_sig_05']:.0f}% p<.05\n"
                    f"t({int(s['n_images'])-1}) = {s['t_vs_zero']:.2f}, "
                    f"p = {s['p_vs_zero']:.3f}",
                    xy=(0.05, 0.97),
                    xycoords="axes fraction",
                    fontsize=7.5,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
                )

            ax.set_xlim(0.6, 1.4)
            ax.set_xticks([])
            ax.set_ylabel("Pearson r  (within-image)", fontsize=8.5)
            ax.set_title(
                f"{dv_label}\n[{pred_label}]",
                fontsize=9,
                fontweight="bold",
                pad=5,
            )
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right", "bottom"]].set_visible(False)

    fig.suptitle(
        "Per-image correlations: Encoding SVG → memory recall\n"
        "(each point = one image; t-test vs zero across images)",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-image SVG → memory correlations (diagnostic)."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to analysis_enc.csv (Module 4 output).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=6,
        help="Minimum participants per image to include in correlation (default: 6).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path, dtype={"StimID": str, "SubjectID": str})

    # Exclude low_n trials if column present
    if "low_n_enc" in df.columns:
        before = len(df)
        df = df[~df["low_n_enc"]].copy()
        logger.info(f"Excluded {before - len(df)} low_n rows. Remaining: {len(df)}")

    logger.info(
        f"Loaded {len(df)} rows, "
        f"{df['SubjectID'].nunique()} participants, "
        f"{df['StimID'].nunique()} images."
    )

    # Per-image correlations
    logger.info(f"Computing per-image correlations (min_n={args.min_n}) ...")
    corr_df = compute_per_image_corrs(df, min_n=args.min_n)

    if corr_df.empty:
        logger.error("No correlations computed — check min_n or input data.")
        sys.exit(1)

    # Summary
    summary_df = summarise_corrs(corr_df)

    # Log summary
    logger.info("\n=== Per-image correlation summary (t-test of rs vs 0) ===")
    for _, row in summary_df.iterrows():
        sig = (
            "***"
            if row["p_vs_zero"] < 0.001
            else (
                "**"
                if row["p_vs_zero"] < 0.01
                else "*" if row["p_vs_zero"] < 0.05 else "ns"
            )
        )
        logger.info(
            f"  [{row['predictor']}] {row['dv']}: "
            f"mean r={row['mean_r']:+.3f}, "
            f"{row['pct_positive']:.0f}% positive, "
            f"{row['pct_sig_05']:.0f}% p<.05, "
            f"t({int(row['n_images'])-1})={row['t_vs_zero']:.2f}, "
            f"p={row['p_vs_zero']:.4f} {sig}"
        )

    # Write outputs
    corr_csv = output_dir / "per_image_corr.csv"
    corr_df.to_csv(corr_csv, index=False)
    logger.info(f"\n  Written → {corr_csv.name}")

    summary_csv = output_dir / "per_image_corr_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"  Written → {summary_csv.name}")

    plot_per_image_corrs(corr_df, summary_df, output_dir / "per_image_corr.png")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
