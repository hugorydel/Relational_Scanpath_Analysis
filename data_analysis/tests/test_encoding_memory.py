"""
tests/test_encoding_memory.py
==============================
Standalone test: encoding-time scanpath quality → episodic memory.

Motivation
----------
Module 4 exploratory results show that encoding SVG (both inter and all-edges)
predicts n_relational_correct at β ≈ 0.38, p < 0.01, while LCS is null.
This module examines that association in detail with partial regression plots
and a summary coefficient chart.

Predictors (encoding phase)
---------------------------
  svg_z_enc  — core SVG z-score (interactional + spatial + functional edges)

Note: LCS is excluded — it measures enc→dec AOI sequence overlap, making it
a retrieval-phase measure rather than a pure encoding measure.

DVs
---
  n_relational_correct  = n_action_relation_correct + n_spatial_relation_correct
  n_objects_correct     = n_object_identity_correct + n_object_attribute_correct

Covariates (partialled out from both predictor and DV before plotting)
----------------------------------------------------------------------
  n_fixations_enc, aoi_prop_enc, mean_salience_enc, C(SubjectID)

Output
------
  output/analysis/encoding_memory.png
    2×3 grid of partial regression scatters (rows=DV, cols=predictor)
    + 1 summary bar chart of OLS β coefficients with 95% CIs

Usage
-----
    python tests/test_encoding_memory.py
    python tests/test_encoding_memory.py --features path/to/trial_features_all.csv
    python tests/test_encoding_memory.py --scores   path/to/memory_scores.csv
    python tests/test_encoding_memory.py --output   path/to/output/analysis
    python tests/test_encoding_memory.py --no-plot
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PREDICTORS = [
    ("svg_z_enc", "Encoding SVG (core)"),
]

DVS = [
    ("n_relational_correct", "Relational correct\n(action + spatial)"),
    ("n_objects_correct", "Objects correct\n(identity + attribute)"),
]

COVARIATES = ["n_fixations_enc", "aoi_prop_enc", "mean_salience_enc"]

# Palette per DV
DV_COLOURS = {
    "n_relational_correct": "#084594",
    "n_objects_correct": "#a63603",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load(features_path: Path, scores_path: Path) -> pd.DataFrame:
    """
    Join encoding trial features with memory scores on SubjectID × StimID.
    Returns a trial-level DataFrame with predictor, covariate, and DV columns.
    """
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores not found: {scores_path}")

    features = pd.read_csv(features_path, dtype={"StimID": str, "SubjectID": str})
    enc = features[features["Phase"] == "encoding"].copy()

    # Rename to _enc suffix so they're unambiguous
    enc = enc.rename(
        columns={
            "svg_z": "svg_z_enc",
            "n_fixations": "n_fixations_enc",
            "aoi_prop": "aoi_prop_enc",
            "mean_salience": "mean_salience_enc",
            "low_n": "low_n_enc",
        }
    )

    scores = pd.read_csv(scores_path, dtype={"StimID": str, "SubjectID": str})

    # Derive aggregate DVs from 20-column schema if needed
    def _sum_cols(df, *cols):
        present = [c for c in cols if c in df.columns]
        return df[present].sum(axis=1) if present else pd.Series(0, index=df.index)

    if "n_relational_correct" not in scores.columns:
        scores["n_relational_correct"] = _sum_cols(
            scores, "n_action_relation_correct", "n_spatial_relation_correct"
        )
    if "n_objects_correct" not in scores.columns:
        scores["n_objects_correct"] = _sum_cols(
            scores, "n_object_identity_correct", "n_object_attribute_correct"
        )

    score_keep = [
        "SubjectID",
        "StimID",
        "n_relational_correct",
        "n_objects_correct",
        "empty_response",
    ]
    scores = scores[[c for c in score_keep if c in scores.columns]]

    merged = enc.merge(scores, on=["SubjectID", "StimID"], how="inner")

    # Exclude low-n encoding trials
    if "low_n_enc" in merged.columns:
        before = len(merged)
        merged = merged[~merged["low_n_enc"]].copy()
        print(f"  Excluded {before - len(merged)} low-n encoding trials.")

    pred_cols = [p for p, _ in PREDICTORS]
    dv_cols = [d for d, _ in DVS]
    keep_cols = ["SubjectID", "StimID"] + pred_cols + COVARIATES + dv_cols
    merged = merged[[c for c in keep_cols if c in merged.columns]].reset_index(
        drop=True
    )

    print(
        f"  {len(merged)} trials, {merged['SubjectID'].nunique()} participants, "
        f"{merged['StimID'].nunique()} stimuli."
    )
    return merged


# ---------------------------------------------------------------------------
# Partial residualisation
# ---------------------------------------------------------------------------


def _residualise(df: pd.DataFrame, target: str) -> pd.Series:
    """
    Return residuals of `target` after regressing on COVARIATES + C(SubjectID).
    Rows with any NaN in the required columns are set to NaN in output.
    """
    req = [target] + [c for c in COVARIATES if c in df.columns]
    mask = df[req].notna().all(axis=1)
    out = pd.Series(np.nan, index=df.index)

    sub = df[mask].copy()
    if len(sub) < 10:
        return out

    cov_terms = " + ".join(c for c in COVARIATES if c in df.columns)
    formula = f"{target} ~ {cov_terms} + C(SubjectID)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = smf.ols(formula, data=sub).fit()
            out.loc[mask] = res.resid.values
        except Exception as e:
            print(f"    [WARN] residualisation of {target} failed: {e}")
    return out


# ---------------------------------------------------------------------------
# OLS on residuals
# ---------------------------------------------------------------------------


def _ols(x: np.ndarray, y: np.ndarray):
    """Return (slope, intercept, r, p) from simple OLS on non-NaN pairs."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:
        return np.nan, np.nan, np.nan, np.nan
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    return slope, intercept, r, p


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(coef_records: list) -> None:
    print("\n" + "=" * 72)
    print("ENCODING SCANPATH → MEMORY  (partial regression, covariates removed)")
    print("=" * 72)
    print(f"  {'Predictor':<28} {'DV':<26} {'β':>8} {'r':>7} {'p':>8}  sig")
    print("-" * 72)
    for rec in coef_records:
        sig = (
            "***"
            if rec["p"] < 0.001
            else "**" if rec["p"] < 0.01 else "*" if rec["p"] < 0.05 else "ns"
        )
        print(
            f"  {rec['pred_label']:<28} {rec['dv_label']:<26} "
            f"  {rec['beta']:>6.3f}  {rec['r']:>6.3f}  {rec['p']:>7.4f}  {sig}"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot(df: pd.DataFrame, coef_records: list, output_path: Path) -> None:
    n_dv = len(DVS)
    n_pred = len(PREDICTORS)

    # Main grid (partial regression panels) + summary row below
    fig = plt.figure(figsize=(5.5 * n_pred, 4.5 * n_dv + 3.5))
    gs = fig.add_gridspec(
        n_dv + 1,
        n_pred,
        height_ratios=[1] * n_dv + [0.75],
        hspace=0.48,
        wspace=0.35,
    )

    axes_grid = [
        [fig.add_subplot(gs[r, c]) for c in range(n_pred)] for r in range(n_dv)
    ]
    ax_summary = fig.add_subplot(gs[n_dv, :])

    # ── Partial regression panels ─────────────────────────────────────────
    for dv_i, (dv_col, dv_label) in enumerate(DVS):
        colour = DV_COLOURS.get(dv_col, "#636363")
        dv_resid = _residualise(df, dv_col)

        for pred_i, (pred_col, pred_label) in enumerate(PREDICTORS):
            ax = axes_grid[dv_i][pred_i]

            if pred_col not in df.columns or dv_col not in df.columns:
                ax.set_visible(False)
                continue

            pred_resid = _residualise(df, pred_col)
            x = pred_resid.values
            y = dv_resid.values

            mask = ~(np.isnan(x) | np.isnan(y))
            ax.scatter(
                x[mask], y[mask], color=colour, alpha=0.55, s=28, linewidths=0, zorder=3
            )

            slope, intercept, r, p = _ols(x, y)
            if not np.isnan(slope):
                x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                ax.plot(
                    x_line,
                    intercept + slope * x_line,
                    color=colour,
                    linewidth=1.8,
                    zorder=4,
                )
                sig = (
                    "***"
                    if p < 0.001
                    else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                )
                ax.annotate(
                    f"r = {r:+.3f}, p = {p:.3f} {sig}",
                    xy=(0.05, 0.93),
                    xycoords="axes fraction",
                    fontsize=8,
                    color=colour,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75),
                )

            ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.axvline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.set_xlabel(f"{pred_label}\n(covariate-residualised)", fontsize=8)
            ax.set_ylabel(f"{dv_label}\n(covariate-residualised)", fontsize=8)
            ax.tick_params(labelsize=7.5)
            ax.spines[["top", "right"]].set_visible(False)

            if dv_i == 0:
                ax.set_title(pred_label, fontsize=9, fontweight="bold", pad=6)

    # ── Summary bar chart ─────────────────────────────────────────────────
    # One group of bars per predictor; within each group, one bar per DV
    n_groups = n_pred
    n_bars = n_dv
    group_w = 0.65
    bar_w = group_w / n_bars

    for g_i, (pred_col, pred_label) in enumerate(PREDICTORS):
        for b_i, (dv_col, dv_label) in enumerate(DVS):
            rec = next(
                (
                    r
                    for r in coef_records
                    if r["pred_col"] == pred_col and r["dv_col"] == dv_col
                ),
                None,
            )
            if rec is None or np.isnan(rec["beta"]):
                continue

            x_pos = g_i + (b_i - (n_bars - 1) / 2) * bar_w
            colour = DV_COLOURS.get(dv_col, "#636363")
            ax_summary.bar(
                x_pos,
                rec["beta"],
                width=bar_w * 0.85,
                color=colour,
                alpha=0.80,
                label=dv_label.replace("\n", " ") if g_i == 0 else "_nolegend_",
            )

            # 95% CI from OLS SE (not stored separately — approximate from r, n)
            # Use t-based CI: se_beta ≈ sqrt((1-r²)/(n-2)) * (sd_y/sd_x) but
            # since we're on residuals, sd ratio ≈ 1. Store beta_se in records.
            if "beta_se" in rec and not np.isnan(rec["beta_se"]):
                ci = 1.96 * rec["beta_se"]
                ax_summary.errorbar(
                    x_pos,
                    rec["beta"],
                    yerr=ci,
                    fmt="none",
                    color="black",
                    linewidth=1.2,
                    capsize=3,
                )

            sig = (
                "***"
                if rec["p"] < 0.001
                else "**" if rec["p"] < 0.01 else "*" if rec["p"] < 0.05 else ""
            )
            if sig:
                y_tip = rec["beta"] + (
                    1.96 * rec.get("beta_se", 0) if "beta_se" in rec else 0
                )
                ax_summary.text(
                    x_pos,
                    y_tip + 0.01,
                    sig,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=colour,
                )

    ax_summary.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_summary.set_xticks(range(n_groups))
    ax_summary.set_xticklabels([pl for _, pl in PREDICTORS], fontsize=8.5)
    ax_summary.set_ylabel("β (partial regression)", fontsize=8.5)
    ax_summary.set_title(
        "Summary: encoding predictor → memory (covariate-adjusted)",
        fontsize=9,
        fontweight="bold",
        pad=6,
    )
    ax_summary.legend(fontsize=8, framealpha=0.7, loc="upper right")
    ax_summary.spines[["top", "right"]].set_visible(False)
    ax_summary.tick_params(labelsize=8)

    fig.suptitle(
        "Encoding scanpath → episodic memory\n"
        "Partial regression plots (covariates: n_fixations, aoi_prop, mean_salience, SubjectID)",
        fontsize=10,
        y=1.01,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Encoding scanpath → memory: partial regression plots."
    )
    parser.add_argument("--features", default=str(_DEFAULT_FEATURES))
    parser.add_argument("--scores", default=str(_DEFAULT_SCORES))
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Encoding scanpath → episodic memory")
    print("=" * 60)

    df = _load(Path(args.features), Path(args.scores))

    # Residualise DVs once
    dv_resids = {dv_col: _residualise(df, dv_col) for dv_col, _ in DVS}

    # Fit OLS on residuals for each predictor × DV pair, collect stats
    coef_records = []
    for pred_col, pred_label in PREDICTORS:
        pred_resid = (
            _residualise(df, pred_col)
            if pred_col in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        for dv_col, dv_label in DVS:
            x = pred_resid.values
            y = dv_resids[dv_col].values
            slope, intercept, r, p = _ols(x, y)

            # SE of slope from residuals
            mask = ~(np.isnan(x) | np.isnan(y))
            n = mask.sum()
            beta_se = np.nan
            if n > 3 and not np.isnan(r):
                se_r = np.sqrt((1 - r**2) / (n - 2))
                # slope SE ≈ se_r * (sd_y / sd_x); on residuals sd≈1 but compute exactly
                sd_x = np.std(x[mask], ddof=1)
                sd_y = np.std(y[mask], ddof=1)
                if sd_x > 0:
                    beta_se = se_r * (sd_y / sd_x)

            coef_records.append(
                {
                    "pred_col": pred_col,
                    "pred_label": pred_label,
                    "dv_col": dv_col,
                    "dv_label": dv_label.replace("\n", " "),
                    "n": int(n),
                    "beta": slope,
                    "beta_se": beta_se,
                    "r": r,
                    "p": p,
                }
            )

    _print_summary(coef_records)

    if not args.no_plot:
        _plot(df, coef_records, Path(args.output) / "encoding_memory.png")


if __name__ == "__main__":
    main()
