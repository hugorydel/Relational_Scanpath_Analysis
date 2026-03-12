"""
tests/test_decoding_memory_detail.py
=====================================
Decoding-time scanpath → episodic memory, with fine-grained DV breakdown.

Produces four figures:

  1. decoding_memory_relational_split.png
       DVs: n_action_relation_correct vs n_spatial_relation_correct (side by side)

  2. decoding_memory_relational_combined.png
       DV: n_action_relation_correct + n_spatial_relation_correct

  3. decoding_memory_objects_combined.png
       DV: n_object_identity_correct + n_object_attribute_correct

  4. decoding_memory_objects_identity.png
       DV: n_object_identity_correct only (no features)

Predictors (decoding phase)
---------------------------
  svg_z_dec  — core SVG z-score (interactional + spatial + functional edges)

Covariates partialled out from both predictor and DV
-----------------------------------------------------
  n_fixations_dec, aoi_prop_dec, mean_salience_dec, C(SubjectID)

Usage
-----
    python tests/test_decoding_memory_detail.py
    python tests/test_decoding_memory_detail.py --features path/to/trial_features_all.csv
    python tests/test_decoding_memory_detail.py --scores   path/to/memory_scores.csv
    python tests/test_decoding_memory_detail.py --output   path/to/output/analysis
    python tests/test_decoding_memory_detail.py --no-plot
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
    ("svg_z_dec", "Decoding SVG (core)"),
]

COVARIATES = ["n_fixations_dec", "aoi_prop_dec", "mean_salience_dec"]

# Single combined figure: relational + objects side by side
FIGURES = [
    (
        "combined",
        "Decoding scanpath → memory",
        [
            (
                "n_relational_correct",
                "Relational correct\n(action + spatial)",
                "#084594",
            ),
            ("n_objects_correct", "Objects correct\n(identity + attribute)", "#a63603"),
        ],
    ),
]

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
    dec = dec.rename(
        columns={
            "svg_z": "svg_z_dec",
            "n_fixations": "n_fixations_dec",
            "aoi_prop": "aoi_prop_dec",
            "mean_salience": "mean_salience_dec",
            "low_n": "low_n_dec",
        }
    )

    scores = pd.read_csv(scores_path, dtype={"StimID": str, "SubjectID": str})

    # Derive aggregate columns if not already present
    def _sum(df, *cols):
        present = [c for c in cols if c in df.columns]
        return df[present].sum(axis=1) if present else pd.Series(0, index=df.index)

    if "n_relational_correct" not in scores.columns:
        scores["n_relational_correct"] = _sum(
            scores, "n_action_relation_correct", "n_spatial_relation_correct"
        )
    if "n_objects_correct" not in scores.columns:
        scores["n_objects_correct"] = _sum(
            scores, "n_object_identity_correct", "n_object_attribute_correct"
        )

    # All DV columns needed across all figures
    dv_cols = [
        "n_action_relation_correct",
        "n_spatial_relation_correct",
        "n_relational_correct",
        "n_objects_correct",
        "n_object_identity_correct",
    ]
    score_keep = ["SubjectID", "StimID"] + [c for c in dv_cols if c in scores.columns]
    scores = scores[score_keep]

    merged = dec.merge(scores, on=["SubjectID", "StimID"], how="inner")

    if "low_n_dec" in merged.columns:
        before = len(merged)
        merged = merged[~merged["low_n_dec"]].copy()
        print(f"  Excluded {before - len(merged)} low-n decoding trials.")
    # Exclude wrong-image trials (participant described a different image)
    if "wrong_image" in merged.columns:
        before = len(merged)
        merged = merged[merged["wrong_image"] != 1].copy()
        print(f"  Excluded {before - len(merged)} wrong-image trials.")

    pred_cols = [p for p, _ in PREDICTORS]
    keep = ["SubjectID", "StimID"] + pred_cols + COVARIATES + dv_cols
    merged = merged[[c for c in keep if c in merged.columns]].reset_index(drop=True)

    print(
        f"  {len(merged)} trials, {merged['SubjectID'].nunique()} participants, "
        f"{merged['StimID'].nunique()} stimuli."
    )
    return merged


# ---------------------------------------------------------------------------
# Residualisation + OLS
# ---------------------------------------------------------------------------


def _residualise(df: pd.DataFrame, target: str) -> pd.Series:
    req = [target] + [c for c in COVARIATES if c in df.columns]
    mask = df[req].notna().all(axis=1)
    out = pd.Series(np.nan, index=df.index)
    sub = df[mask].copy()
    if len(sub) < 10:
        return out
    formula = " + ".join(c for c in COVARIATES if c in df.columns)
    formula = f"{target} ~ {formula} + C(SubjectID)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = smf.ols(formula, data=sub).fit()
            out.loc[mask] = res.resid.values
        except Exception as e:
            print(f"    [WARN] residualisation of {target} failed: {e}")
    return out


def _ols(x: np.ndarray, y: np.ndarray):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:
        return np.nan, np.nan, np.nan, np.nan
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    return slope, intercept, r, p


def _beta_se(x, y, r, slope):
    mask = ~(np.isnan(x) | np.isnan(y))
    n = mask.sum()
    if n < 4 or np.isnan(r):
        return np.nan
    sd_x = np.std(x[mask], ddof=1)
    sd_y = np.std(y[mask], ddof=1)
    se_r = np.sqrt((1 - r**2) / (n - 2))
    return se_r * (sd_y / sd_x) if sd_x > 0 else np.nan


# ---------------------------------------------------------------------------
# Single figure
# ---------------------------------------------------------------------------


def _make_figure(
    df: pd.DataFrame,
    title: str,
    dv_specs: list,  # [(dv_col, dv_label, colour), ...]
    output_path: Path,
    pred_resids: dict,  # pre-computed predictor residuals
) -> list[dict]:
    """
    Draw one figure: n_pred columns × n_dv rows of partial regression panels,
    plus a summary bar chart.  Returns coef_records for console printing.
    """
    n_pred = len(PREDICTORS)
    n_dv = len(dv_specs)

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

    coef_records = []

    for dv_i, (dv_col, dv_label, colour) in enumerate(dv_specs):
        if dv_col not in df.columns:
            for pred_i in range(n_pred):
                axes_grid[dv_i][pred_i].set_visible(False)
            continue

        dv_resid = _residualise(df, dv_col)

        for pred_i, (pred_col, pred_label) in enumerate(PREDICTORS):
            ax = axes_grid[dv_i][pred_i]
            pr = pred_resids.get(pred_col, pd.Series(np.nan, index=df.index))
            x, y = pr.values, dv_resid.values

            mask = ~(np.isnan(x) | np.isnan(y))
            ax.scatter(
                x[mask], y[mask], color=colour, alpha=0.55, s=28, linewidths=0, zorder=3
            )

            slope, intercept, r, p = _ols(x, y)
            se = _beta_se(x, y, r, slope)

            coef_records.append(
                {
                    "pred_col": pred_col,
                    "pred_label": pred_label,
                    "dv_col": dv_col,
                    "dv_label": dv_label.replace("\n", " "),
                    "n": int(mask.sum()),
                    "beta": slope,
                    "beta_se": se,
                    "r": r,
                    "p": p,
                    "colour": colour,
                }
            )

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

    # Summary bar chart
    n_groups = n_pred
    n_bars = n_dv
    bar_w = 0.65 / max(n_bars, 1)

    for g_i, (pred_col, pred_label) in enumerate(PREDICTORS):
        for b_i, (dv_col, dv_label, colour) in enumerate(dv_specs):
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
            ax_summary.bar(
                x_pos,
                rec["beta"],
                width=bar_w * 0.85,
                color=colour,
                alpha=0.80,
                label=dv_label.replace("\n", " ") if g_i == 0 else "_nolegend_",
            )
            if not np.isnan(rec.get("beta_se", np.nan)):
                ax_summary.errorbar(
                    x_pos,
                    rec["beta"],
                    yerr=1.96 * rec["beta_se"],
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
                y_tip = rec["beta"] + 1.96 * rec.get("beta_se", 0)
                ax_summary.text(
                    x_pos,
                    y_tip + 0.005,
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
    ax_summary.set_title("Summary coefficients", fontsize=9, fontweight="bold", pad=6)
    if n_dv > 1:
        ax_summary.legend(fontsize=8, framealpha=0.7, loc="upper right")
    ax_summary.spines[["top", "right"]].set_visible(False)
    ax_summary.tick_params(labelsize=8)

    fig.suptitle(
        f"{title}\nPartial regression (covariates: n_fixations, aoi_prop, mean_salience, SubjectID)",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {output_path.name}")
    return coef_records


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(all_records: list[dict], fig_title: str) -> None:
    print(f"\n  {fig_title}")
    print("  " + "-" * 68)
    print(f"  {'Predictor':<28} {'DV':<30} {'β':>7} {'r':>7} {'p':>8}  sig")
    print("  " + "-" * 68)
    for rec in all_records:
        sig = (
            "***"
            if rec["p"] < 0.001
            else "**" if rec["p"] < 0.01 else "*" if rec["p"] < 0.05 else "ns"
        )
        print(
            f"  {rec['pred_label']:<28} {rec['dv_label']:<30} "
            f"  {rec['beta']:>5.3f}  {rec['r']:>6.3f}  {rec['p']:>7.4f}  {sig}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Decoding scanpath → fine-grained memory breakdown."
    )
    parser.add_argument("--features", default=str(_DEFAULT_FEATURES))
    parser.add_argument("--scores", default=str(_DEFAULT_SCORES))
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Decoding scanpath → memory (detailed breakdown)")
    print("=" * 60)

    df = _load(Path(args.features), Path(args.scores))

    # Pre-compute predictor residuals once (shared across all figures)
    pred_resids = {
        pred_col: _residualise(df, pred_col)
        for pred_col, _ in PREDICTORS
        if pred_col in df.columns
    }

    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    for suffix, title, dv_specs in FIGURES:
        out_path = Path(args.output) / f"decoding_memory_{suffix}.png"
        records = (
            _make_figure(df, title, dv_specs, out_path, pred_resids)
            if not args.no_plot
            else _compute_only(df, dv_specs, pred_resids)
        )
        _print_summary(records, title)

    print()


def _compute_only(df, dv_specs, pred_resids) -> list[dict]:
    """Compute coefficients without plotting (--no-plot path)."""
    records = []
    for dv_col, dv_label, colour in dv_specs:
        if dv_col not in df.columns:
            continue
        dv_resid = _residualise(df, dv_col)
        for pred_col, pred_label in PREDICTORS:
            pr = pred_resids.get(pred_col, pd.Series(np.nan, index=df.index))
            x, y = pr.values, dv_resid.values
            slope, intercept, r, p = _ols(x, y)
            se = _beta_se(x, y, r, slope)
            mask = ~(np.isnan(x) | np.isnan(y))
            records.append(
                {
                    "pred_col": pred_col,
                    "pred_label": pred_label,
                    "dv_col": dv_col,
                    "dv_label": dv_label.replace("\n", " "),
                    "n": int(mask.sum()),
                    "beta": slope,
                    "beta_se": se,
                    "r": r,
                    "p": p,
                    "colour": colour,
                }
            )
    return records


if __name__ == "__main__":
    main()
