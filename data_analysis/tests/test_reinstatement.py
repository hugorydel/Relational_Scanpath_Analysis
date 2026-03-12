"""
tests/test_reinstatement.py
============================
Does encoding→decoding scanpath reinstatement predict episodic memory?

Predictors (stored on decoding rows in trial_features_all.csv):
  lcs_enc_dec  — normalised LCS between encoding and decoding object sequences
  tau_enc_dec  — Kendall's tau between encoding and decoding first-occurrence
                 orderings (shared objects only)

DVs (from memory_scores.csv):
  n_relational_correct  — action + spatial relational recall
  n_objects_correct     — object identity + attribute recall

Model (per predictor × DV):
  OLS with covariates n_fixations_dec, aoi_prop_dec, mean_salience_dec,
  n_shared_enc_dec (controls for how many objects were revisited),
  and C(SubjectID) fixed effects.
  Partial residual scatter plotted for each combination.

Output
------
  output/analysis/reinstatement.png
    2×2 grid: LCS vs tau (columns) × relational vs objects (rows).
    Each panel: partial residual scatter + regression line + r, p annotation.
  Console table of β, r, p per predictor × DV.

Usage
-----
    python tests/test_reinstatement.py
    python tests/test_reinstatement.py --features path/to/trial_features_all.csv
    python tests/test_reinstatement.py --scores   path/to/memory_scores.csv
    python tests/test_reinstatement.py --output   path/to/output/analysis
    python tests/test_reinstatement.py --no-plot
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
from scipy.stats import pearsonr

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
    ("lcs_enc_dec", "LCS\n(enc→dec)"),
    ("tau_enc_dec", "Kendall's τ\n(enc→dec)"),
]

DVS = [
    ("n_relational_correct", "Relational correct\n(action + spatial)", "#084594"),
    ("n_objects_correct", "Objects correct\n(identity + attribute)", "#a63603"),
]

COVARIATES = [
    "n_fixations_dec",
    "aoi_prop_dec",
    "mean_salience_dec",
    "n_shared_enc_dec",  # controls for overlap in visited objects
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

    # LCS/tau live on decoding rows
    dec = features[features["Phase"] == "decoding"].copy()
    dec = dec.rename(
        columns={
            "n_fixations": "n_fixations_dec",
            "aoi_prop": "aoi_prop_dec",
            "mean_salience": "mean_salience_dec",
            "low_n": "low_n_dec",
        }
    )

    scores = pd.read_csv(scores_path, dtype={"StimID": str, "SubjectID": str})

    # Derive combined DVs if not already present
    if "n_relational_correct" not in scores.columns:
        action = scores.get(
            "n_action_relation_correct", pd.Series(0, index=scores.index)
        )
        spatial = scores.get(
            "n_spatial_relation_correct", pd.Series(0, index=scores.index)
        )
        scores["n_relational_correct"] = action + spatial

    if "n_objects_correct" not in scores.columns:
        identity = scores.get(
            "n_object_identity_correct", pd.Series(0, index=scores.index)
        )
        attribute = scores.get(
            "n_object_attribute_correct", pd.Series(0, index=scores.index)
        )
        scores["n_objects_correct"] = identity + attribute

    score_keep = ["SubjectID", "StimID", "n_relational_correct", "n_objects_correct"]
    scores = scores[[c for c in score_keep if c in scores.columns]]

    merged = dec.merge(scores, on=["SubjectID", "StimID"], how="inner")

    # Exclude low-n decoding trials (too few transitions for reliable LCS)
    if "low_n_dec" in merged.columns:
        before = len(merged)
        merged = merged[~merged["low_n_dec"]].copy()
        print(f"  Excluded {before - len(merged)} low-n decoding trials.")

    # Drop rows where LCS or tau is NaN
    pred_cols = [p for p, _ in PREDICTORS]
    before = len(merged)
    merged = merged.dropna(subset=pred_cols).reset_index(drop=True)
    if len(merged) < before:
        print(f"  Dropped {before - len(merged)} rows with NaN predictors.")

    print(
        f"  {len(merged)} trials, {merged['SubjectID'].nunique()} participants, "
        f"{merged['StimID'].nunique()} stimuli."
    )
    return merged


# ---------------------------------------------------------------------------
# Model fitting (partial regression)
# ---------------------------------------------------------------------------


def _fit(df: pd.DataFrame, pred_col: str, dv_col: str) -> dict:
    cov_str = " + ".join(c for c in COVARIATES if c in df.columns)
    formula = f"{dv_col} ~ {pred_col} + {cov_str} + C(SubjectID)"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = smf.ols(formula, data=df).fit()
        except Exception as e:
            print(f"  [ERROR] {pred_col} → {dv_col}: {e}")
            return {}

    beta = result.params.get(pred_col, np.nan)
    p = result.pvalues.get(pred_col, np.nan)

    # Partial residuals for scatter
    cov_formula = f"{dv_col} ~ {cov_str} + C(SubjectID)"
    pred_formula = f"{pred_col} ~ {cov_str} + C(SubjectID)"
    try:
        res_dv = smf.ols(cov_formula, data=df).fit().resid
        res_pred = smf.ols(pred_formula, data=df).fit().resid
        r, _ = pearsonr(res_pred, res_dv)
    except Exception:
        res_pred = res_dv = np.full(len(df), np.nan)
        r = np.nan

    return {
        "beta": beta,
        "p": p,
        "r": r,
        "res_pred": res_pred,
        "res_dv": res_dv,
        "n": int(result.nobs),
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(fit_results: dict) -> None:
    print("\n" + "=" * 72)
    print("REINSTATEMENT → MEMORY (partial regression)")
    print("Covariates: n_fixations_dec, aoi_prop_dec, mean_salience_dec,")
    print("            n_shared_enc_dec, C(SubjectID)")
    print("=" * 72)
    print(f"  {'Predictor':<22} {'DV':<42} {'β':>8} {'r':>7} {'p':>8}  sig")
    print("-" * 72)
    for (pred_col, pred_label), (dv_col, dv_label, _) in [
        (p, d) for p in PREDICTORS for d in DVS
    ]:
        fit = fit_results.get((pred_col, dv_col), {})
        b = fit.get("beta", np.nan)
        r = fit.get("r", np.nan)
        p = fit.get("p", np.nan)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(
            f"  {pred_label.replace(chr(10),' '):<22} "
            f"{dv_label.replace(chr(10),' '):<42} "
            f"{b:>+8.3f} {r:>+7.3f} {p:>8.4f}  {sig}"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(df: pd.DataFrame, fit_results: dict, output_path: Path) -> None:
    n_rows = len(DVS)
    n_cols = len(PREDICTORS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))

    for col_idx, (pred_col, pred_label) in enumerate(PREDICTORS):
        for row_idx, (dv_col, dv_label, colour) in enumerate(DVS):
            ax = axes[row_idx][col_idx]
            fit = fit_results.get((pred_col, dv_col), {})

            if not fit or np.isnan(fit.get("r", np.nan)):
                ax.set_visible(False)
                continue

            res_pred = fit["res_pred"]
            res_dv = fit["res_dv"]
            r, p, b = fit["r"], fit["p"], fit["beta"]
            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )

            ax.scatter(res_pred, res_dv, color=colour, alpha=0.35, s=22, linewidths=0)

            # Regression line on residuals
            m, c = np.polyfit(res_pred, res_dv, 1)
            x_line = np.linspace(res_pred.min(), res_pred.max(), 100)
            ax.plot(x_line, m * x_line + c, color=colour, linewidth=2, alpha=0.9)

            ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.4)
            ax.axvline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.4)

            ax.annotate(
                f"r = {r:+.3f}, p = {p:.3f} {sig}",
                xy=(0.05, 0.93),
                xycoords="axes fraction",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

            ax.set_xlabel(
                f"{pred_label.replace(chr(10), ' ')}\n" f"(covariate-residualised)",
                fontsize=8.5,
            )
            ax.set_ylabel(
                f"{dv_label.replace(chr(10), ' ')}\n" f"(covariate-residualised)",
                fontsize=8.5,
            )
            ax.set_title(
                f"{pred_label.replace(chr(10), ' ')} → "
                f"{dv_label.replace(chr(10), ' ')}",
                fontsize=9,
                fontweight="bold",
                pad=6,
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)

    fig.suptitle(
        "Scanpath reinstatement (enc→dec) → episodic memory\n"
        "Partial regression (covariates: fixations, AOI prop, salience, "
        "shared objects, SubjectID)",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {output_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Scanpath reinstatement (LCS + tau) → episodic memory."
    )
    parser.add_argument("--features", default=str(_DEFAULT_FEATURES))
    parser.add_argument("--scores", default=str(_DEFAULT_SCORES))
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Scanpath reinstatement → episodic memory")
    print("=" * 60)

    df = _load(Path(args.features), Path(args.scores))

    fit_results = {}
    for pred_col, pred_label in PREDICTORS:
        for dv_col, dv_label, _ in DVS:
            fit_results[(pred_col, dv_col)] = _fit(df, pred_col, dv_col)

    _print_summary(fit_results)

    if not args.no_plot:
        _plot(df, fit_results, Path(args.output) / "reinstatement.png")


if __name__ == "__main__":
    main()
