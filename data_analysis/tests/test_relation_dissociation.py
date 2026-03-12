"""
tests/test_relational_dissociation.py
======================================
Formal test of whether encoding SVG predicts action vs spatial relational
recall differentially.

Approach
--------
Stack n_action_relation_correct and n_spatial_relation_correct into long
format, adding a binary factor relation_type (0=action, 1=spatial).
Fit an OLS model with:

    score ~ svg_pred + relation_type + svg_pred:relation_type
            + covariates + C(SubjectID) + C(StimID)

C(StimID) is included because each stimulus contributes one action and one
spatial score — the two rows per trial are not independent.

The focal term is svg_pred:relation_type. A significant positive coefficient
means the SVG → memory slope is steeper for spatial than action relations.

Run for both svg_z_inter_enc and svg_z_all_enc separately.

Output
------
  Console table of interaction term β, t, p for each SVG variant.
  output/analysis/relational_dissociation.png
    Side-by-side slope plot: predicted score as a function of SVG,
    separately for action (dark) and spatial (light), for each SVG variant.

Usage
-----
    python tests/test_relational_dissociation.py
    python tests/test_relational_dissociation.py --features path/to/trial_features_all.csv
    python tests/test_relational_dissociation.py --scores   path/to/memory_scores.csv
    python tests/test_relational_dissociation.py --output   path/to/output/analysis
    python tests/test_relational_dissociation.py --no-plot
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
SVG_PREDICTORS = [
    ("svg_z_inter_enc", "Encoding SVG (interactional)"),
    ("svg_z_all_enc", "Encoding SVG (all-edges)"),
]

COVARIATES = ["n_fixations_enc", "aoi_prop_enc", "mean_salience_enc"]

COLOURS = {
    "action": "#084594",
    "spatial": "#6baed6",
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
    enc = features[features["Phase"] == "encoding"].copy()
    enc = enc.rename(
        columns={
            "svg_z_inter": "svg_z_inter_enc",
            "svg_z_all": "svg_z_all_enc",
            "n_fixations": "n_fixations_enc",
            "aoi_prop": "aoi_prop_enc",
            "mean_salience": "mean_salience_enc",
            "low_n": "low_n_enc",
        }
    )

    scores = pd.read_csv(scores_path, dtype={"StimID": str, "SubjectID": str})

    score_keep = [
        "SubjectID",
        "StimID",
        "n_action_relation_correct",
        "n_spatial_relation_correct",
    ]
    scores = scores[[c for c in score_keep if c in scores.columns]]

    merged = enc.merge(scores, on=["SubjectID", "StimID"], how="inner")

    if "low_n_enc" in merged.columns:
        before = len(merged)
        merged = merged[~merged["low_n_enc"]].copy()
        print(f"  Excluded {before - len(merged)} low-n encoding trials.")

    pred_cols = [p for p, _ in SVG_PREDICTORS]
    keep = (
        ["SubjectID", "StimID"]
        + pred_cols
        + COVARIATES
        + ["n_action_relation_correct", "n_spatial_relation_correct"]
    )
    merged = merged[[c for c in keep if c in merged.columns]].reset_index(drop=True)

    print(
        f"  {len(merged)} wide trials, {merged['SubjectID'].nunique()} participants, "
        f"{merged['StimID'].nunique()} stimuli."
    )
    return merged


# ---------------------------------------------------------------------------
# Long-format conversion
# ---------------------------------------------------------------------------


def _to_long(wide: pd.DataFrame, svg_col: str) -> pd.DataFrame:
    """
    Stack action and spatial scores into long format.
    Each wide row becomes two long rows, distinguished by relation_type (0/1).
    """
    id_cols = ["SubjectID", "StimID", svg_col] + [
        c for c in COVARIATES if c in wide.columns
    ]

    action = wide[id_cols + ["n_action_relation_correct"]].copy()
    action = action.rename(columns={"n_action_relation_correct": "score"})
    action["relation_type"] = 0  # action = 0

    spatial = wide[id_cols + ["n_spatial_relation_correct"]].copy()
    spatial = spatial.rename(columns={"n_spatial_relation_correct": "score"})
    spatial["relation_type"] = 1  # spatial = 1

    long = pd.concat([action, spatial], ignore_index=True)
    long = long.dropna(subset=["score", svg_col])

    # Z-score the SVG predictor within the long table
    mu, sd = long[svg_col].mean(), long[svg_col].std()
    long[f"{svg_col}_z"] = (long[svg_col] - mu) / sd if sd > 0 else 0.0

    return long


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def _fit_interaction(long: pd.DataFrame, svg_col: str) -> dict:
    """
    Fit: score ~ svg_z * relation_type + covariates + C(SubjectID) + C(StimID)

    Returns a dict with focal terms extracted.
    """
    svg_z = f"{svg_col}_z"
    cov_str = " + ".join(c for c in COVARIATES if c in long.columns)
    formula = (
        f"score ~ {svg_z} * relation_type"
        f" + {cov_str}"
        f" + C(SubjectID) + C(StimID)"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = smf.ols(formula, data=long).fit()
        except Exception as e:
            print(f"  [ERROR] Model fit failed: {e}")
            return {}

    def _extract(term):
        # Find term in params (handles statsmodels name formatting)
        matches = [k for k in result.params.index if term in k]
        if not matches:
            return dict(beta=np.nan, se=np.nan, t=np.nan, p=np.nan)
        k = matches[0]
        return dict(
            beta=result.params[k],
            se=result.bse[k],
            t=result.tvalues[k],
            p=result.pvalues[k],
        )

    return {
        "result": result,
        "svg_main": _extract(f"{svg_z}]"),
        "type_main": _extract("relation_type"),
        "interaction": _extract(f"{svg_z}:relation_type"),
        "n_obs": int(result.nobs),
        "r2": result.rsquared,
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(fit_results: dict) -> None:
    print("\n" + "=" * 72)
    print("ACTION vs SPATIAL DISSOCIATION — SVG × relation_type interaction")
    print(
        "Model: score ~ SVG_z * relation_type + covariates + C(SubjectID) + C(StimID)"
    )
    print("       relation_type: 0 = action, 1 = spatial")
    print("=" * 72)
    print(f"  {'SVG predictor':<30} {'term':<22} {'β':>8} {'t':>7} {'p':>8}  sig")
    print("-" * 72)

    for svg_col, svg_label in SVG_PREDICTORS:
        fit = fit_results.get(svg_col, {})
        if not fit:
            print(f"  {svg_label:<30}  [FAILED]")
            continue

        for term_key, term_label in [
            ("svg_main", "SVG main effect"),
            ("type_main", "relation_type main"),
            ("interaction", "SVG × relation_type"),
        ]:
            rec = fit.get(term_key, {})
            b, t, p = (
                rec.get("beta", np.nan),
                rec.get("t", np.nan),
                rec.get("p", np.nan),
            )
            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            label_col = svg_label if term_key == "svg_main" else ""
            print(
                f"  {label_col:<30} {term_label:<22} {b:>+8.3f} {t:>7.3f} {p:>8.4f}  {sig}"
            )

        print(f"  {'':30} {'n_obs':<22} {fit['n_obs']:>8}  R²={fit['r2']:.3f}")
        print()

    print("=" * 72)
    print("\n  Interpretation of SVG × relation_type:")
    print("  Positive β → SVG predicts spatial recall MORE than action recall")
    print("  Negative β → SVG predicts action recall MORE than spatial recall")
    print("  ns         → no evidence of differential prediction")


# ---------------------------------------------------------------------------
# Plot: predicted slopes per relation type
# ---------------------------------------------------------------------------


def _plot(wide: pd.DataFrame, fit_results: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(
        1, len(SVG_PREDICTORS), figsize=(6 * len(SVG_PREDICTORS), 5)
    )
    if len(SVG_PREDICTORS) == 1:
        axes = [axes]

    for ax, (svg_col, svg_label) in zip(axes, SVG_PREDICTORS):
        fit = fit_results.get(svg_col, {})
        if not fit or "result" not in fit:
            ax.set_visible(False)
            continue

        result = fit["result"]
        svg_z = f"{svg_col}_z"
        long = _to_long(wide, svg_col)

        # SVG z range for prediction line
        x_range = np.linspace(long[svg_z].min(), long[svg_z].max(), 100)

        # Build prediction DataFrames at mean covariates, reference SubjectID/StimID
        ref_subj = long["SubjectID"].mode()[0]
        ref_stim = long["StimID"].mode()[0]
        cov_means = {c: long[c].mean() for c in COVARIATES if c in long.columns}

        for rel_type, label, colour in [
            (0, "Action relation", COLOURS["action"]),
            (1, "Spatial relation", COLOURS["spatial"]),
        ]:
            pred_df = pd.DataFrame(
                {
                    svg_z: x_range,
                    "relation_type": rel_type,
                    "SubjectID": ref_subj,
                    "StimID": ref_stim,
                    **cov_means,
                }
            )
            try:
                y_pred = result.predict(pred_df)
            except Exception:
                continue

            ax.plot(x_range, y_pred, color=colour, linewidth=2.2, label=label)

            # Scatter raw data points
            sub = long[long["relation_type"] == rel_type]
            ax.scatter(
                sub[svg_z], sub["score"], color=colour, alpha=0.25, s=20, linewidths=0
            )

        # Annotate interaction term
        ix = fit.get("interaction", {})
        b, p = ix.get("beta", np.nan), ix.get("p", np.nan)
        if not np.isnan(b):
            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            ax.annotate(
                f"interaction β = {b:+.3f}, p = {p:.3f} {sig}",
                xy=(0.05, 0.06),
                xycoords="axes fraction",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

        ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.axvline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_xlabel(f"{svg_label} (z-scored)", fontsize=9)
        ax.set_ylabel("Recall score (raw count)", fontsize=9)
        ax.set_title(svg_label, fontsize=10, fontweight="bold", pad=7)
        ax.legend(fontsize=8.5, framealpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "Action vs Spatial relational recall — encoding SVG dissociation test\n"
        "Lines = model-predicted slopes at mean covariates; "
        "dots = raw trial scores",
        fontsize=10,
        y=1.02,
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
        description="Formal dissociation test: encoding SVG × relation type."
    )
    parser.add_argument("--features", default=str(_DEFAULT_FEATURES))
    parser.add_argument("--scores", default=str(_DEFAULT_SCORES))
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Action vs Spatial dissociation test")
    print("=" * 60)

    wide = _load(Path(args.features), Path(args.scores))

    fit_results = {}
    for svg_col, svg_label in SVG_PREDICTORS:
        print(f"\n  Fitting model for {svg_label} ...")
        long = _to_long(wide, svg_col)
        print(
            f"  Long format: {len(long)} rows "
            f"({long['relation_type'].eq(0).sum()} action, "
            f"{long['relation_type'].eq(1).sum()} spatial)"
        )
        fit_results[svg_col] = _fit_interaction(long, svg_col)

    _print_summary(fit_results)

    if not args.no_plot:
        _plot(wide, fit_results, Path(args.output) / "relational_dissociation.png")


if __name__ == "__main__":
    main()
