"""
tests/test_relation_object_dissociation.py
==========================================
Formal test of whether encoding SVG predicts relational memory more than
object memory.

Motivation
----------
The encoding SVG → memory effect is currently non-specific: both relational
and object DVs are significant with similar β. This test asks whether the
SVG → memory slope is meaningfully steeper for relational content.

Approach
--------
Stack n_relational_correct and n_objects_correct into long format, adding a
binary factor memory_type (0=objects, 1=relational).
Fit an OLS model with:

    score ~ svg_z_enc * memory_type
            + covariates + C(SubjectID) + C(StimID)

C(StimID) is included because each stimulus contributes one relational and
one object score — the two rows per trial are not independent.

The focal term is svg_z_enc:memory_type. A significant positive coefficient
means the SVG → memory slope is steeper for relational than object memory —
i.e. relational scanning specifically benefits relational recall beyond its
general effect on memory quality.

Covariates: n_fixations_enc, aoi_prop_enc, mean_salience_relational_enc

Output
------
  Console table of all model terms (main effects + interaction).
  output/analysis/relation_object_dissociation.png
    Predicted slopes for relational vs object memory as a function of
    encoding SVG, at mean covariates.

Usage
-----
    python tests/test_relation_object_dissociation.py
    python tests/test_relation_object_dissociation.py --features path/to/trial_features_all.csv
    python tests/test_relation_object_dissociation.py --scores   path/to/memory_scores.csv
    python tests/test_relation_object_dissociation.py --output   path/to/output/analysis
    python tests/test_relation_object_dissociation.py --no-plot
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
    ("svg_z_enc", "Encoding SVG (core)"),
]

COVARIATES = ["n_fixations_enc", "aoi_prop_enc", "mean_salience_relational_enc"]

COLOURS = {
    "objects": "#a63603",
    "relational": "#084594",
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
            "svg_z": "svg_z_enc",
            "n_fixations": "n_fixations_enc",
            "aoi_prop": "aoi_prop_enc",
            "mean_salience": "mean_salience_enc",
            "mean_salience_relational": "mean_salience_relational_enc",
            "low_n": "low_n_enc",
        }
    )

    scores = pd.read_csv(scores_path, dtype={"StimID": str, "SubjectID": str})

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

    score_keep = [
        "SubjectID",
        "StimID",
        "n_relational_correct",
        "n_objects_correct",
        "wrong_image",
    ]
    scores = scores[[c for c in score_keep if c in scores.columns]]

    merged = enc.merge(scores, on=["SubjectID", "StimID"], how="inner")

    if "low_n_enc" in merged.columns:
        before = len(merged)
        merged = merged[~merged["low_n_enc"]].copy()
        print(f"  Excluded {before - len(merged)} low-n encoding trials.")
    if "wrong_image" in merged.columns:
        before = len(merged)
        merged = merged[merged["wrong_image"] != 1].copy()
        print(f"  Excluded {before - len(merged)} wrong-image trials.")

    pred_cols = [p for p, _ in SVG_PREDICTORS]
    keep = (
        ["SubjectID", "StimID"]
        + pred_cols
        + COVARIATES
        + ["n_relational_correct", "n_objects_correct"]
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
    Stack relational and object scores into long format.
    memory_type: 0 = objects, 1 = relational.

    Each DV is z-scored within the wide table before stacking so that both
    are on a common variance-standardised scale. Without this, the raw counts
    differ in mean and spread across question types (scale confound), making
    the main effect of memory_type uninterpretable and the interaction noisy.
    """
    wide = wide.copy()
    id_cols = ["SubjectID", "StimID", svg_col] + [
        c for c in COVARIATES if c in wide.columns
    ]

    for col in ["n_objects_correct", "n_relational_correct"]:
        mu, sd = wide[col].mean(), wide[col].std()
        wide[f"{col}_z"] = (wide[col] - mu) / sd if sd > 0 else 0.0

    obj = wide[id_cols + ["n_objects_correct_z"]].copy()
    obj = obj.rename(columns={"n_objects_correct_z": "score"})
    obj["memory_type"] = 0

    rel = wide[id_cols + ["n_relational_correct_z"]].copy()
    rel = rel.rename(columns={"n_relational_correct_z": "score"})
    rel["memory_type"] = 1

    long = pd.concat([obj, rel], ignore_index=True)
    long = long.dropna(subset=["score", svg_col])

    mu, sd = long[svg_col].mean(), long[svg_col].std()
    long[f"{svg_col}_z"] = (long[svg_col] - mu) / sd if sd > 0 else 0.0

    return long


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def _fit_interaction(long: pd.DataFrame, svg_col: str) -> dict:
    svg_z = f"{svg_col}_z"
    cov_str = " + ".join(c for c in COVARIATES if c in long.columns)
    formula = (
        f"score ~ {svg_z} * memory_type" f" + {cov_str}" f" + C(SubjectID) + C(StimID)"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = smf.ols(formula, data=long).fit()
        except Exception as e:
            print(f"  [ERROR] Model fit failed: {e}")
            return {}

    def _extract(term):
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
        "svg_main": _extract(svg_z),
        "type_main": _extract("memory_type"),
        "interaction": _extract(f"{svg_z}:memory_type"),
        "n_obs": int(result.nobs),
        "r2": result.rsquared,
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(fit_results: dict) -> None:
    print("\n" + "=" * 72)
    print("RELATIONAL vs OBJECT MEMORY — SVG × memory_type interaction")
    print("Model: score ~ SVG_z * memory_type + covariates + C(SubjectID) + C(StimID)")
    print("       memory_type: 0 = objects, 1 = relational")
    print("=" * 72)
    print(f"  {'SVG predictor':<30} {'term':<24} {'β':>8} {'t':>7} {'p':>8}  sig")
    print("-" * 72)

    for svg_col, svg_label in SVG_PREDICTORS:
        fit = fit_results.get(svg_col, {})
        if not fit:
            print(f"  {svg_label:<30}  [FAILED]")
            continue

        for term_key, term_label in [
            ("svg_main", "SVG main effect"),
            ("type_main", "memory_type main"),
            ("interaction", "SVG × memory_type"),
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
                f"  {label_col:<30} {term_label:<24} {b:>+8.3f} {t:>7.3f} {p:>8.4f}  {sig}"
            )

        print(f"  {'':30} {'n_obs':<24} {fit['n_obs']:>8}  R²={fit['r2']:.3f}")
        print()

    print("=" * 72)
    print("\n  Interpretation of SVG × memory_type:")
    print("  (Both DVs z-scored before stacking — βs are in SD units)")
    print("  Positive β → SVG predicts relational recall MORE than object recall")
    print("  Negative β → SVG predicts object recall MORE than relational recall")
    print("  ns         → no evidence of differential prediction (non-specific effect)")


# ---------------------------------------------------------------------------
# Plot
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

        x_range = np.linspace(long[svg_z].min(), long[svg_z].max(), 100)
        ref_subj = long["SubjectID"].mode()[0]
        ref_stim = long["StimID"].mode()[0]
        cov_means = {c: long[c].mean() for c in COVARIATES if c in long.columns}

        for mem_type, label, colour in [
            (0, "Object memory", COLOURS["objects"]),
            (1, "Relational memory", COLOURS["relational"]),
        ]:
            pred_df = pd.DataFrame(
                {
                    svg_z: x_range,
                    "memory_type": mem_type,
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

            sub = long[long["memory_type"] == mem_type]
            ax.scatter(
                sub[svg_z],
                sub["score"],
                color=colour,
                alpha=0.25,
                s=20,
                linewidths=0,
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
        ax.set_xlabel(f"{svg_label} (z-scored within long table)", fontsize=9)
        ax.set_ylabel("Recall score (z-scored within type)", fontsize=9)
        ax.set_title(svg_label, fontsize=10, fontweight="bold", pad=7)
        ax.legend(fontsize=8.5, framealpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "Relational vs Object memory — encoding SVG specificity test\n"
        "Lines = model-predicted slopes at mean covariates; dots = z-scored trial scores",
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
        description="Relational vs object memory dissociation: encoding SVG specificity."
    )
    parser.add_argument("--features", default=str(_DEFAULT_FEATURES))
    parser.add_argument("--scores", default=str(_DEFAULT_SCORES))
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Relational vs Object memory dissociation test")
    print("=" * 60)

    wide = _load(Path(args.features), Path(args.scores))

    fit_results = {}
    for svg_col, svg_label in SVG_PREDICTORS:
        print(f"\n  Fitting model for {svg_label} ...")
        long = _to_long(wide, svg_col)
        print(
            f"  Long format: {len(long)} rows "
            f"({long['memory_type'].eq(0).sum()} object, "
            f"{long['memory_type'].eq(1).sum()} relational)"
        )
        fit_results[svg_col] = _fit_interaction(long, svg_col)

    _print_summary(fit_results)

    if not args.no_plot:
        _plot(wide, fit_results, Path(args.output) / "relation_object_dissociation.png")


if __name__ == "__main__":
    main()
