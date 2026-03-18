"""
pipeline/module_4/output.py
============================
Step 6: Summarise results and write outputs.

  - _extract_coef_table : normalises coefficients from LMM or OLS result
  - _descriptives       : proportion DV summary stats
  - _forest_plot        : two-panel forest plot (H2 | Exploratory)
  - summarise           : master function
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .constants import (
    DV_OBJECTS,
    DV_RELATIONS,
    DV_TOTAL,
    ENC_BETWEEN_COVARIATES,
    ENC_COVARIATES,
    MODEL_SPECS,
)

logger = logging.getLogger(__name__)

# Nuisance terms to skip in the forest plot
_SKIP_TERMS = (
    {f"{c}_z" for c in ENC_COVARIATES}
    | {f"{c}_z" for c in ENC_BETWEEN_COVARIATES}
    | {"Group Var", "StimID Var", "Intercept"}
)

_PALETTE = {
    "H1_svg_enc": "#636363",
    "H2_total": "#2166ac",
    "H2_relations": "#084594",
    "H2_objects": "#a63603",
    "EXP_dissociation": "#74c476",
}


# ---------------------------------------------------------------------------
# Coefficient table extraction
# ---------------------------------------------------------------------------


def _extract_coef_table(name: str, result) -> pd.DataFrame:
    """
    Build tidy coefficient DataFrame from LMM or OLS result.
    C(SubjectID) and C(StimID) dummy rows are stripped.
    """
    ci = result.conf_int()
    params = result.params
    mask = ~params.index.str.startswith("C(SubjectID)") & ~params.index.str.startswith(
        "C(StimID)"
    )
    return pd.DataFrame(
        {
            "model": name,
            "term": params.index[mask],
            "coef": params.values[mask],
            "std_err": result.bse.values[mask],
            "z_or_t": result.tvalues.values[mask],
            "p": result.pvalues.values[mask],
            "ci_lower": ci.iloc[:, 0].values[mask],
            "ci_upper": ci.iloc[:, 1].values[mask],
        }
    )


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------


def _descriptives(filtered: dict) -> str:
    enc = filtered.get("enc", pd.DataFrame())
    lines = ["\n=== Encoding SVG and proportion DV descriptives ==="]

    if "svg_z_enc" in enc.columns:
        vals = enc["svg_z_enc"].dropna()
        lines.append(
            f"  svg_z_enc:     n={len(vals)}, mean={vals.mean():.3f}, "
            f"sd={vals.std():.3f}, %>0={100*(vals>0).mean():.1f}%"
        )

    for col, label in [
        (DV_TOTAL, "prop_total    "),
        (DV_RELATIONS, "prop_relations"),
        (DV_OBJECTS, "prop_objects  "),
    ]:
        if col in enc.columns:
            vals = enc[col].dropna()
            lines.append(
                f"  {label}: n={len(vals)}, mean={vals.mean():.3f}, "
                f"sd={vals.std():.3f}, range=[{vals.min():.3f}, {vals.max():.3f}]"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Partial regression plot
# ---------------------------------------------------------------------------

_DV_SPECS = [
    (DV_TOTAL, "Total recall\n(all correct nodes)", "#2166ac"),
    (DV_RELATIONS, "Relational recall\n(action + spatial)", "#084594"),
    (DV_OBJECTS, "Object recall\n(identity + attribute)", "#a63603"),
]


def _residualise(df: pd.DataFrame, target: str) -> pd.Series:
    """
    Return residuals of `target` after regressing on
    ENC_COVARIATES + ENC_BETWEEN_COVARIATES + C(SubjectID).
    Rows with any NaN in required columns are set to NaN in output.
    """
    import warnings

    all_covs = ENC_COVARIATES + ENC_BETWEEN_COVARIATES
    req = [target] + [c for c in all_covs if c in df.columns]
    mask = df[req].notna().all(axis=1)
    out = pd.Series(np.nan, index=df.index)
    sub = df[mask].copy()
    if len(sub) < 10:
        return out
    cov_terms = " + ".join(f"{c}_z" for c in all_covs if f"{c}_z" in sub.columns)
    formula = f"{target} ~ {cov_terms} + C(SubjectID)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = smf.ols(formula, data=sub).fit()
            out.loc[mask] = res.resid.values
        except Exception as e:
            logger.warning(f"  residualisation of {target} failed: {e}")
    return out


def _partial_regression_plot(enc: pd.DataFrame, output_path: Path) -> None:
    """
    Three-panel partial regression plot: encoding SVG (within-image) → each DV.
    X axis: svg_z_enc_within residualised against ENC_COVARIATES +
            ENC_BETWEEN_COVARIATES + C(SubjectID).
    Y axis: each proportion DV residualised against the same set.
    This matches exactly what the LMM H2 models estimate.
    """
    svg_col = "svg_z_enc_within"
    if svg_col not in enc.columns:
        logger.warning(
            "  _partial_regression_plot: svg_z_enc_within missing — skipping."
        )
        return

    # Ensure z-scored covariate columns exist
    all_covs = ENC_COVARIATES + ENC_BETWEEN_COVARIATES
    for c in all_covs:
        if f"{c}_z" not in enc.columns and c in enc.columns:
            mu, sd = enc[c].mean(), enc[c].std()
            enc = enc.copy()
            enc[f"{c}_z"] = (enc[c] - mu) / sd if sd > 0 else 0.0

    svg_resid = _residualise(enc, svg_col)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

    for ax, (dv_col, dv_label, colour) in zip(axes, _DV_SPECS):
        if dv_col not in enc.columns:
            ax.set_visible(False)
            continue

        dv_resid = _residualise(enc, dv_col)
        x = svg_resid.values
        y = dv_resid.values
        mask = ~(np.isnan(x) | np.isnan(y))

        ax.scatter(
            x[mask], y[mask], color=colour, alpha=0.35, s=22, linewidths=0, zorder=3
        )

        if mask.sum() >= 5:
            slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
            x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(
                x_line,
                intercept + slope * x_line,
                color=colour,
                linewidth=2.0,
                zorder=4,
            )
            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            ax.annotate(
                f"r = {r:+.3f},  p = {p:.3f}  {sig}",
                xy=(0.05, 0.93),
                xycoords="axes fraction",
                fontsize=8.5,
                color=colour,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8),
            )

        ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.axvline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_xlabel(
            "Encoding SVG (within-image)\n(covariate-residualised)", fontsize=9
        )
        ax.set_ylabel(f"{dv_label}\n(covariate-residualised)", fontsize=9)
        ax.set_title(dv_label.replace("\n", " "), fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Partial regression: Encoding SVG (within-image) → memory recall (proportion DVs)\n"
        "Covariates removed: n_fixations, aoi_prop, mean_salience_relational,"
        " svg_z_enc_image_mean, SubjectID",
        fontsize=9,
        y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")


# ---------------------------------------------------------------------------
# Prediction helpers for model-predicted figures
# ---------------------------------------------------------------------------


def _fe_params_and_cov(result):
    """
    Return (fe_params Series, fe_cov DataFrame) for fixed effects only,
    stripping out variance component rows from LMM or returning OLS equivalents.
    """
    params = result.params
    # Exclude variance component rows (contain 'Var' or 'scale')
    mask = ~params.index.str.contains(r"Var|scale", regex=True, na=False)
    fe_names = params.index[mask]
    fe_params = params[fe_names]

    try:
        full_cov = result.cov_params()
        available = [n for n in fe_names if n in full_cov.index]
        fe_cov = full_cov.loc[available, available]
        fe_params = fe_params[available]
    except Exception:
        fe_cov = pd.DataFrame(
            np.diag(result.bse[fe_names].values ** 2),
            index=fe_names,
            columns=fe_names,
        )
    return fe_params, fe_cov


def _marginal_predict(result, design_fn, x_grid: np.ndarray) -> tuple:
    """
    Marginal fixed-effects predictions over x_grid.

    design_fn(x) → dict {param_name: coefficient_value} for one x value.
    All params not returned by design_fn default to 0.

    Returns (preds, ci_lo, ci_hi) as numpy arrays.
    """
    fe_params, fe_cov = _fe_params_and_cov(result)

    preds, ci_lo, ci_hi = [], [], []
    for x in x_grid:
        design = design_fn(x)
        row = np.array([design.get(n, 0.0) for n in fe_params.index])
        pred = float(row @ fe_params.values)
        var  = float(row @ fe_cov.values @ row)
        se   = np.sqrt(max(var, 0.0))
        preds.append(pred)
        ci_lo.append(pred - 1.96 * se)
        ci_hi.append(pred + 1.96 * se)

    return np.array(preds), np.array(ci_lo), np.array(ci_hi)


# ---------------------------------------------------------------------------
# Figure 1 — Vertical violin of encoding SVG
# ---------------------------------------------------------------------------


def _figure1_violin(enc: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Vertical violin plot of trial-level svg_z_enc values.
    Single 'Encoding' category on x-axis — future-proof for adding Decoding.
    Returns the figure data DataFrame.
    """
    svg_col = "svg_z_enc"
    if svg_col not in enc.columns:
        logger.warning("  _figure1_violin: svg_z_enc missing — skipping.")
        return pd.DataFrame()

    vals = enc[[svg_col, "SubjectID", "StimID"]].dropna(subset=[svg_col]).copy()
    vals["phase"] = "Encoding"

    grand_mean = vals[svg_col].mean()
    pct_above  = 100 * (vals[svg_col] > 0).mean()

    fig, ax = plt.subplots(figsize=(4.5, 6))

    # Violin
    vp = ax.violinplot(
        vals[svg_col].values,
        positions=[0],
        widths=0.55,
        showmedians=False,
        showextrema=False,
    )
    for body in vp["bodies"]:
        body.set_facecolor("#2166ac")
        body.set_alpha(0.35)
        body.set_edgecolor("#2166ac")
        body.set_linewidth(1.2)

    # Jittered strip
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.07, 0.07, len(vals))
    ax.scatter(
        jitter, vals[svg_col].values,
        color="#2166ac", alpha=0.25, s=12, linewidths=0, zorder=3,
    )

    # IQR box
    q25, q75 = np.percentile(vals[svg_col], [25, 75])
    median    = np.median(vals[svg_col])
    ax.vlines(0, q25, q75, color="#2166ac", linewidth=5, alpha=0.7, zorder=4)
    ax.scatter([0], [median], color="white", s=30, zorder=5)

    # Reference lines
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--",
               alpha=0.6, label="Permutation chance (0)", zorder=2)
    ax.axhline(grand_mean, color="#2166ac", linewidth=1.4, linestyle="-",
               alpha=0.85, label=f"Grand mean = {grand_mean:.2f} SD", zorder=2)

    # Annotations
    ax.annotate(
        f"Mean = {grand_mean:.2f} SD\n{pct_above:.0f}% > chance",
        xy=(0.97, 0.97), xycoords="axes fraction",
        fontsize=9, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9),
    )

    ax.set_xticks([0])
    ax.set_xticklabels(["Encoding"], fontsize=10)
    ax.set_ylabel("Relational scanpath strength\n(z-score above permutation baseline)", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.85, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")

    return vals[["SubjectID", "StimID", svg_col, "phase"]].copy()


# ---------------------------------------------------------------------------
# Figure 3 — Model-predicted total recall
# ---------------------------------------------------------------------------


def _figure3_predicted_total(
    enc: pd.DataFrame,
    results: dict,
    output_path: Path,
) -> pd.DataFrame:
    """
    Model-predicted total recall proportion as a function of within-image SVG.
    Fixed effects only; all covariates at their standardised mean (0).
    Returns the figure data DataFrame.
    """
    entry = results.get("H2_total")
    if entry is None or entry[0] is None:
        logger.warning("  _figure3_predicted_total: H2_total result missing — skipping.")
        return pd.DataFrame()

    result, _ = entry
    svg_col = "svg_z_enc_within_z"

    if svg_col not in enc.columns:
        logger.warning("  _figure3_predicted_total: svg_z_enc_within_z missing — skipping.")
        return pd.DataFrame()

    vals = enc[svg_col].dropna().values
    x_grid = np.linspace(np.percentile(vals, 2.5), np.percentile(vals, 97.5), 120)

    def design(x):
        return {"Intercept": 1.0, svg_col: x}

    preds, ci_lo, ci_hi = _marginal_predict(result, design, x_grid)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ax.fill_between(x_grid, ci_lo, ci_hi, color="#2166ac", alpha=0.18, zorder=2)
    ax.plot(x_grid, preds, color="#2166ac", linewidth=2.2, zorder=3)

    ax.axvline(0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel(
        "Encoding relational scanpath strength within image (z)",
        fontsize=9,
    )
    ax.set_ylabel("Predicted total recall proportion", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")

    return pd.DataFrame({
        "svg_z_enc_within_z": x_grid,
        "pred_total":          preds,
        "ci_lower":            ci_lo,
        "ci_upper":            ci_hi,
    })


# ---------------------------------------------------------------------------
# Figure 4 — Model-predicted object vs relational recall
# ---------------------------------------------------------------------------


def _figure4_content_comparison(
    enc: pd.DataFrame,
    results: dict,
    output_path: Path,
) -> pd.DataFrame:
    """
    Two parallel model-predicted lines from the dissociation model:
    object recall (rust) and relational recall (blue), both vs within-image SVG.
    Shows equal slopes and baseline gap.
    Returns the figure data DataFrame.
    """
    entry = results.get("EXP_dissociation")
    if entry is None or entry[0] is None:
        logger.warning("  _figure4_content_comparison: EXP_dissociation result missing — skipping.")
        return pd.DataFrame()

    result, _ = entry
    svg_col = "svg_z_enc_within_z"

    if svg_col not in enc.columns:
        logger.warning("  _figure4_content_comparison: svg_z_enc_within_z missing — skipping.")
        return pd.DataFrame()

    vals = enc[svg_col].dropna().values
    x_grid = np.linspace(np.percentile(vals, 2.5), np.percentile(vals, 97.5), 120)

    # Objects: memory_type = 0
    def design_obj(x):
        return {"Intercept": 1.0, svg_col: x, "memory_type": 0.0,
                f"{svg_col}:memory_type": 0.0}

    # Relations: memory_type = 1
    def design_rel(x):
        return {"Intercept": 1.0, svg_col: x, "memory_type": 1.0,
                f"{svg_col}:memory_type": x}

    preds_obj, ci_lo_obj, ci_hi_obj = _marginal_predict(result, design_obj, x_grid)
    preds_rel, ci_lo_rel, ci_hi_rel = _marginal_predict(result, design_rel, x_grid)

    colour_obj = "#a63603"
    colour_rel = "#2166ac"

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Objects
    ax.fill_between(x_grid, ci_lo_obj, ci_hi_obj, color=colour_obj, alpha=0.18, zorder=2)
    ax.plot(x_grid, preds_obj, color=colour_obj, linewidth=2.2, zorder=3)

    # Relations
    ax.fill_between(x_grid, ci_lo_rel, ci_hi_rel, color=colour_rel, alpha=0.18, zorder=2)
    ax.plot(x_grid, preds_rel, color=colour_rel, linewidth=2.2, zorder=3)

    # Direct line labels at right edge
    x_label = x_grid[-1] + 0.05
    ax.text(x_label, float(preds_obj[-1]), "Object recall",
            color=colour_obj, fontsize=8.5, va="center")
    ax.text(x_label, float(preds_rel[-1]), "Relational recall",
            color=colour_rel, fontsize=8.5, va="center")

    ax.axvline(0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel(
        "Encoding relational scanpath strength within image (z)",
        fontsize=9,
    )
    ax.set_ylabel("Predicted recall proportion", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    # Extra right margin for labels
    xlo, xhi = ax.get_xlim()
    ax.set_xlim(xlo, xhi + (xhi - xlo) * 0.22)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")

    return pd.DataFrame({
        "svg_z_enc_within_z":  x_grid,
        "pred_objects":         preds_obj,
        "ci_lower_objects":     ci_lo_obj,
        "ci_upper_objects":     ci_hi_obj,
        "pred_relations":       preds_rel,
        "ci_lower_relations":   ci_lo_rel,
        "ci_upper_relations":   ci_hi_rel,
    })


# ---------------------------------------------------------------------------
# Supplementary — per-image mean relationality
# ---------------------------------------------------------------------------


def _supp_per_image(enc: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Sorted dot plot of per-image mean encoding SVG (svg_z_enc_image_mean).
    Shows how much each image structurally pulls relational scanning.
    Returns the figure data DataFrame.
    """
    svg_col = "svg_z_enc"
    if svg_col not in enc.columns or "StimID" not in enc.columns:
        logger.warning("  _supp_per_image: missing columns — skipping.")
        return pd.DataFrame()

    img_stats = (
        enc.groupby("StimID")[svg_col]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
        .sort_values("mean")
        .reset_index(drop=True)
    )
    img_stats["se"] = img_stats["sd"] / np.sqrt(img_stats["n"])
    tc = stats.t.ppf(0.975, df=img_stats["n"] - 1)
    img_stats["ci_lo"] = img_stats["mean"] - tc * img_stats["se"]
    img_stats["ci_hi"] = img_stats["mean"] + tc * img_stats["se"]

    n_img = len(img_stats)
    fig, ax = plt.subplots(figsize=(5.5, max(4.5, n_img * 0.32)))

    for i, row in img_stats.iterrows():
        c = "#2166ac" if row["mean"] > 0 else "#a63603"
        ax.errorbar(
            row["mean"], i,
            xerr=[[row["mean"] - row["ci_lo"]], [row["ci_hi"] - row["mean"]]],
            fmt="o", color=c, markersize=5, linewidth=1.2, capsize=2.5, alpha=0.8,
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    grand = img_stats["mean"].mean()
    ax.axvline(grand, color="#636363", linewidth=1.0, linestyle="-", alpha=0.7)
    ax.set_yticks(range(n_img))
    ax.set_yticklabels(img_stats["StimID"].values, fontsize=7)
    ax.set_xlabel("Mean encoding SVG (z-score, ± 95% CI)", fontsize=9)
    ax.set_title("Per-image mean relational scanpath strength\n(sorted ascending)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")

    return img_stats[["StimID", "mean", "sd", "n", "ci_lo", "ci_hi"]].rename(
        columns={"mean": "svg_z_enc_image_mean", "sd": "svg_z_enc_image_sd",
                 "ci_lo": "ci_lower", "ci_hi": "ci_upper"}
    )


def _save_partial_regression_data(enc: pd.DataFrame, output_path: Path) -> None:
    """Save residualised data used by the partial regression supplementary figure."""
    svg_col = "svg_z_enc_within"
    rows = {}
    targets = [svg_col, DV_TOTAL, DV_RELATIONS, DV_OBJECTS]
    for t in targets:
        if t in enc.columns:
            rows[t + "_resid"] = _residualise(enc, t).values
    if rows:
        df_out = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_path, index=False)
        logger.info("  Written → figure_data/supp_partial_regression_data.csv")


# ---------------------------------------------------------------------------
# Master summarise function
# ---------------------------------------------------------------------------


def summarise(
    results: dict,
    filtered: dict,
    output_dir: Path,
    plot: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Write all Module 4 outputs to output_dir.

    Directory layout
    ----------------
    output_dir/
      model_summaries.txt          — full model output text
      figures/                     — main paper figures
      supplementary/               — supplementary figures
      figure_data/                 — CSVs to reproduce each figure

    Returns the concatenated coefficient DataFrame.
    """
    logger.info("\nStep 6: Writing outputs ...")

    figures_dir     = output_dir / "figures"
    supp_dir        = output_dir / "supplementary"
    fig_data_dir    = output_dir / "figure_data"

    for d in (output_dir, figures_dir, supp_dir, fig_data_dir):
        d.mkdir(parents=True, exist_ok=True)

    enc = filtered.get("enc", pd.DataFrame())

    # ── Model summaries text ────────────────────────────────────────────────
    all_coefs      = []
    summary_lines  = [_descriptives(filtered)]

    for group in ["H1", "H2", "Exploratory"]:
        summary_lines.append(f"\n\n{'#'*60}\n# {group} MODELS\n{'#'*60}")

        for name, _, desc, _, grp in MODEL_SPECS:
            if grp != group:
                continue

            entry = results.get(name)
            result, mode = entry if entry else (None, "skipped")

            summary_lines.append(f"\n{'='*60}\n{name}: {desc}\n{'='*60}")
            if result is None:
                summary_lines.append(f"SKIPPED / FAILED (mode={mode})\n")
                continue

            summary_lines.append(f"Mode: {mode}")
            coef_df = _extract_coef_table(name, result)
            all_coefs.append(coef_df)
            summary_lines.append(str(result.summary()))

            if hasattr(result, "rsquared"):
                summary_lines.append(f"R²: {result.rsquared:.4f}")
            if hasattr(result, "converged"):
                summary_lines.append(f"Converged: {result.converged}")

            focus_terms = (
                ["Intercept"]
                if group == "H1"
                else [
                    t
                    for t in coef_df["term"].str.strip()
                    if (
                        t.endswith("_z")
                        and t not in {f"{c}_z" for c in ENC_COVARIATES}
                        and t not in {f"{c}_z" for c in ENC_BETWEEN_COVARIATES}
                    )
                    or t == "memory_type"
                    or "memory_type" in t
                ]
            )
            for term in focus_terms:
                row = coef_df[coef_df["term"].str.strip() == term]
                if row.empty:
                    continue
                b, p   = row["coef"].values[0], row["p"].values[0]
                lo, hi = row["ci_lower"].values[0], row["ci_upper"].values[0]
                sig = ("***" if p < 0.001 else "**" if p < 0.01
                       else "*" if p < 0.05 else "ns")
                logger.info(
                    f"    {term}: β={b:.4f} [{lo:.4f}, {hi:.4f}], p={p:.4f} {sig}"
                )

    coef_all = pd.concat(all_coefs, ignore_index=True) if all_coefs else pd.DataFrame()

    with open(output_dir / "model_summaries.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    logger.info("  Written → model_summaries.txt")

    if not plot or enc.empty:
        return coef_all

    # ── Main figures ────────────────────────────────────────────────────────

    # Figure 1: violin
    fig1_data = _figure1_violin(enc, figures_dir / "figure1_violin.png")
    if not fig1_data.empty:
        fig1_data.to_csv(fig_data_dir / "figure1_data.csv", index=False)
        logger.info("  Written → figure_data/figure1_data.csv")

    # Figure 3: predicted total recall
    fig3_data = _figure3_predicted_total(enc, results, figures_dir / "figure3_total_recall.png")
    if not fig3_data.empty:
        fig3_data.to_csv(fig_data_dir / "figure3_data.csv", index=False)
        logger.info("  Written → figure_data/figure3_data.csv")

    # Figure 4: object vs relational recall
    fig4_data = _figure4_content_comparison(enc, results, figures_dir / "figure4_content_comparison.png")
    if not fig4_data.empty:
        fig4_data.to_csv(fig_data_dir / "figure4_data.csv", index=False)
        logger.info("  Written → figure_data/figure4_data.csv")

    # ── Supplementary figures ────────────────────────────────────────────────

    # Partial regression
    _partial_regression_plot(enc, supp_dir / "supp_partial_regression.png")
    # Save residualised data for partial regression
    _save_partial_regression_data(enc, fig_data_dir / "supp_partial_regression_data.csv")

    # Per-image mean relationality
    img_data = _supp_per_image(enc, supp_dir / "supp_per_image_relationality.png")
    if not img_data.empty:
        img_data.to_csv(fig_data_dir / "supp_per_image_data.csv", index=False)
        logger.info("  Written → figure_data/supp_per_image_data.csv")

    return coef_all