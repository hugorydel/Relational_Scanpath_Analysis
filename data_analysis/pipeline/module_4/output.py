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
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .constants import (
    DEC_BETWEEN_COVARIATES,
    DEC_COVARIATES,
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
    Build tidy coefficient DataFrame from LMM, OLS, or t-test result.
    C(SubjectID) and C(StimID) dummy rows are stripped.
    """
    # t-test result: params is a plain dict
    if isinstance(result.params, dict):
        terms = list(result.params.keys())
        ci_df = result.conf_int()
        return pd.DataFrame(
            {
                "model": name,
                "term": terms,
                "coef": [result.params[t] for t in terms],
                "std_err": [result.bse[t] for t in terms],
                "z_or_t": [result.tvalues[t] for t in terms],
                "p": [result.pvalues[t] for t in terms],
                "ci_lower": ci_df["0"].values,
                "ci_upper": ci_df["1"].values,
            }
        )

    # statsmodels result: params is a Series
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


def compute_descriptives(filtered: dict, output_dir: Path) -> tuple:
    """
    Compute participant-level descriptive statistics for all key variables.

    Aggregation order:
      1. Average trial-level values to one mean per participant (removes
         within-participant dependence and matches the unit of inference).
      2. Across those N participant means compute grand M, SD, and 95% CI.

    Recall proportion DVs are multiplied by 100 for % reporting.

    Writes descriptives.csv to output_dir.

    Returns
    -------
    (log_text : str, desc_df : pd.DataFrame)
    """
    enc = filtered.get("enc", pd.DataFrame())
    dec = filtered.get("dec", pd.DataFrame())
    lines = ["\n=== Descriptive statistics (participant-level means) ==="]
    rows = []

    def _agg(df, col, scale=1.0):
        """Aggregate col to participant means, return stats dict or None."""
        if df.empty or col not in df.columns or "SubjectID" not in df.columns:
            return None
        subj_means = df.groupby("SubjectID")[col].mean().dropna() * scale
        n = len(subj_means)
        if n < 2:
            return None
        m = subj_means.mean()
        sd = subj_means.std(ddof=1)
        se = subj_means.sem()
        ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
        return {
            "n": n,
            "M": m,
            "SD": sd,
            "SE": se,
            "CI_lower": ci_lo,
            "CI_upper": ci_hi,
        }

    _VARS = [
        ("svg_z_enc", enc, 1.0, "Encoding SVG (z)"),
        ("svg_z_dec", dec, 1.0, "Decoding SVG (z)"),
        (DV_TOTAL, enc, 100.0, "Total recall (%)"),
        (DV_RELATIONS, enc, 100.0, "Relational recall (%)"),
        (DV_OBJECTS, enc, 100.0, "Object recall (%)"),
    ]

    for col, df, scale, label in _VARS:
        s = _agg(df, col, scale)
        if s is None:
            lines.append(f"  {label:<25}  — data unavailable")
            continue
        lines.append(
            f"  {label:<25}  N={s['n']:2d}  "
            f"M={s['M']:7.3f}  SD={s['SD']:6.3f}  "
            f"95% CI [{s['CI_lower']:7.3f}, {s['CI_upper']:7.3f}]"
        )
        rows.append({"variable": label, **s})

    desc_df = pd.DataFrame(rows)
    if not desc_df.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "descriptives.csv"
        desc_df.to_csv(out_path, index=False)
        logger.info("  Written → descriptives.csv")

    return "\n".join(lines), desc_df


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

    all_cov_names = (
        [c.replace("_enc", "").replace("_z", "") for c in ENC_COVARIATES]
        + [c.replace("_z", "") for c in ENC_BETWEEN_COVARIATES]
        + ["SubjectID"]
    )
    fig.suptitle(
        "Partial regression: Encoding SVG (within-image) → memory recall (proportion DVs)\n"
        f"Covariates removed: {', '.join(all_cov_names)}",
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
        var = float(row @ fe_cov.values @ row)
        se = np.sqrt(max(var, 0.0))
        preds.append(pred)
        ci_lo.append(pred - 1.96 * se)
        ci_hi.append(pred + 1.96 * se)

    return np.array(preds), np.array(ci_lo), np.array(ci_hi)


# ---------------------------------------------------------------------------
# Assumption checks for LMMs (H2 + Exploratory models)
# ---------------------------------------------------------------------------

# Predictor columns used in VIF computation per table type
_VIF_COLS_ENC = [
    "svg_z_enc_within_z",
    "n_fixations_enc_z",
    "mean_salience_relational_enc_z",
    "svg_z_enc_image_mean_z",
]
_VIF_COLS_DEC = [
    "svg_z_dec_within_z",
    "n_fixations_dec_z",
    "mean_salience_relational_dec_z",
    "svg_z_dec_image_mean_z",
]
_VIF_COLS_LONG = _VIF_COLS_ENC  # same predictors as enc


def _compute_vif(df: pd.DataFrame, predictor_cols: list) -> dict:
    """
    Compute Variance Inflation Factor for each predictor.
    Adds a constant column internally (required by statsmodels VIF).
    Returns {col: vif_value}. Any column absent from df gets NaN.
    """
    available = [c for c in predictor_cols if c in df.columns]
    sub = df[available].dropna()
    if sub.empty or len(available) < 2:
        return {c: np.nan for c in predictor_cols}

    sub = sub.assign(_const=1.0)
    X = sub[available + ["_const"]].values
    vif = {}
    for i, col in enumerate(available):
        try:
            vif[col] = float(variance_inflation_factor(X, i))
        except Exception:
            vif[col] = np.nan
    for col in predictor_cols:
        if col not in vif:
            vif[col] = np.nan
    return vif


def _check_lmm_assumptions(
    name: str,
    result,
    df: pd.DataFrame,
    supp_dir: Path,
    vif_cols: list,
) -> dict:
    """
    Run assumption checks for one fitted LMM and save diagnostic plots.

    Checks
    ------
    1. Normality of residuals  — Q-Q plot (visual assessment; Shapiro-Wilk
                                 is not used at this N as it is hypersensitive
                                 to trivial deviations above ~100 observations)
    2. Homoscedasticity        — residuals-vs-fitted plot
    3. Multicollinearity       — VIF per predictor

    Linearity is assessed visually via the partial regression plots
    already generated as supplementary figures.

    Returns a dict of check results for assembly into assumption_checks.csv.
    Plots are saved to supp_dir/assumption_checks/.
    """
    checks_dir = supp_dir / "assumption_checks"
    checks_dir.mkdir(parents=True, exist_ok=True)

    row = {"model": name}

    # ── Residuals ─────────────────────────────────────────────────────────
    try:
        resid = result.resid
        fitted = result.fittedvalues
    except AttributeError:
        logger.warning(
            f"  {name}: no residuals available — skipping assumption checks."
        )
        row["max_vif"] = np.nan
        return row

    resid_arr = np.asarray(resid.dropna())

    # ── 1. Q-Q plot (normality assessed visually) ─────────────────────────
    # Shapiro-Wilk is not used for LMM residuals: at N > ~100 it becomes
    # hypersensitive to trivially small deviations and p-values are
    # uninterpretable. Q-Q plots are the standard approach at this scale.
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    (osm, osr), (slope, intercept, r) = stats.probplot(resid_arr, dist="norm")
    ax.scatter(osm, osr, color="#2166ac", alpha=0.5, s=18, linewidths=0)
    ax.plot(
        [osm[0], osm[-1]],
        [slope * osm[0] + intercept, slope * osm[-1] + intercept],
        color="#b2182b",
        linewidth=1.5,
    )
    ax.set_xlabel("Theoretical quantiles", fontsize=9)
    ax.set_ylabel("Sample quantiles", fontsize=9)
    ax.set_title(
        f"Q-Q plot of residuals: {name}\n(normality assessed visually)", fontsize=8.5
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    qq_path = checks_dir / f"{name}_qq.png"
    plt.savefig(qq_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. Residuals-vs-fitted plot ───────────────────────────────────────
    fitted_arr = np.asarray(fitted.loc[resid.index].dropna())
    min_len = min(len(resid_arr), len(fitted_arr))
    resid_arr_aligned = resid_arr[:min_len]
    fitted_arr_aligned = fitted_arr[:min_len]

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.scatter(
        fitted_arr_aligned,
        resid_arr_aligned,
        color="#2166ac",
        alpha=0.35,
        s=14,
        linewidths=0,
    )
    ax.axhline(0, color="#b2182b", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Fitted values", fontsize=9)
    ax.set_ylabel("Residuals", fontsize=9)
    ax.set_title(f"Residuals vs Fitted: {name}", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    rvf_path = checks_dir / f"{name}_resid_vs_fitted.png"
    plt.savefig(rvf_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4. VIF ────────────────────────────────────────────────────────────
    vif_dict = _compute_vif(df, vif_cols)
    for col, val in vif_dict.items():
        row[f"vif_{col}"] = round(val, 3) if not np.isnan(val) else np.nan
    valid_vifs = [v for v in vif_dict.values() if not np.isnan(v)]
    row["max_vif"] = round(max(valid_vifs), 3) if valid_vifs else np.nan

    vif_flag = row["max_vif"] > 5 if not np.isnan(row["max_vif"]) else False
    logger.info(
        f"  {name} assumption checks: "
        f"Q-Q plot saved, "
        f"max VIF={row['max_vif']} ({'HIGH — check multicollinearity' if vif_flag else 'OK'})"
    )

    return row


# ---------------------------------------------------------------------------
# Appendix parameter tables
# ---------------------------------------------------------------------------


def _write_parameter_tables(
    results: dict,
    output_dir: Path,
) -> None:
    """
    Write full fixed-effects parameter tables for appendix reporting.

    Appendix Table 2 — H2 models (encoding + decoding)
    Appendix Table 3 — Exploratory dissociation model

    Coefficients are rounded to 4dp so near-zero values are
    visible rather than appearing as .000.
    """
    h2_names = [
        "H2_total",
        "H2_relations",
        "H2_objects",
        "H2_dec_total",
        "H2_dec_relations",
        "H2_dec_objects",
    ]
    exp_names = ["EXP_dissociation"]

    def _extract(names):
        rows = []
        for name in names:
            entry = results.get(name)
            if entry is None or entry[0] is None:
                continue
            result, mode = entry
            if mode not in ("lmm", "ols"):
                continue
            coef_df = _extract_coef_table(name, result)
            # Filter out random effects variance rows
            coef_df = coef_df[
                ~coef_df["term"].str.contains(
                    r"Var|scale|C\(SubjectID\)|C\(StimID\)", regex=True, na=False
                )
            ].copy()
            # Round to 4dp for display
            for col in ["coef", "std_err", "ci_lower", "ci_upper"]:
                coef_df[col] = coef_df[col].round(4)
            coef_df["p"] = coef_df["p"].round(4)
            # Convergence note
            optimizer = getattr(result, "optimizer_used", "unknown")
            n_obs = getattr(result, "n_obs", "?")
            n_dropped = getattr(result, "n_dropped", "?")
            coef_df["optimizer"] = optimizer
            coef_df["n_obs"] = n_obs
            coef_df["n_dropped"] = n_dropped
            rows.append(coef_df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    app2 = _extract(h2_names)
    app3 = _extract(exp_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not app2.empty:
        path = output_dir / "appendix_table2_h2_params.csv"
        app2.to_csv(path, index=False)
        logger.info("  Written → appendix_table2_h2_params.csv")
    if not app3.empty:
        path = output_dir / "appendix_table3_exploratory_params.csv"
        app3.to_csv(path, index=False)
        logger.info("  Written → appendix_table3_exploratory_params.csv")


# ---------------------------------------------------------------------------
# APA shared style helper
# ---------------------------------------------------------------------------


def _apply_apa_style(ax, font="Arial", fontsize=10):
    """Apply APA-7 axis style: left/bottom spines only, Arial font, no gridlines."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=fontsize, direction="out")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(font)
        label.set_fontsize(fontsize)
    ax.xaxis.label.set_fontfamily(font)
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontfamily(font)
    ax.yaxis.label.set_fontsize(fontsize)
    ax.grid(False)


# ---------------------------------------------------------------------------
# Figure 1 — Box and whisker: encoding and decoding SVG
# ---------------------------------------------------------------------------


def _figure1_boxplot(
    enc: pd.DataFrame,
    output_path: Path,
    dec: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    APA-style side-by-side boxplots of trial-level SVG z-scores.
    Encoding on left, Decoding on right.
    Permutation baseline (y = 0) shown as dashed line.
    Returns figure data DataFrame.
    """
    svg_enc = "svg_z_enc"
    if svg_enc not in enc.columns:
        logger.warning("  _figure1_boxplot: svg_z_enc missing — skipping.")
        return pd.DataFrame()

    enc_vals = enc[svg_enc].dropna().values
    has_dec = dec is not None and "svg_z_dec" in dec.columns
    dec_vals = dec["svg_z_dec"].dropna().values if has_dec else np.array([])

    datasets = [enc_vals] + ([dec_vals] if has_dec else [])
    labels = ["Encoding"] + (["Decoding"] if has_dec else [])
    positions = list(range(len(datasets)))

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    bp = ax.boxplot(
        datasets,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="white", color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        flierprops=dict(
            marker="o",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=3,
            linewidth=0.5,
        ),
        showfliers=True,
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relational SVG Score")
    _apply_apa_style(ax)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")

    rows = [
        {"phase": lbl, "svg_z": v} for lbl, arr in zip(labels, datasets) for v in arr
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 2 — Model-predicted total recall
# ---------------------------------------------------------------------------


def _figure2_predicted_total(
    enc: pd.DataFrame,
    results: dict,
    output_path: Path,
) -> pd.DataFrame:
    """
    APA-style model-predicted total recall (%) as a function of within-image
    encoding SVG. Greyscale line + shaded 95% CI band.
    Returns figure data DataFrame.
    """
    entry = results.get("H2_total")
    if entry is None or entry[0] is None:
        logger.warning(
            "  _figure2_predicted_total: H2_total result missing — skipping."
        )
        return pd.DataFrame()

    result, _ = entry
    svg_col = "svg_z_enc_within_z"
    if svg_col not in enc.columns:
        logger.warning(
            "  _figure2_predicted_total: svg_z_enc_within_z missing — skipping."
        )
        return pd.DataFrame()

    vals = enc[svg_col].dropna().values
    x_grid = np.linspace(np.percentile(vals, 2.5), np.percentile(vals, 97.5), 120)

    def design(x):
        return {"Intercept": 1.0, svg_col: x}

    preds, ci_lo, ci_hi = _marginal_predict(result, design, x_grid)

    # Convert proportions → percentages
    preds_pct = preds * 100
    ci_lo_pct = ci_lo * 100
    ci_hi_pct = ci_hi * 100

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.fill_between(x_grid, ci_lo_pct, ci_hi_pct, color="#BBBBBB", alpha=0.6, zorder=2)
    ax.plot(x_grid, preds_pct, color="#333333", linewidth=1.8, zorder=3)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5, zorder=1)
    ax.set_xlabel("Relational SVG Score at Encoding")
    ax.set_ylabel("Predicted Total Recall (%)")
    _apply_apa_style(ax)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")

    return pd.DataFrame(
        {
            "svg_z_enc_within_z": x_grid,
            "pred_total_pct": preds_pct,
            "ci_lower_pct": ci_lo_pct,
            "ci_upper_pct": ci_hi_pct,
        }
    )


# ---------------------------------------------------------------------------
# Figure 3 — Model-predicted relational vs object recall
# ---------------------------------------------------------------------------


def _figure3_content_comparison(
    enc: pd.DataFrame,
    results: dict,
    output_path: Path,
) -> pd.DataFrame:
    """
    APA-style two-line plot: model-predicted relational recall (dark grey)
    and object recall (light grey) as a function of within-image encoding SVG.
    Y-axis in percentages. Returns figure data DataFrame.
    """
    entry = results.get("EXP_dissociation")
    if entry is None or entry[0] is None:
        logger.warning(
            "  _figure3_content_comparison: EXP_dissociation result missing — skipping."
        )
        return pd.DataFrame()

    result, _ = entry
    svg_col = "svg_z_enc_within_z"
    if svg_col not in enc.columns:
        logger.warning(
            "  _figure3_content_comparison: svg_z_enc_within_z missing — skipping."
        )
        return pd.DataFrame()

    vals = enc[svg_col].dropna().values
    x_grid = np.linspace(np.percentile(vals, 2.5), np.percentile(vals, 97.5), 120)

    def design_rel(x):
        return {
            "Intercept": 1.0,
            svg_col: x,
            "memory_type": 1.0,
            f"{svg_col}:memory_type": x,
        }

    def design_obj(x):
        return {
            "Intercept": 1.0,
            svg_col: x,
            "memory_type": 0.0,
            f"{svg_col}:memory_type": 0.0,
        }

    preds_rel, ci_lo_rel, ci_hi_rel = _marginal_predict(result, design_rel, x_grid)
    preds_obj, ci_lo_obj, ci_hi_obj = _marginal_predict(result, design_obj, x_grid)

    # Convert to percentages
    preds_rel *= 100
    ci_lo_rel *= 100
    ci_hi_rel *= 100
    preds_obj *= 100
    ci_lo_obj *= 100
    ci_hi_obj *= 100

    dark = "#333333"  # relational recall
    light = "#999999"  # object recall

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Relational — dark grey
    ax.fill_between(x_grid, ci_lo_rel, ci_hi_rel, color=dark, alpha=0.15, zorder=2)
    ax.plot(
        x_grid,
        preds_rel,
        color=dark,
        linewidth=1.8,
        label="Relational recall",
        zorder=3,
    )

    # Object — light grey
    ax.fill_between(x_grid, ci_lo_obj, ci_hi_obj, color=light, alpha=0.25, zorder=2)
    ax.plot(
        x_grid, preds_obj, color=light, linewidth=1.8, label="Object recall", zorder=3
    )

    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5, zorder=1)

    ax.set_xlabel("Relational SVG Score at Encoding")
    ax.set_ylabel("Predicted Recall (%)")

    legend = ax.legend(frameon=False, fontsize=10, loc="upper left", handlelength=1.5)
    for text in legend.get_texts():
        text.set_fontfamily("Arial")

    _apply_apa_style(ax)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")

    return pd.DataFrame(
        {
            "svg_z_enc_within_z": x_grid,
            "pred_relations_pct": preds_rel,
            "ci_lower_rel_pct": ci_lo_rel,
            "ci_upper_rel_pct": ci_hi_rel,
            "pred_objects_pct": preds_obj,
            "ci_lower_obj_pct": ci_lo_obj,
            "ci_upper_obj_pct": ci_hi_obj,
        }
    )


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
            row["mean"],
            i,
            xerr=[[row["mean"] - row["ci_lo"]], [row["ci_hi"] - row["mean"]]],
            fmt="o",
            color=c,
            markersize=5,
            linewidth=1.2,
            capsize=2.5,
            alpha=0.8,
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    grand = img_stats["mean"].mean()
    ax.axvline(grand, color="#636363", linewidth=1.0, linestyle="-", alpha=0.7)
    ax.set_yticks(range(n_img))
    ax.set_yticklabels(img_stats["StimID"].values, fontsize=7)
    ax.set_xlabel("Mean encoding SVG (z-score, ± 95% CI)", fontsize=9)
    ax.set_title(
        "Per-image mean relational scanpath strength\n(sorted ascending)", fontsize=9
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")

    return img_stats[["StimID", "mean", "sd", "n", "ci_lo", "ci_hi"]].rename(
        columns={
            "mean": "svg_z_enc_image_mean",
            "sd": "svg_z_enc_image_sd",
            "ci_lo": "ci_lower",
            "ci_hi": "ci_upper",
        }
    )


def _save_partial_regression_data(enc: pd.DataFrame, output_path: Path) -> None:
    """Save residualised data used by the partial regression supplementary figure."""
    svg_col = "svg_z_enc_within"
    rows = {}
    for t in [svg_col, DV_TOTAL, DV_RELATIONS, DV_OBJECTS]:
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

    figures_dir = output_dir / "figures"
    supp_dir = output_dir / "supplementary"
    fig_data_dir = output_dir / "figure_data"

    for d in (output_dir, figures_dir, supp_dir, fig_data_dir):
        d.mkdir(parents=True, exist_ok=True)

    enc = filtered.get("enc", pd.DataFrame())
    dec = filtered.get("dec", pd.DataFrame())
    completeness = filtered.get("_data_completeness", {})

    # Write full tables for use by test scripts
    if not enc.empty:
        enc.to_csv(fig_data_dir / "analysis_enc.csv", index=False)
        logger.info("  Written → figure_data/analysis_enc.csv")
    if not dec.empty:
        dec.to_csv(fig_data_dir / "analysis_dec.csv", index=False)
        logger.info("  Written → figure_data/analysis_dec.csv")

    # ── Assumption checks (H2 + Exploratory LMMs) ───────────────────────────
    _VIF_MAP = {
        "enc": _VIF_COLS_ENC,
        "dec": _VIF_COLS_DEC,
        "enc_long": _VIF_COLS_LONG,
    }
    assumption_rows = []
    for name, _, _, table_key, group in MODEL_SPECS:
        if group not in ("H2", "Exploratory"):
            continue
        entry = results.get(name)
        if entry is None or entry[0] is None:
            continue
        result, mode = entry
        if mode != "lmm":
            continue
        df_for_checks = filtered.get(table_key, pd.DataFrame())
        vif_cols = _VIF_MAP.get(table_key, _VIF_COLS_ENC)
        row = _check_lmm_assumptions(name, result, df_for_checks, supp_dir, vif_cols)
        assumption_rows.append(row)

    if assumption_rows:
        assump_df = pd.DataFrame(assumption_rows)
        assump_path = output_dir / "assumption_checks.csv"
        assump_df.to_csv(assump_path, index=False)
        logger.info("  Written → assumption_checks.csv")

    # ── Appendix parameter tables ────────────────────────────────────────────
    _write_parameter_tables(results, output_dir)

    # ── Model summaries text ─────────────────────────────────────────────────
    desc_text, _ = compute_descriptives(filtered, output_dir)

    all_coefs = []
    summary_lines = [desc_text]

    # Data completeness block
    if completeness:
        summary_lines.append("\n=== Data completeness ===")
        for phase, initial_key, low_n_key, wrong_key, final_key in [
            ("Encoding", "enc_initial", "enc_low_n", "enc_wrong_image", "enc_final"),
            ("Decoding", "dec_initial", "dec_low_n", "dec_wrong_image", "dec_final"),
        ]:
            if initial_key in completeness:
                summary_lines.append(
                    f"  {phase}: {completeness[initial_key]} initial rows  "
                    f"→  {completeness.get(low_n_key, 0)} removed (low_n)  "
                    f"+ {completeness.get(wrong_key, 0)} removed (wrong image)  "
                    f"→  {completeness.get(final_key, '?')} final rows"
                )

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

            # For H1 t-test results, append Cohen's d and normality checks
            if group == "H1" and hasattr(result, "cohens_d"):
                h1_extra = [
                    f"  Cohen's d      : {result.cohens_d:.3f}",
                    f"  Shapiro-Wilk   : W={result.sw_stat:.3f}, p={result.sw_p:.4f}"
                    + (
                        "  *** NORMALITY VIOLATED ***"
                        if result.normality_violated
                        else "  (normality holds)"
                    ),
                    f"  Wilcoxon W     : {result.wx_stat}, p={result.wx_p:.4f}"
                    + (
                        "  [robustness check — normality violated]"
                        if result.normality_violated
                        else "  [robustness check]"
                    ),
                ]
                summary_lines.extend(h1_extra)

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
                        and t not in {f"{c}_z" for c in DEC_COVARIATES}
                        and t not in {f"{c}_z" for c in DEC_BETWEEN_COVARIATES}
                    )
                    or t == "memory_type"
                    or "memory_type" in t
                ]
            )
            for term in focus_terms:
                row = coef_df[coef_df["term"].str.strip() == term]
                if row.empty:
                    continue
                b, p = row["coef"].values[0], row["p"].values[0]
                lo, hi = row["ci_lower"].values[0], row["ci_upper"].values[0]
                sig = (
                    "***"
                    if p < 0.001
                    else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                )
                # Flag coefficients that would round to .000 at 3dp
                if abs(b) < 0.0005:
                    logger.warning(
                        f"    {term}: coefficient rounds to .000 at 3dp "
                        f"(raw={b:.6f}) — report as b < .001 or use 4dp in writeup"
                    )
                logger.info(
                    f"    {term}: β={b:.4f} [{lo:.4f}, {hi:.4f}], p={p:.4f} {sig}"
                )

    coef_all = pd.concat(all_coefs, ignore_index=True) if all_coefs else pd.DataFrame()

    # ── Convergence summary ──────────────────────────────────────────────────
    summary_lines.append("\n\n" + "=" * 60)
    summary_lines.append("CONVERGENCE SUMMARY")
    summary_lines.append("=" * 60)
    for name, _, desc, _, group in MODEL_SPECS:
        entry = results.get(name)
        if entry is None:
            continue
        result, mode = entry
        if mode == "lmm" and result is not None:
            optimizer = getattr(result, "optimizer_used", "unknown")
            converged = getattr(result, "converged", "?")
            n_obs = getattr(result, "n_obs", "?")
            n_dropped = getattr(result, "n_dropped", "?")
            summary_lines.append(
                f"  {name:<30} mode={mode}  optimizer={optimizer}  "
                f"converged={converged}  n_obs={n_obs}  n_dropped={n_dropped}"
            )
        elif mode == "ttest" and result is not None:
            summary_lines.append(
                f"  {name:<30} mode={mode}  n_participants={result.nobs}"
            )
        else:
            summary_lines.append(f"  {name:<30} mode={mode}")

    with open(output_dir / "model_summaries.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    logger.info("  Written → model_summaries.txt")

    if not plot or enc.empty:
        return coef_all

    # ── Main figures ────────────────────────────────────────────────────────

    # Figure 1: box and whisker — encoding and decoding SVG
    fig1_data = _figure1_boxplot(
        enc,
        figures_dir / "figure1_boxplot.png",
        dec=dec if not dec.empty else None,
    )
    if not fig1_data.empty:
        fig1_data.to_csv(fig_data_dir / "figure1_data.csv", index=False)
        logger.info("  Written → figure_data/figure1_data.csv")

    # Figure 2: model-predicted total recall (encoding SVG)
    fig2_data = _figure2_predicted_total(
        enc, results, figures_dir / "figure2_predicted_total.png"
    )
    if not fig2_data.empty:
        fig2_data.to_csv(fig_data_dir / "figure2_data.csv", index=False)
        logger.info("  Written → figure_data/figure2_data.csv")

    # Figure 3: model-predicted relational vs object recall
    fig3_data = _figure3_content_comparison(
        enc, results, figures_dir / "figure3_content_comparison.png"
    )
    if not fig3_data.empty:
        fig3_data.to_csv(fig_data_dir / "figure3_data.csv", index=False)
        logger.info("  Written → figure_data/figure3_data.csv")

    # ── Supplementary figures ────────────────────────────────────────────────

    # Encoding partial regression (supports linearity assumption claim)
    _partial_regression_plot(enc, supp_dir / "supp_partial_regression.png")
    _save_partial_regression_data(
        enc, fig_data_dir / "supp_partial_regression_data.csv"
    )

    # Per-image mean relationality (supports methods description)
    img_data = _supp_per_image(enc, supp_dir / "supp_per_image_relationality.png")
    if not img_data.empty:
        img_data.to_csv(fig_data_dir / "supp_per_image_data.csv", index=False)
        logger.info("  Written → figure_data/supp_per_image_data.csv")

    return coef_all
