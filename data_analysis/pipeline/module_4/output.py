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

from .constants import DV_OBJECTS, DV_RELATIONS, DV_TOTAL, ENC_COVARIATES, MODEL_SPECS

logger = logging.getLogger(__name__)

# Nuisance terms to skip in the forest plot
_SKIP_TERMS = {f"{c}_z" for c in ENC_COVARIATES} | {
    "Group Var",
    "StimID Var",
    "Intercept",
}

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
    Return residuals of `target` after regressing on ENC_COVARIATES + C(SubjectID).
    Rows with any NaN in required columns are set to NaN in output.
    """
    import warnings

    req = [target] + [c for c in ENC_COVARIATES if c in df.columns]
    mask = df[req].notna().all(axis=1)
    out = pd.Series(np.nan, index=df.index)
    sub = df[mask].copy()
    if len(sub) < 10:
        return out
    cov_terms = " + ".join(f"{c}_z" for c in ENC_COVARIATES if f"{c}_z" in sub.columns)
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
    Three-panel partial regression plot: encoding SVG → each proportion DV.
    Both SVG and DV are residualised against ENC_COVARIATES + C(SubjectID)
    before plotting, so the trend line shows the unique relationship
    independent of nuisance variables.
    """
    svg_col = "svg_z_enc"
    # Need z-scored covariates in the df
    for c in ENC_COVARIATES:
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
        ax.set_xlabel("Encoding SVG\n(covariate-residualised)", fontsize=9)
        ax.set_ylabel(f"{dv_label}\n(covariate-residualised)", fontsize=9)
        ax.set_title(dv_label.replace("\n", " "), fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Partial regression: Encoding SVG → memory recall (proportion DVs)\n"
        "Covariates removed: n_fixations, aoi_prop, mean_salience_relational, SubjectID",
        fontsize=9,
        y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------


def _forest_plot(coef_df: pd.DataFrame, output_path: Path) -> None:
    """Two-panel forest plot: H2 main effects | Exploratory."""
    plot_groups = ["H2", "Exploratory"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, group in zip(axes, plot_groups):
        to_plot = []
        for name, _, _, _, grp in MODEL_SPECS:
            if grp != group:
                continue
            sub = coef_df[coef_df["model"] == name].copy()
            if sub.empty:
                continue
            sub = sub[
                ~sub["term"].str.strip().isin(_SKIP_TERMS)
                & ~sub["term"].str.strip().str.startswith("C(")
            ]
            for _, r in sub.iterrows():
                to_plot.append(
                    {
                        "label": f"{name}\n({r['term'].strip()})",
                        "coef": r["coef"],
                        "lo": r["ci_lower"],
                        "hi": r["ci_upper"],
                        "p": r["p"],
                        "color": _PALETTE.get(name, "#636363"),
                    }
                )

        if not to_plot:
            ax.set_visible(False)
            continue

        ypos = list(range(len(to_plot)))[::-1]
        for y, row in zip(ypos, to_plot):
            ax.errorbar(
                row["coef"],
                y,
                xerr=[[row["coef"] - row["lo"]], [row["hi"] - row["coef"]]],
                fmt="o",
                color=row["color"],
                markersize=6,
                linewidth=1.6,
                capsize=3,
            )
            sig = (
                "***"
                if row["p"] < 0.001
                else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
            )
            if sig:
                ax.text(
                    row["hi"] + 0.01,
                    y,
                    sig,
                    va="center",
                    fontsize=10,
                    color=row["color"],
                )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels([r["label"] for r in to_plot], fontsize=8)
        ax.set_xlabel("β  (proportion units, 0-1 scale)", fontsize=9)
        ax.set_title(group, fontsize=11, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Module 4 — Encoding SVG → memory recall proportions\n"
        "pilot: OLS+C(SubjectID) | final: LMM+(1|SubjectID)+(1|StimID)",
        fontsize=9,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")


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
    Returns the concatenated coefficient DataFrame.
    """
    logger.info("\nStep 6: Writing outputs ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analysis tables
    enc = filtered.get("enc", pd.DataFrame())
    if not enc.empty:
        enc.to_csv(output_dir / "analysis_enc.csv", index=False)
    logger.info("  Written → analysis_enc.csv")

    all_coefs = []
    summary_lines = [_descriptives(filtered)]

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

            # Log focal term(s)
            focus_terms = (
                ["Intercept"]
                if group == "H1"
                else [
                    t
                    for t in coef_df["term"].str.strip()
                    if t.endswith("_z")
                    and t not in {f"{c}_z" for c in ENC_COVARIATES}
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
                logger.info(
                    f"    {term}: β={b:.4f} [{lo:.4f}, {hi:.4f}], " f"p={p:.4f} {sig}"
                )

    coef_all = pd.concat(all_coefs, ignore_index=True) if all_coefs else pd.DataFrame()
    if not coef_all.empty:
        coef_all.to_csv(output_dir / "model_coefficients.csv", index=False)
        logger.info("  Written → model_coefficients.csv")

    with open(output_dir / "model_summaries.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    logger.info("  Written → model_summaries.txt")

    if plot and not coef_all.empty:
        _forest_plot(coef_all, output_dir / "forest_plot.png")

    if plot and not enc.empty:
        _partial_regression_plot(enc, output_dir / "partial_regression.png")

    return coef_all
