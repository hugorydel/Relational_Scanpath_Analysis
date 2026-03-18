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
# H1 figure — trial distribution + per-participant consistency
# ---------------------------------------------------------------------------


def _h1_figure(enc: pd.DataFrame, results: dict, output_path: Path) -> None:
    """
    Two-panel H1 figure.

    Left  — density of all trial-level svg_z_enc values with 0 (chance)
             and grand mean marked. Area above 0 shaded.
    Right — per-participant mean ± 95% CI dot plot, sorted ascending.

    Together these show: (a) the distribution is shifted well above chance,
    and (b) this is consistent across every participant.
    """
    svg_col = "svg_z_enc"
    if svg_col not in enc.columns:
        logger.warning("  _h1_figure: missing svg_z_enc — skipping.")
        return

    vals_all = enc[svg_col].dropna().values
    grand_mean = vals_all.mean()
    grand_sd = vals_all.std()
    pct_above = 100 * (vals_all > 0).mean()

    # Per-participant means + 95% CI
    records = []
    for subj, grp in enc.groupby("SubjectID"):
        vals = grp[svg_col].dropna().values
        n = len(vals)
        if n < 2:
            continue
        m = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(n)
        tc = stats.t.ppf(0.975, df=n - 1)
        records.append(
            {"SubjectID": subj, "mean": m, "lo": m - tc * se, "hi": m + tc * se}
        )
    if not records:
        logger.warning("  _h1_figure: no per-participant data — skipping.")
        return
    df_pp = pd.DataFrame(records).sort_values("mean").reset_index(drop=True)
    n_subj = len(df_pp)

    colour = "#2166ac"
    fig, (ax_dist, ax_pp) = plt.subplots(
        1,
        2,
        figsize=(13, max(5, n_subj * 0.38)),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    # ── Left: distribution ────────────────────────────────────────────────
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(vals_all, bw_method=0.3)
    x_grid = np.linspace(vals_all.min() - 0.5, vals_all.max() + 0.5, 400)
    y_kde = kde(x_grid)

    ax_dist.plot(x_grid, y_kde, color=colour, linewidth=2, zorder=4)
    # Shade above 0
    mask_above = x_grid >= 0
    ax_dist.fill_between(
        x_grid[mask_above],
        y_kde[mask_above],
        color=colour,
        alpha=0.25,
        zorder=3,
        label=f"{pct_above:.0f}% of trials > 0",
    )
    ax_dist.fill_between(
        x_grid[~mask_above],
        y_kde[~mask_above],
        color="#a63603",
        alpha=0.18,
        zorder=3,
    )
    ax_dist.axvline(
        0,
        color="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.6,
        label="Permutation chance (0)",
    )
    ax_dist.axvline(
        grand_mean,
        color=colour,
        linewidth=1.4,
        linestyle="-",
        alpha=0.85,
        label=f"Grand mean = {grand_mean:.2f} SD",
    )
    ax_dist.set_xlabel("Encoding SVG (z-score above permutation baseline)", fontsize=9)
    ax_dist.set_ylabel("Density", fontsize=9)
    ax_dist.set_title(
        "H1: Distribution of relational scanning\nacross all encoding trials",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax_dist.annotate(
        f"mean = {grand_mean:.2f} SD  (σ = {grand_sd:.2f})\n{pct_above:.0f}% of trials above chance",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        fontsize=8.5,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.88),
    )
    ax_dist.legend(fontsize=8, framealpha=0.85)
    ax_dist.spines[["top", "right"]].set_visible(False)
    ax_dist.tick_params(labelsize=8)

    # ── Right: per-participant dot plot ───────────────────────────────────
    for i, row in df_pp.iterrows():
        c = colour if row["mean"] > 0 else "#a63603"
        ax_pp.errorbar(
            row["mean"],
            i,
            xerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
            fmt="o",
            color=c,
            markersize=6,
            linewidth=1.4,
            capsize=3,
            alpha=0.85,
        )
    ax_pp.axvline(
        grand_mean,
        color="#636363",
        linewidth=1.2,
        linestyle="-",
        alpha=0.7,
        label=f"Grand mean ({grand_mean:.2f})",
    )
    ax_pp.axvline(
        0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="Chance (0)"
    )
    ax_pp.set_yticks(range(n_subj))
    ax_pp.set_yticklabels(
        [
            r["SubjectID"].replace("Encode-Decode_Experiment-", "P")
            for _, r in df_pp.iterrows()
        ],
        fontsize=8,
    )
    ax_pp.set_xlabel("Mean encoding SVG (z-scored, ± 95% CI)", fontsize=9)
    ax_pp.set_title(
        "H1: Per-participant relational scanning\n(sorted ascending; blue = above chance)",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax_pp.legend(fontsize=8, framealpha=0.85)
    ax_pp.spines[["top", "right"]].set_visible(False)
    ax_pp.tick_params(labelsize=8)

    fig.suptitle(
        "H1: Encoding scanpaths are reliably more relational than the permutation baseline",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")


# ---------------------------------------------------------------------------
# H2 figure — quartile bin plot + per-image correlation distribution
# ---------------------------------------------------------------------------


def _h2_figure(
    enc: pd.DataFrame,
    results: dict,
    output_path: Path,
) -> None:
    """
    Two-panel H2 figure.

    Left  — Quartile bins of svg_z_enc_within on X; mean recall proportion
             (± 95% CI) on Y for prop_total, prop_relations, prop_objects.
             LMM β and p annotated per DV.
    Right — Per-image correlation distribution: strip + box of per-image
             Pearson r (svg_z_enc_within vs each DV), with one-sample
             t-test vs zero annotated.
    """
    import warnings

    from scipy.stats import pearsonr, ttest_1samp

    svg_within = "svg_z_enc_within"
    dvs = [
        (DV_TOTAL, "Total recall", "#2166ac"),
        (DV_RELATIONS, "Relational recall", "#084594"),
        (DV_OBJECTS, "Object recall", "#a63603"),
    ]

    if svg_within not in enc.columns:
        logger.warning("  _h2_figure: svg_z_enc_within missing — skipping.")
        return

    fig, (ax_bin, ax_img) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: quartile bin plot ───────────────────────────────────────────
    enc_q = enc.copy()
    try:
        enc_q["svg_quartile"] = pd.qcut(
            enc_q[svg_within],
            q=4,
            labels=["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"],
        )
    except ValueError:
        # Fallback if too few unique values
        enc_q["svg_quartile"] = pd.qcut(
            enc_q[svg_within],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )

    q_labels = enc_q["svg_quartile"].cat.categories.tolist()
    x_pos = np.arange(len(q_labels))
    offsets = [-0.22, 0, 0.22]

    for (dv_col, dv_label, colour), offset in zip(dvs, offsets):
        if dv_col not in enc_q.columns:
            continue
        means, los, his = [], [], []
        for ql in q_labels:
            grp = enc_q.loc[enc_q["svg_quartile"] == ql, dv_col].dropna().values
            if len(grp) < 3:
                means.append(np.nan)
                los.append(np.nan)
                his.append(np.nan)
                continue
            m = grp.mean()
            se = grp.std(ddof=1) / np.sqrt(len(grp))
            tc = stats.t.ppf(0.975, df=len(grp) - 1)
            means.append(m)
            los.append(m - tc * se)
            his.append(m + tc * se)

        means = np.array(means, dtype=float)
        los = np.array(los, dtype=float)
        his = np.array(his, dtype=float)

        ax_bin.errorbar(
            x_pos + offset,
            means,
            yerr=[means - los, his - means],
            fmt="o-",
            color=colour,
            markersize=6,
            linewidth=1.8,
            capsize=3,
            label=dv_label,
        )

    # Annotate LMM betas
    model_name_map = {
        DV_TOTAL: "H2_total",
        DV_RELATIONS: "H2_relations",
        DV_OBJECTS: "H2_objects",
    }
    annot_lines = []
    for dv_col, dv_label, _ in dvs:
        mname = model_name_map.get(dv_col)
        if mname and mname in results:
            res, _ = results[mname]
            if res is not None:
                try:
                    b = res.params.get(
                        "svg_z_enc_within_z", res.params.get(svg_within + "_z", np.nan)
                    )
                    p = res.pvalues.get(
                        "svg_z_enc_within_z", res.pvalues.get(svg_within + "_z", np.nan)
                    )
                    sig = (
                        "***"
                        if p < 0.001
                        else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    )
                    annot_lines.append(f"{dv_label}: β={b:+.3f} {sig}")
                except Exception:
                    pass
    if annot_lines:
        ax_bin.annotate(
            "\n".join(annot_lines),
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.88),
        )

    ax_bin.set_xticks(x_pos)
    ax_bin.set_xticklabels(q_labels, fontsize=9)
    ax_bin.set_xlabel(
        "Within-image SVG quartile\n(relational scanning relative to image mean)",
        fontsize=9,
    )
    ax_bin.set_ylabel("Mean recall proportion (± 95% CI)", fontsize=9)
    ax_bin.set_title(
        "H2: Relational scanning predicts memory\nacross encoding SVG quartiles",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax_bin.legend(fontsize=8.5, framealpha=0.85)
    ax_bin.spines[["top", "right"]].set_visible(False)
    ax_bin.tick_params(labelsize=8)

    # ── Right: per-image correlation distribution ─────────────────────────
    stim_ids = enc["StimID"].unique()
    offsets_img = [-0.22, 0, 0.22]

    for col_i, ((dv_col, dv_label, colour), x_off) in enumerate(zip(dvs, offsets_img)):
        if dv_col not in enc.columns:
            continue
        rs = []
        for stim_id in stim_ids:
            sub = enc[enc["StimID"] == stim_id][[svg_within, dv_col]].dropna()
            if len(sub) < 6:
                continue
            r, _ = pearsonr(sub[svg_within], sub[dv_col])
            rs.append(r)
        if not rs:
            continue
        rs = np.array(rs)
        x_center = col_i

        # Box
        ax_img.boxplot(
            rs,
            positions=[x_center],
            widths=0.3,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="white", linewidth=2),
            boxprops=dict(facecolor=colour, alpha=0.35),
            whiskerprops=dict(color=colour, alpha=0.6),
            capprops=dict(color=colour, alpha=0.6),
        )
        # Strip
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(rs))
        ax_img.scatter(
            np.ones(len(rs)) * x_center + jitter,
            rs,
            color=colour,
            alpha=0.6,
            s=26,
            linewidths=0,
            zorder=4,
        )
        # t-test annotation
        t, p_t = ttest_1samp(rs, 0)
        sig = (
            "***"
            if p_t < 0.001
            else "**" if p_t < 0.01 else "*" if p_t < 0.05 else "ns"
        )
        ax_img.annotate(
            f"mean r={rs.mean():+.3f}\n{sig}",
            xy=(x_center, rs.max() + 0.03),
            ha="center",
            fontsize=8,
            color=colour,
        )

    ax_img.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_img.set_xticks([0, 1, 2])
    ax_img.set_xticklabels(
        [dv_label for _, dv_label, _ in dvs],
        fontsize=9,
    )
    ax_img.set_ylabel("Per-image Pearson r\n(within-image SVG vs recall)", fontsize=9)
    ax_img.set_title(
        "H2: Effect is consistent across images\n(each point = one image, N=30)",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax_img.spines[["top", "right"]].set_visible(False)
    ax_img.tick_params(labelsize=8)

    fig.suptitle(
        "H2: Within-image relational scanning predicts episodic memory recall",
        fontsize=11,
        fontweight="bold",
        y=1.01,
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
        _partial_regression_plot(
            enc, output_dir / "supplementary_partial_regression.png"
        )
        _h1_figure(enc, results, output_dir / "h1_relational_scanning.png")
        _h2_figure(enc, results, output_dir / "h2_svg_memory.png")

    return coef_all
