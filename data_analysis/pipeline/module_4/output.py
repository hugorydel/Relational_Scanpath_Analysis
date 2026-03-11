"""
pipeline/module4/output.py
===========================
Step 6: Summarise results and write all outputs.

  - _extract_coef_table : normalises coefficients from LMM or OLS result
  - _h1_descriptives    : raw SVG descriptive stats string
  - _forest_plot        : three-panel forest plot (H1 | H2 | Exploratory)
  - summarise           : master function — writes CSVs, txt, and plot
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .constants import DEC_COVARIATES, DEFAULT_SCORES_PATH, ENC_COVARIATES, MODEL_SPECS
from .pilot_diagnostics import make_pilot_diagnostics

logger = logging.getLogger(__name__)

# Terms to skip in the forest plot (nuisance / variance components)
_SKIP_TERMS = {f"{c}_z" for c in DEC_COVARIATES + ENC_COVARIATES} | {
    "Group Var",
    "StimID Var",
}

# Colour palette for confirmatory models; exploratory defaults to grey
_PALETTE = {
    "H1a_svg_inter": "#084594",
    "H1b_svg_inter": "#4292c6",
    "H1a_svg_all": "#084594",
    "H1b_svg_all": "#4292c6",
    "H2_relational_inter": "#99000d",
    "H2_relational_all": "#ef3b2c",
    "H2_objects_inter": "#a63603",
    "H2_objects_all": "#fd8d3c",
}


# ---------------------------------------------------------------------------
# Coefficient table extraction
# ---------------------------------------------------------------------------


def _extract_coef_table(name: str, result) -> pd.DataFrame:
    """
    Build a tidy coefficient DataFrame from either a MixedLMResults (LMM)
    or an OLS RegressionResultsWrapper (pilot mode).

    C(SubjectID) dummy rows are stripped — they are nuisance parameters and
    would clutter the coefficients CSV and forest plot.
    """
    ci = result.conf_int()
    params = result.params
    mask = ~params.index.str.startswith("C(SubjectID)")
    return pd.DataFrame(
        {
            "model": name,
            "term": params.index[mask],
            "coef": params.values[mask],
            "std_err": result.bse.values[mask],
            "z": result.tvalues.values[mask],
            "p": result.pvalues.values[mask],
            "ci_lower": ci.iloc[:, 0].values[mask],
            "ci_upper": ci.iloc[:, 1].values[mask],
        }
    )


# ---------------------------------------------------------------------------
# H1 descriptives
# ---------------------------------------------------------------------------


def _h1_descriptives(filtered: dict) -> str:
    lines = [
        "\n=== H1 Descriptives: raw decoding SVG z-scores "
        "(before covariate adjustment) ==="
    ]
    for key, col in [("dec_inter", "svg_z_inter_dec"), ("dec_all", "svg_z_all_dec")]:
        df = filtered.get(key, pd.DataFrame())
        vals = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        if len(vals) == 0:
            lines.append(f"  {col}: no data")
        else:
            lines.append(
                f"  {col}: n={len(vals)}, mean={vals.mean():.3f}, "
                f"sd={vals.std():.3f}, median={vals.median():.3f}, "
                f"% > 0: {100*(vals > 0).mean():.1f}%"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------


def _forest_plot(coef_df: pd.DataFrame, output_path: Path) -> None:
    """Three-panel forest plot: H1 intercepts | H2 | Exploratory."""
    groups = ["H1", "H2", "Exploratory"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))

    for ax, group in zip(axes, groups):
        to_plot = []
        for name, _, _, _, grp in MODEL_SPECS:
            if grp != group:
                continue
            sub = coef_df[coef_df["model"] == name].copy()
            if sub.empty:
                continue
            if group == "H1":
                sub = sub[sub["term"].str.strip() == "Intercept"]
            else:
                sub = sub[
                    ~sub["term"].str.strip().isin(_SKIP_TERMS)
                    & ~sub["term"].str.strip().str.startswith("C(")
                    & (sub["term"].str.strip() != "Intercept")
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
                    row["hi"] + 0.02,
                    y,
                    sig,
                    va="center",
                    fontsize=10,
                    color=row["color"],
                )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels([r["label"] for r in to_plot], fontsize=7.5)
        ax.set_xlabel("β  (or intercept for H1)", fontsize=9)
        ax.set_title(group, fontsize=11, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        "Module 4 results — pilot: OLS+C(SubjectID) | final: LMM+(1|SubjectID)+(1|StimID)",
        fontsize=8,
        y=1.01,
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
    features_path: "Path | None" = None,
    scores_path: "Path | None" = None,
) -> pd.DataFrame:
    """
    Write all Module 4 outputs to output_dir:
      analysis_dec_inter.csv, analysis_enc_all.csv, analysis_replay.csv
      model_coefficients.csv
      model_summaries.txt
      forest_plot.png          (if plot=True and any models converged)
      pilot_diagnostics.png    (always, if features_path and scores_path given)

    Returns the concatenated coefficient DataFrame.
    """
    logger.info("\nStep 6: Writing outputs ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analysis tables
    for key in ("dec_inter", "enc_all", "replay"):
        df = filtered.get(key, pd.DataFrame())
        if not df.empty:
            df.to_csv(output_dir / f"analysis_{key}.csv", index=False)
    logger.info("  Written → analysis_*.csv")

    all_coefs = []
    summary_lines = [_h1_descriptives(filtered)]

    for group in ["H1", "H2", "Exploratory"]:
        summary_lines.append(f"\n\n{'#'*60}\n# {group} MODELS\n{'#'*60}")

        for name, _, desc, _, grp in MODEL_SPECS:
            if grp != group:
                continue

            entry = results.get(name)
            result, _ = entry if entry else (None, False)

            summary_lines.append(f"\n{'='*60}\n{name}: {desc}\n{'='*60}")
            if result is None:
                summary_lines.append("SKIPPED / FAILED\n")
                continue

            coef_df = _extract_coef_table(name, result)
            all_coefs.append(coef_df)
            summary_lines.append(str(result.summary()))

            # Append fit quality line (llf / R² depending on model type)
            if hasattr(result, "llf") and result.llf is not None:
                summary_lines.append(f"\nLog-likelihood: {result.llf:.4f}")
            if hasattr(result, "rsquared"):
                summary_lines.append(f"R²: {result.rsquared:.4f}")
            if hasattr(result, "converged"):
                summary_lines.append(f"Converged: {result.converged}")

            # Log the focal term(s) to console
            focus = (
                ["Intercept"]
                if group == "H1"
                else [
                    t
                    for t in coef_df["term"].str.strip()
                    if t.endswith("_z")
                    and t not in {f"{c}_z" for c in DEC_COVARIATES + ENC_COVARIATES}
                ]
            )
            for term in focus:
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
                    f"    {term}: β={b:.4f} [{lo:.4f}, {hi:.4f}], p={p:.4f} {sig}"
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

    if plot and features_path is not None and scores_path is not None:
        try:
            make_pilot_diagnostics(
                features_path,
                scores_path,
                output_dir / "pilot_diagnostics.png",
            )
        except Exception as e:
            logger.warning(f"  pilot_diagnostics skipped: {e}")

    return coef_all
