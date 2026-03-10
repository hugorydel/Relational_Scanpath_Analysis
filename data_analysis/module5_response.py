"""
module5_exploratory_response.py
================================
Test module: correlates decoding relational alignment (SVG z-score) with the
length of the free exploratory response, adjusted for non-alphanumeric characters.

Rationale
---------
If participants whose eye movements trace relational structure during retrieval
(high svg_z_inter_dec) are actually reconstructing the scene more richly, this
should manifest in more detailed free-recall responses — independent of any
forced-choice accuracy ceiling.

Response length measure
-----------------------
`alnum_len`: count of alphanumeric characters in FreeResponse after stripping
all non-alphanumeric characters (punctuation, spaces, digits-if-desired).
This avoids penalising responses that happen to use more punctuation, and is
robust to the run-together typing style observed in the pilot data.

Analyses
--------
1. Per-participant Spearman ρ (svg_z_inter_dec vs alnum_len, svg_z_all_dec vs alnum_len)
   — works at n=10 trials per participant.

2. Pooled partial Spearman: within-participant mean-centred alnum_len vs
   mean-centred svg scores, pooled across all participants.
   Equivalent to a partial correlation controlling for participant identity.

3. Mixed-effects LMM (when n_participants >= 3):
   alnum_len ~ svg_z_inter_dec_z + n_fixations_dec_z + aoi_prop_dec_z
               + mean_salience_dec_z + (1|SubjectID) + (1|StimID)
   Both inter and all-edges SVG variants.

4. Scatter plots: one panel per participant, pooled panel, with Spearman ρ
   and p-value annotated.

Inputs
------
    DATA_BEHAVIORAL_DIR / *_exploratory.csv  (FreeResponse column)
    OUTPUT_FEATURES_DIR / trial_features_all.csv  (decoding SVG + covariates)

Outputs
-------
    output/exploratory_response/exploratory_response_df.csv
    output/exploratory_response/per_participant_correlations.csv
    output/exploratory_response/model_coefficients.csv   (if LMM run)
    output/exploratory_response/model_summaries.txt      (if LMM run)
    output/exploratory_response/scatter_plot.png

Usage
-----
    python module5_exploratory_response.py
    python module5_exploratory_response.py --features path/to/trial_features_all.csv
    python module5_exploratory_response.py --no-plot
"""

import argparse
import logging
import re
import warnings
from pathlib import Path

import config
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

logger = logging.getLogger(__name__)

MIN_PARTICIPANTS_FOR_LMM = 3
DEC_COVARIATES = ["n_fixations_dec", "aoi_prop_dec", "mean_salience_dec"]


# ---------------------------------------------------------------------------
# Step 1: Load exploratory response files
# ---------------------------------------------------------------------------


def load_exploratory_responses(behavioral_dir: Path) -> pd.DataFrame:
    """
    Load all *_exploratory.csv files and compute alnum_len.

    alnum_len = number of alphanumeric characters in FreeResponse,
    i.e. len after stripping everything that is not [A-Za-z0-9].
    """
    logger.info("Step 1: Loading exploratory response files ...")
    files = sorted(behavioral_dir.glob("*_exploratory.csv"))

    if not files:
        raise FileNotFoundError(f"No *_exploratory.csv files found in {behavioral_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, dtype={"StimID": str, "SubjectID": str})
        if "FreeResponse" not in df.columns:
            logger.warning(f"  {f.name}: no FreeResponse column — skipping.")
            continue
        df["alnum_len"] = df["FreeResponse"].apply(
            lambda x: (
                len(re.sub(r"[^A-Za-z0-9]", "", str(x))) if pd.notna(x) else np.nan
            )
        )
        dfs.append(df)
        logger.info(
            f"  {f.name}: {len(df)} trials, "
            f"alnum_len {df['alnum_len'].min():.0f}–{df['alnum_len'].max():.0f}"
        )

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(
        f"  Total: {len(combined)} exploratory trials from "
        f"{combined['SubjectID'].nunique()} participant(s)"
    )
    return combined


# ---------------------------------------------------------------------------
# Step 2: Join with decoding SVG features
# ---------------------------------------------------------------------------


def build_analysis_df(exploratory: pd.DataFrame, features_path: Path) -> pd.DataFrame:
    """
    Join exploratory responses with decoding-phase SVG features.

    Takes decoding rows from trial_features_all.csv, renames SVG and
    covariate columns with _dec suffix, and merges on SubjectID × StimID.
    Rows without a matching decoding feature row are dropped and logged.
    """
    logger.info("Step 2: Joining with decoding features ...")

    features = pd.read_csv(features_path, dtype={"StimID": str, "SubjectID": str})
    dec = features[features["Phase"] == "decoding"].copy()

    dec = dec.rename(
        columns={
            "svg_z_all": "svg_z_all_dec",
            "svg_z_inter": "svg_z_inter_dec",
            "n_fixations": "n_fixations_dec",
            "aoi_prop": "aoi_prop_dec",
            "mean_salience": "mean_salience_dec",
            "low_n": "low_n_dec",
        }
    )

    keep = (
        ["SubjectID", "StimID", "svg_z_all_dec", "svg_z_inter_dec", "low_n_dec"]
        + DEC_COVARIATES
        + ["dec_total_correct", "q1_accuracy", "q2_accuracy"]
    )
    dec_sub = dec[[c for c in keep if c in dec.columns]]

    before = len(exploratory)
    df = exploratory.merge(dec_sub, on=["SubjectID", "StimID"], how="inner")
    dropped = before - len(df)
    if dropped:
        logger.warning(
            f"  {dropped} exploratory trial(s) had no matching decoding feature "
            f"row and were dropped."
        )

    logger.info(
        f"  Analysis df: {len(df)} trials, "
        f"{df['SubjectID'].nunique()} participant(s), "
        f"{df['StimID'].nunique()} stimuli"
    )
    return df


# ---------------------------------------------------------------------------
# Step 3: Exclusions and standardisation
# ---------------------------------------------------------------------------


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply exclusions and z-score continuous predictors.

    Exclusions:
        - low_n_dec: too few fixations for reliable SVG estimate
        - missing alnum_len

    Standardisation is within the analysis sample so the LMM intercept is
    interpretable as the mean at an average trial.
    """
    logger.info("Step 3: Exclusions and standardisation ...")
    n = len(df)

    df = df[~df["low_n_dec"]].copy()
    logger.info(f"  low_n_dec filter: {n} → {len(df)}")

    df = df[df["alnum_len"].notna()].copy()
    logger.info(f"  missing alnum_len: → {len(df)} usable trials")

    for col in ["svg_z_inter_dec", "svg_z_all_dec"] + DEC_COVARIATES:
        if col not in df.columns:
            continue
        mu, sd = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / sd if sd > 0 else 0.0

    # Mean-centred version of alnum_len for pooled partial correlation
    df["alnum_len_centred"] = df["alnum_len"] - df.groupby("SubjectID")[
        "alnum_len"
    ].transform("mean")

    return df


# ---------------------------------------------------------------------------
# Step 4: Correlations
# ---------------------------------------------------------------------------


def run_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run per-participant and pooled Spearman correlations.

    Returns a DataFrame with one row per (participant, svg_variant) pair
    plus a 'pooled_partial' row.
    """
    logger.info("Step 4: Spearman correlations ...")
    rows = []

    svg_variants = [
        ("svg_z_inter_dec", "interactional"),
        ("svg_z_all_dec", "all-edges"),
    ]

    # Per-participant
    for sid, grp in df.groupby("SubjectID"):
        for col, label in svg_variants:
            sub = grp[["alnum_len", col]].dropna()
            if len(sub) < 4:
                logger.warning(
                    f"  {sid} × {label}: only {len(sub)} valid trials — "
                    f"correlation unreliable, reporting anyway."
                )
            if len(sub) < 3:
                continue
            rho, p = stats.spearmanr(sub["alnum_len"], sub[col])
            rows.append(
                {
                    "SubjectID": sid,
                    "svg_variant": label,
                    "n": len(sub),
                    "rho": rho,
                    "p": p,
                    "type": "per_participant",
                }
            )
            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            logger.info(
                f"  {sid} [{label}]: ρ={rho:.3f}, p={p:.3f} {sig}  (n={len(sub)})"
            )

    # Pooled partial: correlate within-participant mean-centred scores
    for col, label in svg_variants:
        centred_svg_col = f"{col}_centred"
        df[centred_svg_col] = df[col] - df.groupby("SubjectID")[col].transform("mean")
        sub = df[["alnum_len_centred", centred_svg_col]].dropna()
        if len(sub) < 5:
            logger.warning(f"  Pooled [{label}]: too few rows for pooled correlation.")
            continue
        rho, p = stats.spearmanr(sub["alnum_len_centred"], sub[centred_svg_col])
        rows.append(
            {
                "SubjectID": "pooled_partial",
                "svg_variant": label,
                "n": len(sub),
                "rho": rho,
                "p": p,
                "type": "pooled_partial",
            }
        )
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        logger.info(
            f"  Pooled partial [{label}]: ρ={rho:.3f}, p={p:.3f} {sig}  (n={len(sub)})"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5: LMM (when n_participants >= MIN_PARTICIPANTS_FOR_LMM)
# ---------------------------------------------------------------------------


def run_lmm(df: pd.DataFrame) -> dict:
    """
    Fit mixed-effects models predicting alnum_len from decoding SVG.

    Only runs if enough participants are present.
    Returns dict of {model_name: result | None}.
    """
    n_subj = df["SubjectID"].nunique()
    if n_subj < MIN_PARTICIPANTS_FOR_LMM:
        logger.info(
            f"Step 5: LMM skipped — only {n_subj} participant(s) "
            f"(need >= {MIN_PARTICIPANTS_FOR_LMM})."
        )
        return {}

    logger.info(f"Step 5: Fitting LMMs ({n_subj} participants) ...")
    cov_z = " + ".join(f"{c}_z" for c in DEC_COVARIATES)
    models = {
        "LMM_svg_inter": f"alnum_len ~ svg_z_inter_dec_z + {cov_z}",
        "LMM_svg_all": f"alnum_len ~ svg_z_all_dec_z + {cov_z}",
        "LMM_combined": f"alnum_len ~ svg_z_inter_dec_z + svg_z_all_dec_z + {cov_z}",
    }

    results = {}
    for name, formula in models.items():
        logger.info(f"  {name}: alnum_len ~ {formula.split('~')[1].strip()}")
        # Collect required columns
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
        req = list(
            {"SubjectID", "StimID", "alnum_len"}
            | {t for t in tokens if t in df.columns}
        )
        model_df = df[[c for c in req if c in df.columns]].dropna()

        if len(model_df) < 15 or model_df["SubjectID"].nunique() < 2:
            logger.warning(f"  {name}: insufficient data — skipping.")
            results[name] = None
            continue

        vc = {"StimID": "0 + C(StimID)"} if model_df["StimID"].nunique() > 1 else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = smf.mixedlm(
                    formula,
                    data=model_df,
                    groups="SubjectID",
                    vc_formula=vc if vc else None,
                )
                result = model.fit(reml=True, method="lbfgs", maxiter=300)
                results[name] = result
                logger.info(f"    Converged: {result.converged}")

                # Log primary predictor
                for term in ["svg_z_inter_dec_z", "svg_z_all_dec_z"]:
                    if term not in result.params:
                        continue
                    b, p = result.params[term], result.pvalues[term]
                    ci = result.conf_int()
                    lo, hi = ci.loc[term]
                    sig = (
                        "***"
                        if p < 0.001
                        else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    )
                    logger.info(
                        f"    {term}: β={b:.3f} [{lo:.3f}, {hi:.3f}], p={p:.4f} {sig}"
                    )
            except Exception as e:
                logger.error(f"  {name}: failed — {e}")
                results[name] = None

    return results


# ---------------------------------------------------------------------------
# Step 6: Output
# ---------------------------------------------------------------------------


def write_outputs(
    df: pd.DataFrame,
    corr_df: pd.DataFrame,
    lmm_results: dict,
    output_dir: Path,
    plot: bool = True,
) -> None:
    logger.info("Step 6: Writing outputs ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "exploratory_response_df.csv", index=False)
    corr_df.to_csv(output_dir / "per_participant_correlations.csv", index=False)
    logger.info(
        "  Written → exploratory_response_df.csv, per_participant_correlations.csv"
    )

    if lmm_results:
        coef_rows = []
        summary_lines = []
        for name, result in lmm_results.items():
            summary_lines.append(f"\n{'='*60}\n{name}\n{'='*60}")
            if result is None:
                summary_lines.append("SKIPPED / FAILED")
                continue
            summary_lines.append(str(result.summary()))
            ci = result.conf_int()
            for term in result.params.index:
                coef_rows.append(
                    {
                        "model": name,
                        "term": term,
                        "coef": result.params[term],
                        "std_err": result.bse[term],
                        "p": result.pvalues[term],
                        "ci_lower": ci.loc[term, 0],
                        "ci_upper": ci.loc[term, 1],
                    }
                )
        pd.DataFrame(coef_rows).to_csv(
            output_dir / "model_coefficients.csv", index=False
        )
        with open(output_dir / "model_summaries.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        logger.info("  Written → model_coefficients.csv, model_summaries.txt")

    if plot:
        _scatter_plot(df, corr_df, output_dir / "scatter_plot.png")


def _scatter_plot(
    df: pd.DataFrame,
    corr_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    One panel per participant + one pooled-partial panel.
    Each panel: alnum_len (y) vs svg_z_inter_dec (x), coloured by svg_z_all_dec.
    """
    subjects = sorted(df["SubjectID"].unique())
    n_panels = len(subjects) + 1  # +1 for pooled
    ncols = min(n_panels, 3)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False
    )
    axes_flat = [ax for row in axes for ax in row]

    col = "svg_z_inter_dec"
    col_label = "svg_z_inter_dec (decoding)"

    def _annotate(ax, rho, p, n):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(
            0.05,
            0.95,
            f"ρ={rho:.2f}, p={p:.3f} {sig}\nn={n}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    # Per-participant panels
    for i, sid in enumerate(subjects):
        ax = axes_flat[i]
        sub = df[df["SubjectID"] == sid][["alnum_len", col]].dropna()
        ax.scatter(
            sub[col],
            sub["alnum_len"],
            alpha=0.7,
            s=40,
            color="#2171b5",
            edgecolors="white",
            linewidth=0.5,
        )

        if len(sub) >= 3:
            m, b = np.polyfit(sub[col], sub["alnum_len"], 1)
            x_line = np.linspace(sub[col].min(), sub[col].max(), 100)
            ax.plot(
                x_line,
                m * x_line + b,
                color="#084594",
                linewidth=1.5,
                linestyle="--",
                alpha=0.8,
            )
            corr_row = corr_df[
                (corr_df["SubjectID"] == sid)
                & (corr_df["svg_variant"] == "interactional")
            ]
            if not corr_row.empty:
                _annotate(
                    ax, corr_row["rho"].values[0], corr_row["p"].values[0], len(sub)
                )

        short_sid = sid.split("Experiment-")[-1] if "Experiment-" in sid else sid
        ax.set_title(short_sid, fontsize=10)
        ax.set_xlabel(col_label, fontsize=8)
        ax.set_ylabel("Response length (alphanum chars)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # Pooled partial panel
    ax_pool = axes_flat[len(subjects)]
    centred_col = f"{col}_centred"
    if centred_col in df.columns:
        sub = df[["alnum_len_centred", centred_col]].dropna()
        ax_pool.scatter(
            sub[centred_col],
            sub["alnum_len_centred"],
            alpha=0.6,
            s=35,
            color="#99000d",
            edgecolors="white",
            linewidth=0.5,
        )
        if len(sub) >= 5:
            m, b = np.polyfit(sub[centred_col], sub["alnum_len_centred"], 1)
            x_line = np.linspace(sub[centred_col].min(), sub[centred_col].max(), 100)
            ax_pool.plot(
                x_line,
                m * x_line + b,
                color="#67000d",
                linewidth=1.5,
                linestyle="--",
                alpha=0.8,
            )
            corr_row = corr_df[
                (corr_df["SubjectID"] == "pooled_partial")
                & (corr_df["svg_variant"] == "interactional")
            ]
            if not corr_row.empty:
                _annotate(
                    ax_pool,
                    corr_row["rho"].values[0],
                    corr_row["p"].values[0],
                    len(sub),
                )
        ax_pool.set_title("Pooled (within-participant centred)", fontsize=10)
        ax_pool.set_xlabel(f"{col_label} (centred)", fontsize=8)
        ax_pool.set_ylabel("Response length (centred)", fontsize=8)
        ax_pool.spines[["top", "right"]].set_visible(False)

    # Hide unused panels
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    plt.suptitle(
        "Decoding relational alignment vs exploratory response length\n"
        "(svg_z_inter_dec; alnum chars only)",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Written → {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATEFMT,
    )

    parser = argparse.ArgumentParser(
        description="Module 5: Decoding SVG vs exploratory response length."
    )
    parser.add_argument(
        "--features",
        default=str(config.OUTPUT_FEATURES_DIR / "trial_features_all.csv"),
        help="Path to trial_features_all.csv",
    )
    parser.add_argument(
        "--behavioral-dir",
        default=str(config.OUTPUT_BEHAVIORAL_DIR),
        help="Directory containing *_exploratory.csv files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.OUTPUT_DIR / "exploratory_response"),
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Module 5: Decoding SVG vs exploratory response length")
    logger.info("=" * 60)

    exploratory = load_exploratory_responses(Path(args.behavioral_dir))
    df = build_analysis_df(exploratory, Path(args.features))
    df = prepare(df)
    corr_df = run_correlations(df)
    lmm_results = run_lmm(df)
    write_outputs(
        df, corr_df, lmm_results, Path(args.output_dir), plot=not args.no_plot
    )

    logger.info("\n" + "=" * 60)
    logger.info("Module 5 complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
