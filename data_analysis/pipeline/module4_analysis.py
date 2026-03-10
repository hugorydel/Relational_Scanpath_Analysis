"""
module4_analysis.py
===================
Module 4: Mixed-effects analysis of relational retrieval and episodic memory.

Hypotheses
----------
H1 — Decoding relational alignment exists
    Participants predictably trace relational structure during episodic
    retrieval, above and beyond low-level fixation characteristics.
    Test: svg_z_inter_dec reliably > 0 (raw H1a, then covariate-adjusted H1b).

H2 — Decoding relational alignment scaffolds episodic memory
    Higher relational alignment during retrieval predicts better free-recall
    of relational details (primary) and object details (secondary).
    Primary  : svg_z_inter_dec → n_relational_correct
    Secondary: svg_z_inter_dec → n_objects_correct

Exploratory
-----------
    Confabulation : dec SVG → n_relational_incorrect
    Writing length: dec SVG → writing_length
    Encoding SVG + LCS/tau → n_relational_correct

Memory scoring
--------------
    Manually scored per trial in data_scoring/memory_scores.csv.
    Columns: SubjectID, StimID,
             n_relational_correct, n_relational_incorrect,
             n_objects_correct, n_objects_incorrect, empty_response (0/1).
    Derived at load time: n_relational_total, any_relational_correct.
    writing_length is derived from the free-response text in module 3
    and carried through trial_features_all.csv.

Pipeline
--------
Step 1  — Load trial_features_all.csv + memory_scores.csv
Step 2  — Build analysis tables
            dec      : decoding rows, SVG columns renamed *_dec, memory scores joined
            enc      : encoding rows, SVG/covariate columns renamed *_enc
            wide     : enc + dec joined for exploratory models
Step 3  — Hypothesis-specific exclusions
Step 4  — Standardise predictors within each filtered table
Step 5  — Fit models
Step 6  — Summarise and write outputs
            output/analysis/analysis_*.csv
            output/analysis/model_coefficients.csv
            output/analysis/model_summaries.txt
            output/analysis/forest_plot.png

Usage
-----
    python module4_analysis.py
    python module4_analysis.py --input path/to/trial_features_all.csv
    python module4_analysis.py --scores path/to/memory_scores.csv
    python module4_analysis.py --no-plot
"""

import argparse
import logging
import re
import sys
import warnings
from pathlib import Path

import config
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum n_shared_enc_dec for replay models.
# Set to 2 for pilot (n=3). Raise to 3 for final analysis.
MIN_N_SHARED_REPLAY = 2

DEC_COVARIATES = ["n_fixations_dec", "aoi_prop_dec", "mean_salience_dec"]
ENC_COVARIATES = ["n_fixations_enc", "aoi_prop_enc", "mean_salience_enc"]
COVARIATES = DEC_COVARIATES  # backwards-compatible alias for run_pipeline.py

# Memory DVs — from manual scoring joined at load time
DV_RELATIONAL = "n_relational_correct"
DV_OBJECTS = "n_objects_correct"
DV_CONFAB = "n_relational_incorrect"
DV_LENGTH = "writing_length"

# Default path for manual memory scores
DEFAULT_SCORES_PATH = config.ROOT_DIR / "data_scoring" / "memory_scores.csv"

# (name, primary_predictor, description, table_key, hypothesis_group)
MODEL_SPECS = [
    # H1 — decoding SVG as outcome
    (
        "H1a_svg_inter",
        "1",
        "H1a raw: Decoding interactional SVG — intercept only",
        "dec_inter",
        "H1",
    ),
    (
        "H1a_svg_all",
        "1",
        "H1a raw: Decoding all-edges SVG — intercept only",
        "dec_all",
        "H1",
    ),
    (
        "H1b_svg_inter",
        "covariates",
        "H1b adjusted: Decoding interactional SVG — covariate-adjusted intercept",
        "dec_inter",
        "H1",
    ),
    (
        "H1b_svg_all",
        "covariates",
        "H1b adjusted: Decoding all-edges SVG — covariate-adjusted intercept",
        "dec_all",
        "H1",
    ),
    # H2 primary — dec SVG → relational recall
    (
        "H2_relational_inter",
        "svg_z_inter_dec_z",
        "H2 primary: Decoding interactional SVG → relational recall",
        "dec_inter",
        "H2",
    ),
    (
        "H2_relational_all",
        "svg_z_all_dec_z",
        "H2 primary (all): Decoding all-edges SVG → relational recall",
        "dec_all",
        "H2",
    ),
    # H2 secondary — dec SVG → object recall (dissociation)
    (
        "H2_objects_inter",
        "svg_z_inter_dec_z",
        "H2 secondary: Decoding interactional SVG → object recall",
        "dec_inter",
        "H2",
    ),
    (
        "H2_objects_all",
        "svg_z_all_dec_z",
        "H2 secondary (all): Decoding all-edges SVG → object recall",
        "dec_all",
        "H2",
    ),
    # Exploratory — confabulation (does SVG predict wrong relational details?)
    (
        "EXP_confab_inter",
        "svg_z_inter_dec_z",
        "Exploratory: Decoding interactional SVG → relational confabulation",
        "dec_inter",
        "Exploratory",
    ),
    (
        "EXP_confab_all",
        "svg_z_all_dec_z",
        "Exploratory: Decoding all-edges SVG → relational confabulation",
        "dec_all",
        "Exploratory",
    ),
    # Exploratory — writing length (overall recall fluency)
    (
        "EXP_length_inter",
        "svg_z_inter_dec_z",
        "Exploratory: Decoding interactional SVG → writing length",
        "dec_inter",
        "Exploratory",
    ),
    (
        "EXP_length_all",
        "svg_z_all_dec_z",
        "Exploratory: Decoding all-edges SVG → writing length",
        "dec_all",
        "Exploratory",
    ),
    # Exploratory — encoding → relational recall
    (
        "EXP_enc_svg_inter",
        "svg_z_inter_enc_z",
        "Exploratory: Encoding interactional SVG → relational recall",
        "enc_inter",
        "Exploratory",
    ),
    (
        "EXP_enc_svg_all",
        "svg_z_all_enc_z",
        "Exploratory: Encoding all-edges SVG → relational recall",
        "enc_all",
        "Exploratory",
    ),
    (
        "EXP_enc_lcs",
        "lcs_enc_dec_z",
        "Exploratory: LCS sequence overlap → relational recall",
        "enc_all",
        "Exploratory",
    ),
    (
        "EXP_enc_combined",
        "svg_z_inter_enc_z + lcs_enc_dec_z",
        "Exploratory: Encoding SVG + LCS jointly → relational recall",
        "enc_inter",
        "Exploratory",
    ),
    # Exploratory — replay quality → relational recall
    (
        "EXP_replay_lcs",
        "lcs_enc_dec_z",
        "Exploratory: LCS (replay-quality filtered) → relational recall",
        "replay",
        "Exploratory",
    ),
    (
        "EXP_replay_tau",
        "tau_enc_dec_z",
        "Exploratory: Tau (replay-quality filtered) → relational recall",
        "replay",
        "Exploratory",
    ),
]


# ---------------------------------------------------------------------------
# Step 1: Load
# ---------------------------------------------------------------------------


def load_data(input_path: Path) -> pd.DataFrame:
    logger.info(f"Step 1: Loading {input_path.name} ...")
    if not input_path.exists():
        raise FileNotFoundError(f"Features file not found: {input_path}")
    df = pd.read_csv(input_path, dtype={"StimID": str, "SubjectID": str})
    logger.info(
        f"  Loaded {len(df)} rows, {df['SubjectID'].nunique()} participants, "
        f"{df['StimID'].nunique()} stimuli."
    )
    return df


def load_memory_scores(scores_path: Path) -> pd.DataFrame:
    """
    Load manually scored memory data and compute derived columns.

    Input columns (from scorer):
        SubjectID, StimID,
        n_relational_correct, n_relational_incorrect,
        n_objects_correct, n_objects_incorrect,
        empty_response  (0/1)

    Derived columns added here:
        n_relational_total      = n_relational_correct + n_relational_incorrect
        any_relational_correct  = 1 if n_relational_correct >= 1 else 0
    """
    logger.info(f"  Loading memory scores from {scores_path.name} ...")
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Memory scores file not found: {scores_path}\n"
            f"Score responses and save to {scores_path} before running Module 4."
        )

    scores = pd.read_csv(scores_path, dtype={"StimID": str, "SubjectID": str})

    required = [
        "SubjectID",
        "StimID",
        "n_relational_correct",
        "n_relational_incorrect",
        "n_objects_correct",
        "n_objects_incorrect",
        "empty_response",
    ]
    missing = [c for c in required if c not in scores.columns]
    if missing:
        raise ValueError(f"Memory scores file missing columns: {missing}")

    scores["n_relational_total"] = (
        scores["n_relational_correct"] + scores["n_relational_incorrect"]
    )
    scores["any_relational_correct"] = (scores["n_relational_correct"] >= 1).astype(int)

    logger.info(
        f"  Loaded {len(scores)} scored trials, "
        f"{scores['SubjectID'].nunique()} participants."
    )
    return scores


# ---------------------------------------------------------------------------
# Step 2: Build analysis tables
# ---------------------------------------------------------------------------


def build_analysis_tables(df: pd.DataFrame, memory_scores: pd.DataFrame) -> dict:
    logger.info("Step 2: Building analysis tables ...")

    enc_raw = df[df["Phase"] == "encoding"].copy()
    dec_raw = df[df["Phase"] == "decoding"].copy()

    # Decoding table — rename phase-specific columns with _dec suffix
    dec = dec_raw.rename(
        columns={
            "svg_z_all": "svg_z_all_dec",
            "svg_z_inter": "svg_z_inter_dec",
            "n_fixations": "n_fixations_dec",
            "aoi_prop": "aoi_prop_dec",
            "mean_salience": "mean_salience_dec",
            "low_n": "low_n_dec",
            "n_transitions": "n_transitions_dec",
        }
    ).reset_index(drop=True)

    # Join manual memory scores onto decoding rows
    dec = dec.merge(memory_scores, on=["SubjectID", "StimID"], how="left")
    n_unscored = dec[DV_RELATIONAL].isna().sum()
    if n_unscored > 0:
        logger.warning(
            f"  {n_unscored} decoding row(s) have no memory score — "
            f"check that memory_scores.csv covers all participants/stimuli."
        )

    # Encoding table — rename with _enc suffix
    enc = enc_raw.rename(
        columns={
            "svg_z_all": "svg_z_all_enc",
            "svg_z_inter": "svg_z_inter_enc",
            "n_fixations": "n_fixations_enc",
            "aoi_prop": "aoi_prop_enc",
            "mean_salience": "mean_salience_enc",
            "low_n": "low_n_enc",
            "n_transitions": "n_transitions_enc",
        }
    ).reset_index(drop=True)

    # Wide table for exploratory enc → memory models
    MEMORY_COLS = [
        DV_RELATIONAL,
        DV_OBJECTS,
        DV_CONFAB,
        "n_objects_incorrect",
        "n_relational_total",
        "any_relational_correct",
        "empty_response",
        DV_LENGTH,
    ]
    enc_keep = [
        "SubjectID",
        "StimID",
        "svg_z_all_enc",
        "svg_z_inter_enc",
        "n_fixations_enc",
        "aoi_prop_enc",
        "mean_salience_enc",
        "low_n_enc",
    ]
    dec_keep = [
        "SubjectID",
        "StimID",
        "svg_z_all_dec",
        "svg_z_inter_dec",
        "n_fixations_dec",
        "aoi_prop_dec",
        "mean_salience_dec",
        "low_n_dec",
        "n_shared_enc_dec",
        "lcs_enc_dec",
        "tau_enc_dec",
    ] + [c for c in MEMORY_COLS if c in dec.columns]

    enc_sub = enc[[c for c in enc_keep if c in enc.columns]]
    dec_sub = dec[[c for c in dec_keep if c in dec.columns]]
    wide = enc_sub.merge(dec_sub, on=["SubjectID", "StimID"], how="inner")

    logger.info(f"  dec table:  {len(dec)} rows  ({n_unscored} unscored)")
    logger.info(f"  enc table:  {len(enc)} rows")
    logger.info(f"  wide table: {len(wide)} rows")

    return {"dec": dec, "enc": enc, "wide": wide}


# ---------------------------------------------------------------------------
# Step 3: Hypothesis-specific exclusions
# ---------------------------------------------------------------------------


def apply_exclusions(tables: dict) -> dict:
    logger.info("Step 3: Applying hypothesis-specific exclusions ...")

    dec = tables["dec"]
    wide = tables["wide"]

    def _log(name, before, after, reason):
        logger.info(
            f"  {name}: {before} → {after} " f"(removed {before - after}: {reason})"
        )

    # H1 / H2 trial-level: filter on decoding low_n only
    n = len(dec)
    dec_inter = dec[~dec["low_n_dec"]].copy()
    dec_all = dec[~dec["low_n_dec"]].copy()
    _log("dec_inter / dec_all", n, len(dec_inter), "low_n_dec=True")

    # Exploratory enc → memory: filter wide table on encoding low_n only
    n_w = len(wide)
    enc_inter = wide[~wide["low_n_enc"]].copy()
    enc_all = wide[~wide["low_n_enc"]].copy()
    _log("enc_inter / enc_all", n_w, len(enc_inter), "low_n_enc=True")

    # Exploratory replay: additionally require n_shared >= MIN and non-null tau
    replay = wide[
        (~wide["low_n_dec"])
        & (~wide["low_n_enc"])
        & (wide["n_shared_enc_dec"] >= MIN_N_SHARED_REPLAY)
    ].copy()
    n_before_tau = len(replay)
    replay = replay[replay["tau_enc_dec"].notna()].copy()
    _log(
        "replay",
        n_w,
        n_before_tau,
        f"low_n_dec + low_n_enc + n_shared < {MIN_N_SHARED_REPLAY}",
    )
    _log("replay", n_before_tau, len(replay), "missing tau_enc_dec")

    logger.info(
        f"  Note: MIN_N_SHARED_REPLAY={MIN_N_SHARED_REPLAY} (pilot). "
        f"Raise to 3 for final analysis."
    )

    return {
        "dec_inter": dec_inter,
        "dec_all": dec_all,
        "enc_inter": enc_inter,
        "enc_all": enc_all,
        "replay": replay,
    }


# ---------------------------------------------------------------------------
# Step 4: Standardise
# ---------------------------------------------------------------------------


def standardise_tables(filtered: dict) -> dict:
    """
    Z-score continuous predictors within each filtered table.
    Global standardisation within table: intercept in H1 models equals the
    mean SVG score at an average (covariate=0) trial.
    Adds '{col}_z' columns; originals kept.
    """
    logger.info("Step 4: Standardising predictors ...")

    cols_by_table = {
        "dec_inter": ["svg_z_inter_dec"] + DEC_COVARIATES,
        "dec_all": ["svg_z_all_dec"] + DEC_COVARIATES,
        "enc_inter": ["svg_z_inter_enc", "lcs_enc_dec"] + ENC_COVARIATES,
        "enc_all": ["svg_z_all_enc", "lcs_enc_dec"] + ENC_COVARIATES,
        "replay": ["lcs_enc_dec", "tau_enc_dec"] + DEC_COVARIATES,
    }

    for key, cols in cols_by_table.items():
        df = filtered[key]
        for col in cols:
            if col not in df.columns:
                continue
            mu, sd = df[col].mean(), df[col].std()
            df[f"{col}_z"] = (df[col] - mu) / sd if sd > 0 else 0.0
        filtered[key] = df

    return filtered


# ---------------------------------------------------------------------------
# Step 5: Fit models
# ---------------------------------------------------------------------------


def _fit_lmm(
    name: str,
    formula: str,
    dv: str,
    df: pd.DataFrame,
    vc_trial: bool = False,
) -> tuple:
    """
    Returns (result, trial_re_used: bool).
    Fit one LMM:  dv ~ formula + (1|SubjectID) + (1|StimID) [+ (1|TrialID)]

    SubjectID is passed as `groups`; StimID and optionally TrialID are
    crossed random intercepts via vc_formula.
    """
    full_formula = f"{dv} ~ {formula}"

    # Identify required columns from the formula
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
    req = list({dv, "SubjectID", "StimID"} | {t for t in tokens if t in df.columns})
    if vc_trial and "TrialID" in df.columns:
        req.append("TrialID")

    model_df = df[[c for c in req if c in df.columns]].dropna()

    n_obs = len(model_df)
    n_subj = model_df["SubjectID"].nunique()
    n_stim = model_df["StimID"].nunique() if "StimID" in model_df else 0

    logger.info(f"  {name}: n={n_obs} obs, {n_subj} subj, {n_stim} stim")

    if n_obs < 20 or n_subj < 2:
        logger.warning(f"  {name}: insufficient data — skipping.")
        return None, False

    vc = {"StimID": "0 + C(StimID)"}
    if vc_trial and "TrialID" in model_df.columns:
        vc["TrialID"] = "0 + C(TrialID)"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.mixedlm(
                full_formula,
                data=model_df,
                groups="SubjectID",
                vc_formula=vc,
            )
            result = model.fit(reml=True, method="lbfgs", maxiter=300)
        except Exception as e:
            # TrialID RE often causes a singular matrix at small n because
            # each trial has exactly 2 observations (Q1 + Q2), making the
            # variance component unidentifiable. Fall back to SubjectID +
            # StimID only and flag it clearly.
            if vc_trial and "TrialID" in vc and "ingular" in str(e):
                logger.warning(
                    f"  {name}: TrialID RE caused singular matrix "
                    f"(expected at n<10 participants) — retrying without TrialID RE. "
                    f"Add TrialID RE for final analysis."
                )
                vc_fallback = {k: v for k, v in vc.items() if k != "TrialID"}
                try:
                    model = smf.mixedlm(
                        full_formula,
                        data=model_df,
                        groups="SubjectID",
                        vc_formula=vc_fallback if vc_fallback else None,
                    )
                    result = model.fit(reml=True, method="lbfgs", maxiter=300)
                    vc_trial = False  # TrialID RE was dropped in fallback
                except Exception as e2:
                    logger.error(f"  {name}: fallback also failed — {e2}")
                    return None, False
            else:
                logger.error(f"  {name}: fitting failed — {e}")
                return None, False

    logger.info(f"    Converged: {result.converged}")
    return result, vc_trial


def _formula_for(name: str, primary_pred: str) -> tuple[str, str, bool]:
    """
    Return (formula_string, dv_column, use_trial_re).

    All H2 and Exploratory models are trial-level (no question table),
    so use_trial_re is always False.
    """
    cov_dec_z = " + ".join(f"{c}_z" for c in DEC_COVARIATES)
    cov_enc_z = " + ".join(f"{c}_z" for c in ENC_COVARIATES)

    # H1 models: DV is the SVG score itself
    if name.startswith("H1"):
        dv = "svg_z_inter_dec" if "inter" in name else "svg_z_all_dec"
        formula = "1" if primary_pred == "1" else cov_dec_z
        return formula, dv, False

    # H2 — relational recall (primary)
    if name.startswith("H2_relational"):
        return f"{primary_pred} + {cov_dec_z}", DV_RELATIONAL, False

    # H2 — object recall (secondary / dissociation)
    if name.startswith("H2_objects"):
        return f"{primary_pred} + {cov_dec_z}", DV_OBJECTS, False

    # Exploratory — confabulation
    if "confab" in name:
        return f"{primary_pred} + {cov_dec_z}", DV_CONFAB, False

    # Exploratory — writing length
    if "length" in name:
        return f"{primary_pred} + {cov_dec_z}", DV_LENGTH, False

    # Exploratory — encoding → relational recall
    if "enc" in name:
        if "combined" in name:
            formula = f"svg_z_inter_enc_z + lcs_enc_dec_z + {cov_enc_z}"
        elif "lcs" in name:
            formula = f"lcs_enc_dec_z + {cov_enc_z}"
        elif "inter" in name:
            formula = f"svg_z_inter_enc_z + {cov_enc_z}"
        else:
            formula = f"svg_z_all_enc_z + {cov_enc_z}"
        return formula, DV_RELATIONAL, False

    # Exploratory — replay quality → relational recall
    if "replay" in name:
        pred_z = "tau_enc_dec_z" if "tau" in name else "lcs_enc_dec_z"
        return f"{pred_z} + {cov_dec_z}", DV_RELATIONAL, False

    raise ValueError(f"Cannot determine formula for model: {name}")


def fit_all_models(filtered: dict) -> dict:
    logger.info("\nStep 5: Fitting models ...")
    results = {}

    for name, primary_pred, desc, table_key, group in MODEL_SPECS:
        logger.info(f"\n  [{group}] {name}: {desc}")
        df = filtered.get(table_key, pd.DataFrame())
        if df.empty:
            logger.warning(f"  {name}: table '{table_key}' is empty — skipping.")
            results[name] = (None, False)
            continue

        formula, dv, use_trial_re = _formula_for(name, primary_pred)
        logger.info(f"    {dv} ~ {formula}")
        results[name] = _fit_lmm(name, formula, dv, df, vc_trial=use_trial_re)
        # results[name] is now a (MixedLMResults | None, trial_re_used: bool) tuple

    return results


# ---------------------------------------------------------------------------
# Step 6: Summarise and output
# ---------------------------------------------------------------------------


def _extract_coef_table(name: str, result: MixedLMResults) -> pd.DataFrame:
    ci = result.conf_int()
    return pd.DataFrame(
        {
            "model": name,
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "z": result.tvalues.values,
            "p": result.pvalues.values,
            "ci_lower": ci.iloc[:, 0].values,
            "ci_upper": ci.iloc[:, 1].values,
        }
    )


def _h1_descriptives(filtered: dict) -> str:
    lines = [
        "\n=== H1 Descriptives: raw decoding SVG z-scores (before covariate adjustment) ==="
    ]
    for key, col, label in [
        ("dec_inter", "svg_z_inter_dec", "svg_z_inter_dec"),
        ("dec_all", "svg_z_all_dec", "svg_z_all_dec"),
    ]:
        df = filtered.get(key, pd.DataFrame())
        vals = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        if len(vals) == 0:
            lines.append(f"  {label}: no data")
            continue
        lines.append(
            f"  {label}: n={len(vals)}, mean={vals.mean():.3f}, "
            f"sd={vals.std():.3f}, median={vals.median():.3f}, "
            f"% > 0: {100*(vals > 0).mean():.1f}%"
        )
    return "\n".join(lines)


def summarise(
    results: dict,
    filtered: dict,
    output_dir: Path,
    plot: bool = True,
) -> pd.DataFrame:
    logger.info("\nStep 6: Writing outputs ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save analysis tables
    for key in ("dec_inter", "enc_all", "replay"):
        df = filtered.get(key, pd.DataFrame())
        if not df.empty:
            df.to_csv(output_dir / f"analysis_{key}.csv", index=False)
    logger.info("  Written → analysis_*.csv")

    all_coefs = []
    summary_lines = [_h1_descriptives(filtered)]

    for group in ["H1", "H2", "Exploratory"]:
        summary_lines.append(f"\n\n{'#'*60}\n# {group} MODELS\n{'#'*60}")
        for name, primary_pred, desc, _, grp in MODEL_SPECS:
            if grp != group:
                continue
            entry = results.get(name)
            result, trial_re_used = entry if entry else (None, False)
            summary_lines.append(f"\n{'='*60}\n{name}: {desc}\n{'='*60}")
            if result is None:
                summary_lines.append("SKIPPED / FAILED\n")
                continue

            coef_df = _extract_coef_table(name, result)
            all_coefs.append(coef_df)
            summary_lines.append(str(result.summary()))
            summary_lines.append(f"\nLog-likelihood: {result.llf:.4f}")
            summary_lines.append(f"Converged: {result.converged}")

            focus = (
                ["Intercept"]
                if group == "H1"
                else [
                    t
                    for t in coef_df["term"].str.strip()
                    if t.endswith("_z")
                    and not any(t == f"{c}_z" for c in DEC_COVARIATES + ENC_COVARIATES)
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
        _forest_plot(coef_all, output_dir / "forest_plot.png", results=results)

    return coef_all


def _forest_plot(
    coef_df: pd.DataFrame, output_path: Path, results: dict = None
) -> None:
    """Three-panel forest plot: H1 intercepts | H2 | Exploratory."""
    skip_terms = {f"{c}_z" for c in DEC_COVARIATES + ENC_COVARIATES} | {
        "Group Var",
        "StimID Var",
    }

    palette = {
        "H1a_svg_inter": "#084594",
        "H1b_svg_inter": "#4292c6",
        "H1a_svg_all": "#084594",
        "H1b_svg_all": "#4292c6",
        "H2_relational_inter": "#99000d",
        "H2_relational_all": "#ef3b2c",
        "H2_objects_inter": "#a63603",
        "H2_objects_all": "#fd8d3c",
    }

    groups = ["H1", "H2", "Exploratory"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))

    for ax, group in zip(axes, groups):
        specs = [s for s in MODEL_SPECS if s[4] == group]
        to_plot = []

        for name, _, _, _, _ in specs:
            sub = coef_df[coef_df["model"] == name].copy()
            if sub.empty:
                continue
            if group == "H1":
                sub = sub[sub["term"].str.strip() == "Intercept"]
            else:
                sub = sub[
                    ~sub["term"].str.strip().isin(skip_terms)
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
                        "color": palette.get(name, "#636363"),
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
        "Module 4 results \u2014 (1|SubjectID) + (1|StimID)",
        fontsize=9,
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
        description="Module 4: Mixed-effects analysis of relational retrieval → memory."
    )
    parser.add_argument(
        "--input",
        default=str(config.OUTPUT_FEATURES_DIR / "trial_features_all.csv"),
    )
    parser.add_argument(
        "--scores",
        default=str(DEFAULT_SCORES_PATH),
        help="Path to manually scored memory_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.OUTPUT_DIR / "analysis"),
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Module 4: Relational retrieval → episodic memory")
    logger.info("=" * 60)

    raw_df = load_data(Path(args.input))
    memory_scores = load_memory_scores(Path(args.scores))
    tables = build_analysis_tables(raw_df, memory_scores)
    filtered = apply_exclusions(tables)
    filtered = standardise_tables(filtered)
    results = fit_all_models(filtered)
    summarise(results, filtered, Path(args.output_dir), plot=not args.no_plot)

    logger.info("\n" + "=" * 60)
    logger.info("Module 4 complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
