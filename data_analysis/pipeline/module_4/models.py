"""
pipeline/module_4/models.py
============================
Step 5: Model fitting.

All models are encoding-phase only. DVs are proportion columns (0-1).

Pilot mode  (n_subj < PILOT_SUBJ_THRESHOLD): OLS + C(SubjectID)
Full mode   (n_subj >= PILOT_SUBJ_THRESHOLD): LMM + (1|SubjectID) + (1|StimID)

The dissociation model (EXP_dissociation) is handled separately via a
long-format table with a memory_type binary factor.
"""

import logging
import re
import warnings

import pandas as pd
import statsmodels.formula.api as smf

from .constants import (
    DV_OBJECTS,
    DV_RELATIONS,
    DV_TOTAL,
    ENC_BETWEEN_COVARIATES,
    ENC_COVARIATES,
    MODEL_SPECS,
    PILOT_SUBJ_THRESHOLD,
)

logger = logging.getLogger(__name__)

_COV_ENC_Z = " + ".join(f"{c}_z" for c in ENC_COVARIATES)
_COV_BETWEEN_Z = " + ".join(f"{c}_z" for c in ENC_BETWEEN_COVARIATES)
_COV_FULL_Z = f"{_COV_ENC_Z} + {_COV_BETWEEN_Z}"


# ---------------------------------------------------------------------------
# Formula construction
# ---------------------------------------------------------------------------


def _formula_for(name: str, primary_pred: str) -> tuple:
    """
    Return (rhs_formula, dv_column).
    rhs_formula does not include the DV.
    """

    # H1: DV is the SVG score itself; intercept test
    if name.startswith("H1"):
        dv = "svg_z_enc"
        rhs = "1"
        return rhs, dv

    # H2 / Exploratory — main encoding models
    # Primary predictor: within-image centred SVG (participant deviation from image mean)
    # Full covariates: within-trial covariates + between-image svg_z_enc_image_mean
    if name == "H2_total":
        return f"{primary_pred} + {_COV_FULL_Z}", DV_TOTAL

    if name == "H2_relations":
        return f"{primary_pred} + {_COV_FULL_Z}", DV_RELATIONS

    if name == "H2_objects":
        return f"{primary_pred} + {_COV_FULL_Z}", DV_OBJECTS

    if name == "EXP_dissociation":
        return (
            f"{primary_pred} + {_COV_FULL_Z}",
            "score",
        )

    raise ValueError(f"Cannot determine formula for model: {name}")


# ---------------------------------------------------------------------------
# Single-model fitting
# ---------------------------------------------------------------------------


def _fit_one(name: str, formula: str, dv: str, df: pd.DataFrame) -> tuple:
    """
    Fit one model. Returns (result | None, mode: str).
    mode is "ols_pilot" or "lmm".
    """
    full_formula = f"{dv} ~ {formula}"

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
    req = list({dv, "SubjectID", "StimID"} | {t for t in tokens if t in df.columns})
    model_df = df[[c for c in req if c in df.columns]].dropna()

    n_obs = len(model_df)
    n_subj = model_df["SubjectID"].nunique()
    n_stim = model_df["StimID"].nunique() if "StimID" in model_df else 0
    logger.info(f"  {name}: n={n_obs} obs, {n_subj} subj, {n_stim} stim")

    if n_obs < 20 or n_subj < 2:
        logger.warning(f"  {name}: insufficient data — skipping.")
        return None, "skipped"

    if n_subj < PILOT_SUBJ_THRESHOLD:
        return _fit_ols_pilot(name, full_formula, model_df)
    else:
        return _fit_lmm(name, full_formula, model_df)


def _fit_ols_pilot(name: str, full_formula: str, model_df: pd.DataFrame) -> tuple:
    pilot_formula = full_formula + " + C(SubjectID)"
    logger.warning(
        f"  {name}: n_subj < {PILOT_SUBJ_THRESHOLD} — "
        "using OLS + C(SubjectID) (pilot mode)."
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = smf.ols(pilot_formula, data=model_df).fit()
        except Exception as e:
            logger.error(f"  {name}: OLS fit failed — {e}")
            return None, "failed"
    logger.info(f"    OLS R²={result.rsquared:.3f}, F p={result.f_pvalue:.4f}")
    return result, "ols_pilot"


def _fit_lmm(name: str, full_formula: str, model_df: pd.DataFrame) -> tuple:
    vc = {"StimID": "0 + C(StimID)"}
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
            logger.error(f"  {name}: LMM failed — {e}")
            return None, "failed"
    logger.info(f"    LMM converged: {result.converged}")
    return result, "lmm"


# ---------------------------------------------------------------------------
# Fit all models
# ---------------------------------------------------------------------------


def fit_all_models(filtered: dict) -> dict:
    """
    Iterate MODEL_SPECS, fit each model, return results dict.
    Keys: model name → (result | None, mode: str)
    """
    logger.info("\nStep 5: Fitting models ...")
    results = {}

    for name, primary_pred, desc, table_key, group in MODEL_SPECS:
        logger.info(f"\n  [{group}] {name}: {desc}")
        df = filtered.get(table_key, pd.DataFrame())
        if df.empty:
            logger.warning(f"  {name}: table '{table_key}' is empty — skipping.")
            results[name] = (None, "skipped")
            continue

        formula, dv = _formula_for(name, primary_pred)
        logger.info(f"    {dv} ~ {formula}")
        results[name] = _fit_one(name, formula, dv, df)

    return results
