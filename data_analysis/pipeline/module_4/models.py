"""
pipeline/module4/models.py
===========================
Step 5: Model fitting.

Contains the entire fitting strategy:
  - _formula_for : maps a MODEL_SPECS entry to a statsmodels formula string
  - _fit_one     : fits a single model with pilot/full-n branching
  - fit_all_models : iterates MODEL_SPECS and collects results

Pilot mode (n_subj < PILOT_SUBJ_THRESHOLD)
-------------------------------------------
statsmodels' REML C-level optimizer hard-crashes (segfault) before Python's
try/except can fire when n_subj is small.  Below threshold we fall back to
OLS with C(SubjectID) as a fixed effect; this controls for between-subject
variance without random-effects machinery and cannot crash.

Full mode (n_subj >= PILOT_SUBJ_THRESHOLD)
-------------------------------------------
LMM with (1|SubjectID) as groups and crossed (1|StimID) via vc_formula.
"""

import logging
import re
import warnings

import pandas as pd
import statsmodels.formula.api as smf

from .constants import (
    DEC_COVARIATES,
    DV_CONFAB,
    DV_LENGTH,
    DV_OBJECTS,
    DV_RELATIONAL,
    ENC_COVARIATES,
    MODEL_SPECS,
    PILOT_SUBJ_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formula construction
# ---------------------------------------------------------------------------


def _formula_for(name: str, primary_pred: str) -> tuple[str, str, bool]:
    """
    Return (formula_fragment, dv_column, use_trial_re).

    formula_fragment does NOT include the DV — _fit_one prepends it.
    use_trial_re is always False for trial-level models (all current specs).
    """
    cov_dec_z = " + ".join(f"{c}_z" for c in DEC_COVARIATES)
    cov_enc_z = " + ".join(f"{c}_z" for c in ENC_COVARIATES)

    # H1: DV is the SVG score itself; test whether intercept > 0
    if name.startswith("H1"):
        dv = "svg_z_inter_dec" if "inter" in name else "svg_z_all_dec"
        formula = "1" if primary_pred == "1" else cov_dec_z
        return formula, dv, False

    # H2 — relational recall (primary outcome)
    if name.startswith("H2_relational"):
        return f"{primary_pred} + {cov_dec_z}", DV_RELATIONAL, False

    # H2 — object recall (dissociation check)
    if name.startswith("H2_objects"):
        return f"{primary_pred} + {cov_dec_z}", DV_OBJECTS, False

    # Exploratory — confabulation
    if "confab" in name:
        return f"{primary_pred} + {cov_dec_z}", DV_CONFAB, False

    # Exploratory — writing length
    if "length" in name:
        return f"{primary_pred} + {cov_dec_z}", DV_LENGTH, False

    # Exploratory — encoding predictors → relational recall
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


# ---------------------------------------------------------------------------
# Single-model fitting
# ---------------------------------------------------------------------------


def _fit_one(
    name: str,
    formula: str,
    dv: str,
    df: pd.DataFrame,
    vc_trial: bool = False,
) -> tuple:
    """
    Fit one model and return (result, trial_re_used: bool).

    Selects between OLS (pilot) and LMM (full) based on n_subj.
    Returns (None, False) on failure or insufficient data.
    """
    full_formula = f"{dv} ~ {formula}"

    # Build minimal DataFrame (only columns needed by the formula)
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

    # ── Branch: pilot OLS vs full LMM ────────────────────────────────────
    if n_subj < PILOT_SUBJ_THRESHOLD:
        return _fit_ols_pilot(name, full_formula, dv, model_df)
    else:
        return _fit_lmm(name, full_formula, dv, model_df, vc_trial)


def _fit_ols_pilot(
    name: str,
    full_formula: str,
    dv: str,
    model_df: pd.DataFrame,
) -> tuple:
    """
    OLS + C(SubjectID) fixed effect for pilot runs (n_subj < threshold).

    C(SubjectID) absorbs between-subject variance without any random-effects
    machinery, so it cannot trigger the statsmodels C-level REML segfault.

    Standard (non-robust) SEs are used deliberately: HC3 calls into
    statsmodels' C extension and can itself segfault on Windows with small n.
    For pilot data (n<10 subjects) OLS p-values are descriptive anyway.
    """
    pilot_formula = full_formula + " + C(SubjectID)"
    logger.warning(
        f"  {name}: n_subj < {PILOT_SUBJ_THRESHOLD} — "
        "using OLS + C(SubjectID) fixed effect (pilot mode). "
        f"Switch to LMM for final analysis (n>={PILOT_SUBJ_THRESHOLD})."
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = smf.ols(pilot_formula, data=model_df).fit()
        except Exception as e:
            logger.error(f"  {name}: OLS pilot fit failed — {e}")
            return None, False
    logger.info(f"    OLS R²={result.rsquared:.3f}, F p={result.f_pvalue:.4f}")
    return result, False


def _fit_lmm(
    name: str,
    full_formula: str,
    dv: str,
    model_df: pd.DataFrame,
    vc_trial: bool,
) -> tuple:
    """
    Full LMM: (1|SubjectID) + crossed (1|StimID) [+ optional (1|TrialID)].

    Falls back to dropping TrialID RE if it causes a singular Hessian
    (expected when each trial has very few observations).
    """
    vc: dict = {"StimID": "0 + C(StimID)"}
    if vc_trial and "TrialID" in model_df.columns:
        vc["TrialID"] = "0 + C(TrialID)"

    def _try(vc_formula):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                full_formula,
                data=model_df,
                groups="SubjectID",
                vc_formula=vc_formula if vc_formula else None,
            )
            return model.fit(reml=True, method="lbfgs", maxiter=300)

    try:
        result = _try(vc)
    except Exception as e:
        if vc_trial and "TrialID" in vc and "ingular" in str(e):
            logger.warning(
                f"  {name}: TrialID RE singular — retrying without TrialID RE."
            )
            vc_no_trial = {k: v for k, v in vc.items() if k != "TrialID"}
            try:
                result = _try(vc_no_trial if vc_no_trial else None)
                vc_trial = False
            except Exception as e2:
                logger.error(f"  {name}: LMM fallback also failed — {e2}")
                return None, False
        else:
            logger.error(f"  {name}: LMM fitting failed — {e}")
            return None, False

    logger.info(f"    Converged: {result.converged}")
    return result, vc_trial


# ---------------------------------------------------------------------------
# Fit all models
# ---------------------------------------------------------------------------


def fit_all_models(filtered: dict) -> dict:
    """
    Iterate MODEL_SPECS, fit each model, and return a results dict.

    Returns
    -------
    dict mapping model name → (result | None, trial_re_used: bool)
    """
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
        results[name] = _fit_one(name, formula, dv, df, vc_trial=use_trial_re)

    return results
