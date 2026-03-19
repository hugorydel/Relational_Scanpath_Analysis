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
    DEC_BETWEEN_COVARIATES,
    DEC_COVARIATES,
    DV_OBJECTS,
    DV_RELATIONS,
    DV_TOTAL,
    ENC_BETWEEN_COVARIATES,
    ENC_COVARIATES,
    MODEL_SPECS,
    PILOT_SUBJ_THRESHOLD,
)

logger = logging.getLogger(__name__)

_COV_ENC_Z      = " + ".join(f"{c}_z" for c in ENC_COVARIATES)
_COV_BETWEEN_Z  = " + ".join(f"{c}_z" for c in ENC_BETWEEN_COVARIATES)
_COV_FULL_Z     = f"{_COV_ENC_Z} + {_COV_BETWEEN_Z}"

_COV_DEC_Z      = " + ".join(f"{c}_z" for c in DEC_COVARIATES)
_COV_DEC_BTW_Z  = " + ".join(f"{c}_z" for c in DEC_BETWEEN_COVARIATES)
_COV_DEC_FULL_Z = f"{_COV_DEC_Z} + {_COV_DEC_BTW_Z}"


# ---------------------------------------------------------------------------
# Formula construction
# ---------------------------------------------------------------------------


def _formula_for(name: str, primary_pred: str) -> tuple:
    """
    Return (rhs_formula, dv_column).
    rhs_formula does not include the DV.
    """
    # H1: intercept test — DV is the SVG score for that phase
    if name == "H1_svg_enc":
        return "1", "svg_z_enc"
    if name == "H1_svg_dec":
        return "1", "svg_z_dec"

    # H2 encoding models
    if name == "H2_total":
        return f"{primary_pred} + {_COV_FULL_Z}", DV_TOTAL
    if name == "H2_relations":
        return f"{primary_pred} + {_COV_FULL_Z}", DV_RELATIONS
    if name == "H2_objects":
        return f"{primary_pred} + {_COV_FULL_Z}", DV_OBJECTS

    # Exploratory dissociation
    if name == "EXP_dissociation":
        return f"{primary_pred} + {_COV_FULL_Z}", "score"

    # H2 decoding models
    if name == "H2_dec_total":
        return f"{primary_pred} + {_COV_DEC_FULL_Z}", DV_TOTAL
    if name == "H2_dec_relations":
        return f"{primary_pred} + {_COV_DEC_FULL_Z}", DV_RELATIONS
    if name == "H2_dec_objects":
        return f"{primary_pred} + {_COV_DEC_FULL_Z}", DV_OBJECTS

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
    # Intercept-only H1 models: StimID vc causes SE collapse — use SubjectID only
    is_intercept_only = full_formula.split("~")[1].strip() == "1"
    vc = {} if is_intercept_only else {"StimID": "0 + C(StimID)"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.mixedlm(
                full_formula,
                data=model_df,
                groups="SubjectID",
                vc_formula=vc if vc else None,
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


import types
from scipy import stats as scipy_stats


def _fit_h1_ttest(name: str, svg_col: str, df: pd.DataFrame) -> tuple:
    """
    One-sample t-test of per-participant mean SVG against 0.

    Aggregates to participant means first (removes within-participant
    dependence), then tests whether the mean of those means > 0.

    Returns a lightweight namespace that mimics the statsmodels result
    interface used in summarise() and _descriptives():
        .params["Intercept"]   — grand mean
        .pvalues["Intercept"]  — two-tailed p (one-sided is p/2)
        .tvalues["Intercept"]  — t statistic
        .bse["Intercept"]      — SE of the mean
        .conf_int()            — DataFrame with [0, 1] columns
        .summary()             — text summary string
        .converged             — True
    """
    vals = df[svg_col].dropna()
    if len(vals) < 2:
        logger.warning(f"  {name}: insufficient data for t-test — skipping.")
        return None, "skipped"

    # Aggregate to participant means
    subj_means = df.groupby("SubjectID")[svg_col].mean().dropna()
    n = len(subj_means)
    grand_mean = subj_means.mean()
    se = subj_means.sem()
    t_stat, p_two = scipy_stats.ttest_1samp(subj_means, 0)
    ci = scipy_stats.t.interval(0.95, df=n - 1, loc=grand_mean, scale=se)

    pct_above = 100 * (vals > 0).mean()
    logger.info(
        f"  {name}: one-sample t-test on {n} participant means\n"
        f"    grand mean={grand_mean:.3f}, SE={se:.3f}, "
        f"t({n-1})={t_stat:.3f}, p={p_two:.4f} (two-tailed)\n"
        f"    {pct_above:.1f}% of trials above permutation baseline"
    )

    # Build duck-typed result namespace
    result = types.SimpleNamespace()
    result.params    = {"Intercept": grand_mean}
    result.pvalues   = {"Intercept": p_two}
    result.tvalues   = {"Intercept": t_stat}
    result.bse       = {"Intercept": se}
    result.converged = True
    result.nobs      = n
    result.df_resid  = n - 1

    def _conf_int(alpha=0.05):
        ci_lo, ci_hi = scipy_stats.t.interval(
            1 - alpha, df=n - 1, loc=grand_mean, scale=se
        )
        return pd.DataFrame(
            {"0": [ci_lo], "1": [ci_hi]}, index=["Intercept"]
        )
    result.conf_int = _conf_int

    def _summary():
        return (
            f"H1 one-sample t-test (participant means vs 0)\n"
            f"  N participants : {n}\n"
            f"  Grand mean SVG : {grand_mean:.4f}  (SE={se:.4f})\n"
            f"  t({n-1})        : {t_stat:.3f}\n"
            f"  p (two-tailed) : {p_two:.4f}\n"
            f"  95% CI         : [{ci[0]:.4f}, {ci[1]:.4f}]\n"
            f"  % trials > 0   : {pct_above:.1f}%"
        )
    result.summary = _summary

    return result, "ttest"


def fit_all_models(filtered: dict) -> dict:
    """
    Iterate MODEL_SPECS, fit each model, return results dict.
    Keys: model name → (result | None, mode: str)

    H1 models use a one-sample t-test on per-participant means.
    All other models use LMM (or OLS pilot fallback).
    """
    logger.info("\nStep 5: Fitting models ...")
    results = {}

    _H1_SVG_COLS = {
        "H1_svg_enc": ("enc", "svg_z_enc"),
        "H1_svg_dec": ("dec", "svg_z_dec"),
    }

    for name, primary_pred, desc, table_key, group in MODEL_SPECS:
        logger.info(f"\n  [{group}] {name}: {desc}")
        df = filtered.get(table_key, pd.DataFrame())
        if df.empty:
            logger.warning(f"  {name}: table \'{table_key}\' is empty — skipping.")
            results[name] = (None, "skipped")
            continue

        # H1: one-sample t-test on participant means
        if name in _H1_SVG_COLS:
            _, svg_col = _H1_SVG_COLS[name]
            results[name] = _fit_h1_ttest(name, svg_col, df)
            continue

        formula, dv = _formula_for(name, primary_pred)
        logger.info(f"    {dv} ~ {formula}")
        results[name] = _fit_one(name, formula, dv, df)

    return results