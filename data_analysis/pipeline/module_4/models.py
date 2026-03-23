"""
pipeline/module_4/models.py
============================
Step 5: Model fitting.

All models are encoding-phase or decoding-phase only. DVs are proportion
columns (0-1).

H1 models are handled separately via one-sample t-tests on per-participant
means. All other models use LMM with crossed random effects:
    (1 | SubjectID) + (1 | StimID)

The dissociation model (EXP_dissociation) is handled via the long-format table
with a memory_type factor.
"""

import logging
import re
import types
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

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

_COV_ENC_Z = " + ".join(f"{c}_z" for c in ENC_COVARIATES)
_COV_BETWEEN_Z = " + ".join(f"{c}_z" for c in ENC_BETWEEN_COVARIATES)
_COV_FULL_Z = f"{_COV_ENC_Z} + {_COV_BETWEEN_Z}"

_COV_DEC_Z = " + ".join(f"{c}_z" for c in DEC_COVARIATES)
_COV_DEC_BTW_Z = " + ".join(f"{c}_z" for c in DEC_BETWEEN_COVARIATES)
_COV_DEC_FULL_Z = f"{_COV_DEC_Z} + {_COV_DEC_BTW_Z}"


# ---------------------------------------------------------------------------
# Formula construction
# ---------------------------------------------------------------------------


def _formula_for(name: str, primary_pred: str) -> tuple:
    """
    Return (rhs_formula, dv_column).
    rhs_formula does not include the DV.
    """
    if name == "H1_svg_enc":
        return "1", "svg_z_enc"
    if name == "H1_svg_dec":
        return "1", "svg_z_dec"

    if name == "H2_total":
        return f"{primary_pred} + {_COV_FULL_Z}", DV_TOTAL
    if name == "H2_relations":
        return f"{primary_pred} + {_COV_FULL_Z}", DV_RELATIONS
    if name == "H2_objects":
        return f"{primary_pred} + {_COV_FULL_Z}", DV_OBJECTS

    if name == "EXP_dissociation":
        return f"{primary_pred} + {_COV_FULL_Z}", "score"

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
    Fit one non-H1 model via LMM.
    Returns (result | None, mode: str).
    """
    full_formula = f"{dv} ~ {formula}"

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
    req = [dv, "SubjectID", "StimID"] + [t for t in tokens if t in df.columns]
    req = list(dict.fromkeys(req))

    missing_required_cols = [c for c in req if c not in df.columns]
    if missing_required_cols:
        logger.warning(f"  {name}: missing required columns: {missing_required_cols}")

    model_df = df[[c for c in req if c in df.columns]].copy()
    n_before = len(model_df)
    na_counts = model_df.isna().sum()
    model_df = model_df.dropna()
    n_obs = len(model_df)
    dropped = n_before - n_obs

    n_subj = model_df["SubjectID"].nunique() if "SubjectID" in model_df else 0
    n_stim = model_df["StimID"].nunique() if "StimID" in model_df else 0
    logger.info(f"  {name}: n={n_obs} obs, {n_subj} subj, {n_stim} stim")

    if dropped:
        nonzero_na = na_counts[na_counts > 0].sort_values(ascending=False)
        logger.info(
            f"    dropped {dropped} row(s) with missing values: "
            + ", ".join(f"{col}={int(cnt)}" for col, cnt in nonzero_na.items())
        )

    if n_obs < 20 or n_subj < 2 or n_stim < 2:
        logger.warning(f"  {name}: insufficient data — skipping.")
        return None, "skipped"

    return _fit_lmm(name, full_formula, model_df)


def _fit_lmm(name: str, full_formula: str, model_df: pd.DataFrame) -> tuple:
    """
    LMM with crossed random effects: (1|SubjectID) + (1|StimID).

    lbfgs is excluded: on Windows it calls C exit() on certain data
    configurations, killing the process silently before any Python code
    can react. nm (Nelder-Mead) and powell are pure-Python and always safe.
    Estimates are identical to lbfgs — only speed differs.
    """
    # Intercept-only H1 models: StimID vc causes SE collapse — SubjectID only
    is_intercept_only = full_formula.split("~")[1].strip() == "1"
    vc = {} if is_intercept_only else {"StimID": "0 + C(StimID)"}

    _OPTIMIZERS = [
        ("nm", {"maxiter": 1000}),
        ("powell", {"maxiter": 1000}),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method, fit_kwargs in _OPTIMIZERS:
            try:
                model = smf.mixedlm(
                    full_formula,
                    data=model_df,
                    groups="SubjectID",
                    vc_formula=vc if vc else None,
                )
                result = model.fit(reml=True, method=method, **fit_kwargs)
                if not result.converged:
                    logger.warning(
                        f"  {name}: {method} did not converge — trying next optimizer"
                    )
                    continue
                logger.info(f"    LMM converged ({method}): {result.converged}")
                return result, "lmm"
            except Exception as e:
                logger.warning(
                    f"  {name}: {method} failed ({str(e)[:120]}) — trying next optimizer"
                )

    logger.error(f"  {name}: LMM failed across all optimizers — returning None")
    return None, "failed"


# ---------------------------------------------------------------------------
# H1 one-sample t-tests
# ---------------------------------------------------------------------------


def _fit_h1_ttest(name: str, svg_col: str, df: pd.DataFrame) -> tuple:
    """
    One-sample t-test of per-participant mean SVG against 0.

    Aggregates to participant means first (removes within-participant
    dependence), then tests whether the mean of those means > 0.

    Also computes:
      - Cohen's d  = grand_mean / SD of participant means (one-sample, null=0)
      - Shapiro-Wilk normality test on participant means
      - Wilcoxon signed-rank test (non-parametric robustness check),
        run unconditionally so the result is always available for reporting.

    Returns a lightweight namespace that mimics the statsmodels result
    interface used elsewhere in the module.
    """
    vals = df[svg_col].dropna()
    if len(vals) < 2:
        logger.warning(f"  {name}: insufficient data for t-test — skipping.")
        return None, "skipped"

    subj_means = df.groupby("SubjectID")[svg_col].mean().dropna()
    n = len(subj_means)
    grand_mean = subj_means.mean()
    sd = subj_means.std(ddof=1)
    se = subj_means.sem()
    t_stat, p_two = scipy_stats.ttest_1samp(subj_means, 0)
    ci = scipy_stats.t.interval(0.95, df=n - 1, loc=grand_mean, scale=se)

    # Cohen's d for one-sample t-test against zero: d = mean / SD
    cohens_d = grand_mean / sd if sd > 0 else np.nan

    # Shapiro-Wilk normality test on participant means
    sw_stat, sw_p = scipy_stats.shapiro(subj_means)
    normality_violated = sw_p < 0.05

    # Wilcoxon signed-rank test (always run as robustness check)
    try:
        wx_stat, wx_p = scipy_stats.wilcoxon(subj_means, alternative="two-sided")
    except Exception as e:
        logger.warning(f"  {name}: Wilcoxon failed ({e}) — storing NaN.")
        wx_stat, wx_p = np.nan, np.nan

    pct_above = 100 * (vals > 0).mean()

    logger.info(
        f"  {name}: one-sample t-test on {n} participant means\n"
        f"    grand mean={grand_mean:.3f}, SD={sd:.3f}, SE={se:.3f}, "
        f"t({n-1})={t_stat:.3f}, p={p_two:.4f} (two-tailed)\n"
        f"    Cohen's d={cohens_d:.3f}\n"
        f"    Shapiro-Wilk: W={sw_stat:.3f}, p={sw_p:.4f} "
        f"({'VIOLATED' if normality_violated else 'OK'})\n"
        f"    Wilcoxon signed-rank: W={wx_stat}, p={wx_p:.4f}\n"
        f"    {pct_above:.1f}% of trials above permutation baseline"
    )

    result = types.SimpleNamespace()
    result.params = {"Intercept": grand_mean}
    result.pvalues = {"Intercept": p_two}
    result.tvalues = {"Intercept": t_stat}
    result.bse = {"Intercept": se}
    result.converged = True
    result.nobs = n
    result.df_resid = n - 1

    # Additional fields for output.py
    result.cohens_d = cohens_d
    result.sd_subj_means = sd
    result.sw_stat = sw_stat
    result.sw_p = sw_p
    result.normality_violated = normality_violated
    result.wx_stat = wx_stat
    result.wx_p = wx_p
    result.pct_above = pct_above

    def _conf_int(alpha=0.05):
        ci_lo, ci_hi = scipy_stats.t.interval(
            1 - alpha, df=n - 1, loc=grand_mean, scale=se
        )
        return pd.DataFrame({"0": [ci_lo], "1": [ci_hi]}, index=["Intercept"])

    result.conf_int = _conf_int

    def _summary():
        normality_note = (
            f"  Shapiro-Wilk   : W={sw_stat:.3f}, p={sw_p:.4f}  *** NORMALITY VIOLATED ***\n"
            f"  Wilcoxon W     : {wx_stat}, p={wx_p:.4f}  (non-parametric robustness check)\n"
            if normality_violated
            else f"  Shapiro-Wilk   : W={sw_stat:.3f}, p={sw_p:.4f}  (normality holds)\n"
            f"  Wilcoxon W     : {wx_stat}, p={wx_p:.4f}  (non-parametric robustness check)\n"
        )
        return (
            f"H1 one-sample t-test (participant means vs 0)\n"
            f"  N participants : {n}\n"
            f"  Grand mean SVG : {grand_mean:.4f}  (SD={sd:.4f}, SE={se:.4f})\n"
            f"  t({n-1})        : {t_stat:.3f}\n"
            f"  p (two-tailed) : {p_two:.4f}\n"
            f"  95% CI         : [{ci[0]:.4f}, {ci[1]:.4f}]\n"
            f"  Cohen's d      : {cohens_d:.3f}\n"
            f"  % trials > 0   : {pct_above:.1f}%\n" + normality_note
        )

    result.summary = _summary
    return result, "ttest"


# ---------------------------------------------------------------------------
# Fit all models
# ---------------------------------------------------------------------------


def fit_all_models(filtered: dict) -> dict:
    """
    Iterate MODEL_SPECS, fit each model, return results dict.
    Keys: model name → (result | None, mode: str)

    H1 models use a one-sample t-test on per-participant means.
    All other models use LMM with crossed random effects.
    """
    logger.info("\nStep 5: Fitting models ...")
    results = {}

    h1_svg_cols = {
        "H1_svg_enc": ("enc", "svg_z_enc"),
        "H1_svg_dec": ("dec", "svg_z_dec"),
    }

    for name, primary_pred, desc, table_key, group in MODEL_SPECS:
        logger.info(f"\n  [{group}] {name}: {desc}")
        df = filtered.get(table_key, pd.DataFrame())
        if df.empty:
            logger.warning(f"  {name}: table '{table_key}' is empty — skipping.")
            results[name] = (None, "skipped")
            continue

        if name in h1_svg_cols:
            _, svg_col = h1_svg_cols[name]
            results[name] = _fit_h1_ttest(name, svg_col, df)
            continue

        formula, dv = _formula_for(name, primary_pred)
        logger.info(f"    {dv} ~ {formula}")
        results[name] = _fit_one(name, formula, dv, df)

    return results
