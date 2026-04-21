"""
sensitivity_memorability.py
===========================
Sensitivity analysis: do the H2 results hold after adding image memorability
(ResMem score) as an additional between-image covariate?

Memorability is a per-image predictor (one value per StimID) sourced from
stimuli_dataset.json. It is z-scored and added alongside the existing
between-image covariate (svg_z_enc_image_mean_z) in every H2 model.

This version fixes a row-alignment bug by constructing the model dataframe
from ALL formula-referenced columns, dropping missing rows BEFORE MixedLM,
and resetting the index. Without that, statsmodels can silently drop rows
internally and trigger index-out-of-bounds errors.
"""

import argparse
import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import config
from pipeline.module_3.scene_graph import load_stimulus_metadata
from pipeline.module_4 import (
    apply_exclusions,
    build_analysis_tables,
    load_data,
    load_memory_scores,
    standardise_tables,
)
from pipeline.module_4.constants import (
    DEC_BETWEEN_COVARIATES,
    DEC_COVARIATES,
    DEFAULT_FLAGS_PATH,
    DEFAULT_SCORES_PATH,
    DV_OBJECTS,
    DV_RELATIONS,
    DV_TOTAL,
    ENC_BETWEEN_COVARIATES,
    ENC_COVARIATES,
)

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
)
logger = logging.getLogger(__name__)

OUT_DIR = config.OUTPUT_DIR / "diagnostics"

# ---------------------------------------------------------------------------
# Covariate strings (mirrors constants.py)
# ---------------------------------------------------------------------------

_COV_ENC_Z = " + ".join(f"{c}_z" for c in ENC_COVARIATES)
_COV_BTW_Z = " + ".join(f"{c}_z" for c in ENC_BETWEEN_COVARIATES)
_COV_DEC_Z = " + ".join(f"{c}_z" for c in DEC_COVARIATES)
_COV_DEC_BTW_Z = " + ".join(f"{c}_z" for c in DEC_BETWEEN_COVARIATES)

_COV_ENC_FULL = f"{_COV_ENC_Z} + {_COV_BTW_Z}"
_COV_ENC_MEM = f"{_COV_ENC_Z} + {_COV_BTW_Z} + memorability_z"
_COV_DEC_FULL = f"{_COV_DEC_Z} + {_COV_DEC_BTW_Z}"
_COV_DEC_MEM = f"{_COV_DEC_Z} + {_COV_DEC_BTW_Z} + memorability_z"

# ---------------------------------------------------------------------------
# Model specs for this sensitivity script
# (name, primary_pred, dv, table, covariates_without_mem, covariates_with_mem)
# ---------------------------------------------------------------------------

_MODELS = [
    # Encoding H2
    ("H2_total", "svg_z_enc_within_z", DV_TOTAL, "enc", _COV_ENC_FULL, _COV_ENC_MEM),
    (
        "H2_relations",
        "svg_z_enc_within_z",
        DV_RELATIONS,
        "enc",
        _COV_ENC_FULL,
        _COV_ENC_MEM,
    ),
    (
        "H2_objects",
        "svg_z_enc_within_z",
        DV_OBJECTS,
        "enc",
        _COV_ENC_FULL,
        _COV_ENC_MEM,
    ),
    # Exploratory dissociation (long format)
    (
        "EXP_dissoc",
        "svg_z_enc_within_z:memory_type",
        "score",
        "enc_long",
        f"svg_z_enc_within_z * memory_type + {_COV_ENC_FULL}",
        f"svg_z_enc_within_z * memory_type + {_COV_ENC_MEM}",
    ),
    # Decoding H2
    (
        "H2_dec_total",
        "svg_z_dec_within_z",
        DV_TOTAL,
        "dec",
        _COV_DEC_FULL,
        _COV_DEC_MEM,
    ),
    (
        "H2_dec_relations",
        "svg_z_dec_within_z",
        DV_RELATIONS,
        "dec",
        _COV_DEC_FULL,
        _COV_DEC_MEM,
    ),
    (
        "H2_dec_objects",
        "svg_z_dec_within_z",
        DV_OBJECTS,
        "dec",
        _COV_DEC_FULL,
        _COV_DEC_MEM,
    ),
]

# ---------------------------------------------------------------------------
# Helper: build safe model dataframe from full formula
# ---------------------------------------------------------------------------


def _prepare_model_df(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    """
    Build the exact analysis dataframe required by a formula:
      - collect all referenced columns that exist in df
      - keep SubjectID and StimID for random effects
      - drop rows with ANY missing values in required columns
      - reset index to prevent statsmodels alignment/index bugs
    """
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
    req = ["SubjectID", "StimID"] + [t for t in tokens if t in df.columns]
    req = list(dict.fromkeys(req))
    model_df = df[[c for c in req if c in df.columns]].copy()
    model_df = model_df.dropna().reset_index(drop=True)
    return model_df


def _log_missing(df: pd.DataFrame, formula: str, name: str) -> None:
    """Log missingness across all columns required by the formula."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
    req = ["SubjectID", "StimID"] + [t for t in tokens if t in df.columns]
    req = list(dict.fromkeys(req))
    miss = df[req].isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        logger.info(f"  {name} missingness before dropna:\n{miss.to_string()}")


# ---------------------------------------------------------------------------
# LMM helper
# ---------------------------------------------------------------------------


def _fit_lmm(name: str, formula: str, df: pd.DataFrame):
    """
    Fit LMM with crossed random effects.

    Tries:
      1) groups=SubjectID + vc_formula StimID (matches main pipeline)
      2) participant random intercept only (fallback)

    Returns
    -------
    (result | None, converged: bool, mode: str)
    """
    df = df.reset_index(drop=True).copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for method in ["nm", "powell"]:
            # Attempt 1: crossed random effects
            try:
                model = smf.mixedlm(
                    formula,
                    data=df,
                    groups="SubjectID",
                    vc_formula={"StimID": "0 + C(StimID)"},
                )
                result = model.fit(reml=True, method=method, maxiter=2000)
                if result.converged:
                    return result, True, f"crossed_{method}"
            except Exception as e:
                logger.info(f"  [{name}] crossed_{method} failed: {e}")

            # Attempt 2: participant random intercept only
            try:
                model = smf.mixedlm(
                    formula,
                    data=df,
                    groups="SubjectID",
                )
                result = model.fit(reml=True, method=method, maxiter=2000)
                if result.converged:
                    logger.info(
                        f"  [{name}] fell back to participant-only random intercept ({method})"
                    )
                    return result, True, f"subject_only_{method}"
            except Exception as e:
                logger.info(f"  [{name}] subject_only_{method} failed: {e}")

    return None, False, "failed"


def _extract(result, term: str):
    """Pull b, SE, CI, p for a given term from a fitted result."""
    if result is None:
        return dict(b=np.nan, se=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)

    try:
        b = float(result.params[term])
        se = float(result.bse[term])
        p = float(result.pvalues[term])
        ci = result.conf_int()

        # Be robust to integer- or string-named CI columns
        lo = float(ci.loc[term].iloc[0])
        hi = float(ci.loc[term].iloc[1])

        return dict(b=b, se=se, ci_lo=lo, ci_hi=hi, p=p)
    except KeyError:
        return dict(b=np.nan, se=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)


def _sig(p: float) -> str:
    if np.isnan(p):
        return "na"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Load memorability from stimuli_dataset.json
# ---------------------------------------------------------------------------


def load_memorability() -> pd.Series:
    """
    Return a pd.Series indexed by StimID (str) with the ResMem memorability
    score for each stimulus.
    """
    metadata = load_stimulus_metadata()
    records = {
        stim_id: entry.get("memorability", np.nan)
        for stim_id, entry in metadata.items()
    }
    s = pd.Series(records, name="memorability")
    s.index.name = "StimID"
    logger.info(
        f"Memorability scores loaded: {s.notna().sum()}/{len(s)} stimuli, "
        f"M={s.mean():.3f}, SD={s.std():.3f}, range=[{s.min():.3f}, {s.max():.3f}]"
    )
    return s


def add_memorability(filtered: dict, mem: pd.Series) -> dict:
    """
    Join memorability onto each table and add memorability_z column.
    Operates on copies so the original filtered dict is unchanged.
    """
    result = {}
    for key, df in filtered.items():
        if key.startswith("_") or df is None or df.empty:
            result[key] = df
            continue

        df = df.copy()
        df = df.merge(mem.reset_index(), on="StimID", how="left")

        mu = df["memorability"].mean()
        sd = df["memorability"].std()
        df["memorability_z"] = (df["memorability"] - mu) / sd if sd > 0 else 0.0

        missing = df["memorability"].isna().sum()
        if missing:
            logger.warning(
                f"  [{key}] {missing} rows have no memorability score — "
                "they will be dropped by model preparation."
            )

        result[key] = df

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis: H2 results with memorability covariate."
    )
    parser.add_argument(
        "--input",
        default=str(config.OUTPUT_FEATURES_DIR / "trial_features_all.csv"),
    )
    parser.add_argument("--scores", default=str(DEFAULT_SCORES_PATH))
    parser.add_argument("--flags", default=str(DEFAULT_FLAGS_PATH))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and prepare data (identical to main pipeline) ────────────────
    logger.info("Loading data ...")
    raw_df = load_data(Path(args.input))
    mem_scores = load_memory_scores(Path(args.scores))
    tables = build_analysis_tables(raw_df, mem_scores)
    filtered = apply_exclusions(tables, flags_path=Path(args.flags))
    filtered = standardise_tables(filtered)

    n_participants = filtered["enc"]["SubjectID"].nunique()
    logger.info(f"N participants in analysis: {n_participants}")

    # ── Load and join memorability ────────────────────────────────────────
    logger.info("\nLoading memorability scores ...")
    mem = load_memorability()
    filtered_m = add_memorability(filtered, mem)

    # ── Fit all models, with and without memorability ─────────────────────
    rows = []
    summary_lines = [
        "=" * 72,
        "SENSITIVITY ANALYSIS: H2 models with memorability covariate",
        f"N participants = {n_participants}",
        "=" * 72,
        "",
        "For each model, the primary predictor estimate is shown with and",
        "without memorability (ResMem score) as a between-image covariate.",
        "",
    ]

    col_hdr = (
        f"  {'Model':<20}  {'b (orig)':>10}  {'95% CI':^18}  "
        f"{'p':>7}  {'sig':>3}  │  "
        f"{'b (+mem)':>10}  {'95% CI':^18}  {'p':>7}  {'sig':>3}  "
        f"{'mem_b':>8}  {'mem_p':>7}"
    )
    divider = "  " + "-" * (len(col_hdr) - 2)

    for section, model_names in [
        (
            "ENCODING (H2 + Exploratory)",
            ["H2_total", "H2_relations", "H2_objects", "EXP_dissoc"],
        ),
        ("DECODING (H2)", ["H2_dec_total", "H2_dec_relations", "H2_dec_objects"]),
    ]:
        summary_lines += [section, col_hdr, divider]

        for spec in _MODELS:
            name, pred_term, dv, table_key, cov_base, cov_mem = spec
            if name not in model_names:
                continue

            df_base_raw = filtered.get(table_key, pd.DataFrame())
            df_mem_raw = filtered_m.get(table_key, pd.DataFrame())

            if df_base_raw.empty:
                logger.warning(f"  {name}: table '{table_key}' is empty — skipping.")
                continue

            if name == "EXP_dissoc":
                formula_base = f"score ~ {cov_base}"
                formula_mem = f"score ~ {cov_mem}"
                dv_col = "score"
            else:
                formula_base = f"{dv} ~ {pred_term} + {cov_base}"
                formula_mem = f"{dv} ~ {pred_term} + {cov_mem}"
                dv_col = dv

            # Log missingness across the full model columns
            _log_missing(df_base_raw, formula_base, f"{name} base")
            _log_missing(df_mem_raw, formula_mem, f"{name} +mem")

            # Prepare model frames safely
            model_df_base = _prepare_model_df(df_base_raw, formula_base)
            model_df_mem = _prepare_model_df(df_mem_raw, formula_mem)

            logger.info(
                f"\n  {name}: n_base={len(model_df_base)}, n_mem={len(model_df_mem)}"
            )
            logger.info(
                f"    formula (base): {formula_base.split('~')[1].strip()[:120]}"
            )
            logger.info(
                f"    formula (+mem): {formula_mem.split('~')[1].strip()[:120]}"
            )

            r_base, conv_base, mode_base = _fit_lmm(name, formula_base, model_df_base)
            if not conv_base:
                logger.warning(f"    {name} base model did not converge")
            else:
                logger.info(f"    {name} base model converged via {mode_base}")
            e_base = _extract(r_base, pred_term)

            r_mem, conv_mem, mode_mem = _fit_lmm(
                name + "_mem", formula_mem, model_df_mem
            )
            if not conv_mem:
                logger.warning(f"    {name} +memorability model did not converge")
            else:
                logger.info(f"    {name} +memorability model converged via {mode_mem}")
            e_mem = _extract(r_mem, pred_term)
            e_mem_cov = _extract(r_mem, "memorability_z")

            rows.append(
                {
                    "model": name,
                    "predictor": pred_term,
                    "dv": dv_col,
                    "n_obs_base": len(model_df_base),
                    "n_obs_mem": len(model_df_mem),
                    "fit_mode_base": mode_base,
                    "fit_mode_mem": mode_mem,
                    "b_base": (
                        round(e_base["b"], 4) if not np.isnan(e_base["b"]) else np.nan
                    ),
                    "se_base": (
                        round(e_base["se"], 4) if not np.isnan(e_base["se"]) else np.nan
                    ),
                    "ci_lo_base": (
                        round(e_base["ci_lo"], 4)
                        if not np.isnan(e_base["ci_lo"])
                        else np.nan
                    ),
                    "ci_hi_base": (
                        round(e_base["ci_hi"], 4)
                        if not np.isnan(e_base["ci_hi"])
                        else np.nan
                    ),
                    "p_base": (
                        round(e_base["p"], 4) if not np.isnan(e_base["p"]) else np.nan
                    ),
                    "b_mem": (
                        round(e_mem["b"], 4) if not np.isnan(e_mem["b"]) else np.nan
                    ),
                    "se_mem": (
                        round(e_mem["se"], 4) if not np.isnan(e_mem["se"]) else np.nan
                    ),
                    "ci_lo_mem": (
                        round(e_mem["ci_lo"], 4)
                        if not np.isnan(e_mem["ci_lo"])
                        else np.nan
                    ),
                    "ci_hi_mem": (
                        round(e_mem["ci_hi"], 4)
                        if not np.isnan(e_mem["ci_hi"])
                        else np.nan
                    ),
                    "p_mem": (
                        round(e_mem["p"], 4) if not np.isnan(e_mem["p"]) else np.nan
                    ),
                    "mem_b": (
                        round(e_mem_cov["b"], 4)
                        if not np.isnan(e_mem_cov["b"])
                        else np.nan
                    ),
                    "mem_p": (
                        round(e_mem_cov["p"], 4)
                        if not np.isnan(e_mem_cov["p"])
                        else np.nan
                    ),
                }
            )

            def _fmt(e):
                return (
                    f"{e['b']:+.3f}  [{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}]  "
                    f"p={e['p']:.3f}  {_sig(e['p'])}"
                )

            summary_lines.append(
                f"  {name:<20}  {_fmt(e_base)}  │  {_fmt(e_mem)}"
                f"  mem_b={e_mem_cov['b']:+.3f}  mem_p={e_mem_cov['p']:.3f}"
            )

        summary_lines.append("")

    # ── Interpretation note ───────────────────────────────────────────────
    summary_lines += [
        "=" * 72,
        "INTERPRETATION GUIDE",
        "=" * 72,
        "  b (orig)  : coefficient from the original model (no memorability)",
        "  b (+mem)  : same coefficient after adding memorability as covariate",
        "  mem_b     : memorability coefficient itself (positive = more",
        "              memorable images recalled better)",
        "",
        "  If b (+mem) ≈ b (orig) and significance is unchanged: the SVG→recall",
        "  effect is independent of inherent image memorability.",
        "",
        "  If b (+mem) < b (orig) and p increases: part of the effect was",
        "  confounded with memorability.",
        "=" * 72,
    ]

    # ── Write outputs ─────────────────────────────────────────────────────
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    txt_path = OUT_DIR / "sensitivity_memorability_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info(f"\nSummary written → {txt_path}")

    csv_path = OUT_DIR / "sensitivity_memorability_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Results CSV   → {csv_path}")


if __name__ == "__main__":
    main()
