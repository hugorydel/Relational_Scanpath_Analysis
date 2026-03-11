"""
pipeline/module4/loader.py
===========================
Steps 1–4: data loading, table construction, exclusions, standardisation.

All functions are pure data-in → data-out (no model fitting, no plotting).
"""

import logging
from pathlib import Path

import pandas as pd

from .constants import (
    DEC_COVARIATES,
    DV_CONFAB,
    DV_LENGTH,
    DV_OBJECTS,
    DV_RELATIONAL,
    ENC_COVARIATES,
    MIN_N_SHARED_REPLAY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1a: Load trial features
# ---------------------------------------------------------------------------


def load_data(input_path: Path) -> pd.DataFrame:
    """Load trial_features_all.csv produced by Module 3."""
    logger.info(f"Step 1: Loading {input_path.name} ...")
    if not input_path.exists():
        raise FileNotFoundError(f"Features file not found: {input_path}")
    df = pd.read_csv(input_path, dtype={"StimID": str, "SubjectID": str})
    logger.info(
        f"  Loaded {len(df)} rows, {df['SubjectID'].nunique()} participants, "
        f"{df['StimID'].nunique()} stimuli."
    )
    return df


# ---------------------------------------------------------------------------
# Step 1b: Load memory scores
# ---------------------------------------------------------------------------

# Content types and statuses used by the new 20-column scorer schema.
_NEW_CONTENT_TYPES = [
    "object_identity",
    "object_attribute",
    "action_relation",
    "spatial_relation",
    "scene_gist",
]
_NEW_STATUSES = ["correct", "incorrect", "inference", "repeat"]
_NEW_COUNT_COLS = [f"n_{ct}_{st}" for ct in _NEW_CONTENT_TYPES for st in _NEW_STATUSES]


def _derive_legacy_columns(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Given a new-schema scores DataFrame, compute the legacy aggregate columns
    that the rest of the pipeline expects:
        n_relational_correct   = action_relation + spatial_relation (correct)
        n_relational_incorrect = action_relation + spatial_relation (incorrect)
        n_objects_correct      = object_identity + object_attribute (correct)
        n_objects_incorrect    = object_identity + object_attribute (incorrect)

    Also adds convenience totals:
        n_gist_correct     = n_scene_gist_correct
        n_inference_total  = sum of all *_inference columns
        n_repeat_total     = sum of all *_repeat columns
    """
    # Ensure every expected count column is present (fills zeros for any
    # content types added after the CSV header was first written).
    for col in _NEW_COUNT_COLS:
        if col not in scores.columns:
            scores[col] = 0

    scores["n_relational_correct"] = (
        scores["n_action_relation_correct"] + scores["n_spatial_relation_correct"]
    )
    scores["n_relational_incorrect"] = (
        scores["n_action_relation_incorrect"] + scores["n_spatial_relation_incorrect"]
    )
    scores["n_objects_correct"] = (
        scores["n_object_identity_correct"] + scores["n_object_attribute_correct"]
    )
    scores["n_objects_incorrect"] = (
        scores["n_object_identity_incorrect"] + scores["n_object_attribute_incorrect"]
    )
    scores["n_gist_correct"] = scores["n_scene_gist_correct"]
    scores["n_inference_total"] = scores[
        [f"n_{ct}_inference" for ct in _NEW_CONTENT_TYPES]
    ].sum(axis=1)
    scores["n_repeat_total"] = scores[
        [f"n_{ct}_repeat" for ct in _NEW_CONTENT_TYPES]
    ].sum(axis=1)
    return scores


def load_memory_scores(scores_path: Path) -> pd.DataFrame:
    """
    Load manually scored memory data and compute derived columns.

    Supports two scorer schemas:

    Legacy (4-column):
        SubjectID, StimID,
        n_relational_correct, n_relational_incorrect,
        n_objects_correct, n_objects_incorrect, empty_response

    New (20-column, content_type × status):
        SubjectID, StimID, empty_response, [wrong_image],
        n_{content_type}_{status}  (5 types × 4 statuses = 20 columns)

    In both cases, the following derived columns are added:
        n_relational_total     = n_relational_correct + n_relational_incorrect
        any_relational_correct = 1 if n_relational_correct >= 1 else 0
    """
    logger.info(f"  Loading memory scores from {scores_path.name} ...")
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Memory scores file not found: {scores_path}\n"
            "Score responses with the scoring app and save before running Module 4."
        )

    scores = pd.read_csv(scores_path, dtype={"StimID": str, "SubjectID": str})

    # Detect schema by spot-checking the first four new-schema columns.
    is_new_schema = all(c in scores.columns for c in _NEW_COUNT_COLS[:4])

    if is_new_schema:
        logger.info("  Detected new 20-column scorer schema — deriving legacy columns.")
        scores = _derive_legacy_columns(scores)
    else:
        logger.info("  Detected legacy 4-column scorer schema.")
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


def build_analysis_tables(
    df: pd.DataFrame, memory_scores: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """
    Split trial_features_all into decoding, encoding, and wide tables,
    then join memory scores onto the decoding table.

    Returns
    -------
    dict with keys: "dec", "enc", "wide"
    """
    logger.info("Step 2: Building analysis tables ...")

    enc_raw = df[df["Phase"] == "encoding"].copy()
    dec_raw = df[df["Phase"] == "decoding"].copy()

    # ── Decoding table ────────────────────────────────────────────────────
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

    dec = dec.merge(memory_scores, on=["SubjectID", "StimID"], how="left")
    n_unscored = dec[DV_RELATIONAL].isna().sum()
    if n_unscored > 0:
        logger.warning(
            f"  {n_unscored} decoding row(s) have no memory score — "
            "check that memory_scores.csv covers all participants/stimuli."
        )

    # ── Encoding table ────────────────────────────────────────────────────
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

    # ── Wide table (enc + dec joined) for exploratory models ─────────────
    memory_cols = [
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
    ] + [c for c in memory_cols if c in dec.columns]

    wide = enc[[c for c in enc_keep if c in enc.columns]].merge(
        dec[[c for c in dec_keep if c in dec.columns]],
        on=["SubjectID", "StimID"],
        how="inner",
    )

    logger.info(f"  dec table:  {len(dec)} rows  ({n_unscored} unscored)")
    logger.info(f"  enc table:  {len(enc)} rows")
    logger.info(f"  wide table: {len(wide)} rows")

    return {"dec": dec, "enc": enc, "wide": wide}


# ---------------------------------------------------------------------------
# Step 3: Hypothesis-specific exclusions
# ---------------------------------------------------------------------------


def apply_exclusions(tables: dict) -> dict:
    """
    Apply per-hypothesis trial-level exclusions and return filtered tables.

    Returns
    -------
    dict with keys: dec_inter | dec_all | enc_inter | enc_all | replay
    """
    logger.info("Step 3: Applying hypothesis-specific exclusions ...")

    dec = tables["dec"]
    wide = tables["wide"]

    def _log(name, before, after, reason):
        logger.info(
            f"  {name}: {before} → {after} (removed {before - after}: {reason})"
        )

    # H1 / H2: filter decoding table on low_n_dec
    n = len(dec)
    dec_inter = dec[~dec["low_n_dec"]].copy()
    dec_all = dec[~dec["low_n_dec"]].copy()
    _log("dec_inter / dec_all", n, len(dec_inter), "low_n_dec=True")

    # Exploratory enc → memory: filter wide table on low_n_enc
    n_w = len(wide)
    enc_inter = wide[~wide["low_n_enc"]].copy()
    enc_all = wide[~wide["low_n_enc"]].copy()
    _log("enc_inter / enc_all", n_w, len(enc_inter), "low_n_enc=True")

    # Replay: additionally require n_shared >= MIN and non-null tau
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
        "Raise to 3 for final analysis."
    )

    return {
        "dec_inter": dec_inter,
        "dec_all": dec_all,
        "enc_inter": enc_inter,
        "enc_all": enc_all,
        "replay": replay,
    }


# ---------------------------------------------------------------------------
# Step 4: Standardise predictors
# ---------------------------------------------------------------------------


def standardise_tables(filtered: dict) -> dict:
    """
    Z-score continuous predictors within each filtered table.
    Adds `{col}_z` columns; originals are kept.
    Global standardisation within table ensures the intercept in H1 models
    equals the mean SVG score at an average-covariate trial.
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
