"""
pipeline/module_4/loader.py
============================
Steps 1-4: data loading, table construction, exclusions, standardisation.

Memory DVs come from recall_by_category.csv (output of aggregate_recall.py),
not from the manual scoring UI. Three proportion DVs are computed:

    prop_total     = n_correct_nodes_recalled / max_per_stim
    prop_relations = (n_action_relation_recalled + n_spatial_relation_recalled) / max_per_stim
    prop_objects   = (n_object_identity_recalled + n_object_attribute_recalled) / max_per_stim

Each denominator is the empirical maximum across all participants for that
StimID, separately per DV. This normalises for both image complexity and
image memorability, placing all images on a 0-1 scale.

If max=0 for a given StimID × DV (nobody recalled any relations for that
image), those cells become NaN and are silently excluded by model dropna.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DV_OBJECTS, DV_RELATIONS, DV_TOTAL, ENC_COVARIATES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1a: Load trial features
# ---------------------------------------------------------------------------


def load_data(input_path: Path) -> pd.DataFrame:
    """Load trial_features_all.csv produced by Module 3."""
    logger.info(f"Step 1a: Loading {input_path.name} ...")
    if not input_path.exists():
        raise FileNotFoundError(f"Features file not found: {input_path}")
    df = pd.read_csv(input_path, dtype={"StimID": str, "SubjectID": str})
    logger.info(
        f"  Loaded {len(df)} rows, {df['SubjectID'].nunique()} participants, "
        f"{df['StimID'].nunique()} stimuli."
    )
    return df


# ---------------------------------------------------------------------------
# Step 1b: Load recall_by_category.csv and compute proportion DVs
# ---------------------------------------------------------------------------


def load_memory_scores(scores_path: Path) -> pd.DataFrame:
    """
    Load recall_by_category.csv (from aggregate_recall.py) and compute the
    three empirically-normalised proportion DVs.

    Returns a DataFrame with columns:
        SubjectID, StimID,
        prop_total, prop_relations, prop_objects
    plus the underlying raw counts for diagnostics.
    """
    logger.info(f"Step 1b: Loading recall scores from {scores_path.name} ...")
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Recall scores file not found: {scores_path}\n"
            "Run aggregate_recall.py before Module 4."
        )

    df = pd.read_csv(scores_path, dtype={"StimID": str, "SubjectID": str})

    # Derive combined raw counts
    # Relations: action + spatial
    for col in ["n_action_relation_recalled", "n_spatial_relation_recalled"]:
        if col not in df.columns:
            logger.warning(f"  Column {col} missing from scores — treating as 0.")
            df[col] = 0
    df["n_relations_recalled"] = (
        df["n_action_relation_recalled"] + df["n_spatial_relation_recalled"]
    )

    # Objects: identity + attribute
    for col in ["n_object_identity_recalled", "n_object_attribute_recalled"]:
        if col not in df.columns:
            logger.warning(f"  Column {col} missing from scores — treating as 0.")
            df[col] = 0
    df["n_objects_recalled"] = (
        df["n_object_identity_recalled"] + df["n_object_attribute_recalled"]
    )

    # Total correct nodes
    if "n_correct_nodes_recalled" not in df.columns:
        raise ValueError("recall_by_category.csv missing n_correct_nodes_recalled.")
    df["n_total_recalled"] = df["n_correct_nodes_recalled"]

    # ------------------------------------------------------------------
    # Empirical per-StimID maxima (separately per DV)
    # ------------------------------------------------------------------
    raw_cols = {
        DV_TOTAL: "n_total_recalled",
        DV_RELATIONS: "n_relations_recalled",
        DV_OBJECTS: "n_objects_recalled",
    }

    for prop_col, raw_col in raw_cols.items():
        stim_max = df.groupby("StimID")[raw_col].max().rename(f"_max_{raw_col}")
        df = df.merge(stim_max, on="StimID", how="left")
        max_col = f"_max_{raw_col}"

        # Proportion: NaN where max=0 (nobody recalled anything for that DV/stim)
        df[prop_col] = np.where(
            df[max_col] > 0,
            df[raw_col] / df[max_col],
            np.nan,
        )

        n_nan = df[prop_col].isna().sum()
        n_zero_max = (df[max_col] == 0).sum()
        if n_zero_max > 0:
            logger.warning(
                f"  {prop_col}: {n_zero_max} rows have max=0 for their StimID "
                f"({n_nan} NaN entries — excluded from that DV's models)."
            )
        df = df.drop(columns=[max_col])

    logger.info(
        f"  Loaded {len(df)} scored rows, "
        f"{df['SubjectID'].nunique()} participants, "
        f"{df['StimID'].nunique()} stimuli."
    )

    # Log proportion descriptives
    for prop_col in [DV_TOTAL, DV_RELATIONS, DV_OBJECTS]:
        vals = df[prop_col].dropna()
        logger.info(
            f"  {prop_col}: n={len(vals)}, "
            f"mean={vals.mean():.3f}, sd={vals.std():.3f}, "
            f"range=[{vals.min():.3f}, {vals.max():.3f}]"
        )

    keep_cols = [
        "SubjectID",
        "StimID",
        DV_TOTAL,
        DV_RELATIONS,
        DV_OBJECTS,
        "n_total_recalled",
        "n_relations_recalled",
        "n_objects_recalled",
    ]
    return df[[c for c in keep_cols if c in df.columns]]


# ---------------------------------------------------------------------------
# Step 2: Build analysis tables
# ---------------------------------------------------------------------------


def build_analysis_tables(df: pd.DataFrame, memory_scores: pd.DataFrame) -> dict:
    """
    Build encoding analysis table by joining memory proportions onto the
    encoding phase of trial_features_all.csv.

    Also builds a long-format table for the dissociation model
    (SVG × memory_type: relations vs objects).

    Returns dict with keys: "enc", "enc_long"
    """
    logger.info("Step 2: Building analysis tables ...")

    enc = df[df["Phase"] == "encoding"].copy()
    enc = enc.rename(
        columns={
            "svg_z": "svg_z_enc",
            "n_fixations": "n_fixations_enc",
            "aoi_prop": "aoi_prop_enc",
            "mean_salience": "mean_salience_enc",
            "mean_salience_relational": "mean_salience_relational_enc",
            "low_n": "low_n_enc",
        }
    ).reset_index(drop=True)

    enc = enc.merge(memory_scores, on=["SubjectID", "StimID"], how="left")

    n_unscored = enc[DV_TOTAL].isna().sum()
    if n_unscored > 0:
        logger.warning(
            f"  {n_unscored} encoding rows have no recall score — "
            "check that recall_by_category.csv covers all participants/stimuli."
        )

    # ------------------------------------------------------------------
    # Long-format table for dissociation model
    # Stack prop_relations and prop_objects with a memory_type factor.
    # Both DVs are already on a 0-1 scale so no further normalisation needed.
    # ------------------------------------------------------------------
    id_cols = (
        ["SubjectID", "StimID", "svg_z_enc"]
        + [c for c in ENC_COVARIATES if c in enc.columns]
        + ["low_n_enc"]
    )

    rel_rows = enc[[c for c in id_cols if c in enc.columns] + [DV_RELATIONS]].copy()
    rel_rows = rel_rows.rename(columns={DV_RELATIONS: "score"})
    rel_rows["memory_type"] = 1  # relations = 1

    obj_rows = enc[[c for c in id_cols if c in enc.columns] + [DV_OBJECTS]].copy()
    obj_rows = obj_rows.rename(columns={DV_OBJECTS: "score"})
    obj_rows["memory_type"] = 0  # objects = 0

    enc_long = pd.concat([rel_rows, obj_rows], ignore_index=True)

    logger.info(f"  enc table:      {len(enc)} rows")
    logger.info(
        f"  enc_long table: {len(enc_long)} rows "
        f"({rel_rows['score'].notna().sum()} relation, "
        f"{obj_rows['score'].notna().sum()} object rows with data)"
    )

    return {"enc": enc, "enc_long": enc_long}


# ---------------------------------------------------------------------------
# Step 3: Exclusions
# ---------------------------------------------------------------------------


def apply_exclusions(tables: dict) -> dict:
    """
    Remove low-n encoding trials (too few AOI transitions for reliable SVG).
    """
    logger.info("Step 3: Applying exclusions ...")

    enc = tables["enc"]
    before = len(enc)
    enc_filtered = enc[~enc["low_n_enc"]].copy()
    logger.info(
        f"  enc: {before} → {len(enc_filtered)} "
        f"(removed {before - len(enc_filtered)}: low_n_enc=True)"
    )

    # Apply same exclusion to long table
    enc_long = tables["enc_long"]
    before_long = len(enc_long)
    enc_long_filtered = enc_long[~enc_long["low_n_enc"]].copy()
    logger.info(
        f"  enc_long: {before_long} → {len(enc_long_filtered)} "
        f"(removed {before_long - len(enc_long_filtered)}: low_n_enc=True)"
    )

    return {"enc": enc_filtered, "enc_long": enc_long_filtered}


# ---------------------------------------------------------------------------
# Step 4: Standardise predictors
# ---------------------------------------------------------------------------


def standardise_tables(filtered: dict) -> dict:
    """
    Z-score continuous predictors within each table.
    Adds {col}_z columns; originals are kept.
    """
    logger.info("Step 4: Standardising predictors ...")

    cols_to_standardise = ["svg_z_enc"] + ENC_COVARIATES

    for key in ("enc", "enc_long"):
        df = filtered[key]
        for col in cols_to_standardise:
            if col not in df.columns:
                continue
            mu, sd = df[col].mean(), df[col].std()
            df[f"{col}_z"] = (df[col] - mu) / sd if sd > 0 else 0.0
        filtered[key] = df

    return filtered
