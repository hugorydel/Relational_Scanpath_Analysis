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

from .constants import (
    DEFAULT_FLAGS_PATH,
    DEC_BETWEEN_COVARIATES,
    DEC_COVARIATES,
    DV_OBJECTS,
    DV_RELATIONS,
    DV_TOTAL,
    ENC_BETWEEN_COVARIATES,
    ENC_COVARIATES,
)

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

    # Decoding table — blank-screen recall phase
    dec = df[df["Phase"] == "decoding"].copy()
    dec = dec.rename(
        columns={
            "svg_z": "svg_z_dec",
            "n_fixations": "n_fixations_dec",
            "aoi_prop": "aoi_prop_dec",
            "mean_salience": "mean_salience_dec",
            "mean_salience_relational": "mean_salience_relational_dec",
            "low_n": "low_n_dec",
        }
    ).reset_index(drop=True)
    dec = dec.merge(memory_scores, on=["SubjectID", "StimID"], how="left")
    logger.info(f"  dec table:      {len(dec)} rows")

    return {"enc": enc, "enc_long": enc_long, "dec": dec}


# ---------------------------------------------------------------------------
# Step 3: Exclusions
# ---------------------------------------------------------------------------


def apply_exclusions(tables: dict, flags_path: Path | None = None) -> dict:
    """
    Remove low-n encoding trials (too few AOI transitions for reliable SVG)
    and wrong-image pairs flagged via review_wrong_images.py.

    flags_path : optional path to wrong_image_flags.csv.
                 Defaults to DEFAULT_FLAGS_PATH; silently skipped if absent.
    """
    logger.info("Step 3: Applying exclusions ...")

    enc = tables["enc"]
    before = len(enc)
    enc_filtered = enc[~enc["low_n_enc"]].copy()
    logger.info(
        f"  enc: {before} → {len(enc_filtered)} "
        f"(removed {before - len(enc_filtered)}: low_n_enc=True)"
    )

    # ------------------------------------------------------------------
    # Wrong-image exclusions
    # ------------------------------------------------------------------
    if flags_path is None:
        flags_path = DEFAULT_FLAGS_PATH

    if flags_path.exists():
        flags_df = pd.read_csv(flags_path, dtype={"SubjectID": str, "StimID": str})
        wrong = set(
            zip(
                flags_df.loc[flags_df["flagged"] == 1, "SubjectID"],
                flags_df.loc[flags_df["flagged"] == 1, "StimID"],
            )
        )
        if wrong:
            mask = enc_filtered.apply(
                lambda r: (r["SubjectID"], r["StimID"]) in wrong, axis=1
            )
            enc_filtered = enc_filtered[~mask].copy()
            logger.info(
                f"  enc: removed {mask.sum()} wrong-image pairs "
                f"({len(wrong)} unique flagged pairs in flags file)."
            )
        else:
            logger.info("  No wrong-image pairs flagged — nothing extra removed.")
    else:
        logger.info(
            f"  No flags file found at {flags_path.name} — skipping wrong-image exclusion."
        )

    # ------------------------------------------------------------------
    # Within-image SVG mean-centering (L1/L2 decomposition)
    # Computed here, AFTER exclusions, so image means are based only on
    # participants with reliable SVG estimates (low_n_enc=False).
    #
    # svg_z_enc_image_mean : per-StimID mean of svg_z_enc across valid participants
    # svg_z_enc_within     : participant deviation from that image mean
    # ------------------------------------------------------------------
    stim_svg_mean = (
        enc_filtered.groupby("StimID")["svg_z_enc"]
        .mean()
        .rename("svg_z_enc_image_mean")
    )
    enc_filtered = enc_filtered.merge(stim_svg_mean, on="StimID", how="left")
    enc_filtered["svg_z_enc_within"] = (
        enc_filtered["svg_z_enc"] - enc_filtered["svg_z_enc_image_mean"]
    )
    logger.info(
        f"  SVG decomposition (post-exclusion): "
        f"svg_z_enc_within mean={enc_filtered['svg_z_enc_within'].mean():.4f} "
        f"(should be ~0), sd={enc_filtered['svg_z_enc_within'].std():.4f}"
    )

    # ------------------------------------------------------------------
    # Per-image variance controls
    #
    # svg_z_enc_within_sd : SD of within-image SVG across participants.
    #   Images where everyone scans similarly (low SD) contribute less
    #   reliable predictor signal — e.g. 2348899 (tie), svg_mean=3.23.
    #
    # prop_total_image_sd : SD of prop_total across participants per image.
    #   Images where recall is near-uniform (low SD) have little outcome
    #   variance for the regression to detect — e.g. gist-dominant images.
    # ------------------------------------------------------------------
    stim_svg_within_sd = (
        enc_filtered.groupby("StimID")["svg_z_enc_within"]
        .std()
        .rename("svg_z_enc_within_sd")
    )
    enc_filtered = enc_filtered.merge(stim_svg_within_sd, on="StimID", how="left")

    if DV_TOTAL in enc_filtered.columns:
        stim_prop_sd = (
            enc_filtered.groupby("StimID")[DV_TOTAL]
            .std()
            .rename("prop_total_image_sd")
        )
        enc_filtered = enc_filtered.merge(stim_prop_sd, on="StimID", how="left")
    else:
        enc_filtered["prop_total_image_sd"] = np.nan

    logger.info(
        f"  Variance controls: "
        f"svg_within_sd mean={enc_filtered['svg_z_enc_within_sd'].mean():.3f}, "
        f"prop_total_sd mean={enc_filtered['prop_total_image_sd'].mean():.3f}"
    )

    # Apply same exclusions to long table
    enc_long = tables["enc_long"]
    before_long = len(enc_long)
    enc_long_filtered = enc_long[~enc_long["low_n_enc"]].copy()

    if flags_path.exists() and wrong:
        mask_long = enc_long_filtered.apply(
            lambda r: (r["SubjectID"], r["StimID"]) in wrong, axis=1
        )
        enc_long_filtered = enc_long_filtered[~mask_long].copy()

    logger.info(
        f"  enc_long: {before_long} → {len(enc_long_filtered)} "
        f"(low_n + wrong-image exclusions)"
    )

    # Propagate decomposition + variance columns into enc_long via SubjectID × StimID
    decomp_cols = enc_filtered[
        ["SubjectID", "StimID", "svg_z_enc_within", "svg_z_enc_image_mean",
         "svg_z_enc_within_sd", "prop_total_image_sd"]
    ]
    enc_long_filtered = enc_long_filtered.merge(
        decomp_cols, on=["SubjectID", "StimID"], how="left"
    )

    # ------------------------------------------------------------------
    # Decoding exclusions + within-image SVG decomposition
    # ------------------------------------------------------------------
    dec = tables.get("dec", pd.DataFrame())
    if not dec.empty:
        before_dec = len(dec)
        dec_filtered = dec[~dec["low_n_dec"]].copy()
        logger.info(
            f"  dec: {before_dec} → {len(dec_filtered)} "
            f"(removed {before_dec - len(dec_filtered)}: low_n_dec=True)"
        )

        if flags_path.exists() and wrong:
            mask_dec = dec_filtered.apply(
                lambda r: (r["SubjectID"], r["StimID"]) in wrong, axis=1
            )
            dec_filtered = dec_filtered[~mask_dec].copy()
            logger.info(f"  dec: removed {mask_dec.sum()} wrong-image pairs.")

        stim_svg_dec_mean = (
            dec_filtered.groupby("StimID")["svg_z_dec"]
            .mean()
            .rename("svg_z_dec_image_mean")
        )
        dec_filtered = dec_filtered.merge(stim_svg_dec_mean, on="StimID", how="left")
        dec_filtered["svg_z_dec_within"] = (
            dec_filtered["svg_z_dec"] - dec_filtered["svg_z_dec_image_mean"]
        )
        logger.info(
            f"  Dec SVG decomposition: "
            f"svg_z_dec_within mean={dec_filtered['svg_z_dec_within'].mean():.4f} "
            f"(should be ~0), sd={dec_filtered['svg_z_dec_within'].std():.4f}"
        )
    else:
        dec_filtered = dec

    return {"enc": enc_filtered, "enc_long": enc_long_filtered, "dec": dec_filtered}


# ---------------------------------------------------------------------------
# Step 4: Standardise predictors
# ---------------------------------------------------------------------------


def standardise_tables(filtered: dict) -> dict:
    """
    Z-score continuous predictors within each table.
    Adds {col}_z columns; originals are kept.
    """
    logger.info("Step 4: Standardising predictors ...")

    enc_cols = ["svg_z_enc", "svg_z_enc_within"] + ENC_COVARIATES + ENC_BETWEEN_COVARIATES
    dec_cols = ["svg_z_dec", "svg_z_dec_within"] + DEC_COVARIATES + DEC_BETWEEN_COVARIATES

    table_col_map = {
        "enc":      enc_cols,
        "enc_long": enc_cols,
        "dec":      dec_cols,
    }

    for key, cols in table_col_map.items():
        df = filtered.get(key)
        if df is None or df.empty:
            continue
        for col in cols:
            if col not in df.columns:
                continue
            mu, sd = df[col].mean(), df[col].std()
            df[f"{col}_z"] = (df[col] - mu) / sd if sd > 0 else 0.0
        filtered[key] = df

    return filtered