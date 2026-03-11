"""
module3_features.py
===================
Module 3: Feature extraction for the relational replay analysis.

Runs per-participant, in sequence:
    Step 3  — AOI assignment (screen→image transform, point-in-polygon,
               proximity fallback, saliency sampling)
    Step 4  — Build object sequences per trial
    Step 5  — SVG alignment z-scores (all relations + interactional only)
    Step 6  — Symbolic LCS and Kendall's tau (enc vs dec)
    Step 7  — Per-trial covariates (fixation count, AOI proportion,
               mean saliency)
    Step 8  — Join behavioral data (encoding Q1/Q2 accuracy per image in
               wide format; decoding free-recall RT and writing length)
    Step 9  — Assemble trial_features.csv (60 rows per participant:
               30 encoding rows — one per image, both questions merged;
               30 decoding rows — one per image)
    Step 10 — Validate outputs

Outputs (per participant):
    output/features/{SubjectID}_fixations_aoi.csv   — AOI-enriched fixations
    output/features/{SubjectID}_trial_features.csv  — 90-row feature table

Final output (all participants concatenated):
    output/features/trial_features_all.csv

Usage:
    # All participants
    python module3_features.py

    # Single participant
    python module3_features.py --subject Encode-Decode_Experiment-1-1

    # Force recompute AOI even if cached
    python module3_features.py --force-aoi
"""

import argparse
import logging
import re
import sys
import warnings
from pathlib import Path

import config
import numpy as np
import pandas as pd
from pipeline.misc import get_subject_ids, setup_logging
from pipeline.module_3.aoi import assign_aoi
from pipeline.module_3.metrics import (
    build_object_sequence,
    kendall_tau_shared,
    svg_alignment,
    symbolic_lcs,
)
from pipeline.module_3.scene_graph import build_graph_index

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Step 3: AOI assignment
# ---------------------------------------------------------------------------


def _run_aoi(
    subject_id: str,
    fixations_df: pd.DataFrame,
    aoi_path: Path,
    force: bool,
) -> pd.DataFrame:
    if aoi_path.exists() and not force:
        logger.info(f"  Step 3: Loading cached AOI fixations ...")
        return pd.read_csv(aoi_path, dtype={"StimID": str})

    logger.info(f"  Step 3: AOI assignment ...")
    enriched = assign_aoi(fixations_df)
    aoi_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(aoi_path, index=False)
    logger.info(f"    Written → {aoi_path.name}")
    return enriched


# ---------------------------------------------------------------------------
# Step 4–5: Sequences and SVG alignment
# ---------------------------------------------------------------------------


def _build_sequences_and_svg(
    fixations_aoi: pd.DataFrame,
    graph_index: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    For each trial (StimID × Phase), build the object sequence
    and compute SVG alignment scores.

    Returns a DataFrame with one row per trial containing sequence and SVG columns.
    """
    logger.info(f"  Steps 4–5: Sequences and SVG alignment ...")

    records = []

    groups = fixations_aoi.groupby(["StimID", "Phase"], dropna=False, sort=False)

    for (stim_id, phase), group in groups:
        stim_id = str(stim_id)
        seq = build_object_sequence(group)

        edges_all = graph_index["all"].get(stim_id, set())
        edges_inter = graph_index["interactional"].get(stim_id, set())

        svg_all = svg_alignment(
            seq, edges_all, n_permutations=config.SVG_N_PERMUTATIONS, rng=rng
        )
        svg_inter = svg_alignment(
            seq, edges_inter, n_permutations=config.SVG_N_PERMUTATIONS, rng=rng
        )

        records.append(
            {
                "StimID": stim_id,
                "Phase": phase,
                "_sequence": seq,  # temporary — dropped before output
                "svg_z_all": svg_all["svg_z"],
                "svg_obs_all": svg_all["svg_obs"],
                "svg_null_mean_all": svg_all["svg_null_mean"],
                "svg_null_std_all": svg_all["svg_null_std"],
                "svg_z_inter": svg_inter["svg_z"],
                "svg_obs_inter": svg_inter["svg_obs"],
                "svg_null_mean_inter": svg_inter["svg_null_mean"],
                "svg_null_std_inter": svg_inter["svg_null_std"],
                "n_transitions": svg_all["n_transitions"],
                "n_relational_all": svg_all["n_relational"],
                "n_relational_inter": svg_inter["n_relational"],
                "low_n": svg_all["low_n"],
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 6: Pairwise LCS and Kendall's tau
# ---------------------------------------------------------------------------


def _compute_pairwise_metrics(trial_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each StimID, compute LCS and Kendall's tau between:
        enc vs dec  (primary — encoding scanpath vs retrieval scanpath)

    Results are stored on the decoding row for each StimID.
    """
    logger.info(f"  Step 6: Pairwise LCS / Kendall's tau ...")

    # Index sequences by (StimID, Phase) — no ViewingNumber needed
    seq_index = {
        (str(row["StimID"]), row["Phase"]): row["_sequence"]
        for _, row in trial_df.iterrows()
    }

    pairwise_cols = [
        "lcs_enc_dec",
        "lcs_len_enc_dec",
        "tau_enc_dec",
        "tau_p_enc_dec",
        "n_shared_enc_dec",
    ]
    for col in pairwise_cols:
        trial_df[col] = np.nan

    for stim_id in trial_df["StimID"].unique():
        stim_id = str(stim_id)

        seq_enc = seq_index.get((stim_id, "encoding"), [])
        seq_dec = seq_index.get((stim_id, "decoding"), [])

        if seq_enc and seq_dec:
            lcs = symbolic_lcs(seq_enc, seq_dec)
            tau = kendall_tau_shared(seq_enc, seq_dec)
            mask = (trial_df["StimID"] == stim_id) & (trial_df["Phase"] == "decoding")
            trial_df.loc[mask, "lcs_enc_dec"] = lcs["lcs_score"]
            trial_df.loc[mask, "lcs_len_enc_dec"] = lcs["lcs_length"]
            trial_df.loc[mask, "tau_enc_dec"] = tau["tau"]
            trial_df.loc[mask, "tau_p_enc_dec"] = tau["tau_p"]
            trial_df.loc[mask, "n_shared_enc_dec"] = tau["n_shared"]

    return trial_df


# ---------------------------------------------------------------------------
# Step 7: Per-trial covariates
# ---------------------------------------------------------------------------


def _compute_covariates(
    fixations_aoi: pd.DataFrame,
    trial_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute fixation count, AOI proportion, and mean saliency per trial.
    Joined onto trial_df by StimID × Phase.
    """
    logger.info(f"  Step 7: Computing covariates ...")

    covariate_rows = []

    groups = fixations_aoi.groupby(["StimID", "Phase"], dropna=False, sort=False)

    for (stim_id, phase), group in groups:
        n_fixations = len(group)
        n_aoi = (group["AssignmentMethod"] != "none").sum()
        aoi_prop = n_aoi / n_fixations if n_fixations > 0 else np.nan
        mean_salience = group["SalienceAtFixation"].mean()

        covariate_rows.append(
            {
                "StimID": str(stim_id),
                "Phase": phase,
                "n_fixations": n_fixations,
                "n_aoi": n_aoi,
                "aoi_prop": round(aoi_prop, 4) if not np.isnan(aoi_prop) else np.nan,
                "mean_salience": (
                    round(mean_salience, 6) if not np.isnan(mean_salience) else np.nan
                ),
            }
        )

    cov_df = pd.DataFrame(covariate_rows)
    cov_df["StimID"] = cov_df["StimID"].astype(str)
    trial_df["StimID"] = trial_df["StimID"].astype(str)

    return trial_df.merge(cov_df, on=["StimID", "Phase"], how="left")


# ---------------------------------------------------------------------------
# Step 8: Join behavioral data
# ---------------------------------------------------------------------------


def _join_behavioral(
    subject_id: str,
    trial_df: pd.DataFrame,
) -> pd.DataFrame:
    logger.info(f"  Step 8: Joining behavioral data ...")

    enc_path = config.OUTPUT_BEHAVIORAL_DIR / f"{subject_id}_encoding.csv"
    dec_path = config.OUTPUT_BEHAVIORAL_DIR / f"{subject_id}_decoding.csv"

    # ------------------------------------------------------------------
    # Encoding behavioral — 60 rows (2 per image), pivot to wide (30 rows)
    # q1 = the question shown earlier (lower TrialIndex for that StimID)
    # q2 = the question shown later  (higher TrialIndex for that StimID)
    # ------------------------------------------------------------------
    if enc_path.exists():
        enc_beh = pd.read_csv(enc_path, dtype={"StimID": str})
        enc_beh = enc_beh.sort_values(["StimID", "TrialIndex"])

        enc_wide_rows = []
        for stim_id, grp in enc_beh.groupby("StimID", sort=False):
            grp = grp.reset_index(drop=True)
            row = {"StimID": stim_id}
            for i, (_, qrow) in enumerate(grp.iterrows(), start=1):
                row[f"enc_q{i}_accuracy"] = qrow.get("Accuracy", np.nan)
                row[f"enc_q{i}_rt_ms"] = qrow.get("RT_ms", np.nan)
                row[f"enc_q{i}_trial_index"] = qrow.get("TrialIndex", np.nan)
            enc_wide_rows.append(row)

        enc_wide = pd.DataFrame(enc_wide_rows)
        trial_df = trial_df.merge(enc_wide, on="StimID", how="left")
    else:
        logger.warning(f"  Encoding behavioral not found: {enc_path.name}")
        for col in [
            "enc_q1_accuracy",
            "enc_q1_rt_ms",
            "enc_q1_trial_index",
            "enc_q2_accuracy",
            "enc_q2_rt_ms",
            "enc_q2_trial_index",
        ]:
            trial_df[col] = np.nan

    # ------------------------------------------------------------------
    # Decoding behavioral — 30 rows (1 per image), flat join
    # Adds: dec_rt_ms, dec_trial_index, writing_length
    # writing_length = alphanumeric character count of the free response
    # ------------------------------------------------------------------
    if dec_path.exists():
        dec_beh = pd.read_csv(dec_path, dtype={"StimID": str})

        dec_beh["writing_length"] = dec_beh["FreeResponse"].apply(
            lambda x: len(re.sub(r"[^A-Za-z0-9]", "", str(x))) if pd.notna(x) else 0
        )

        dec_join = dec_beh[["StimID", "RT_ms", "TrialIndex", "writing_length"]].rename(
            columns={
                "RT_ms": "dec_rt_ms",
                "TrialIndex": "dec_trial_index",
            }
        )
        trial_df = trial_df.merge(dec_join, on="StimID", how="left")
    else:
        logger.warning(f"  Decoding behavioral not found: {dec_path.name}")
        for col in ["dec_rt_ms", "dec_trial_index", "writing_length"]:
            trial_df[col] = np.nan

    return trial_df


# ---------------------------------------------------------------------------
# Step 9: Assemble
# ---------------------------------------------------------------------------


def _assemble(subject_id: str, trial_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"  Step 9: Assembling trial features ...")

    trial_df = trial_df.drop(columns=["_sequence"], errors="ignore")
    trial_df.insert(0, "SubjectID", subject_id)

    # Nullify phase-specific behavioral columns on the wrong phase rows.
    # The StimID-level merge in Step 8 propagates dec columns onto encoding
    # rows and enc columns onto decoding rows — correct values, wrong rows.
    dec_only_cols = [
        c for c in trial_df.columns if c.startswith("dec_") or c == "writing_length"
    ]
    enc_only_cols = [c for c in trial_df.columns if c.startswith("enc_")]
    trial_df.loc[trial_df["Phase"] == "encoding", dec_only_cols] = np.nan
    trial_df.loc[trial_df["Phase"] == "decoding", enc_only_cols] = np.nan

    # Round floats for readability
    float_cols = trial_df.select_dtypes(include=[np.floating]).columns
    trial_df[float_cols] = trial_df[float_cols].round(6)

    return trial_df


# ---------------------------------------------------------------------------
# Step 10: Validate
# ---------------------------------------------------------------------------


def _validate(subject_id: str, trial_df: pd.DataFrame) -> bool:
    logger.info(f"  Step 10: Validating ...")
    ok = True

    # Row count — 30 encoding + 30 decoding = 60
    expected_rows = config.N_ENCODING_TRIALS + config.N_DECODING_TRIALS
    if len(trial_df) != expected_rows:
        logger.warning(f"    Row count: {len(trial_df)} (expected {expected_rows})")
        ok = False
    else:
        logger.info(f"    Row count: {len(trial_df)} ✓")

    # low_n trials
    n_low = trial_df["low_n"].sum()
    if n_low > 0:
        logger.warning(
            f"    low_n trials: {n_low} (< {config.MIN_VALID_TRANSITIONS} transitions)"
        )

    # NaN rates for key columns
    key_cols = ["svg_z_all", "svg_z_inter", "n_fixations", "aoi_prop", "mean_salience"]
    for col in key_cols:
        if col not in trial_df.columns:
            continue
        n_nan = trial_df[col].isna().sum()
        if n_nan > 0:
            logger.warning(f"    {col}: {n_nan} NaN values")

    # writing_length present on decoding rows
    dec_rows = trial_df[trial_df["Phase"] == "decoding"]
    if "writing_length" in trial_df.columns:
        n_missing = dec_rows["writing_length"].isna().sum()
        if n_missing > 0:
            logger.warning(f"    writing_length: {n_missing} missing on decoding rows")
        else:
            logger.info(
                f"    writing_length: complete ✓  "
                f"(mean={dec_rows['writing_length'].mean():.1f} chars)"
            )

    return ok


# ---------------------------------------------------------------------------
# Per-participant runner
# ---------------------------------------------------------------------------


def process_subject(subject_id, force_aoi=False, rng=None) -> pd.DataFrame:
    logger.info(f"\n{'='*60}")
    logger.info(f"Module 3: {subject_id}")
    logger.info(f"{'='*60}")

    rng = rng or np.random.default_rng(42)

    # Input paths
    fixations_path = config.OUTPUT_EYETRACKING_DIR / f"{subject_id}_fixations.csv"
    aoi_path = config.OUTPUT_FEATURES_DIR / f"{subject_id}_fixations_aoi.csv"
    output_path = config.OUTPUT_FEATURES_DIR / f"{subject_id}_trial_features.csv"

    if not fixations_path.exists():
        logger.error(f"  Fixations file not found: {fixations_path}")
        return pd.DataFrame()

    # Load fixations
    fixations_df = pd.read_csv(fixations_path, dtype={"StimID": str})
    logger.info(f"  Loaded {len(fixations_df)} fixations.")

    # Load graph index (cached after first call)
    graph_index = build_graph_index()

    # Step 3 — AOI assignment
    fixations_aoi = _run_aoi(subject_id, fixations_df, aoi_path, force=force_aoi)

    # Steps 4–5 — sequences + SVG alignment
    trial_df = _build_sequences_and_svg(fixations_aoi, graph_index, rng)

    # Step 6 — pairwise LCS / tau
    trial_df = _compute_pairwise_metrics(trial_df)

    # Step 7 — covariates
    trial_df = _compute_covariates(fixations_aoi, trial_df)

    # Step 8 — behavioral join
    trial_df = _join_behavioral(subject_id, trial_df)

    # Step 9 — assemble
    trial_df = _assemble(subject_id, trial_df)

    # Step 10 — validate
    _validate(subject_id, trial_df)

    # Write per-participant output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trial_df.to_csv(output_path, index=False)
    logger.info(f"  Written → {output_path.name}")

    return trial_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    setup_logging(level=config.LOG_LEVEL)

    parser = argparse.ArgumentParser(
        description="Module 3: Extract trial-level features for relational replay analysis."
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Run for a single subject ID (default: all discovered subjects)",
    )
    parser.add_argument(
        "--force-aoi",
        action="store_true",
        help="Recompute AOI assignment even if cached _fixations_aoi.csv exists",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for SVG permutation null (default: 42)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.subject:
        subject_ids = [args.subject]
    else:
        subject_ids = get_subject_ids()
        logger.info(f"Discovered {len(subject_ids)} subjects.")

    all_dfs = []
    failed = []

    for subject_id in subject_ids:
        try:
            df = process_subject(subject_id, force_aoi=args.force_aoi, rng=rng)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            logger.error(f"  FAILED {subject_id}: {e}", exc_info=True)
            failed.append(subject_id)

    # Concatenate and write combined output
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = config.OUTPUT_FEATURES_DIR / "trial_features_all.csv"
        combined.to_csv(combined_path, index=False)
        logger.info(f"\nCombined output: {len(combined)} rows → {combined_path}")

    if failed:
        logger.warning(f"\nFailed subjects ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
