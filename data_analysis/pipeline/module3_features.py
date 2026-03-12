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
    Step 8  — Join behavioral data (encoding + decoding accuracy,
               confidence, RT per question)
    Step 9  — Assemble trial_features.csv (60 rows per participant)
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

        # Core edges: interactional + spatial + functional (social/emotional excluded)
        edges_core = graph_index["all"].get(stim_id, set())

        svg = svg_alignment(
            seq, edges_core, n_permutations=config.SVG_N_PERMUTATIONS, rng=rng
        )

        records.append(
            {
                "StimID": stim_id,
                "Phase": phase,
                "_sequence": seq,  # temporary — dropped before output
                "svg_z": svg["svg_z"],
                "svg_obs": svg["svg_obs"],
                "svg_null_mean": svg["svg_null_mean"],
                "svg_null_std": svg["svg_null_std"],
                "n_transitions": svg["n_transitions"],
                "n_relational": svg["n_relational"],
                "low_n": svg["low_n"],
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

    enc_cols = [
        "StimID",
        "CueText",
        "Accuracy",
        "RT_ms",
        "Response",
        "CorrectKey",
        "TrialIndex",
    ]

    # Encoding behavioral
    if enc_path.exists():
        enc_beh = pd.read_csv(enc_path, dtype={"StimID": str})
        enc_beh = enc_beh[[c for c in enc_cols if c in enc_beh.columns]].copy()
        enc_beh = enc_beh.rename(
            columns={
                "Accuracy": "enc_accuracy",
                "RT_ms": "enc_rt_ms",
                "Response": "enc_response",
                "CorrectKey": "enc_correct_key",
                "TrialIndex": "enc_trial_index",
                "CueText": "CueText",
            }
        )
        enc_merge_cols = [c for c in enc_beh.columns if c != "CueText"]
        trial_df = trial_df.merge(enc_beh[enc_merge_cols], on="StimID", how="left")
    else:
        logger.warning(f"  Encoding behavioral not found: {enc_path.name}")
        for col in [
            "enc_accuracy",
            "enc_rt_ms",
            "enc_response",
            "enc_correct_key",
            "enc_trial_index",
        ]:
            trial_df[col] = np.nan

    # Decoding behavioral — long format (2 rows per image, one per question)
    # Pivot to wide format keyed by StimID so we can join onto the trial table
    if dec_path.exists():
        dec_beh = pd.read_csv(dec_path, dtype={"StimID": str})

        # Separate Q1 and Q2 rows and rename columns with question prefix
        wide_rows = []
        for stim_id, grp in dec_beh.groupby("StimID"):
            row = {"StimID": stim_id}
            for _, qrow in grp.iterrows():
                qid = qrow["QuestionID"].lower()  # "q1" or "q2"
                row[f"{qid}_accuracy"] = qrow.get("Accuracy", np.nan)
                row[f"{qid}_rt_ms"] = qrow.get("RT_ms", np.nan)
                row[f"{qid}_confidence"] = qrow.get("Confidence", np.nan)
                row[f"{qid}_present_order"] = qrow.get("PresentOrder", np.nan)
            wide_rows.append(row)

        dec_wide = pd.DataFrame(wide_rows)

        # Aggregate convenience columns
        acc_cols = [c for c in dec_wide.columns if c.endswith("_accuracy")]
        dec_wide["dec_total_correct"] = dec_wide[acc_cols].sum(axis=1, skipna=True)

        dec_wide["dec_trial_index"] = (
            dec_beh.drop_duplicates(subset="StimID")
            .set_index("StimID")["TrialIndex"]
            .reindex(dec_wide["StimID"])
            .values
        )

        trial_df = trial_df.merge(dec_wide, on="StimID", how="left")
    else:
        logger.warning(f"  Decoding behavioral not found: {dec_path.name}")
        for col in [
            "q1_accuracy",
            "q1_rt_ms",
            "q1_confidence",
            "q1_present_order",
            "q2_accuracy",
            "q2_rt_ms",
            "q2_confidence",
            "q2_present_order",
            "dec_total_correct",
            "dec_trial_index",
        ]:
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
        c for c in trial_df.columns if c.startswith(("q1_", "q2_", "dec_"))
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
    key_cols = ["svg_z", "n_fixations", "aoi_prop", "mean_salience"]
    for col in key_cols:
        if col not in trial_df.columns:
            continue
        n_nan = trial_df[col].isna().sum()
        if n_nan > 0:
            logger.warning(f"    {col}: {n_nan} NaN values")

    # Decoding accuracy present
    dec_rows = trial_df[trial_df["Phase"] == "decoding"]
    if "dec_total_correct" in trial_df.columns:
        n_missing = dec_rows["dec_total_correct"].isna().sum()
        if n_missing > 0:
            logger.warning(
                f"    dec_total_correct: {n_missing} missing on decoding rows"
            )
        else:
            logger.info(
                f"    dec_total_correct: complete ✓  "
                f"(mean={dec_rows['dec_total_correct'].mean():.2f}/2)"
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
