"""
module3_features.py
===================
Module 3: Feature extraction for the relational replay analysis.

Runs per-participant, in sequence:
    Step 3  — AOI assignment (screen→image transform, point-in-polygon,
               proximity fallback, saliency sampling)
    Step 4  — Build object sequences per trial
    Step 5  — SVG alignment z-scores (all relations + interactional only)
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
    compute_transition_salience,
    svg_alignment,
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

        sal = compute_transition_salience(group, edges_core)

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
                "mean_salience_relational": sal["mean_salience_relational"],
                "mean_salience_nonrelational": sal["mean_salience_nonrelational"],
                "n_relational_fixations": sal["n_relational_fixations"],
                "n_nonrelational_fixations": sal["n_nonrelational_fixations"],
            }
        )

    return pd.DataFrame(records)


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

    # Encoding behavioral — 2 rows per StimID (Q1 and Q2, ordered by row)
    # Pivot to wide format so the merge is 1-to-1 with trial_df rows.
    if enc_path.exists():
        enc_beh = pd.read_csv(enc_path, dtype={"StimID": str})
        wide_enc_rows = []
        for stim_id, grp in enc_beh.groupby("StimID", sort=False):
            grp = grp.reset_index(drop=True)
            row = {"StimID": str(stim_id)}
            for i, (_, qrow) in enumerate(grp.iterrows()):
                qid = f"q{i+1}"
                row[f"enc_{qid}_accuracy"] = qrow.get("Accuracy", np.nan)
                row[f"enc_{qid}_rt_ms"] = qrow.get("RT_ms", np.nan)
                row[f"enc_{qid}_response"] = qrow.get("Response", np.nan)
                row[f"enc_{qid}_correct_key"] = qrow.get("CorrectKey", np.nan)
                row[f"enc_{qid}_question"] = qrow.get("Question", "")
            row["enc_trial_index"] = (
                grp["TrialIndex"].iloc[0] if "TrialIndex" in grp.columns else np.nan
            )
            # Convenience: proportion correct across both questions
            acc_vals = [row[k] for k in row if k.endswith("_accuracy")]
            row["enc_total_correct"] = sum(
                v for v in acc_vals if not (isinstance(v, float) and np.isnan(v))
            )
            wide_enc_rows.append(row)
        enc_wide = pd.DataFrame(wide_enc_rows)
        trial_df = trial_df.merge(enc_wide, on="StimID", how="left")
    else:
        logger.warning(f"  Encoding behavioral not found: {enc_path.name}")
        for col in [
            "enc_q1_accuracy",
            "enc_q2_accuracy",
            "enc_trial_index",
            "enc_total_correct",
        ]:
            trial_df[col] = np.nan

    # Decoding behavioral — free recall format (1 row per image)
    # Columns: StimID, CueText, CueQuestion, FreeResponse, RT_ms, TrialIndex
    if dec_path.exists():
        dec_beh = pd.read_csv(dec_path, dtype={"StimID": str})
        dec_beh = dec_beh.rename(
            columns={
                "FreeResponse": "dec_free_response",
                "RT_ms": "dec_rt_ms",
                "TrialIndex": "dec_trial_index",
                "CueQuestion": "dec_cue_question",
            }
        )
        keep = [
            "StimID",
            "dec_free_response",
            "dec_rt_ms",
            "dec_trial_index",
            "dec_cue_question",
        ]
        dec_beh = dec_beh[[c for c in keep if c in dec_beh.columns]]
        trial_df = trial_df.merge(dec_beh, on="StimID", how="left")
    else:
        logger.warning(f"  Decoding behavioral not found: {dec_path.name}")
        for col in [
            "dec_free_response",
            "dec_rt_ms",
            "dec_trial_index",
            "dec_cue_question",
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
    dec_only_cols = [c for c in trial_df.columns if c.startswith("dec_")]
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
    key_cols = [
        "svg_z",
        "n_fixations",
        "aoi_prop",
        "mean_salience",
        "mean_salience_relational",
        "mean_salience_nonrelational",
    ]
    for col in key_cols:
        if col not in trial_df.columns:
            continue
        n_nan = trial_df[col].isna().sum()
        if n_nan > 0:
            logger.warning(f"    {col}: {n_nan} NaN values")

    # Decoding free response present
    dec_rows = trial_df[trial_df["Phase"] == "decoding"]
    if "dec_free_response" in trial_df.columns:
        n_missing = dec_rows["dec_free_response"].isna().sum()
        if n_missing > 0:
            logger.warning(
                f"    dec_free_response: {n_missing} missing on decoding rows"
            )
        else:
            logger.info(f"    dec_free_response: complete ✓")

    return ok


# ---------------------------------------------------------------------------
# Per-participant runner
# ---------------------------------------------------------------------------


def process_subject(subject_id, force_aoi=False, force=False, rng=None) -> pd.DataFrame:
    logger.info(f"\n{'='*60}")
    logger.info(f"Module 3: {subject_id}")
    logger.info(f"{'='*60}")

    rng = rng or np.random.default_rng(42)

    # Input paths
    fixations_path = config.OUTPUT_EYETRACKING_DIR / f"{subject_id}_fixations.csv"
    aoi_path = config.OUTPUT_FEATURES_DIR / f"{subject_id}_fixations_aoi.csv"
    output_path = config.OUTPUT_FEATURES_DIR / f"{subject_id}_trial_features.csv"

    # Skip if output already exists and force not requested
    if output_path.exists() and not force:
        logger.info(f"  Already processed — loading cached output.")
        return pd.read_csv(output_path, dtype={"StimID": str, "SubjectID": str})

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
        "--force",
        action="store_true",
        help="Reprocess all subjects even if trial_features.csv already exists.",
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

    failed = []

    for subject_id in subject_ids:
        try:
            process_subject(
                subject_id,
                force_aoi=args.force_aoi,
                force=args.force,
                rng=rng,
            )
        except Exception as e:
            logger.error(f"  FAILED {subject_id}: {e}", exc_info=True)
            failed.append(subject_id)

    # Rebuild combined output from ALL per-participant files on disk,
    # so skipped (cached) participants are included alongside newly processed ones.
    all_files = sorted(config.OUTPUT_FEATURES_DIR.glob("*_trial_features.csv"))
    # Exclude the combined file itself if it exists in the same dir
    all_files = [f for f in all_files if f.name != "trial_features_all.csv"]
    if all_files:
        combined = pd.concat(
            [
                pd.read_csv(f, dtype={"StimID": str, "SubjectID": str})
                for f in all_files
            ],
            ignore_index=True,
        )
        combined_path = config.OUTPUT_FEATURES_DIR / "trial_features_all.csv"
        combined.to_csv(combined_path, index=False)
        logger.info(
            f"\nCombined output: {len(combined)} rows from "
            f"{len(all_files)} participants → {combined_path}"
        )

    if failed:
        logger.warning(f"\nFailed subjects ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
