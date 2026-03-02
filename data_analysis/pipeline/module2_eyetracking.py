"""
Module 2: Eye-Tracking Extractor
==================================
Reads an EyeLink .edf file, segments the continuous fixation/saccade
stream into per-trial windows using TRIALID message boundaries, joins
to StimID via the behavioral CSVs produced by Module 1, and writes
two clean CSVs per participant.

Usage (standalone):
    python module2_eyetracking.py --subject sub01 \\
        --edf-dir data_eyetracking/ \\
        --beh-dir output/behavioral/ \\
        --output-dir output/eyetracking/

Usage (from orchestrator):
    from pipeline.module2_eyetracking import process_subject
    process_subject("sub01",
                    edf_dir="data_eyetracking/",
                    beh_dir="output/behavioral/",
                    output_dir="output/eyetracking/")
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data_analysis.config import DISPLAY_HEIGHT_PX, DISPLAY_WIDTH_PX

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1a: Filter invalid events
# ---------------------------------------------------------------------------


def filter_invalid_events(
    fixations: pd.DataFrame,
    saccades: pd.DataFrame,
    subject_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove fixations and saccades with invalid gaze coordinates.

    Invalid events include:
      - Out-of-bounds gaze (outside screen dimensions)
      - EyeLink blink/tracker-loss artifacts, which are reported as
        extreme coordinate values (e.g. 1e8) and always fall outside
        screen bounds, so the same filter catches both.

    Fixations are filtered on average gaze position (axp, ayp).
    Saccades are filtered on start position (sxp, syp).
    """

    def _valid_mask(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
        return (
            df[x_col].notna()
            & df[y_col].notna()
            & (df[x_col] >= 0)
            & (df[x_col] <= DISPLAY_WIDTH_PX)
            & (df[y_col] >= 0)
            & (df[y_col] <= DISPLAY_HEIGHT_PX)
        )

    n_fix_before = len(fixations)
    n_sacc_before = len(saccades)

    if len(fixations) > 0 and {"axp", "ayp"}.issubset(fixations.columns):
        fix_mask = _valid_mask(fixations, "axp", "ayp")
        fixations = fixations[fix_mask].copy()

    if len(saccades) > 0 and {"sxp", "syp"}.issubset(saccades.columns):
        sacc_mask = _valid_mask(saccades, "sxp", "syp")
        saccades = saccades[sacc_mask].copy()

    n_fix_removed = n_fix_before - len(fixations)
    n_sacc_removed = n_sacc_before - len(saccades)

    if n_fix_removed > 0 or n_sacc_removed > 0:
        logger.info(
            f"  [{subject_id}] Filtered invalid events: "
            f"{n_fix_removed} fixations, {n_sacc_removed} saccades removed "
            f"(out-of-bounds / blink artifacts)."
        )
    else:
        logger.info(
            f"  [{subject_id}] No invalid events found — all coordinates within screen bounds."
        )

    return fixations, saccades


# ---------------------------------------------------------------------------
# Step 1: Load EDF
# ---------------------------------------------------------------------------


def load_edf(edf_path: Path) -> dict:
    """
    Load an EyeLink .edf file via eyelinkio.

    Returns a dict with keys:
        'messages'  : pd.DataFrame  — stime (ms), msg (str)
        'fixations' : pd.DataFrame  — stime, etime, axp, ayp, eye (all ms/px)
        'saccades'  : pd.DataFrame  — stime, etime, sxp, syp, exp, eyp, pv, eye
    """
    try:
        import eyelinkio
    except ImportError:
        raise ImportError(
            "eyelinkio is required for Module 2. Install with: pip install eyelinkio"
        )

    logger.info(f"  Loading EDF: {edf_path.name} ...")
    edf = eyelinkio.read_edf(str(edf_path))
    discrete = edf.discrete

    logger.info(f"  edf.discrete keys: {list(discrete.keys())}")

    def structured_to_df(key: str) -> pd.DataFrame:
        """Convert a numpy structured array from discrete to a DataFrame."""
        if key not in discrete:
            logger.warning(
                f"  Key '{key}' not found in edf.discrete — returning empty DataFrame."
            )
            return pd.DataFrame()
        arr = discrete[key]
        if arr is None or len(arr) == 0:
            return pd.DataFrame()
        if hasattr(arr, "dtype") and arr.dtype.names:
            return pd.DataFrame(arr)
        return pd.DataFrame(arr)

    messages = structured_to_df("messages")
    fixations = structured_to_df("fixations")
    saccades = structured_to_df("saccades")

    # --- Decode bytes → str in messages ---
    if len(messages) > 0:
        msg_col = messages.columns[-1]  # text column is always last
        messages[msg_col] = messages[msg_col].apply(
            lambda x: (
                x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x)
            )
        )
        messages = messages.rename(
            columns={messages.columns[0]: "stime", msg_col: "msg"}
        )

    # --- Convert times from seconds → ms (eyelinkio returns seconds) ---
    def _to_ms(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in cols:
            if col in df.columns:
                # If max value is small it's in seconds; if large already ms
                if df[col].abs().max() < 100_000:
                    df[col] = (df[col] * 1000).round().astype("Int64")
                else:
                    df[col] = df[col].round().astype("Int64")
        return df

    messages = _to_ms(messages, ["stime"])
    fixations = _to_ms(fixations, ["stime", "etime"])
    saccades = _to_ms(saccades, ["stime", "etime"])

    logger.info(
        f"  Loaded: {len(messages)} messages, "
        f"{len(fixations)} fixations, {len(saccades)} saccades."
    )

    return {"messages": messages, "fixations": fixations, "saccades": saccades}


# ---------------------------------------------------------------------------
# Step 2: Build trial table from messages
# ---------------------------------------------------------------------------

# Regex patterns for message parsing
_RE_TRIALID = re.compile(r"^TRIALID\s+(\d+)$")
_RE_PRACTICE_SAMPLE = re.compile(r"!V TRIAL_VAR PracticeList\.Sample\s+(\d+)")
_RE_TEST_SAMPLE = re.compile(r"!V TRIAL_VAR TestList\.Sample\s+(\d+)")


def build_trial_table(messages: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    """
    Parse the messages DataFrame to build a trial table with one row per
    EDF trial (encoding or decoding only — any trial with neither
    PracticeList.Sample nor TestList.Sample is skipped).

    Returns a DataFrame with columns:
        trial_id, trial_start_ms, trial_end_ms, block, trial_index
    """
    if len(messages) == 0:
        logger.error(f"  [{subject_id}] No messages found in EDF.")
        return pd.DataFrame()

    rows = messages.to_dict("records")
    n = len(rows)
    trial_rows = []

    # Identify indices of all TRIALID messages
    trialid_indices = [
        i for i, r in enumerate(rows) if _RE_TRIALID.match(r["msg"].strip())
    ]

    logger.info(f"  [{subject_id}] Found {len(trialid_indices)} TRIALID markers.")

    for pos, idx in enumerate(trialid_indices):
        trial_id = int(_RE_TRIALID.match(rows[idx]["msg"].strip()).group(1))
        trial_start = rows[idx]["stime"]

        # Trial ends at next TRIALID timestamp, or at last message for final trial
        if pos + 1 < len(trialid_indices):
            trial_end = rows[trialid_indices[pos + 1]]["stime"]
        else:
            trial_end = rows[-1]["stime"]

        # Collect messages within this trial block
        block_msgs = [
            rows[j]["msg"]
            for j in range(
                idx, trialid_indices[pos + 1] if pos + 1 < len(trialid_indices) else n
            )
        ]

        # Determine block type and trial index
        practice_match = next(
            (m for msg in block_msgs for m in [_RE_PRACTICE_SAMPLE.search(msg)] if m),
            None,
        )
        test_match = next(
            (m for msg in block_msgs for m in [_RE_TEST_SAMPLE.search(msg)] if m), None
        )

        if practice_match:
            block = "encoding"
            trial_index = int(practice_match.group(1))
        elif test_match:
            block = "decoding"
            trial_index = int(test_match.group(1))
        else:
            # No sample variable found — skip (distractor, exploratory, or unknown)
            logger.debug(
                f"  [{subject_id}] TRIALID {trial_id}: no PracticeList.Sample or "
                f"TestList.Sample found — skipping."
            )
            continue

        trial_rows.append(
            {
                "TrialID": trial_id,
                "TrialIndex": trial_index,
                "Phase": block,
                "trial_start_ms": int(trial_start),
                "trial_end_ms": int(trial_end),
            }
        )

    trial_table = pd.DataFrame(trial_rows)

    if len(trial_table) > 0:
        n_enc = (trial_table["Phase"] == "encoding").sum()
        n_dec = (trial_table["Phase"] == "decoding").sum()
        logger.info(
            f"  [{subject_id}] Trial table built: "
            f"{n_enc} encoding trials, {n_dec} decoding trials."
        )
    else:
        logger.warning(f"  [{subject_id}] Trial table is empty after parsing.")

    return trial_table


# ---------------------------------------------------------------------------
# Step 3: Join to StimID via behavioral CSVs
# ---------------------------------------------------------------------------


def join_stim_ids(
    trial_table: pd.DataFrame,
    beh_dir: Path,
    subject_id: str,
) -> pd.DataFrame:
    """
    Load encoding and decoding behavioral CSVs from Module 1 and join
    StimID + CueText onto the trial table via TrialIndex.
    """
    enc_path = beh_dir / f"{subject_id}_encoding.csv"
    dec_path = beh_dir / f"{subject_id}_decoding.csv"

    for p in [enc_path, dec_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Behavioral CSV not found: {p}\n"
                f"Run Module 1 for {subject_id} first."
            )

    enc_beh = pd.read_csv(enc_path, dtype={"StimID": str})[
        ["TrialIndex", "StimID", "CueText"]
    ]
    dec_beh = pd.read_csv(dec_path, dtype={"StimID": str})[
        ["TrialIndex", "StimID", "CueText"]
    ]

    enc_trials = trial_table[trial_table["Phase"] == "encoding"].copy()
    dec_trials = trial_table[trial_table["Phase"] == "decoding"].copy()

    enc_joined = enc_trials.merge(enc_beh, on="TrialIndex", how="left")
    dec_joined = dec_trials.merge(dec_beh, on="TrialIndex", how="left")

    joined = pd.concat([enc_joined, dec_joined], ignore_index=True)

    # Flag any unmatched rows
    missing = joined["StimID"].isna().sum()
    if missing > 0:
        logger.warning(
            f"  [{subject_id}] {missing} trial(s) could not be matched to a StimID. "
            f"Check TrialIndex alignment between EDF and behavioral CSVs."
        )

    logger.info(
        f"  [{subject_id}] StimID join complete. {len(joined)} trials with StimID."
    )
    return joined


# ---------------------------------------------------------------------------
# Step 3a: Add ViewingNumber to encoding trials
# ---------------------------------------------------------------------------


def add_viewing_number(trial_table: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ViewingNumber column to the trial table.

    For encoding trials: ViewingNumber = 1 for the first appearance of
    each StimID (by TrialID order), 2 for the second appearance.
    For decoding trials: ViewingNumber = None (single retrieval event).

    Assumes each StimID appears at most twice in encoding, which matches
    the experiment design. Logs a warning if any StimID appears more than
    twice (future-proofing for design changes).
    """
    trial_table = trial_table.copy()
    trial_table["ViewingNumber"] = None

    enc_mask = trial_table["Phase"] == "encoding"
    enc = trial_table[enc_mask].sort_values("TrialID")

    # Rank each StimID's appearances in TrialID order
    viewing_nums = enc.groupby("StimID").cumcount() + 1  # 1-indexed

    # Warn if any StimID appears more than twice
    max_views = viewing_nums.max()
    if max_views > 2:
        over = enc.groupby("StimID").size()
        over = over[over > 2]
        logger.warning(
            f"  Some StimIDs appear more than twice in encoding: "
            f"{over.to_dict()} — ViewingNumber will exceed 2."
        )

    trial_table.loc[enc_mask, "ViewingNumber"] = viewing_nums.values
    return trial_table


# ---------------------------------------------------------------------------
# Step 4: Segment fixations and saccades
# ---------------------------------------------------------------------------


def segment_events(
    events: pd.DataFrame,
    trial_table: pd.DataFrame,
    subject_id: str,
    event_type: str,  # "fixation" or "saccade"
) -> pd.DataFrame:
    """
    Vectorised segmentation: assign each event to a trial based on whether
    its stime falls within [trial_start_ms, trial_end_ms].

    Uses pd.IntervalIndex for O(n log n) lookup rather than a Python loop.
    """
    if len(events) == 0 or len(trial_table) == 0:
        return pd.DataFrame()

    # Build an IntervalIndex from the trial table
    intervals = pd.IntervalIndex.from_arrays(
        trial_table["trial_start_ms"],
        trial_table["trial_end_ms"],
        closed="left",
    )

    # For each event, find which interval its stime falls into
    event_stimes = events["stime"].values
    trial_positions = intervals.get_indexer(event_stimes)
    # -1 means the event fell outside all trial windows (between trials)

    in_trial_mask = trial_positions >= 0
    matched_events = events[in_trial_mask].copy()
    matched_positions = trial_positions[in_trial_mask]

    # Attach trial metadata
    meta_cols = ["TrialID", "TrialIndex", "Phase", "ViewingNumber", "StimID", "CueText"]
    for col in meta_cols:
        matched_events[col] = trial_table[col].iloc[matched_positions].values

    n_outside = (~in_trial_mask).sum()
    if n_outside > 0:
        logger.debug(
            f"  [{subject_id}] {n_outside} {event_type}(s) fell outside all trial windows "
            f"(between-trial gaze) — excluded."
        )

    logger.info(
        f"  [{subject_id}] Segmented {len(matched_events)}/{len(events)} "
        f"{event_type}s into trial windows."
    )

    return matched_events


# ---------------------------------------------------------------------------
# Step 5: Validate
# ---------------------------------------------------------------------------


def validate_outputs(
    fixations: pd.DataFrame,
    saccades: pd.DataFrame,
    trial_table: pd.DataFrame,
    subject_id: str,
) -> bool:
    passed = True

    # Report trial counts
    n_enc = (trial_table["Phase"] == "encoding").sum()
    n_dec = (trial_table["Phase"] == "decoding").sum()
    logger.info(f"  [{subject_id}] Encoding trials in EDF : {n_enc}")
    logger.info(f"  [{subject_id}] Decoding trials in EDF : {n_dec}")

    # Cross-check: every trial in trial_table has fixations
    if len(fixations) > 0:
        trials_with_fix = set(zip(fixations["Phase"], fixations["TrialIndex"]))
        for _, row in trial_table.iterrows():
            key = (row["Phase"], row["TrialIndex"])
            if key not in trials_with_fix:
                logger.warning(
                    f"  [{subject_id}] Trial {row['TrialID']} "
                    f"({row['Phase']} TrialIndex={row['TrialIndex']}) "
                    f"has 0 fixations — possible data quality issue."
                )
                passed = False

    # Check for orphaned TrialIndex values (in EDF but not in behavioral)
    orphaned = trial_table[trial_table["StimID"].isna()]
    if len(orphaned) > 0:
        logger.warning(
            f"  [{subject_id}] {len(orphaned)} trial(s) have no matching StimID: "
            f"{orphaned[['TrialID','Phase','TrialIndex']].to_dict('records')}"
        )
        passed = False

    # Confirm StimIDs in fixations match trial table
    if len(fixations) > 0:
        fix_stim_ids = set(fixations["StimID"].dropna().unique())
        table_stim_ids = set(trial_table["StimID"].dropna().unique())
        mismatch = fix_stim_ids - table_stim_ids
        if mismatch:
            logger.warning(
                f"  [{subject_id}] StimIDs in fixations not found in trial table: {mismatch}"
            )
            passed = False

    if passed:
        logger.info(f"  [{subject_id}] All Module 2 validation checks passed.")

    return passed


# ---------------------------------------------------------------------------
# Step 6: Write outputs
# ---------------------------------------------------------------------------


def write_outputs(
    fixations: pd.DataFrame,
    saccades: pd.DataFrame,
    subject_id: str,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Fixations ---
    fix_out = pd.DataFrame()
    if len(fixations) > 0:
        fix_out = pd.DataFrame(
            {
                "SubjectID": subject_id,
                "StimID": fixations["StimID"],
                "CueText": fixations["CueText"],
                "Phase": fixations["Phase"],
                "ViewingNumber": fixations["ViewingNumber"],
                "TrialIndex": fixations["TrialIndex"],
                "TrialID": fixations["TrialID"],
                "FixStart_ms": fixations["stime"],
                "FixEnd_ms": fixations["etime"],
                "Duration_ms": fixations["etime"] - fixations["stime"],
                "GazeX": fixations["axp"].round(2),
                "GazeY": fixations["ayp"].round(2),
            }
        )

    fix_path = output_dir / f"{subject_id}_fixations.csv"
    fix_out.to_csv(fix_path, index=False)
    logger.info(f"  [{subject_id}] Written: {fix_path.name} ({len(fix_out)} rows)")

    # --- Saccades ---
    sacc_out = pd.DataFrame()
    if len(saccades) > 0:
        sacc_out = pd.DataFrame(
            {
                "SubjectID": subject_id,
                "StimID": saccades["StimID"],
                "CueText": saccades["CueText"],
                "Phase": saccades["Phase"],
                "ViewingNumber": saccades["ViewingNumber"],
                "TrialIndex": saccades["TrialIndex"],
                "TrialID": saccades["TrialID"],
                "SaccStart_ms": saccades["stime"],
                "SaccEnd_ms": saccades["etime"],
                "Duration_ms": saccades["etime"] - saccades["stime"],
                "StartX": saccades["sxp"].round(2),
                "StartY": saccades["syp"].round(2),
                "EndX": saccades["exp"].round(2),
                "EndY": saccades["eyp"].round(2),
                "PeakVelocity": saccades["pv"].round(2),
            }
        )

    sacc_path = output_dir / f"{subject_id}_saccades.csv"
    sacc_out.to_csv(sacc_path, index=False)
    logger.info(f"  [{subject_id}] Written: {sacc_path.name} ({len(sacc_out)} rows)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def process_subject(
    subject_id: str,
    edf_dir: str | Path,
    beh_dir: str | Path,
    output_dir: str | Path,
) -> bool:
    """
    Full Module 2 pipeline for a single participant.

    Parameters
    ----------
    subject_id : str
        e.g. 'sub01'
    edf_dir : path-like
        Directory containing {subject_id}.edf
    beh_dir : path-like
        Directory containing Module 1 behavioral CSVs
    output_dir : path-like
        Directory where fixation/saccade CSVs will be written

    Returns
    -------
    bool
        True if completed successfully.
    """
    edf_dir = Path(edf_dir)
    beh_dir = Path(beh_dir)
    output_dir = Path(output_dir)

    edf_path = edf_dir / f"{subject_id}.edf"
    if not edf_path.exists():
        logger.error(f"[{subject_id}] EDF file not found: {edf_path}")
        return False

    logger.info(f"[{subject_id}] Module 2 starting ...")

    # Step 1: Load EDF
    edf_data = load_edf(edf_path)

    # Step 1a: Filter invalid / out-of-bounds / blink-artifact events
    edf_data["fixations"], edf_data["saccades"] = filter_invalid_events(
        edf_data["fixations"], edf_data["saccades"], subject_id
    )

    # Step 2: Build trial table
    trial_table = build_trial_table(edf_data["messages"], subject_id)
    if len(trial_table) == 0:
        logger.error(f"[{subject_id}] Empty trial table — cannot continue.")
        return False

    # Step 3: Join StimIDs from behavioral CSVs
    trial_table = join_stim_ids(trial_table, beh_dir, subject_id)

    # Step 3a: Add ViewingNumber to encoding trials
    trial_table = add_viewing_number(trial_table)

    # Step 4: Segment fixations and saccades
    fixations = segment_events(
        edf_data["fixations"], trial_table, subject_id, "fixation"
    )
    saccades = segment_events(edf_data["saccades"], trial_table, subject_id, "saccade")

    # Step 5: Validate
    validate_outputs(fixations, saccades, trial_table, subject_id)

    # Step 6: Write
    write_outputs(fixations, saccades, subject_id, output_dir)

    logger.info(f"[{subject_id}] Module 2 complete.")
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Module 2: Eye-Tracking Extractor")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g. sub01)")
    parser.add_argument(
        "--edf-dir", default="data_eyetracking/", help="Directory of .edf files"
    )
    parser.add_argument(
        "--beh-dir", default="output/behavioral/", help="Module 1 output directory"
    )
    parser.add_argument(
        "--output-dir", default="output/eyetracking/", help="Output directory"
    )
    args = parser.parse_args()

    process_subject(
        subject_id=args.subject,
        edf_dir=args.edf_dir,
        beh_dir=args.beh_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
