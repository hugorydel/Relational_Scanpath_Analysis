"""
Module 1: Behavioral Processor
===============================
Parses E-Prime .txt log files into four clean, typed CSVs per participant:
    - {SubjectID}_encoding.csv
    - {SubjectID}_distractor.csv
    - {SubjectID}_decoding.csv
    - {SubjectID}_exploratory.csv

Usage (standalone):
    python module1_behavioral.py --input data_behavioral/ --output output/behavioral/

Usage (from orchestrator):
    from pipeline.module1_behavioral import process_subject
    process_subject("sub01", input_dir="data_behavioral/", output_dir="output/behavioral/")
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column selection and renaming maps per block type
# ---------------------------------------------------------------------------

ENCODING_COLS = {
    "StimID": "StimID",
    "CueText": "CueText",
    "Practice4AFC.ACC": "Accuracy",
    "Practice4AFC.RT": "RT_ms",
    "Practice4AFC.RESP": "Response",
    "Practice4AFC.CRESP": "CorrectKey",
    "PracticeList.Sample": "TrialIndex",
}

DISTRACTOR_COLS = {
    "StimID": "StimID",
    "Question": "Question",
    "Distractor4AFC.ACC": "Accuracy",
    "Distractor4AFC.RT": "RT_ms",
    "Distractor4AFC.RESP": "Response",
    "Distractor4AFC.CRESP": "CorrectKey",
    "DistractorList.Sample": "TrialIndex",
}

DECODING_COLS = {
    "StimID": "StimID",
    "CueText": "CueText",
    "Test4AFC.ACC": "Accuracy",
    "Test4AFC.RT": "RT_ms",
    "Test4AFC.RESP": "Response",
    "Test4AFC.CRESP": "CorrectKey",
    "RateConf.RESP": "Confidence",
    "TestList.Sample": "TrialIndex",
}

EXPLORATORY_COLS = {
    "StimID": "StimID",
    "CueText": "CueText",
    "FreeResponseSlide.RESP": "FreeResponse",
    "FreeResponseSlide.RT": "RT_ms",
    "ExploratoryList.Sample": "TrialIndex",
}

# Expected trial counts for validation
EXPECTED_COUNTS = {
    "encoding": 60,
    "decoding": 30,
    "exploratory": 10,
}

DISTRACTOR_COUNT_RANGE = (40, 60)  # time-terminated loop


# ---------------------------------------------------------------------------
# Step 1 & 2: File reading and block parsing
# ---------------------------------------------------------------------------


def _read_file(filepath: Path) -> str:
    """
    Read an E-Prime .txt file, trying encodings in order of likelihood.
    E-Prime exports as UTF-16 LE with BOM by default. If the file was
    manually re-saved it may be UTF-8. We try UTF-16 first, then UTF-8,
    then fall back to latin-1 (which never fails but may mis-read characters).
    """
    for encoding in ("utf-16", "utf-8", "latin-1"):
        try:
            text = filepath.read_text(encoding=encoding)
            if "LogFrame" in text:
                logger.debug(f"  Read {filepath.name} as {encoding}")
                return text
        except (UnicodeDecodeError, UnicodeError):
            continue
    logger.warning(
        f"  Could not detect encoding for {filepath.name}, using utf-8 with replacement."
    )
    return filepath.read_text(encoding="utf-8", errors="replace")


def read_logframes(filepath: Path) -> list[dict]:
    """
    Read the full E-Prime .txt file and parse every LogFrame block into a
    list of dicts. Each dict maps raw key strings to raw value strings.

    Strategy:
    - Split on '*** LogFrame Start ***' to isolate blocks
    - Within each block, split on first ':' only (handles colons in values)
    - Skip blank lines and the LogFrame End marker
    """
    text = _read_file(filepath)

    # Split on LogFrame Start; first chunk is file header — discard it
    raw_blocks = text.split("*** LogFrame Start ***")[1:]

    blocks = []
    for raw in raw_blocks:
        block = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if "*** LogFrame End ***" in line:
                break
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            block[key.strip()] = value.strip()

        if block:
            blocks.append(block)

    logger.debug(f"  Parsed {len(blocks)} LogFrame blocks from {filepath.name}")
    return blocks


# ---------------------------------------------------------------------------
# Step 3: Route blocks by Procedure
# ---------------------------------------------------------------------------

PROCEDURE_MAP = {
    "PracticeProc": "encoding",
    "DistractorProc": "distractor",
    "TestProc": "decoding",
    "ExploratoryProc": "exploratory",
}


def route_blocks(blocks: list[dict], subject_id: str) -> dict[str, list[dict]]:
    """
    Sort parsed blocks into four lists keyed by block type.
    Blocks with a missing or unrecognised Procedure are logged and skipped.
    """
    routed = {"encoding": [], "distractor": [], "decoding": [], "exploratory": []}
    skipped = 0

    for block in blocks:
        procedure = block.get("Procedure", "").strip()
        block_type = PROCEDURE_MAP.get(procedure)
        if block_type is None:
            logger.warning(
                f"  [{subject_id}] Unrecognised Procedure '{procedure}' — block skipped. "
                f"Keys present: {list(block.keys())[:5]}"
            )
            skipped += 1
            continue
        routed[block_type].append(block)

    if skipped:
        logger.warning(
            f"  [{subject_id}] {skipped} block(s) skipped due to unknown Procedure."
        )

    return routed


# ---------------------------------------------------------------------------
# Step 4 & 5: Column extraction, renaming, and type casting
# ---------------------------------------------------------------------------


def extract_columns(blocks: list[dict], col_map: dict, subject_id: str) -> pd.DataFrame:
    """
    From a list of raw block dicts, extract only the keys in col_map,
    rename them, and return a DataFrame.

    Missing keys in a block produce NaN for that cell (not a crash).
    """
    rows = []
    for block in blocks:
        row = {}
        for raw_key, new_key in col_map.items():
            value = block.get(raw_key, pd.NA)
            # Treat empty string as NA
            if value == "":
                value = pd.NA
            row[new_key] = value
        rows.append(row)

    return pd.DataFrame(rows)


def cast_types(df: pd.DataFrame, block_type: str) -> pd.DataFrame:
    """
    Apply appropriate dtypes per block type.
    Errors in casting are coerced to NaN and logged rather than raised.
    """
    df = df.copy()

    # StimID: always string (image ID, not arithmetic)
    if "StimID" in df.columns:
        df["StimID"] = df["StimID"].astype(str)

    # Integer columns — coerce so bad values become NaN not crashes
    int_cols = ["Accuracy", "RT_ms", "TrialIndex", "Response", "CorrectKey"]
    if block_type == "decoding":
        int_cols.append("Confidence")

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # FreeResponse: string, strip whitespace
    if "FreeResponse" in df.columns:
        df["FreeResponse"] = df["FreeResponse"].apply(
            lambda x: x.strip() if isinstance(x, str) else pd.NA
        )

    return df


# ---------------------------------------------------------------------------
# Step 6: Add SubjectID
# ---------------------------------------------------------------------------


def add_subject_id(df: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    df.insert(0, "SubjectID", subject_id)
    return df


# ---------------------------------------------------------------------------
# Step 7: Validation
# ---------------------------------------------------------------------------


def validate_tables(tables: dict[str, pd.DataFrame], subject_id: str) -> bool:
    """
    Run sanity checks on all four tables.
    Returns True if all pass, False if any fail.
    Logs warnings rather than raising exceptions so pipeline can continue.
    """
    passed = True

    # --- Row counts ---
    for block_type, expected in EXPECTED_COUNTS.items():
        actual = len(tables[block_type])
        if actual != expected:
            logger.warning(
                f"  [{subject_id}] {block_type}: expected {expected} rows, got {actual}."
            )
            passed = False

    dist_count = len(tables["distractor"])
    lo, hi = DISTRACTOR_COUNT_RANGE
    if not (lo <= dist_count <= hi):
        logger.warning(
            f"  [{subject_id}] distractor: row count {dist_count} outside expected range {lo}–{hi}."
        )
        passed = False

    # --- No duplicate StimIDs within encoding or decoding ---
    for block_type in ("encoding", "decoding"):
        df = tables[block_type]
        if "StimID" in df.columns:
            dupes = df["StimID"].duplicated().sum()
            if dupes > 0:
                logger.warning(
                    f"  [{subject_id}] {block_type}: {dupes} duplicate StimID(s) found."
                )
                passed = False

    # --- Cross-table StimID checks (only if tables are non-empty) ---
    def _stim_ids(block_type: str) -> set:
        df = tables[block_type]
        if df.empty or "StimID" not in df.columns:
            return set()
        return set(df["StimID"].dropna())

    enc_ids = _stim_ids("encoding")
    dec_ids = _stim_ids("decoding")
    exp_ids = _stim_ids("exploratory")

    if enc_ids and dec_ids:
        missing_in_enc = dec_ids - enc_ids
        if missing_in_enc:
            logger.warning(
                f"  [{subject_id}] Decoding StimIDs not found in encoding: {missing_in_enc}"
            )
            passed = False

    if dec_ids and exp_ids:
        missing_in_dec = exp_ids - dec_ids
        if missing_in_dec:
            logger.warning(
                f"  [{subject_id}] Exploratory StimIDs not found in decoding: {missing_in_dec}"
            )
            passed = False

    if passed:
        logger.info(f"  [{subject_id}] All validation checks passed.")

    return passed


# ---------------------------------------------------------------------------
# Step 8: Write CSVs
# ---------------------------------------------------------------------------


def write_tables(tables: dict[str, pd.DataFrame], subject_id: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for block_type, df in tables.items():
        out_path = output_dir / f"{subject_id}_{block_type}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"  [{subject_id}] Written: {out_path.name} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

COL_MAPS = {
    "encoding": ENCODING_COLS,
    "distractor": DISTRACTOR_COLS,
    "decoding": DECODING_COLS,
    "exploratory": EXPLORATORY_COLS,
}


def process_subject(
    subject_id: str, input_dir: str | Path, output_dir: str | Path
) -> bool:
    """
    Full Module 1 pipeline for a single participant.

    Parameters
    ----------
    subject_id : str
        e.g. 'sub01'
    input_dir : path-like
        Directory containing {subject_id}.txt
    output_dir : path-like
        Directory where CSVs will be written

    Returns
    -------
    bool
        True if completed without errors (validation warnings don't block completion).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    filepath = input_dir / f"{subject_id}.txt"
    if not filepath.exists():
        logger.error(f"[{subject_id}] Input file not found: {filepath}")
        return False

    logger.info(f"[{subject_id}] Processing {filepath.name} ...")

    # Steps 1–2: Parse all blocks from file
    blocks = read_logframes(filepath)

    # Step 3: Route by Procedure
    routed = route_blocks(blocks, subject_id)

    # Steps 4–5: Extract columns and cast types
    tables = {}
    for block_type, block_list in routed.items():
        df = extract_columns(block_list, COL_MAPS[block_type], subject_id)
        df = cast_types(df, block_type)
        # Step 6: Add SubjectID
        df = add_subject_id(df, subject_id)
        tables[block_type] = df

    # Step 7: Validate
    validate_tables(tables, subject_id)

    # Step 8: Write
    write_tables(tables, subject_id, output_dir)

    logger.info(f"[{subject_id}] Module 1 complete.")
    return True


# ---------------------------------------------------------------------------
# CLI entry point (run a batch directly)
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Module 1: Behavioral Processor")
    parser.add_argument(
        "--input", default="data_behavioral/", help="Directory of .txt files"
    )
    parser.add_argument(
        "--output", default="output/behavioral/", help="Output directory"
    )
    parser.add_argument(
        "--subjects", nargs="*", help="Specific subject IDs to process (default: all)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if args.subjects:
        subject_ids = args.subjects
    else:
        subject_ids = sorted(p.stem for p in input_dir.glob("*.txt"))

    if not subject_ids:
        logger.error(f"No .txt files found in {input_dir}")
        return

    logger.info(f"Found {len(subject_ids)} participant(s): {subject_ids}")

    results = {}
    for sid in subject_ids:
        results[sid] = process_subject(sid, input_dir, output_dir)

    # Summary
    passed = [s for s, ok in results.items() if ok]
    failed = [s for s, ok in results.items() if not ok]
    logger.info(f"\n{'='*50}")
    logger.info(f"Completed: {len(passed)}/{len(subject_ids)}")
    if failed:
        logger.warning(f"Failed:    {failed}")


if __name__ == "__main__":
    main()
