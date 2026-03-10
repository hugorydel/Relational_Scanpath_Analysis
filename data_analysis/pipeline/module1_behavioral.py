"""
Module 1: Behavioral Processor
================================
Parses E-Prime .txt log files into four clean, typed CSVs per participant:
    - {SubjectID}_encoding.csv       (60 rows — two questions per image, one row each)
    - {SubjectID}_distractor.csv
    - {SubjectID}_decoding.csv       (30 rows — one free-recall response per image)
    - {SubjectID}_exploratory.csv    (variable — subset of images, free recall)

Encoding format:
    One row per encoding trial (60 total: 30 images × 2 questions each).
    Columns: SubjectID, StimID, CueText, Question, Accuracy, RT_ms, Response,
             CorrectKey, TrialIndex

Decoding format:
    One row per image (30 total). Participant typed a free-recall response
    after seeing the MCQ cue (no forced-choice response was logged).
    FreeResponse is decoded from E-Prime keystroke syntax ({SPACE}, {.}, etc.).
    Columns: SubjectID, StimID, CueText, CueQuestion, FreeResponse, RT_ms, TrialIndex

Usage (standalone):
    python module1_behavioral.py --input data_behavioral/ --output output/behavioral/

Usage (from orchestrator):
    from pipeline.module1_behavioral import process_subject
    process_subject("sub01", input_dir="data_behavioral/", output_dir="output/behavioral/")
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column maps
# ---------------------------------------------------------------------------

ENCODING_COLS = {
    "StimID": "StimID",
    "CueText": "CueText",
    "Question": "Question",
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

EXPLORATORY_COLS = {
    "StimID": "StimID",
    "CueText": "CueText",
    "FreeResponseSlide.RESP": "FreeResponse",
    "FreeResponseSlide.RT": "RT_ms",
    "ExploratoryList.Sample": "TrialIndex",
}

DECODING_COLS = {
    "StimID": "StimID",
    "CueText": "CueText",
    "Question": "CueQuestion",
    "FreeResponseSlide.RESP": "FreeResponse",
    "FreeResponseSlide.RT": "RT_ms",
    "TestList.Sample": "TrialIndex",
}

import config

EXPECTED_COUNTS = {
    "encoding": config.N_ENCODING_TRIALS * config.N_ENCODING_QUESTIONS,  # 60 rows (2 per image)
    "decoding": config.N_DECODING_TRIALS,   # 30 rows (1 free-recall per image)
}
lo, hi = config.N_DISTRACTOR_RANGE
exp_lo, exp_hi = config.N_EXPLORATORY_RANGE


# ---------------------------------------------------------------------------
# Step 1 & 2: File reading and block parsing
# ---------------------------------------------------------------------------


def _read_file(filepath: Path) -> str:
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
    text = _read_file(filepath)
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
# Step 4 & 5: Column extraction and type casting
# ---------------------------------------------------------------------------

# E-Prime keystroke tokens → plain text.
# Covers tokens observed in pilot data; unknown {TOKEN} forms are stripped.
_EPRIME_TOKENS = {
    "{SPACE}": " ",
    "{ENTER}": "",
    "{BACKSPACE}": "",
    "{DELETE}": "",
    "{TAB}": "	",
    "{LEFTARROW}": "",
    "{RIGHTARROW}": "",
    "{UPARROW}": "",
    "{DOWNARROW}": "",
    "{.}": ".",
    "{,}": ",",
    "{'}": "'",
    '{"}': '"',
    "{!}": "!",
    "{?}": "?",
    "{-}": "-",
    "{;}": ";",
    "{:}": ":",
    "{(}": "(",
    "{)}": ")",
    "{/}": "/",
}


def _decode_eprime(raw: str) -> str:
    """Convert E-Prime keystroke-encoded string to plain text."""
    if not isinstance(raw, str):
        return raw
    text = raw
    for token, replacement in _EPRIME_TOKENS.items():
        text = text.replace(token, replacement)
    # Strip any remaining {TOKEN} forms not in the lookup
    import re as _re
    text = _re.sub(r"\{[^}]+\}", "", text)
    return text.strip()




def extract_columns(blocks: list[dict], col_map: dict, subject_id: str) -> pd.DataFrame:
    """Generic extractor for encoding, distractor, and exploratory blocks."""
    rows = []
    for block in blocks:
        row = {}
        for raw_key, new_key in col_map.items():
            value = block.get(raw_key, pd.NA)
            if value == "":
                value = pd.NA
            row[new_key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def extract_decoding(blocks: list[dict], subject_id: str) -> pd.DataFrame:
    """
    Extract decoding (TestProc) blocks into flat format: 1 row per image.

    Decoding is now a free-recall task. The MCQ is shown as a retrieval cue
    but no forced-choice response is logged — only the typed free-recall
    response (FreeResponseSlide.RESP) and its RT.

    The E-Prime keystroke-encoded response (e.g. "two{SPACE}bears{ENTER}")
    is decoded to plain text via _decode_eprime().

    Output columns:
        StimID, CueText, CueQuestion, FreeResponse, RT_ms, TrialIndex
    """
    rows = []
    for block in blocks:
        raw_resp = block.get("FreeResponseSlide.RESP", pd.NA)
        rows.append({
            "StimID":       block.get("StimID",           pd.NA),
            "CueText":      block.get("CueText",          pd.NA),
            "CueQuestion":  block.get("Question",         pd.NA),
            "FreeResponse": _decode_eprime(raw_resp) if pd.notna(raw_resp) else pd.NA,
            "RT_ms":        block.get("FreeResponseSlide.RT", pd.NA),
            "TrialIndex":   block.get("TestList.Sample",  pd.NA),
        })
    return pd.DataFrame(rows)


def cast_types(df: pd.DataFrame, block_type: str) -> pd.DataFrame:
    df = df.copy()

    if "StimID" in df.columns:
        df["StimID"] = df["StimID"].astype(str)

    if block_type == "decoding":
        int_cols = ["TrialIndex", "RT_ms"]
    elif block_type == "exploratory":
        int_cols = ["TrialIndex", "RT_ms"]
    else:
        int_cols = ["Accuracy", "RT_ms", "TrialIndex", "Response", "CorrectKey"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

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
    passed = True

    # Row counts — exact for encoding and decoding
    for block_type, expected in EXPECTED_COUNTS.items():
        actual = len(tables[block_type])
        if actual != expected:
            logger.warning(
                f"  [{subject_id}] {block_type}: expected {expected} rows, got {actual}."
            )
            passed = False

    # Range checks — distractor and exploratory are variable
    dist_count = len(tables["distractor"])
    if not (lo <= dist_count <= hi):
        logger.warning(
            f"  [{subject_id}] distractor: row count {dist_count} outside expected range {lo}–{hi}."
        )
        passed = False

    exp_count = len(tables["exploratory"])
    if not (exp_lo <= exp_count <= exp_hi):
        logger.warning(
            f"  [{subject_id}] exploratory: row count {exp_count} outside expected range {exp_lo}–{exp_hi}."
        )
        passed = False

    # Encoding: each StimID should appear exactly N_ENCODING_QUESTIONS times (2)
    enc_df = tables["encoding"]
    if "StimID" in enc_df.columns:
        counts = enc_df["StimID"].value_counts()
        bad = counts[counts != config.N_ENCODING_QUESTIONS]
        if len(bad) > 0:
            logger.warning(
                f"  [{subject_id}] encoding: {len(bad)} StimID(s) without exactly "
                f"{config.N_ENCODING_QUESTIONS} question rows: {bad.index.tolist()}"
            )
            passed = False

    # Decoding: each StimID should appear exactly N_DECODING_QUESTIONS times
    dec_df = tables["decoding"]
    if "StimID" in dec_df.columns:
        counts = dec_df["StimID"].value_counts()
        bad = counts[counts != 1]
        if len(bad) > 0:
            logger.warning(
                f"  [{subject_id}] decoding: {len(bad)} StimID(s) with unexpected row count: "
                f"{bad.index.tolist()}"
            )
            passed = False
        # Each StimID should appear exactly once (one free-recall row per image)
        dupes = dec_df["StimID"].duplicated().sum()
        if dupes > 0:
            logger.warning(f"  [{subject_id}] decoding: {dupes} duplicate StimID(s).")
            passed = False

    # Cross-table StimID checks
    def _stim_ids(bt: str) -> set:
        d = tables[bt]
        if d.empty or "StimID" not in d.columns:
            return set()
        return set(d["StimID"].dropna().unique())

    enc_ids = _stim_ids("encoding")
    dec_ids = _stim_ids("decoding")
    exp_ids = _stim_ids("exploratory")

    if enc_ids and dec_ids:
        missing = dec_ids - enc_ids
        if missing:
            logger.warning(
                f"  [{subject_id}] Decoding StimIDs not in encoding: {missing}"
            )
            passed = False

    if dec_ids and exp_ids:
        missing = exp_ids - dec_ids
        if missing:
            logger.warning(
                f"  [{subject_id}] Exploratory StimIDs not in decoding: {missing}"
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


def process_subject(
    subject_id: str, input_dir: str | Path, output_dir: str | Path
) -> bool:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    filepath = input_dir / f"{subject_id}.txt"
    if not filepath.exists():
        logger.error(f"[{subject_id}] Input file not found: {filepath}")
        return False

    logger.info(f"[{subject_id}] Processing {filepath.name} ...")

    blocks = read_logframes(filepath)
    routed = route_blocks(blocks, subject_id)

    tables = {}
    col_maps = {
        "encoding": ENCODING_COLS,
        "distractor": DISTRACTOR_COLS,
        "exploratory": EXPLORATORY_COLS,
    }

    for block_type, col_map in col_maps.items():
        df = extract_columns(routed[block_type], col_map, subject_id)
        df = cast_types(df, block_type)
        df = add_subject_id(df, subject_id)
        tables[block_type] = df

    # Decoding uses dedicated extractor
    dec_df = extract_decoding(routed["decoding"], subject_id)
    dec_df = cast_types(dec_df, "decoding")
    dec_df = add_subject_id(dec_df, subject_id)
    tables["decoding"] = dec_df

    validate_tables(tables, subject_id)
    write_tables(tables, subject_id, output_dir)

    logger.info(f"[{subject_id}] Module 1 complete.")
    return True


# ---------------------------------------------------------------------------
# CLI
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
    parser.add_argument("--subjects", nargs="*", help="Subject IDs (default: all)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    subject_ids = args.subjects or sorted(p.stem for p in input_dir.glob("*.txt"))
    if not subject_ids:
        logger.error(f"No .txt files found in {input_dir}")
        return

    logger.info(f"Found {len(subject_ids)} participant(s): {subject_ids}")
    results = {sid: process_subject(sid, input_dir, output_dir) for sid in subject_ids}

    passed = [s for s, ok in results.items() if ok]
    failed = [s for s, ok in results.items() if not ok]
    logger.info(f"\n{'='*50}\nCompleted: {len(passed)}/{len(subject_ids)}")
    if failed:
        logger.warning(f"Failed: {failed}")


if __name__ == "__main__":
    main()