"""
pipeline/scoring/rebuild_scores.py
=====================================
Syncs recall_scores.csv metadata from the edited codebooks.

Use this after manually editing any codebook JSON (flipping status,
correcting content_type/evidence_type, renaming concepts, splitting nodes)
without re-running the expensive LLM scoring step.

What it does
------------
1. Reads every edited codebook in output/codebooks/edited/
2. Reads the existing recall_scores.csv (which holds the LLM recall decisions)
3. Joins them on node_id, replacing stale metadata columns with the
   corrected values from the codebooks
4. Writes a fresh recall_scores.csv (backup of the old one kept as
   recall_scores.bak.csv)

The recall decisions (recalled, matched_phrase) are never touched.

After running this, re-run aggregate_recall.py to get updated category counts.

Handles
-------
- Nodes that were split in the codebook (e.g. 2387122_002 → 002a, 002b):
  The original scored node is matched to the base ID. Split children that
  have no scored counterpart are added with recalled=0, matched_phrase="".
  A warning is printed so you can review.
- Nodes present in recall_scores.csv but no longer in the codebook:
  Dropped with a warning (they were deleted during manual review).
- Nodes present in the codebook but never scored (new additions):
  Added with recalled=0, matched_phrase="".

Usage
-----
    python pipeline/scoring/rebuild_scores.py
    python pipeline/scoring/rebuild_scores.py --dry-run   # show changes, don't write
    python pipeline/scoring/rebuild_scores.py --no-backup # skip backup file
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_DA_DIR = _HERE.parent.parent
sys.path.insert(0, str(_DA_DIR))

import config

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
)
logger = logging.getLogger(__name__)

EDITED_DIR = config.OUTPUT_DIR / "codebooks" / "edited"
SCORING_DIR = config.OUTPUT_DIR / "scoring"
SCORES_CSV = SCORING_DIR / "recall_scores.csv"
BACKUP_CSV = SCORING_DIR / "recall_scores.bak.csv"

METADATA_COLS = ["concept", "content_type", "evidence_type", "status"]
RECALL_COLS = ["recalled", "matched_phrase"]
ALL_COLS = ["SubjectID", "StimID", "node_id"] + METADATA_COLS + RECALL_COLS


# ---------------------------------------------------------------------------
# Load codebooks
# ---------------------------------------------------------------------------


def load_all_codebooks() -> pd.DataFrame:
    """
    Load every edited codebook JSON into a single DataFrame.
    Returns columns: node_id, StimID, concept, content_type, evidence_type, status.
    """
    import json

    frames = []
    for path in sorted(EDITED_DIR.glob("*_codebook.json")):
        stim_id = path.stem.replace("_codebook", "")
        with open(path, encoding="utf-8") as f:
            nodes = json.load(f)
        df = pd.DataFrame(nodes)
        df["StimID"] = stim_id
        frames.append(df)

    if not frames:
        logger.error(f"No edited codebooks found in {EDITED_DIR}")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Loaded {len(combined)} nodes across "
        f"{combined['StimID'].nunique()} codebooks."
    )
    return combined[["node_id", "StimID"] + METADATA_COLS]


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------


def rebuild(dry_run: bool, no_backup: bool) -> None:
    if not SCORES_CSV.exists():
        logger.error(f"recall_scores.csv not found: {SCORES_CSV}")
        sys.exit(1)

    scores = pd.read_csv(SCORES_CSV, dtype={"StimID": str, "SubjectID": str})
    codebook = load_all_codebooks()
    codebook["StimID"] = codebook["StimID"].astype(str)

    # ------------------------------------------------------------------
    # Detect split nodes: codebook IDs that look like originals with a
    # letter suffix (e.g. 2387122_002a) whose base (2387122_002) was scored.
    # ------------------------------------------------------------------
    scored_ids = set(scores["node_id"].unique())
    codebook_ids = set(codebook["node_id"].unique())

    dropped = scored_ids - codebook_ids
    added = codebook_ids - scored_ids

    # Identify split children: added IDs whose base (strip trailing letter) was scored
    splits = {}
    for nid in added:
        base = nid.rstrip("abcdefghijklmnopqrstuvwxyz")
        if base != nid and base in scored_ids:
            splits.setdefault(base, []).append(nid)

    truly_new = added - {nid for children in splits.values() for nid in children}
    truly_dropped = dropped - set(splits.keys())

    if truly_dropped:
        logger.warning(
            f"{len(truly_dropped)} node(s) in recall_scores.csv no longer exist "
            f"in the codebooks — they will be dropped:\n  "
            + "\n  ".join(sorted(truly_dropped))
        )
    if truly_new:
        logger.warning(
            f"{len(truly_new)} new node(s) in codebooks have no recall decision "
            f"— they will be added with recalled=0:\n  "
            + "\n  ".join(sorted(truly_new))
        )
    if splits:
        logger.warning(
            f"{len(splits)} node(s) appear to have been split. "
            f"Original recall decision copied to all children:\n  "
            + "\n  ".join(
                f"{base} -> {children}" for base, children in sorted(splits.items())
            )
        )

    # ------------------------------------------------------------------
    # Build the new scores table
    # ------------------------------------------------------------------

    # Keep only scored nodes that still exist (drop truly_dropped)
    scores_clean = scores[
        scores["node_id"].isin(codebook_ids | set(splits.keys()))
    ].copy()

    # For split nodes: expand one original row into N child rows
    extra_rows = []
    for base, children in splits.items():
        original = scores_clean[scores_clean["node_id"] == base]
        if original.empty:
            continue
        orig_row = original.iloc[0]
        for child_id in children:
            row = orig_row.copy()
            row["node_id"] = child_id
            extra_rows.append(row)
        # Remove the original base node (it's been replaced by children)
        scores_clean = scores_clean[scores_clean["node_id"] != base]

    if extra_rows:
        scores_clean = pd.concat(
            [scores_clean, pd.DataFrame(extra_rows)], ignore_index=True
        )

    # For truly new nodes: add one row per (SubjectID, StimID) combination
    # with recalled=0. We need the full subject × stim pairs already in scores.
    if truly_new:
        subject_stim_pairs = scores[["SubjectID", "StimID"]].drop_duplicates()
        new_meta = codebook[codebook["node_id"].isin(truly_new)]
        new_rows = subject_stim_pairs.merge(new_meta, on="StimID", how="inner")
        new_rows["recalled"] = 0
        new_rows["matched_phrase"] = ""
        scores_clean = pd.concat([scores_clean, new_rows], ignore_index=True)

    # ------------------------------------------------------------------
    # Replace metadata columns from codebook (the whole point of this script)
    # ------------------------------------------------------------------
    codebook_meta = codebook.set_index("node_id")[METADATA_COLS]

    for col in METADATA_COLS:
        scores_clean[col] = scores_clean["node_id"].map(codebook_meta[col])

    # Check for any unmapped metadata (shouldn't happen after above logic)
    unmapped = scores_clean[scores_clean["concept"].isna()]
    if not unmapped.empty:
        logger.warning(
            f"{len(unmapped)} row(s) could not be mapped to codebook metadata "
            f"— they will be dropped. node_ids: "
            f"{unmapped['node_id'].unique().tolist()}"
        )
        scores_clean = scores_clean.dropna(subset=["concept"])

    # Reorder columns and sort
    scores_clean = (
        scores_clean[ALL_COLS]
        .sort_values(["SubjectID", "StimID", "node_id"])
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    old_rows = len(scores)
    new_rows_count = len(scores_clean)
    logger.info(
        f"\nSummary of changes:"
        f"\n  Rows before : {old_rows}"
        f"\n  Rows after  : {new_rows_count}  (delta {new_rows_count - old_rows:+d})"
        f"\n  Nodes dropped  : {len(truly_dropped)}"
        f"\n  Nodes added    : {len(truly_new)}"
        f"\n  Nodes split    : {len(splits)}"
        f"\n  Metadata cols refreshed: {METADATA_COLS}"
    )

    if dry_run:
        logger.info("DRY RUN — no files written.")
        return

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    if not no_backup and SCORES_CSV.exists():
        shutil.copy(SCORES_CSV, BACKUP_CSV)
        logger.info(f"Backup written -> {BACKUP_CSV}")

    scores_clean.to_csv(SCORES_CSV, index=False)
    logger.info(f"recall_scores.csv updated -> {SCORES_CSV}")
    logger.info("Now run: python pipeline/scoring/aggregate_recall.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Sync recall_scores.csv metadata from edited codebooks."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing anything.",
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip writing recall_scores.bak.csv."
    )
    args = parser.parse_args()

    logger.info("Rebuilding recall_scores.csv from edited codebooks ...")
    logger.info(f"  Codebooks : {EDITED_DIR}")
    logger.info(f"  Scores    : {SCORES_CSV}")

    rebuild(dry_run=args.dry_run, no_backup=args.no_backup)


if __name__ == "__main__":
    main()
