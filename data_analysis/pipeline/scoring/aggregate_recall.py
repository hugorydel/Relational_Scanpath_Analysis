"""
pipeline/scoring/aggregate_recall.py
======================================
Post-processing step: aggregate flat per-node recall scores into
per-(SubjectID, StimID) summary counts, broken down by content_type,
evidence_type, and their cross-product.

Reads:  output/scoring/recall_scores.csv   (written by score_recall.py)
Writes: output/scoring/recall_by_category.csv

No API calls. Safe to re-run at any time without touching the scored data.
Re-running after adding new participants or fixing codebook metadata will
produce an updated summary without re-scoring.

Breakdown structure of recall_by_category.csv
-----------------------------------------------
  1. By content_type           (5 types  × 2 cols = 10)
  2. By evidence_type          (3 types  × 2 cols =  6)
  3. By content_type × evidence_type  (15 combos × 2 cols = 30)
  4. Totals                    (4 cols)
  Total: 52 columns (+ SubjectID, StimID = 54)

Only correct-status nodes enter any denominator. incorrect nodes
(participant hallucinations) are never counted as recalled.

Usage
-----
    python pipeline/scoring/aggregate_recall.py
    python pipeline/scoring/aggregate_recall.py --scores path/to/recall_scores.csv
    python pipeline/scoring/aggregate_recall.py --output path/to/output_dir
"""

import argparse
import logging
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

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
CONTENT_TYPES = [
    "object_identity",
    "object_attribute",
    "action_relation",
    "spatial_relation",
    "scene_gist",
]
EVIDENCE_TYPES = ["literal", "latent", "speculative"]

# ---------------------------------------------------------------------------
# Output column order
# ---------------------------------------------------------------------------
CATEGORY_FIELDNAMES = ["SubjectID", "StimID"]

# 1. By content_type
for ct in CONTENT_TYPES:
    CATEGORY_FIELDNAMES += [f"n_{ct}_recalled", f"n_{ct}_total"]

# 2. By evidence_type
for et in EVIDENCE_TYPES:
    CATEGORY_FIELDNAMES += [f"n_{et}_recalled", f"n_{et}_total"]

# 3. By content_type × evidence_type
for ct in CONTENT_TYPES:
    for et in EVIDENCE_TYPES:
        CATEGORY_FIELDNAMES += [f"n_{ct}_{et}_recalled", f"n_{ct}_{et}_total"]

# 4. Totals
CATEGORY_FIELDNAMES += [
    "n_correct_nodes_recalled",
    "n_correct_nodes_total",
    "n_total_recalled",
    "n_total_nodes",
]

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(scores_path: Path, output_path: Path) -> None:
    if not scores_path.exists():
        logger.error(f"recall_scores.csv not found: {scores_path}")
        sys.exit(1)

    df = pd.read_csv(scores_path, dtype={"StimID": str, "SubjectID": str})

    required = {
        "SubjectID",
        "StimID",
        "node_id",
        "content_type",
        "evidence_type",
        "status",
        "recalled",
    }
    missing = required - set(df.columns)
    if missing:
        logger.error(f"recall_scores.csv is missing columns: {missing}")
        sys.exit(1)

    records = []
    for (subj, stim), grp in df.groupby(["SubjectID", "StimID"]):
        rec = {"SubjectID": subj, "StimID": stim}
        correct_grp = grp[grp["status"] == "correct"]

        # 1. By content_type
        for ct in CONTENT_TYPES:
            sub = correct_grp[correct_grp["content_type"] == ct]
            rec[f"n_{ct}_recalled"] = int(sub["recalled"].sum())
            rec[f"n_{ct}_total"] = len(sub)

        # 2. By evidence_type
        for et in EVIDENCE_TYPES:
            sub = correct_grp[correct_grp["evidence_type"] == et]
            rec[f"n_{et}_recalled"] = int(sub["recalled"].sum())
            rec[f"n_{et}_total"] = len(sub)

        # 3. By content_type × evidence_type
        for ct in CONTENT_TYPES:
            for et in EVIDENCE_TYPES:
                sub = correct_grp[
                    (correct_grp["content_type"] == ct)
                    & (correct_grp["evidence_type"] == et)
                ]
                rec[f"n_{ct}_{et}_recalled"] = int(sub["recalled"].sum())
                rec[f"n_{ct}_{et}_total"] = len(sub)

        # 4. Totals
        rec["n_correct_nodes_recalled"] = int(correct_grp["recalled"].sum())
        rec["n_correct_nodes_total"] = len(correct_grp)
        rec["n_total_recalled"] = int(grp["recalled"].sum())
        rec["n_total_nodes"] = len(grp)

        records.append(rec)

    cat_df = pd.DataFrame(records, columns=CATEGORY_FIELDNAMES)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cat_df.to_csv(output_path, index=False)

    logger.info(
        f"recall_by_category.csv written -> {output_path}\n"
        f"  {len(cat_df)} rows  ×  {len(CATEGORY_FIELDNAMES)} columns\n"
        f"  {cat_df['SubjectID'].nunique()} participants  ×  "
        f"{cat_df['StimID'].nunique()} stimuli"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-node recall scores into category summary CSV."
    )
    parser.add_argument(
        "--scores",
        default=str(config.OUTPUT_DIR / "scoring" / "recall_scores.csv"),
        help="Path to recall_scores.csv (default: output/scoring/recall_scores.csv).",
    )
    parser.add_argument(
        "--output",
        default=str(config.OUTPUT_DIR / "scoring" / "recall_by_category.csv"),
        help="Output path for recall_by_category.csv.",
    )
    args = parser.parse_args()

    logger.info("Aggregating recall scores by category ...")
    logger.info(f"  Input  : {args.scores}")
    logger.info(f"  Output : {args.output}")

    aggregate(Path(args.scores), Path(args.output))


if __name__ == "__main__":
    main()
