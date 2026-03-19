"""
pipeline/scoring/generate_second_rater.py
==========================================
Generates AI-2 recall scores for the subset of (SubjectID, StimID) pairs
that were also scored by the human rater.

Uses score_recall.py's scoring logic directly (imports and calls it),
redirecting output to:
    output/scoring/second_AI_rated_responses/recall_scores.csv

Then runs aggregate_recall.py's aggregation to produce:
    output/scoring/second_AI_rated_responses/recall_by_category.csv

Resume behaviour
----------------
Any (SubjectID, StimID) pair already present in
second_AI_rated_responses/recall_scores.csv is skipped.
Run with --force to reprocess everything.

Usage
-----
    python pipeline/scoring/generate_second_rater.py
    python pipeline/scoring/generate_second_rater.py --dry-run
    python pipeline/scoring/generate_second_rater.py --force
    python pipeline/scoring/generate_second_rater.py --max-concurrency 5
    python pipeline/scoring/generate_second_rater.py --model gpt-4.1
"""

import argparse
import asyncio
import csv
import getpass
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DA_DIR = _HERE.parent.parent
sys.path.insert(0, str(_DA_DIR))
sys.path.insert(0, str(_HERE))

import config

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HUMAN_SCORES_CSV = (
    config.OUTPUT_DIR / "scoring" / "human_rated_responses" / "memory_scores.csv"
)
SECOND_AI_DIR = config.OUTPUT_DIR / "scoring" / "second_AI_rated_responses"
SECOND_SCORES_CSV = SECOND_AI_DIR / "recall_scores.csv"
SECOND_BY_CATEGORY_CSV = SECOND_AI_DIR / "recall_by_category.csv"

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_CONCURRENCY = 10


# ---------------------------------------------------------------------------
# Scope helpers
# ---------------------------------------------------------------------------


def load_human_pairs() -> set:
    """
    Return the set of (SubjectID, StimID) pairs scored by the human rater,
    read from human_rated_responses/memory_scores.csv.
    """
    if not HUMAN_SCORES_CSV.exists():
        logger.error(f"Human scores not found: {HUMAN_SCORES_CSV}")
        sys.exit(1)

    pairs = set()
    with open(HUMAN_SCORES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subj = row.get("SubjectID", "").strip()
            stim = row.get("StimID", "").strip()
            if subj and stim:
                pairs.add((subj, stim))

    logger.info(f"Human rater scored {len(pairs)} (SubjectID, StimID) pairs.")
    return pairs


def load_already_scored() -> set:
    """
    Return set of (SubjectID, StimID) pairs already present in
    second_AI_rated_responses/recall_scores.csv.
    """
    already = set()
    if not SECOND_SCORES_CSV.exists():
        return already

    with open(SECOND_SCORES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subj = row.get("SubjectID", "").strip()
            stim = row.get("StimID", "").strip()
            if subj and stim:
                already.add((subj, stim))

    logger.info(
        f"Found {len(already)} already-scored pairs in "
        "second_AI_rated_responses/recall_scores.csv."
    )
    return already


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------


async def run(args) -> None:
    # Import score_recall and redirect its module-level output globals
    # before any scoring logic runs. _write_score_rows and fill_zeros both
    # reference SCORES_CSV and SCORING_DIR from the module namespace at
    # call time, so patching here is sufficient.
    import score_recall

    SECOND_AI_DIR.mkdir(parents=True, exist_ok=True)
    score_recall.SCORING_DIR = SECOND_AI_DIR
    score_recall.SCORES_CSV = SECOND_SCORES_CSV

    # Load full response corpus
    all_responses = score_recall.load_responses()
    if not all_responses:
        logger.error("No responses found.")
        sys.exit(1)

    # Determine which pairs still need scoring
    human_pairs = load_human_pairs()
    already_scored = set() if args.force else load_already_scored()
    todo = human_pairs - already_scored

    logger.info(
        f"Human pairs: {len(human_pairs)} | "
        f"already scored: {len(already_scored)} | "
        f"remaining: {len(todo)}"
    )

    if not todo:
        logger.info(
            "All human-rated pairs already scored. "
            "Use --force to rerun."
        )
    else:
        # Build flat list of (subject_id, stim_id, free_response, nodes)
        pairs = []
        for stim_id, resp_list in sorted(all_responses.items()):
            nodes = score_recall.load_codebook(stim_id)
            if nodes is None:
                continue
            for r in resp_list:
                if (r["subject_id"], stim_id) in todo:
                    pairs.append(
                        (r["subject_id"], stim_id, r["free_response"], nodes)
                    )

        if not pairs:
            logger.error(
                "No matching responses found for the human-rated pairs. "
                "Check that behavioral CSVs cover these participants/stimuli."
            )
            sys.exit(1)

        logger.info(f"Scoring {len(pairs)} pairs ...")

        if args.dry_run:
            for subject_id, stim_id, free_response, nodes in pairs[:3]:
                print(f"\n{'='*70}")
                print(f"DRY RUN — {subject_id} × {stim_id}")
                print(f"  Response : {free_response[:120]}...")
                n_correct = sum(1 for n in nodes if n.get("status") == "correct")
                print(f"  Nodes    : {len(nodes)} ({n_correct} correct)")
                print(f"{'='*70}")
            if len(pairs) > 3:
                print(f"  ... and {len(pairs) - 3} more pairs")
            return

        print("\n" + "=" * 70)
        api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")
        if not api_key.strip():
            logger.error("No API key provided.")
            sys.exit(1)
        print("API key received")
        print("=" * 70 + "\n")

        scorer = score_recall.RecallScorer(
            api_key=api_key.strip(),
            model=args.model,
            max_concurrency=args.max_concurrency,
        )

        tasks = [
            scorer.score_pair(subj, stim, fr, nodes, args.force)
            for subj, stim, fr, nodes in pairs
        ]
        results = await asyncio.gather(*tasks)

        logger.info("\nFilling in recalled=0 for unscored nodes ...")
        score_recall.fill_zeros(
            pairs,
            {stim_id: nodes for _, stim_id, _, nodes in pairs},
        )

        n_ok = sum(1 for r in results if r["status"] in ("ok", "ok_empty"))
        n_failed = sum(1 for r in results if r["status"] == "failed")
        print(f"\n{'='*50}")
        print(f"Done.  ok={n_ok}  failed={n_failed}")
        print(f"Per-node scores -> {SECOND_SCORES_CSV}")
        if n_failed:
            failed_pairs = [
                (r["subject_id"], r["stim_id"])
                for r in results
                if r["status"] == "failed"
            ]
            print(f"Failed pairs: {failed_pairs}")
            print("Re-run (without --force) to retry only failed pairs.")
        print(f"{'='*50}\n")

    # ---------------------------------------------------------------------------
    # Aggregate to recall_by_category.csv
    # ---------------------------------------------------------------------------
    if not SECOND_SCORES_CSV.exists():
        logger.warning(
            "second_AI_rated_responses/recall_scores.csv not found — "
            "skipping aggregation."
        )
        return

    logger.info("Aggregating to recall_by_category.csv ...")
    from aggregate_recall import aggregate

    aggregate(SECOND_SCORES_CSV, SECOND_BY_CATEGORY_CSV)
    logger.info(f"  Written -> {SECOND_BY_CATEGORY_CSV}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate AI-2 recall scores for the subset of pairs "
            "scored by the human rater."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without making API calls.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess pairs already present in second_AI_rated_responses/.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max parallel API calls (default: {DEFAULT_CONCURRENCY}).",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Generating AI-2 recall scores (human-rater subset)")
    logger.info("=" * 60)
    logger.info(f"  Human scores : {HUMAN_SCORES_CSV}")
    logger.info(f"  Output dir   : {SECOND_AI_DIR}")
    logger.info(f"  Model        : {args.model}")
    logger.info(f"  Force        : {args.force}")
    logger.info(f"  Dry run      : {args.dry_run}")
    logger.info(f"  Concurrency  : {args.max_concurrency}")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()