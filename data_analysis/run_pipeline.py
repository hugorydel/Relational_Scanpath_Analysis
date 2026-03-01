"""
run_pipeline.py
===============
Orchestrator for the relational replay analysis pipeline.

Runs all four modules sequentially. A failure in any module for a given
participant is caught, logged, and skipped — the pipeline continues to
the next participant rather than crashing.

Module 4 (merge) runs once at the end across all successfully processed
participants.

Usage
-----
    # Process all participants found in data_behavioral/
    python run_pipeline.py

    # Process specific participants only
    python run_pipeline.py --subjects sub01 sub03

    # Run only specific modules (e.g. re-run features + merge after a fix)
    python run_pipeline.py --modules 3 4

    # Verbose logging
    python run_pipeline.py --log-level DEBUG
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path so 'import config' works from anywhere
sys.path.insert(0, str(Path(__file__).parent))

import config
from pipeline.module1_behavioral import process_subject as run_module1
from pipeline.utils import get_subject_ids, init_output_dirs, setup_logging

# Placeholders — uncomment as each module is implemented
# from pipeline.module2_eyetracking import process_subject as run_module2
# from pipeline.module3_features    import process_subject as run_module3
# from pipeline.module4_merge       import run_merge        as run_module4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-participant runner
# ---------------------------------------------------------------------------


def process_participant(subject_id: str, modules: list[int]) -> dict[int, bool]:
    """
    Run the requested modules for a single participant.

    Returns a dict of {module_number: success_bool}.
    """
    results = {}

    if 1 in modules:
        try:
            ok = run_module1(
                subject_id,
                input_dir=config.DATA_BEHAVIORAL_DIR,
                output_dir=config.OUTPUT_BEHAVIORAL_DIR,
            )
            results[1] = ok
        except Exception as e:
            logger.error(f"[{subject_id}] Module 1 crashed: {e}", exc_info=True)
            results[1] = False

    if 2 in modules:
        logger.warning(f"[{subject_id}] Module 2 not yet implemented — skipping.")
        results[2] = False

    if 3 in modules:
        logger.warning(f"[{subject_id}] Module 3 not yet implemented — skipping.")
        results[3] = False

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Relational Replay Pipeline")
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Subject IDs to process (default: all found in data_behavioral/)",
    )
    parser.add_argument(
        "--modules",
        nargs="*",
        type=int,
        default=[1, 2, 3, 4],
        help="Which modules to run (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level, logfile=config.OUTPUT_DIR / "pipeline.log")
    init_output_dirs()

    # Resolve subject list
    if args.subjects:
        subject_ids = args.subjects
    else:
        subject_ids = get_subject_ids(config.DATA_BEHAVIORAL_DIR)

    if not subject_ids:
        logger.error(f"No .txt files found in {config.DATA_BEHAVIORAL_DIR}. Exiting.")
        sys.exit(1)

    logger.info(
        f"Pipeline starting — {len(subject_ids)} participant(s), modules {args.modules}"
    )
    logger.info(f"Subjects: {subject_ids}")

    # Per-participant modules (1–3)
    per_subject_modules = [m for m in args.modules if m in (1, 2, 3)]
    all_results = {}

    for sid in subject_ids:
        logger.info(f"--- {sid} ---")
        all_results[sid] = process_participant(sid, per_subject_modules)

    # Module 4: merge across all participants
    if 4 in args.modules:
        logger.info("--- Module 4: Final Merge ---")
        logger.warning("Module 4 not yet implemented — skipping.")
        # successful = [s for s, r in all_results.items() if r.get(3, False)]
        # run_module4(subject_ids=successful, output_dir=config.OUTPUT_DIR)

    # ---------------------------------------------------------------------------
    # Summary report
    # ---------------------------------------------------------------------------
    logger.info("\n" + "=" * 55)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 55)

    for module_num in per_subject_modules:
        successes = [s for s, r in all_results.items() if r.get(module_num)]
        failures = [s for s, r in all_results.items() if not r.get(module_num)]
        logger.info(
            f"  Module {module_num}: "
            f"{len(successes)}/{len(subject_ids)} passed"
            + (f" | Failed: {failures}" if failures else "")
        )

    logger.info("=" * 55)


if __name__ == "__main__":
    main()
