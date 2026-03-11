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

import numpy as np

# Ensure project root is on path so 'import config' works from anywhere
sys.path.insert(0, str(Path(__file__).parent))

import config
from pipeline.misc import get_subject_ids, init_output_dirs, setup_logging
from pipeline.module1_behavioral import process_subject as run_module1
from pipeline.module2_eyetracking import process_subject as run_module2
from pipeline.module3_features import process_subject as run_module3
from pipeline.module4_analysis import (
    apply_exclusions,
    build_analysis_tables,
    fit_all_models,
    load_data,
    load_memory_scores,
    standardise_tables,
    summarise,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-participant runner
# ---------------------------------------------------------------------------


def process_participant(
    subject_id: str, modules: list[int], rng: np.random.Generator = None
) -> dict[int, bool]:
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
        try:
            ok = run_module2(
                subject_id,
                edf_dir=config.DATA_EYETRACKING_DIR,
                beh_dir=config.OUTPUT_BEHAVIORAL_DIR,
                output_dir=config.OUTPUT_EYETRACKING_DIR,
            )
            results[2] = ok
        except Exception as e:
            logger.error(f"[{subject_id}] Module 2 crashed: {e}", exc_info=True)
            results[2] = False

    if 3 in modules:
        try:
            ok = run_module3(subject_id, force_aoi=False, rng=rng)
            results[3] = ok is not None and not ok.empty
        except Exception as e:
            logger.error(f"[{subject_id}] Module 3 crashed: {e}", exc_info=True)
            results[3] = False

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _concat_trial_features(
    output_path: Path, subject_ids: list[str] | None = None
) -> None:
    """
    Concatenate per-participant _trial_features.csv files into trial_features_all.csv.

    If subject_ids is provided, only files matching those IDs are included.
    This prevents stale files from previous runs silently entering the analysis.
    """
    import pandas as pd

    feature_files = sorted(config.OUTPUT_FEATURES_DIR.glob("*_trial_features.csv"))
    feature_files = [f for f in feature_files if f.name != "trial_features_all.csv"]

    if subject_ids is not None:
        # Keep only files whose stem contains one of the requested subject IDs
        feature_files = [
            f for f in feature_files if any(sid in f.stem for sid in subject_ids)
        ]
        if not feature_files:
            logger.warning(
                "  No _trial_features.csv files found matching the requested "
                f"subjects — trial_features_all.csv not written."
            )
            return

    if not feature_files:
        logger.warning(
            "  No _trial_features.csv files found — trial_features_all.csv not written."
        )
        return

    dfs = [pd.read_csv(f, dtype={"StimID": str}) for f in feature_files]
    combined = pd.concat(dfs, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(
        f"  trial_features_all.csv written: "
        f"{len(combined)} rows from {len(feature_files)} participants → {output_path.name}"
    )


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
    rng = np.random.default_rng(seed=42)  # For reproducibility in any random steps
    all_results = {}

    for sid in subject_ids:
        logger.info(f"--- {sid} ---")
        all_results[sid] = process_participant(sid, per_subject_modules, rng=rng)

    # Concatenate per-participant trial_features CSVs into one file.
    # This always runs after any module-3 pass so Module 4 has an up-to-date
    # combined file regardless of whether module 3 was in this invocation.
    features_path = config.OUTPUT_FEATURES_DIR / "trial_features_all.csv"
    _concat_trial_features(features_path, subject_ids=subject_ids)

    # Module 4: mixed-effects analysis across all participants
    if 4 in args.modules:
        logger.info("--- Module 4: Mixed-effects analysis ---")
        if not features_path.exists():
            logger.error(
                f"  trial_features_all.csv not found at {features_path}. "
                f"Run Module 3 first to generate it."
            )
        else:
            try:
                output_dir = config.OUTPUT_DIR / "analysis"
                raw_df = load_data(features_path)
                memory_scores = load_memory_scores(config.MEMORY_SCORES_FILE)
                tables = build_analysis_tables(raw_df, memory_scores)
                filtered = apply_exclusions(tables)
                filtered = standardise_tables(filtered)
                results = fit_all_models(filtered)
                summarise(results, filtered, output_dir, plot=True)
                logger.info("  Module 4 complete.")
            except Exception as e:
                logger.error(f"  Module 4 crashed: {e}", exc_info=True)

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
    if 4 in args.modules:
        logger.info(f"  Module 4: see output/analysis/ for results")

    logger.info("=" * 55)


if __name__ == "__main__":
    main()
