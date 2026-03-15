"""
module4_analysis.py
===================
Module 4: Encoding SVG → episodic memory recall (proportion DVs).

Hypotheses
----------
H1 — Is encoding SVG reliably above zero?
    Test: mean svg_z_enc > 0 (intercept-only model).

H2 — Does encoding SVG predict memory?
    H2a: svg_z_enc → prop_total     (all correct nodes recalled / empirical max)
    H2b: svg_z_enc → prop_relations (action + spatial recalled / empirical max)
    H2c: svg_z_enc → prop_objects   (identity + attribute recalled / empirical max)

Exploratory
-----------
    SVG × memory_type dissociation (relations vs objects, long-format interaction)

Proportion DVs
--------------
    Each DV is normalised by the empirical per-image maximum across all
    participants (separately per DV). This controls for image complexity
    and memorability simultaneously, placing all images on a 0-1 scale.

Memory scores source
--------------------
    recall_by_category.csv  (output of aggregate_recall.py)
    NOT the manual scoring UI memory_scores.csv.

Pipeline
--------
Step 1  Load trial_features_all.csv + recall_by_category.csv
Step 2  Build encoding analysis tables
Step 3  Exclusions (low_n_enc)
Step 4  Standardise predictors
Step 5  Fit models
Step 6  Write outputs

Usage
-----
    python module4_analysis.py
    python module4_analysis.py --input  path/to/trial_features_all.csv
    python module4_analysis.py --scores path/to/recall_by_category.csv
    python module4_analysis.py --output-dir path/to/output/analysis
    python module4_analysis.py --no-plot
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.module_4 import (
    apply_exclusions,
    build_analysis_tables,
    fit_all_models,
    load_data,
    load_memory_scores,
    standardise_tables,
    summarise,
)
from pipeline.module_4.constants import DEFAULT_SCORES_PATH

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATEFMT,
    )

    parser = argparse.ArgumentParser(
        description="Module 4: Encoding SVG → episodic memory recall (proportion DVs)."
    )
    parser.add_argument(
        "--input",
        default=str(config.OUTPUT_FEATURES_DIR / "trial_features_all.csv"),
        help="Path to trial_features_all.csv (Module 3 output).",
    )
    parser.add_argument(
        "--scores",
        default=str(DEFAULT_SCORES_PATH),
        help="Path to recall_by_category.csv (aggregate_recall.py output).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.OUTPUT_DIR / "analysis"),
        help="Directory for all output files.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip forest plot generation.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Module 4: Encoding SVG → episodic memory (proportion DVs)")
    logger.info("=" * 60)

    raw_df = load_data(Path(args.input))
    memory_scores = load_memory_scores(Path(args.scores))
    tables = build_analysis_tables(raw_df, memory_scores)
    filtered = apply_exclusions(tables)
    filtered = standardise_tables(filtered)
    results = fit_all_models(filtered)
    summarise(
        results,
        filtered,
        Path(args.output_dir),
        plot=not args.no_plot,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Module 4 complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
