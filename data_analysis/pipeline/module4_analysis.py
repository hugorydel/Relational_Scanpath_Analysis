"""
module4_analysis.py
===================
Module 4: Mixed-effects analysis of relational retrieval and episodic memory.

Hypotheses
----------
H1 — Decoding relational alignment exists
    Participants predictably trace relational structure during episodic
    retrieval, above and beyond low-level fixation characteristics.
    Test: svg_z_inter_dec reliably > 0 (raw H1a, then covariate-adjusted H1b).

H2 — Decoding relational alignment scaffolds episodic memory
    Higher relational alignment during retrieval predicts better free-recall
    of relational details (primary) and object details (secondary).
    Primary  : svg_z_inter_dec → n_relational_correct
    Secondary: svg_z_inter_dec → n_objects_correct

Exploratory
-----------
    Confabulation : dec SVG → n_relational_incorrect
    Writing length: dec SVG → writing_length
    Encoding SVG + LCS/tau → n_relational_correct

Pipeline
--------
Step 1  — Load trial_features_all.csv + memory_scores.csv
Step 2  — Build analysis tables (dec / enc / wide)
Step 3  — Hypothesis-specific exclusions
Step 4  — Standardise predictors within each filtered table
Step 5  — Fit models (OLS+C(SubjectID) for pilot; LMM for final)
Step 6  — Summarise and write outputs:
            output/analysis/analysis_*.csv
            output/analysis/model_coefficients.csv
            output/analysis/model_summaries.txt
            output/analysis/forest_plot.png

Implementation
--------------
Logic is split across pipeline/module4/:
    constants.py  — MODEL_SPECS, DVs, covariate lists, thresholds
    loader.py     — Steps 1-4 (load, build tables, exclusions, standardise)
    models.py     — Step 5 (formula construction, OLS/LMM fitting)
    output.py     — Step 6 (coefficient tables, summaries, forest plot)

Usage
-----
    python module4_analysis.py
    python module4_analysis.py --input  path/to/trial_features_all.csv
    python module4_analysis.py --scores path/to/memory_scores.csv
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
        description="Module 4: Mixed-effects analysis of relational retrieval -> memory."
    )
    parser.add_argument(
        "--input",
        default=str(config.OUTPUT_FEATURES_DIR / "trial_features_all.csv"),
        help="Path to trial_features_all.csv (Module 3 output).",
    )
    parser.add_argument(
        "--scores",
        default=str(DEFAULT_SCORES_PATH),
        help="Path to manually scored memory_scores.csv.",
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
    logger.info("Module 4: Relational retrieval -> episodic memory")
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
        features_path=Path(args.input),
        scores_path=Path(args.scores),
    )

    logger.info("\n" + "=" * 60)
    logger.info("Module 4 complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
