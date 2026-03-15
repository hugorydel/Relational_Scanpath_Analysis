"""
pipeline/module_4/constants.py
==============================
All symbolic names, thresholds, and model specifications for Module 4.

Encoding-only analysis. DVs are empirically normalised proportions (0-1)
computed in loader.py as participant_recalled / max_recalled_per_stim.

No analysis logic lives here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Below this many subjects, statsmodels' REML C optimizer can hard-crash.
# Fall back to OLS + C(SubjectID) fixed effect for pilot runs.
PILOT_SUBJ_THRESHOLD = 10

# ---------------------------------------------------------------------------
# Covariate lists
# ---------------------------------------------------------------------------

ENC_COVARIATES = ["n_fixations_enc", "aoi_prop_enc", "mean_salience_relational_enc"]

# ---------------------------------------------------------------------------
# Dependent variables
# ---------------------------------------------------------------------------
# These are proportion columns computed in loader.py (0-1 scale).

DV_TOTAL = "prop_total"  # all correct nodes recalled / empirical max
DV_RELATIONS = "prop_relations"  # (action + spatial) recalled / empirical max
DV_OBJECTS = "prop_objects"  # (identity + attribute) recalled / empirical max

# ---------------------------------------------------------------------------
# Default I/O paths
# ---------------------------------------------------------------------------

DEFAULT_SCORES_PATH = config.OUTPUT_DIR / "scoring" / "recall_by_category.csv"

# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------
# Each entry: (name, primary_predictor, description, table_key, hypothesis_group)
#
# table_key: "enc" (single encoding table)
# hypothesis_group: H1 | H2 | Exploratory
#
# primary_predictor special values:
#   "1"  → intercept-only (H1 SVG strength test)
#   anything else → statsmodels formula fragment

MODEL_SPECS = [
    # H1 — is encoding SVG reliably above zero?
    (
        "H1_svg_enc",
        "1",
        "H1: Encoding SVG — intercept test (mean SVG > 0?)",
        "enc",
        "H1",
    ),
    # H2 — encoding SVG → memory proportions
    (
        "H2_total",
        "svg_z_enc_z",
        "H2 overall: Encoding SVG → total node recall proportion",
        "enc",
        "H2",
    ),
    (
        "H2_relations",
        "svg_z_enc_z",
        "H2 relations: Encoding SVG → relational recall proportion (action + spatial)",
        "enc",
        "H2",
    ),
    (
        "H2_objects",
        "svg_z_enc_z",
        "H2 objects: Encoding SVG → object recall proportion (identity + attribute)",
        "enc",
        "H2",
    ),
    # Exploratory — dissociation: is the SVG effect larger for relations than objects?
    # Fit via the stacked long-format interaction model (see test_relation_object_dissociation.py)
    # Included here for coefficient logging only; formula handled specially in models.py.
    (
        "EXP_dissociation",
        "svg_z_enc_z * memory_type",
        "Exploratory: SVG × memory_type (relations vs objects dissociation)",
        "enc_long",
        "Exploratory",
    ),
]
