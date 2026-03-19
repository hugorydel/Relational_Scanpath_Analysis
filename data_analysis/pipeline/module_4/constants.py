"""
pipeline/module_4/constants.py
==============================
All symbolic names, thresholds, and model specifications for Module 4.

Encoding and decoding analysis. DVs are empirically normalised proportions (0-1)
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

ENC_COVARIATES = [
    "n_fixations_enc",
    "aoi_prop_enc",
    "mean_salience_relational_enc",
    "enc_total_correct",
]

# Between-image SVG covariate — included in H2/Exploratory models alongside
# the within-image predictor to decompose participant-level from image-level
# SVG variance. Computed in loader.py via per-StimID mean centering.
ENC_BETWEEN_COVARIATES = ["svg_z_enc_image_mean","svg_z_enc_within_sd",
    "prop_total_image_sd"]

# ---------------------------------------------------------------------------
# Decoding covariate lists
# ---------------------------------------------------------------------------
# No enc_total_correct equivalent — that is encoding-task-specific.

DEC_COVARIATES = [
    "n_fixations_dec",
    "aoi_prop_dec",
    "mean_salience_relational_dec",
]

DEC_BETWEEN_COVARIATES = ["svg_z_dec_image_mean"]

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
DEFAULT_FLAGS_PATH = config.OUTPUT_DIR / "scoring" / "wrong_image_flags.csv"

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
    # Primary predictor is the within-image centred SVG (participant deviation
    # from per-image mean). svg_z_enc_image_mean is included as a between-image
    # covariate to cleanly separate the two variance components.
    (
        "H2_total",
        "svg_z_enc_within_z",
        "H2 overall: Encoding SVG (within-image) → total node recall proportion",
        "enc",
        "H2",
    ),
    (
        "H2_relations",
        "svg_z_enc_within_z",
        "H2 relations: Encoding SVG (within-image) → relational recall proportion (action + spatial)",
        "enc",
        "H2",
    ),
    (
        "H2_objects",
        "svg_z_enc_within_z",
        "H2 objects: Encoding SVG (within-image) → object recall proportion (identity + attribute)",
        "enc",
        "H2",
    ),
    # Exploratory — dissociation
    (
        "EXP_dissociation",
        "svg_z_enc_within_z * memory_type",
        "Exploratory: SVG (within-image) × memory_type (relations vs objects dissociation)",
        "enc_long",
        "Exploratory",
    ),
    # H1 — is decoding SVG reliably above zero?
    (
        "H1_svg_dec",
        "1",
        "H1: Decoding SVG — intercept test (mean SVG > 0?)",
        "dec",
        "H1",
    ),
    # H2 — decoding SVG → memory proportions
    (
        "H2_dec_total",
        "svg_z_dec_within_z",
        "H2 decoding overall: Decoding SVG (within-image) → total node recall proportion",
        "dec",
        "H2",
    ),
    (
        "H2_dec_relations",
        "svg_z_dec_within_z",
        "H2 decoding relations: Decoding SVG (within-image) → relational recall proportion (action + spatial)",
        "dec",
        "H2",
    ),
    (
        "H2_dec_objects",
        "svg_z_dec_within_z",
        "H2 decoding objects: Decoding SVG (within-image) → object recall proportion (identity + attribute)",
        "dec",
        "H2",
    ),
]