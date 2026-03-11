"""
pipeline/module4/constants.py
==============================
All symbolic names, thresholds, and model specifications for Module 4.

This is the single place to:
  - Add or rename a model (MODEL_SPECS)
  - Change a DV or covariate list
  - Adjust pilot-mode thresholds

No analysis logic lives here; no imports beyond stdlib and config.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Minimum n_shared_enc_dec fixations for replay models.
# Set to 2 for pilot. Raise to 3 for final analysis.
MIN_N_SHARED_REPLAY = 2

# Below this many subjects, statsmodels' REML C optimizer hard-crashes
# (segfault) before Python's try/except can fire.  We fall back to OLS +
# C(SubjectID) fixed effect for pilot runs.
PILOT_SUBJ_THRESHOLD = 10

# ---------------------------------------------------------------------------
# Covariate lists
# ---------------------------------------------------------------------------

DEC_COVARIATES = ["n_fixations_dec", "aoi_prop_dec", "mean_salience_dec"]
ENC_COVARIATES = ["n_fixations_enc", "aoi_prop_enc", "mean_salience_enc"]

# Backwards-compatible alias used by run_pipeline.py
COVARIATES = DEC_COVARIATES

# ---------------------------------------------------------------------------
# Dependent variables
# ---------------------------------------------------------------------------

DV_RELATIONAL = "n_relational_correct"
DV_OBJECTS = "n_objects_correct"
DV_CONFAB = "n_relational_incorrect"
DV_LENGTH = "writing_length"

# ---------------------------------------------------------------------------
# Default I/O paths
# ---------------------------------------------------------------------------

DEFAULT_SCORES_PATH = config.MEMORY_SCORES_FILE

# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------
# Each entry: (name, primary_predictor, description, table_key, hypothesis_group)
#
# table_key must be one of: dec_inter | dec_all | enc_inter | enc_all | replay
# hypothesis_group must be one of: H1 | H2 | Exploratory
#
# primary_predictor special values:
#   "1"          → intercept-only (H1 raw)
#   "covariates" → covariate-adjusted intercept (H1b)
#   anything else → treated as a statsmodels formula fragment

MODEL_SPECS = [
    # H1 — decoding SVG as outcome
    (
        "H1a_svg_inter",
        "1",
        "H1a raw: Decoding interactional SVG — intercept only",
        "dec_inter",
        "H1",
    ),
    (
        "H1a_svg_all",
        "1",
        "H1a raw: Decoding all-edges SVG — intercept only",
        "dec_all",
        "H1",
    ),
    (
        "H1b_svg_inter",
        "covariates",
        "H1b adjusted: Decoding interactional SVG — covariate-adjusted intercept",
        "dec_inter",
        "H1",
    ),
    (
        "H1b_svg_all",
        "covariates",
        "H1b adjusted: Decoding all-edges SVG — covariate-adjusted intercept",
        "dec_all",
        "H1",
    ),
    # H2 primary — dec SVG → relational recall
    (
        "H2_relational_inter",
        "svg_z_inter_dec_z",
        "H2 primary: Decoding interactional SVG → relational recall",
        "dec_inter",
        "H2",
    ),
    (
        "H2_relational_all",
        "svg_z_all_dec_z",
        "H2 primary (all): Decoding all-edges SVG → relational recall",
        "dec_all",
        "H2",
    ),
    # H2 secondary — dec SVG → object recall (dissociation check)
    (
        "H2_objects_inter",
        "svg_z_inter_dec_z",
        "H2 secondary: Decoding interactional SVG → object recall",
        "dec_inter",
        "H2",
    ),
    (
        "H2_objects_all",
        "svg_z_all_dec_z",
        "H2 secondary (all): Decoding all-edges SVG → object recall",
        "dec_all",
        "H2",
    ),
    # Exploratory — confabulation
    (
        "EXP_confab_inter",
        "svg_z_inter_dec_z",
        "Exploratory: Decoding interactional SVG → relational confabulation",
        "dec_inter",
        "Exploratory",
    ),
    (
        "EXP_confab_all",
        "svg_z_all_dec_z",
        "Exploratory: Decoding all-edges SVG → relational confabulation",
        "dec_all",
        "Exploratory",
    ),
    # Exploratory — writing length (overall recall fluency)
    (
        "EXP_length_inter",
        "svg_z_inter_dec_z",
        "Exploratory: Decoding interactional SVG → writing length",
        "dec_inter",
        "Exploratory",
    ),
    (
        "EXP_length_all",
        "svg_z_all_dec_z",
        "Exploratory: Decoding all-edges SVG → writing length",
        "dec_all",
        "Exploratory",
    ),
    # Exploratory — encoding → relational recall
    (
        "EXP_enc_svg_inter",
        "svg_z_inter_enc_z",
        "Exploratory: Encoding interactional SVG → relational recall",
        "enc_inter",
        "Exploratory",
    ),
    (
        "EXP_enc_svg_all",
        "svg_z_all_enc_z",
        "Exploratory: Encoding all-edges SVG → relational recall",
        "enc_all",
        "Exploratory",
    ),
    (
        "EXP_enc_lcs",
        "lcs_enc_dec_z",
        "Exploratory: LCS sequence overlap → relational recall",
        "enc_all",
        "Exploratory",
    ),
    (
        "EXP_enc_combined",
        "svg_z_inter_enc_z + lcs_enc_dec_z",
        "Exploratory: Encoding SVG + LCS jointly → relational recall",
        "enc_inter",
        "Exploratory",
    ),
    # Exploratory — replay quality → relational recall
    (
        "EXP_replay_lcs",
        "lcs_enc_dec_z",
        "Exploratory: LCS (replay-quality filtered) → relational recall",
        "replay",
        "Exploratory",
    ),
    (
        "EXP_replay_tau",
        "tau_enc_dec_z",
        "Exploratory: Tau (replay-quality filtered) → relational recall",
        "replay",
        "Exploratory",
    ),
]
