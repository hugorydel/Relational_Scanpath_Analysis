# pipeline/module_4/__init__.py
# Public API — import from here in run_pipeline.py and module4_analysis.py.
from .constants import (
    COVARIATES,
    DEC_COVARIATES,
    DV_CONFAB,
    DV_LENGTH,
    DV_OBJECTS,
    DV_RELATIONAL,
    ENC_COVARIATES,
    MIN_N_SHARED_REPLAY,
    MODEL_SPECS,
    PILOT_SUBJ_THRESHOLD,
)
from .loader import (
    apply_exclusions,
    build_analysis_tables,
    load_data,
    load_memory_scores,
    standardise_tables,
)
from .models import fit_all_models
from .output import summarise
from .pilot_diagnostics import make_pilot_diagnostics

__all__ = [
    # constants
    "COVARIATES",
    "DEC_COVARIATES",
    "DV_CONFAB",
    "DV_LENGTH",
    "DV_OBJECTS",
    "DV_RELATIONAL",
    "ENC_COVARIATES",
    "MIN_N_SHARED_REPLAY",
    "MODEL_SPECS",
    "PILOT_SUBJ_THRESHOLD",
    # loader
    "apply_exclusions",
    "build_analysis_tables",
    "load_data",
    "load_memory_scores",
    "standardise_tables",
    # models
    "fit_all_models",
    # output
    "summarise",
    "make_pilot_diagnostics",
]
