"""
config.py
=========
Central configuration for the relational replay pipeline.
All hardcoded values live here — adjust as needed without touching module code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_BEHAVIORAL_DIR = ROOT_DIR / "data_behavioral"
DATA_EYETRACKING_DIR = ROOT_DIR / "data_eyetracking"
DATA_METADATA_DIR = ROOT_DIR / "data_metadata"
RESOURCES_DIR = ROOT_DIR / "resources"

OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_BEHAVIORAL_DIR = OUTPUT_DIR / "behavioral"
OUTPUT_EYETRACKING_DIR = OUTPUT_DIR / "eyetracking"
OUTPUT_FEATURES_DIR = OUTPUT_DIR / "features"

METADATA_FILE = DATA_METADATA_DIR / "stimuli_dataset.json"

# ---------------------------------------------------------------------------
# Experiment design
# ---------------------------------------------------------------------------

N_ENCODING_TRIALS = 60
N_DECODING_TRIALS = 30
N_EXPLORATORY_TRIALS = 10
N_DISTRACTOR_RANGE = (40, 60)  # time-terminated loop, count not fixed

ENCODING_WINDOW_MS = 5000  # image viewing phase duration
DECODING_WINDOW_MS = 7000  # blank screen retrieval phase duration

N_PARTICIPANTS = 30

# ---------------------------------------------------------------------------
# Display geometry (needed for gaze → image pixel coordinate transform)
# ---------------------------------------------------------------------------
# TODO: confirm these against the actual E-Prime display settings

DISPLAY_WIDTH_PX = 1920
DISPLAY_HEIGHT_PX = 1080

# Stimulus images are letterboxed to fit within this bounding box on screen
# (centred horizontally and vertically)
# TODO: confirm stimulus display size used in E-Prime
STIM_DISPLAY_WIDTH_PX = None  # e.g. 1280 — fill in once confirmed
STIM_DISPLAY_HEIGHT_PX = None  # e.g. 960  — fill in once confirmed

# ---------------------------------------------------------------------------
# EyeLink / eye-tracking
# ---------------------------------------------------------------------------

# Minimum fixation duration to include (ms) — EyeLink default parser minimum
MIN_FIXATION_DURATION_MS = 80

# Saccade amplitude threshold below which a transition is flagged as
# a proximity transition (used for sensitivity analysis in Module 3).
# Expressed in pixels at display resolution; convert to degrees downstream
# using PIXELS_PER_DEGREE.
# TODO: set based on confirmed viewing distance and screen size
VIEWING_DISTANCE_CM = None  # e.g. 60
SCREEN_WIDTH_CM = None  # e.g. 52
PIXELS_PER_DEGREE = None  # computed as: (DISPLAY_WIDTH_PX / SCREEN_WIDTH_CM)
#              * (VIEWING_DISTANCE_CM * tan(1°))
# fill in once viewing geometry is confirmed

SHORT_SACCADE_DEG = 1.0  # saccades below this (degrees) flagged as proximity

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# Relation directionality for graph-walk scoring:
# 'undirected' checks both (i→j) and (j→i); 'directed' checks (i→j) only
RELATION_DIRECTIONALITY = "undirected"

# Minimum number of valid AOI-landing fixation pairs required during the
# decoding blank screen for a trial to be included (not flagged insufficient)
MIN_VALID_TRANSITIONS = 2

# Salience map parameters (spectral residual, Hou & Zhang 2007)
SALIENCE_SMOOTHING_SIGMA = 8  # Gaussian smoothing sigma applied after IFFT

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = "INFO"  # DEBUG for verbose output during development
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
LOG_DATEFMT = "%H:%M:%S"
