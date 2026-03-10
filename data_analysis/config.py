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

N_ENCODING_TRIALS = 30
N_ENCODING_QUESTIONS = 2  # questions per image at encoding (two separate trials)
N_DECODING_TRIALS = 30
N_DECODING_QUESTIONS = 1  # free-recall responses per image at decoding
N_EXPLORATORY_RANGE = (0, 0)  # exploratory phase removed; free recall is in decoding
N_DISTRACTOR_RANGE = (1, 148)  # time-terminated loop, count not fixed

ENCODING_WINDOW_MS = 5000  # image viewing phase duration
DECODING_WINDOW_MS = 7000  # blank screen retrieval phase duration

# ---------------------------------------------------------------------------
# Scoring (Module 4 / manual annotation tool)
# ---------------------------------------------------------------------------

DATA_METADATA_IMAGES_DIR = DATA_METADATA_DIR / "images"
OUTPUT_SCORING_DIR = OUTPUT_DIR / "data_scoring"
MEMORY_SCORES_FILE = OUTPUT_SCORING_DIR / "memory_scores.csv"
MEMORY_ANNOTATIONS_FILE = OUTPUT_SCORING_DIR / "memory_annotations.json"


N_PARTICIPANTS = 30

# ---------------------------------------------------------------------------
# Display geometry (needed for gaze → image pixel coordinate transform)
# ---------------------------------------------------------------------------

DISPLAY_WIDTH_PX = 1920
DISPLAY_HEIGHT_PX = 1080

# All stimulus images are standardised to this resolution in image space.
# The screen→image transform is: img_x = gaze_x * (IMAGE_W / DISPLAY_WIDTH_PX)
#                                 img_y = gaze_y * (IMAGE_H / DISPLAY_HEIGHT_PX)
IMAGE_W = 1024
IMAGE_H = 768

# ---------------------------------------------------------------------------
# EyeLink / eye-tracking
# ---------------------------------------------------------------------------

# Minimum fixation duration to include (ms) — EyeLink default parser minimum
MIN_FIXATION_DURATION_MS = 80

# TODO: set based on confirmed viewing distance and screen size
VIEWING_DISTANCE_CM = None
SCREEN_WIDTH_CM = None
PIXELS_PER_DEGREE = None

SHORT_SACCADE_DEG = 1.0  # saccades below this (degrees) flagged as proximity

# ---------------------------------------------------------------------------
# AOI assignment (Module 3, Step 3)
# ---------------------------------------------------------------------------

# Proximity fallback: if a fixation misses all polygons, assign it to the
# nearest object centroid if within this distance (image pixels).
# Encoding uses a tighter threshold — the image is present so near-misses
# are more likely to be real object fixations just outside the polygon edge.
# Decoding uses a looser threshold — gaze is guided by mental reinstatement
# which is noisier, so a wider tolerance is appropriate.
AOI_PROXIMITY_THRESHOLD_ENCODING_PX = 50
AOI_PROXIMITY_THRESHOLD_DECODING_PX = 100

# Minimum number of AOI-assigned fixations in a trial required to compute
# SVG alignment and LCS scores. Trials below this are still output but
# flagged with low_n=True for downstream filtering.
MIN_AOI_FIXATIONS_PER_TRIAL = 3

# Minimum number of valid consecutive AOI transitions (after self-transition
# removal) required to compute a graph-walk score. Trials with only 1
# transition produce a score of either 0 or 1 with no gradation.
MIN_VALID_TRANSITIONS = 2

# ---------------------------------------------------------------------------
# Feature engineering (Module 3, Steps 5–6)
# ---------------------------------------------------------------------------

# Relation directionality for graph-walk scoring:
# 'undirected' checks frozenset({A, B}) — A→B and B→A are the same edge.
# 'directed'   checks (A, B) only — order matters.
RELATION_DIRECTIONALITY = "undirected"

# Number of permutations for the SVG alignment null distribution.
# 1000 gives stable z-scores; increase to 5000 for final analysis if needed.
SVG_N_PERMUTATIONS = 1000

# ---------------------------------------------------------------------------
# Saliency maps (Module 3, Step 2)
# ---------------------------------------------------------------------------

# Gaussian smoothing sigma applied at the 64×64 FFT stage (Hou & Zhang 2007).
# σ=3 gives peaked blobs around objects without over-diffusing.
SALIENCE_SMOOTHING_SIGMA = 3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = "INFO"  # DEBUG for verbose output during development
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
LOG_DATEFMT = "%H:%M:%S"
