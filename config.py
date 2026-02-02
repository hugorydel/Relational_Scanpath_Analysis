#!/usr/bin/env python3
"""
config.py - Configuration file for SVG Relational Dataset Creator
"""

# Paths
SVG_ROOT = "data/svg_sg"
VG_IMAGE_ROOT = r"D:\Relational Scanpath Research\synthetic_visual_genome_data"
OUTPUT_DIR = "data/processed"

# Reference data (replaces old cache system)
PRECOMPUTED_STATS_PATH = (
    "data/precomputed_stats.json"  # Pre-computed filtering statistics
)
PREDICATE_MAP_PATH = "data/predicate_to_category.json"  # Predicate→category mapping

# Filtering thresholds
MIN_MEMORABILITY = 0.75  # (mean memorability score from 0 to 1)
MIN_MASK_AREA_PERCENT = 1.0  # Minimum area percent to count mask in coverage percentage
MIN_OBJECTS = 10
MAX_OBJECTS = 30
MIN_RELATIONS = 10
MIN_COVERAGE_PERCENT = 75.0

# Image parameters
TARGET_SIZE = (1024, 768)

# Data source filter
# Note: SVG contains multiple datasets (VG, ADE20K, etc.)
# We filter by filename pattern: VG images are numeric (1.jpg, 2.jpg, 2345678.jpg)
# Set to empty string "" to disable filtering and process all sources
SOURCE_FILTER = ""  # Filtering done by filename pattern in preprocessing

# Predicate category weights
PREDICATE_WEIGHTS = {
    "spatial": 0.20,  # layout only – faint
    "functional": 0.65,  # affordances / "used for"
    "social": 0.80,  # people–people relations
    "emotional": 0.85,  # people–people with affect
    "interactional": 1.00,  # direct, physical/attentional actions
}

# Debug/testing
MAX_IMAGES = None  # Set to int for quick debugging runs
RANDOM_SEED = 42
VISUALIZE_SAMPLES = 20

# Performance optimization
NUM_WORKERS = None  # None = auto-detect CPU cores, or set to specific number (e.g., 8)

# Diversity selection
EMBEDDING_CACHE_PATH = "data/embedding_cache.pkl"
N_FINAL_IMAGES = (
    200  # Target number of diverse images for diversity filtered stimulus set
)
EMBEDDING_TYPE = "text"  # "image" or "text" - text is better for semantic diversity
SELECTION_METHOD = "greedy"  # "clustering" or "greedy"
SIMILARITY_THRESHOLD = (
    0.7  # For greedy method: max cosine similarity to already selected
)
N_CLUSTERS = None  # For "clustering" method: None = same as N_FINAL_IMAGES
EMBEDDING_BATCH_SIZE = 32  # Batch size for CLIP processing

# OpenAI API
OPENAI_MODEL = "gpt-5.2"

#  ============================================================================
# OPENAI SCORING CONFIGURATION
# ============================================================================

# Eligibility thresholds
CIC_THRESHOLD = 1  # Character Interaction Complexity; i.e. # of interacting characters.
SEP_THRESHOLD = 1  # Spatial Separation
DYN_THRESHOLD = 1  # Dynamic Action
QLT_THRESHOLD = 1  # Image Quality

# Score weights
CIC_WEIGHT = 2.5
SEP_WEIGHT = 1.0
DYN_WEIGHT = 1.5
QLT_WEIGHT = 1.0


def calculate_image_score(cic: int, sep: int, dyn: int, qlt: int) -> dict:
    """
    Calculate eligibility and score for an image based on OpenAI ratings.

    Eligibility: Image must meet all threshold requirements
    Score: Weighted sum of dimensions, zeroed if not eligible

    Args:
        cic: Character Interaction Complexity (0-3)
        sep: Spatial Separation (0-2)
        dyn: Dynamic Action (0-2)
        qlt: Image Quality (0-1)

    Returns:
        Dictionary with 'eligible' (0 or 1) and 'score' (float)
    """
    # Handle missing values - treat as 0
    cic = int(cic) if cic is not None else 0
    sep = int(sep) if sep is not None else 0
    dyn = int(dyn) if dyn is not None else 0
    qlt = int(qlt) if qlt is not None else 0

    # Calculate eligibility
    eligible = (
        1
        if (
            cic >= CIC_THRESHOLD
            and sep >= SEP_THRESHOLD
            and dyn >= DYN_THRESHOLD
            and qlt >= QLT_THRESHOLD
        )
        else 0
    )

    # Calculate score (zeroed if not eligible)
    score = eligible * (
        cic * CIC_WEIGHT + sep * SEP_WEIGHT + dyn * DYN_WEIGHT + qlt * QLT_WEIGHT
    )

    return {"eligible": eligible, "score": score}


# ============================================================================
# SCORED DIVERSITY SELECTION CONFIGURATION
# ============================================================================

# Path to OpenAI scored images
SCORED_IMAGES_PATH = "./data/scored_images/results.jsonl"

# Text embedding configuration
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, good quality
# Alternative models:
# "sentence-transformers/all-mpnet-base-v2"  # Slower, higher quality
# "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual

TEXT_EMBEDDING_CACHE_PATH = "./data/text_embedding_cache.pkl"

# Diversity selection parameters
N_FINAL_SCORED_IMAGES = 70  # Target number of final images
SCORED_SIMILARITY_THRESHOLD = 0.5  # Cosine similarity threshold (0-1)
