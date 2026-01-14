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
MIN_MEMORABILITY = 0.75
MIN_MASK_AREA_PERCENT = 1.0  # Minimum area percent to count mask in coverage percentage
MIN_OBJECTS = 10
MAX_OBJECTS = 30
MIN_RELATIONS = 10
MIN_COVERAGE_PERCENT = 80.0
MIN_INTERACTIONAL_RELATIONS = 1

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
