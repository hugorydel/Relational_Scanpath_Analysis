#!/usr/bin/env python3
"""
config.py - Configuration file for SVG Relational Dataset Creator
"""

# Paths
SVG_ROOT = "data/svg_sg"
VG_IMAGE_ROOT = r"D:\Relational Scanpath Research\synthetic_visual_genome_data"
OUTPUT_DIR = "data/processed"
MEMORABILITY_CACHE_PATH = "data/memorability_cache.json"
SVG_CACHE_PATH = "data/svg_sg_cache.pkl"
PRECOMPUTED_STATS_CACHE_PATH = "data/precomputed_stats_cache.pkl"

# Filtering thresholds
MIN_MEMORABILITY = 0.9
MIN_MASK_AREA_PERCENT = 0.5  # Minimum area percent to count mask in coverage percentage
MIN_OBJECTS = 10
MAX_OBJECTS = 30
MIN_RELATIONS = 10
MIN_COVERAGE_PERCENT = 80.0

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
    "functional": 0.65,  # affordances / “used for”
    "social": 0.80,  # people–people relations
    "emotional": 0.85,  # people–people with affect
    "interactional": 1.00,  # direct, physical/attentional actions
}

# Debug/testing
MAX_IMAGES = None  # Set to int for quick debugging runs
RANDOM_SEED = 42
VISUALIZE_SAMPLES = 20

# Performance optimization
MEMORABILITY_BATCH_SIZE = 32  # Process this many images at once
NUM_WORKERS = None  # None = auto-detect CPU cores, or set to specific number (e.g., 8)
