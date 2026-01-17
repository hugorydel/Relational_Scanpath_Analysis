"""
Utility functions for SVG relational dataset processing
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import OUTPUT_DIR, PRECOMPUTED_STATS_PATH, VG_IMAGE_ROOT


def ensure_jpg(name: str) -> str:
    """
    Ensure filename has .jpg extension (prevents double extensions).

    Args:
        name: Filename or image ID

    Returns:
        Filename with .jpg extension

    Examples:
        >>> ensure_jpg("2345678")
        "2345678.jpg"
        >>> ensure_jpg("2345678.jpg")
        "2345678.jpg"
        >>> ensure_jpg("2345678.JPG")
        "2345678.JPG"
    """
    if not name:
        return name
    return name if Path(name).suffix.lower() in {".jpg", ".jpeg"} else f"{name}.jpg"


def compute_relational_graph(
    valid_objects: List[Dict],
    relations: List[Dict],
    predicate_weights: Dict[str, float],
) -> Tuple[np.ndarray, Dict]:
    """
    Compute weighted relational graph using predicate categories.

    Args:
        valid_objects: Filtered list of objects
        relations: List of relationships
        predicate_weights: Weight mapping for predicate categories

    Returns:
        adjacency_matrix: NxN weighted adjacency matrix
        object_id_map: Mapping from object IDs to indices
    """
    n_objects = len(valid_objects)
    adjacency = np.zeros((n_objects, n_objects))

    # Create object ID to index mapping
    object_id_map = {obj["object_id"]: idx for idx, obj in enumerate(valid_objects)}
    valid_object_ids = set(object_id_map.keys())

    # Process relations
    for rel in relations:
        subj_id = rel.get("subject_id")
        obj_id = rel.get("object_id")
        predicate_category = rel.get("predicate_category", "spatial")

        # Only include relations between valid objects
        if subj_id not in valid_object_ids or obj_id not in valid_object_ids:
            continue

        subj_idx = object_id_map[subj_id]
        obj_idx = object_id_map[obj_id]

        # Get weight from config
        weight = predicate_weights.get(predicate_category, 0.3)

        # Add edge (accumulate if multiple relations)
        adjacency[subj_idx, obj_idx] += weight
        adjacency[obj_idx, subj_idx] += weight  # Make undirected

    # Normalize to [0, 1]
    max_weight = adjacency.max()
    if max_weight > 0:
        adjacency = adjacency / max_weight

    return adjacency, object_id_map


def validate_paths(required_for_processing=True, required_for_diversity=False):
    """
    Validate that required paths exist before processing.

    Args:
        required_for_processing: Check paths needed for preprocessing
        required_for_diversity: Check paths needed for diversity selection

    Returns:
        True if all paths exist, False otherwise (exits with error message)
    """
    errors = []
    warnings = []

    # Always check VG_IMAGE_ROOT - critical for all operations
    vg_path = Path(VG_IMAGE_ROOT)
    if not vg_path.exists():
        errors.append(
            f"❌ VG_IMAGE_ROOT not found: {VG_IMAGE_ROOT}\n"
            f"   This directory contains the Visual Genome images.\n"
            f"   Please ensure your D: drive is connected or update the path in config.py"
        )
    elif not any(vg_path.iterdir()):
        errors.append(
            f"❌ VG_IMAGE_ROOT is empty: {VG_IMAGE_ROOT}\n"
            f"   Please ensure the Visual Genome images are downloaded to this location."
        )

    # Check reference data for preprocessing
    if required_for_processing:
        stats_path = Path(PRECOMPUTED_STATS_PATH)
        if not stats_path.exists():
            errors.append(
                f"❌ Precomputed stats not found: {PRECOMPUTED_STATS_PATH}\n"
                f"   This file contains pre-computed filtering statistics.\n"
                f"   Please ensure precomputed_stats.json exists in the data/ directory.\n"
                f"   You can download it from the project repository or build it using build_reference_data.py"
            )

    # Check processed data for diversity selection
    if required_for_diversity:
        dataset_path = Path(OUTPUT_DIR) / "annotations" / "dataset.json"
        if not dataset_path.exists():
            errors.append(
                f"❌ Processed dataset not found: {dataset_path}\n"
                f"   Please run preprocess_data.py first to generate filtered images."
            )

    # Print warnings
    if warnings:
        print("\n" + "=" * 60)
        print("WARNINGS")
        print("=" * 60)
        for warning in warnings:
            print(warning)
        print("=" * 60 + "\n")

    # Print errors and exit if any
    if errors:
        print("\n" + "=" * 60)
        print("❌ PATH VALIDATION FAILED")
        print("=" * 60)
        for error in errors:
            print(error)
        print("=" * 60)
        print("\nPlease fix the above issues and try again.")
        print("=" * 60 + "\n")
        sys.exit(1)

    return True
