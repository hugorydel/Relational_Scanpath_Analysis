"""
Utility functions for SVG relational dataset processing
"""

from typing import Dict, List, Tuple

import numpy as np


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
