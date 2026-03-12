"""
utils/scene_graph.py
====================
Step 1 of Module 3: Load and index scene graph metadata.

Provides two cached indices built once from stimuli_dataset.json
and shared across all modules:

  build_polygon_index()  — AOI polygons per image, keyed by StimID
  build_graph_index()    — relational edges per image, keyed by StimID

Both functions are LRU-cached — they load and parse the JSON exactly
once per pipeline run regardless of how many subjects are processed.
"""

import functools
import json
import logging
from pathlib import Path
from typing import Optional

import config
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw metadata loader
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def load_stimulus_metadata(metadata_file: Optional[Path] = None) -> dict:
    """
    Load stimuli_dataset.json and return a dict keyed by image_id (string).

    Cached after the first call — all modules share one copy in memory.

    Returns
    -------
    dict
        {
            "2383555": {
                "image_id":   "2383555",
                "objects":    [...],
                "relations":  [...],
                "story":      "...",
                "memorability": 0.82,
                ...
            },
            ...
        }
    """
    metadata_file = Path(metadata_file) if metadata_file else config.METADATA_FILE
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    logger.info(f"Loading stimulus metadata from {metadata_file.name} ...")
    with metadata_file.open(encoding="utf-8") as f:
        raw = json.load(f)

    images = raw.get("images", [])
    keyed = {str(img["image_id"]): img for img in images}
    logger.info(f"  Loaded metadata for {len(keyed)} stimuli.")
    return keyed


# ---------------------------------------------------------------------------
# Internal polygon parser
# ---------------------------------------------------------------------------


def _parse_polygon(flat_coords: list) -> np.ndarray | None:
    """
    Convert a flat [x0, y0, x1, y1, ...] list to an (N, 2) float array.
    Filters out any non-numeric values (e.g. '...' truncation artifacts).
    Returns None if fewer than 3 valid points remain (degenerate polygon).
    Coordinates are in 1024×768 image space — no transform applied.
    """
    nums = [c for c in flat_coords if isinstance(c, (int, float))]
    if len(nums) < 6:
        return None
    xs = nums[0::2]
    ys = nums[1::2]
    n = min(len(xs), len(ys))
    return np.array(list(zip(xs[:n], ys[:n])), dtype=float)


# ---------------------------------------------------------------------------
# Polygon index
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def build_polygon_index(metadata_file: Optional[Path] = None) -> dict:
    """
    Build an AOI polygon index from stimulus metadata.

    Returns
    -------
    dict
        {
            "2383555": [
                {
                    "object_id": 1,
                    "name":      "woman holding a child",
                    "polygon":   np.ndarray of shape (N, 2),  # image pixels
                    "centroid":  (cx, cy),                    # image pixels
                },
                ...
            ],
            ...
        }

    Notes
    -----
    - Only objects with a parseable polygon (≥3 points) are included.
    - Objects excluded by the >1% size filter in the dataset are still
      excluded here because they are simply absent from the objects array.
    - Coordinates are in 1024×768 image space (no screen transform).
    - Cached after first call.
    """
    metadata_file = Path(metadata_file) if metadata_file else None
    metadata = load_stimulus_metadata(metadata_file)

    index = {}
    total_objects = 0
    total_skipped = 0

    for stim_id, entry in metadata.items():
        poly_list = []
        for obj in entry.get("objects", []):
            polygon = _parse_polygon(obj.get("polygon", []))
            if polygon is None:
                total_skipped += 1
                continue
            centroid = (float(polygon[:, 0].mean()), float(polygon[:, 1].mean()))
            poly_list.append(
                {
                    "object_id": obj["object_id"],
                    "name": obj.get("name", ""),
                    "polygon": polygon,
                    "centroid": centroid,
                }
            )
            total_objects += 1
        index[stim_id] = poly_list

    logger.info(
        f"Polygon index built: {total_objects} objects across "
        f"{len(index)} stimuli "
        f"({total_skipped} skipped — degenerate or missing polygons)."
    )
    return index


# ---------------------------------------------------------------------------
# Graph index
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def build_graph_index(metadata_file: Optional[Path] = None) -> dict:
    """
    Build a relational graph index from stimulus metadata.

    Returns
    -------
    dict with two sub-dicts:
        "all"           : {stim_id: set of frozenset({obj_id_a, obj_id_b})}
        "interactional" : {stim_id: set of frozenset({obj_id_a, obj_id_b})}

    Notes
    -----
    - Edges are stored as frozensets (undirected). A→B and B→A are the
      same edge. This matches the undirected treatment agreed for the
      graph-walk metric.
    - Self-loops (A→A) are excluded.
    - "all" contains only relations where predicate_category is one of
      {"interactional", "spatial", "functional"}. Social and emotional
      predicates are excluded as they are the noisiest and least reliably
      annotated categories.
    - "interactional" contains only relations where
      predicate_category == "interactional", retained for reference.
    - The adjacency_matrix field in the JSON is ignored — we recompute
      binary 0/1 edges from the relations array directly.
    - Cached after first call.
    """
    _CORE_CATEGORIES = {"interactional", "spatial", "functional"}

    metadata_file = Path(metadata_file) if metadata_file else None
    metadata = load_stimulus_metadata(metadata_file)

    all_edges = {}
    inter_edges = {}
    total_relations = 0
    total_excluded = 0

    for stim_id, entry in metadata.items():
        edges_all = set()
        edges_inter = set()

        for rel in entry.get("relations", []):
            subj = rel.get("subject_id")
            obj = rel.get("object_id")
            cat = rel.get("predicate_category", "")

            if subj is None or obj is None or subj == obj:
                continue

            total_relations += 1

            if cat not in _CORE_CATEGORIES:
                total_excluded += 1
                continue

            edge = frozenset({subj, obj})
            edges_all.add(edge)
            if cat == "interactional":
                edges_inter.add(edge)

        all_edges[stim_id] = edges_all
        inter_edges[stim_id] = edges_inter

    logger.info(
        f"Graph index built: {total_relations} relations across "
        f"{len(all_edges)} stimuli "
        f"({total_excluded} excluded — social/emotional predicates). "
        f"Unique undirected edges — "
        f"core (inter+spatial+func): {sum(len(v) for v in all_edges.values())}, "
        f"interactional only: {sum(len(v) for v in inter_edges.values())}."
    )
    return {"all": all_edges, "interactional": inter_edges}
