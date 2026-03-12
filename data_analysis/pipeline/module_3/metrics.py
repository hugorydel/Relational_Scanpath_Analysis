"""
module_3/metrics.py
================
Steps 4–6 of Module 3: sequence construction and scanpath metrics.

Step 4 — build_object_sequence()
    Converts AOI-assigned fixations for one trial into an ordered object
    sequence, collapsing consecutive self-transitions.

Step 5 — svg_alignment()
    Computes the SVG alignment z-score for one object sequence against the
    scene's relational graph, using a permutation null distribution.

All functions operate on plain Python lists / numpy arrays and have no
side effects — they can be called independently for testing.
"""

import logging
from typing import Optional

import config
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 4: Build object sequence
# ---------------------------------------------------------------------------


def build_object_sequence(
    trial_fixations: "pd.DataFrame",
) -> list[int]:
    """
    Build a temporally ordered object sequence from fixations in one trial.

    Parameters
    ----------
    trial_fixations : pd.DataFrame
        Subset of fixations_aoi for a single trial (one StimID × Phase ×
        ViewingNumber combination). Must contain columns:
        FixStart_ms, ObjectID.

    Returns
    -------
    list of int
        Ordered object IDs, consecutive duplicates collapsed (self-transitions
        removed). Fixations with ObjectID=NaN are excluded.

    Examples
    --------
    Raw:      [A, A, B, C, C, B, A]
    Returned: [A, B, C, B, A]
    """
    # Filter to AOI-assigned fixations only
    assigned = trial_fixations[trial_fixations["ObjectID"].notna()].copy()

    if assigned.empty:
        return []

    # Sort by fixation onset
    assigned = assigned.sort_values("FixStart_ms")
    obj_ids = assigned["ObjectID"].astype(int).tolist()

    # Collapse consecutive duplicates
    sequence = [obj_ids[0]]
    for obj_id in obj_ids[1:]:
        if obj_id != sequence[-1]:
            sequence.append(obj_id)

    return sequence


# ---------------------------------------------------------------------------
# Step 5: SVG alignment
# ---------------------------------------------------------------------------


def _count_relational_transitions(
    sequence: list[int],
    edges: set,
) -> int:
    """
    Count consecutive pairs (A→B) in sequence where frozenset({A,B}) ∈ edges.
    Self-transitions are already removed by build_object_sequence.
    """
    return sum(
        1 for a, b in zip(sequence[:-1], sequence[1:]) if frozenset({a, b}) in edges
    )


def svg_alignment(
    sequence: list[int],
    edges: set,
    n_permutations: int = None,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Compute SVG alignment z-score for one object sequence.

    The observed proportion of relational transitions is compared against a
    null distribution of the same metric computed on random permutations of
    the *same visited object set* (not all objects in the image). This
    controls for graph density and object selection simultaneously.

    Parameters
    ----------
    sequence : list of int
        Ordered object sequence (consecutive duplicates already collapsed).
    edges : set of frozenset
        Undirected relational edges for this image — from build_graph_index().
        Use "all" or "interactional" sub-index as appropriate.
    n_permutations : int, optional
        Number of random permutations for the null. Defaults to
        config.SVG_N_PERMUTATIONS.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a fresh
        default_rng() is used.

    Returns
    -------
    dict with keys:
        svg_z        : float  — z-score (NaN if std=0 or n_transitions < 1)
        svg_obs      : float  — observed relational proportion
        svg_null_mean: float  — mean of null distribution
        svg_null_std : float  — std of null distribution
        n_transitions: int    — total consecutive pairs in sequence
        n_relational : int    — relational pairs in observed sequence
        low_n        : bool   — True if n_transitions < MIN_VALID_TRANSITIONS

    Notes
    -----
    - Sequences of length ≤ 1 produce all-NaN output (no transitions possible).
    - Permutations are of the visited object set only, preserving sequence
      length. A→A transitions cannot appear (objects are unique in each
      permutation) but A→A was already excluded upstream.
    """
    n_permutations = n_permutations or config.SVG_N_PERMUTATIONS
    rng = rng or np.random.default_rng()

    n_transitions = max(len(sequence) - 1, 0)
    low_n = n_transitions < config.MIN_VALID_TRANSITIONS

    null_result = {
        "svg_z": np.nan,
        "svg_obs": np.nan,
        "svg_null_mean": np.nan,
        "svg_null_std": np.nan,
        "n_transitions": n_transitions,
        "n_relational": 0,
        "low_n": low_n,
    }

    if n_transitions < 1:
        return null_result

    n_relational = _count_relational_transitions(sequence, edges)
    svg_obs = n_relational / n_transitions

    null_result["svg_obs"] = svg_obs
    null_result["n_relational"] = n_relational

    # Permutation null: shuffle the visited object set
    visited = list(set(sequence))  # unique objects actually fixated

    if len(visited) < 2:
        # Only one unique object — all permutations identical, std=0
        null_result["svg_null_mean"] = svg_obs
        null_result["svg_null_std"] = 0.0
        return null_result

    null_scores = np.empty(n_permutations, dtype=np.float32)
    seq_len = len(sequence)

    for i in range(n_permutations):
        # Sample sequence of same length from visited objects (with replacement
        # to respect the original repeat structure, but never A→A consecutive)
        perm = _sample_permutation(visited, seq_len, rng)
        n_rel_perm = _count_relational_transitions(perm, edges)
        null_scores[i] = n_rel_perm / (seq_len - 1)

    null_mean = float(null_scores.mean())
    null_std = float(null_scores.std())

    svg_z = (svg_obs - null_mean) / null_std if null_std > 0 else np.nan

    return {
        "svg_z": svg_z,
        "svg_obs": svg_obs,
        "svg_null_mean": null_mean,
        "svg_null_std": null_std,
        "n_transitions": n_transitions,
        "n_relational": n_relational,
        "low_n": low_n,
    }


def compute_transition_salience(
    trial_fixations: "pd.DataFrame",
    edges: set,
) -> dict:
    """
    Compute mean salience at relational vs non-relational fixations.

    A fixation is classified as "relational" if it is part of any relational
    edge transition — i.e. either the next distinct object visited is
    relationally connected to the current object, or the previous distinct
    object visited was. This tags both sides of each edge traversal.

    Fixations without an AOI assignment (ObjectID NaN) are excluded.
    Fixations with NaN SalienceAtFixation are excluded from the mean but
    still used for transition classification.

    Parameters
    ----------
    trial_fixations : pd.DataFrame
        AOI-assigned fixations for one trial. Must contain:
        FixStart_ms, ObjectID, SalienceAtFixation.
    edges : set of frozenset
        Relational edges for this stimulus, from build_graph_index()["all"].

    Returns
    -------
    dict with keys:
        mean_salience_relational    : float — NaN if no relational fixations
        mean_salience_nonrelational : float — NaN if no non-relational fixations
        n_relational_fixations      : int
        n_nonrelational_fixations   : int
    """
    null = {
        "mean_salience_relational": np.nan,
        "mean_salience_nonrelational": np.nan,
        "n_relational_fixations": 0,
        "n_nonrelational_fixations": 0,
    }

    if "SalienceAtFixation" not in trial_fixations.columns:
        return null

    assigned = trial_fixations[trial_fixations["ObjectID"].notna()].copy()
    if len(assigned) < 2:
        return null

    assigned = assigned.sort_values("FixStart_ms").reset_index(drop=True)
    obj_ids = assigned["ObjectID"].astype(int).values
    salience = pd.to_numeric(assigned["SalienceAtFixation"], errors="coerce").values
    n = len(obj_ids)

    is_relational = np.zeros(n, dtype=bool)

    for i in range(n):
        current = obj_ids[i]

        # Check outgoing transition: next distinct object
        for j in range(i + 1, n):
            if obj_ids[j] != current:
                if frozenset({current, obj_ids[j]}) in edges:
                    is_relational[i] = True
                break

        # Check incoming transition: previous distinct object
        if not is_relational[i]:
            for j in range(i - 1, -1, -1):
                if obj_ids[j] != current:
                    if frozenset({current, obj_ids[j]}) in edges:
                        is_relational[i] = True
                    break

    sal_rel = salience[is_relational]
    sal_nonrel = salience[~is_relational]

    # Exclude NaN from means but preserve counts
    mean_rel = float(np.nanmean(sal_rel)) if np.any(~np.isnan(sal_rel)) else np.nan
    mean_nonrel = (
        float(np.nanmean(sal_nonrel)) if np.any(~np.isnan(sal_nonrel)) else np.nan
    )

    return {
        "mean_salience_relational": mean_rel,
        "mean_salience_nonrelational": mean_nonrel,
        "n_relational_fixations": int(is_relational.sum()),
        "n_nonrelational_fixations": int((~is_relational).sum()),
    }


def _sample_permutation(
    visited: list[int],
    length: int,
    rng: np.random.Generator,
) -> list[int]:
    """
    Sample a random sequence of `length` objects from `visited` such that
    no two consecutive elements are identical (mirrors the real sequence
    structure where self-transitions have been collapsed).

    Uses rejection sampling — fast in practice since visited sets are small
    (typically 5–20 objects) and collision probability is low.
    """
    if len(visited) == 1:
        return visited * length

    result = [rng.choice(visited)]
    for _ in range(length - 1):
        # Resample until we get a different object from the last one
        prev = result[-1]
        candidates = [v for v in visited if v != prev]
        result.append(rng.choice(candidates))

    return result
