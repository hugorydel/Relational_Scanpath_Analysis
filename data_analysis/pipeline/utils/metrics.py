"""
utils/metrics.py
================
Steps 4–6 of Module 3: sequence construction and scanpath metrics.

Step 4 — build_object_sequence()
    Converts AOI-assigned fixations for one trial into an ordered object
    sequence, collapsing consecutive self-transitions.

Step 5 — svg_alignment()
    Computes the SVG alignment z-score for one object sequence against the
    scene's relational graph, using a permutation null distribution.

Step 6 — symbolic_lcs() and kendall_tau_shared()
    Computes normalised LCS and Kendall's tau between two object sequences
    (e.g. encoding vs decoding) as replay-fidelity metrics.

All functions operate on plain Python lists / numpy arrays and have no
side effects — they can be called independently for testing.
"""

import logging
from typing import Optional

import config
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

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


# ---------------------------------------------------------------------------
# Step 6: Symbolic LCS and Kendall's tau
# ---------------------------------------------------------------------------


def _first_occurrence_sequence(sequence: list[int]) -> list[int]:
    """
    Compress sequence to first-occurrence-only ordering.

    Each object appears at most once, at its first position in the input.
    This gives a clean ordinal ranking for Kendall's tau and avoids
    repeat-distortion in LCS.

    Example: [A, B, A, C, B] → [A, B, C]
    """
    seen = set()
    result = []
    for obj in sequence:
        if obj not in seen:
            seen.add(obj)
            result.append(obj)
    return result


def _lcs_length(seq_a: list, seq_b: list) -> int:
    """
    Standard dynamic programming LCS length computation.
    O(m × n) time, O(min(m,n)) space.
    """
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a  # ensure seq_b is shorter for space efficiency

    m, n = len(seq_a), len(seq_b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def symbolic_lcs(
    seq_a: list[int],
    seq_b: list[int],
) -> dict:
    """
    Compute normalised LCS between two object sequences.

    Both sequences are first compressed to first-occurrence-only before
    comparison, removing repeat-distortion.

    Parameters
    ----------
    seq_a, seq_b : list of int
        Object sequences (e.g. encoding and decoding for the same image).

    Returns
    -------
    dict with keys:
        lcs_score : float — LCS length / min(len(seq_a), len(seq_b))
                            NaN if either sequence is empty.
        lcs_length: int   — raw LCS length
        len_a     : int   — length of compressed seq_a
        len_b     : int   — length of compressed seq_b
    """
    fa = _first_occurrence_sequence(seq_a)
    fb = _first_occurrence_sequence(seq_b)

    if not fa or not fb:
        return {
            "lcs_score": np.nan,
            "lcs_length": 0,
            "len_a": len(fa),
            "len_b": len(fb),
        }

    lcs_len = _lcs_length(fa, fb)
    lcs_score = lcs_len / min(len(fa), len(fb))

    return {
        "lcs_score": lcs_score,
        "lcs_length": lcs_len,
        "len_a": len(fa),
        "len_b": len(fb),
    }


def kendall_tau_shared(
    seq_a: list[int],
    seq_b: list[int],
) -> dict:
    """
    Compute Kendall's tau on the shared object subset.

    Objects are ranked by their first occurrence position in each sequence.
    Only objects appearing in both sequences are included. This is the
    discrete analogue of Johansson et al.'s MultiMatch replay metric.

    Parameters
    ----------
    seq_a, seq_b : list of int

    Returns
    -------
    dict with keys:
        tau        : float — Kendall's tau (-1 to 1), NaN if < 2 shared objects
        tau_p      : float — p-value
        n_shared   : int   — number of objects in both sequences
    """
    fa = _first_occurrence_sequence(seq_a)
    fb = _first_occurrence_sequence(seq_b)

    shared = [obj for obj in fa if obj in set(fb)]

    if len(shared) < 2:
        return {"tau": np.nan, "tau_p": np.nan, "n_shared": len(shared)}

    # Rank by first-occurrence position in each compressed sequence
    rank_a = {obj: i for i, obj in enumerate(fa)}
    rank_b = {obj: i for i, obj in enumerate(fb)}

    ranks_a = [rank_a[obj] for obj in shared]
    ranks_b = [rank_b[obj] for obj in shared]

    tau, p = kendalltau(ranks_a, ranks_b)

    return {
        "tau": float(tau),
        "tau_p": float(p),
        "n_shared": len(shared),
    }
