#!/usr/bin/env python3
"""
build_predicate_map.py

Utility script to build a predicate→category mapping from the
SVG-Relations JSONL file and write it to ``data/predicate_to_category.json``.

This mapping is exactly what ``SceneGraphLoader`` in your pipeline expects:
    - keys: predicate strings as they appear in SVG
      (e.g., "wearing", "on", "worn by")
    - values: one of SVG's coarse categories
      ("spatial", "interactional", "functional", "social", "emotional", ...)

Run once with:
    python build_predicate_map.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

from huggingface_hub import hf_hub_download
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------


def update_predicate_counts(
    relations_category: dict,
    pred_counts: Dict[str, Counter],
) -> Tuple[int, int]:
    """
    Update ``pred_counts`` with predicates from one JSONL entry.

    Parameters
    ----------
    relations_category : dict
        The value of the "relations_category" field for one JSONL line.
        Structure (simplified):

        {
          "25152": {          # image id
            "5": {            # subject object id
              "relations": {
                "spatial":      [["15", "above"], ...],
                "interactional":[["11", "wearing"], ...],
                ...
              },
              ...
            },
            ...
          },
          "25153": { ... },
          ...
        }

    pred_counts : dict
        Maps predicate string → Counter of category → count.

    Returns
    -------
    total_relations : int
        Number of relation instances seen in this call.
    unique_predicates : int
        Number of distinct predicate strings touched in this call.
    """
    total_relations = 0
    seen_here = set()

    if not isinstance(relations_category, dict):
        return 0, 0

    # iterate over images
    for img_id, subj_map in relations_category.items():
        if not isinstance(subj_map, dict):
            continue

        # iterate over subject objects
        for subj_id, subj_info in subj_map.items():
            if not isinstance(subj_info, dict):
                continue

            relations = subj_info.get("relations", {})
            if not isinstance(relations, dict):
                continue

            # iterate over categories (spatial, interactional, functional, ...)
            for cat, rel_list in relations.items():
                if not isinstance(rel_list, list):
                    continue

                for rel in rel_list:
                    # rel is typically [object_id, predicate]
                    if not isinstance(rel, (list, tuple)) or len(rel) < 2:
                        continue
                    predicate = rel[1]
                    if not isinstance(predicate, str):
                        continue

                    p = predicate.strip()
                    if not p:
                        continue

                    pred_counts[p][cat] += 1
                    total_relations += 1
                    seen_here.add(p)

    return total_relations, len(seen_here)


# ---------------------------------------------------------------------------
# Mapping construction
# ---------------------------------------------------------------------------


def build_predicate_category_map(min_dominance: float = 0.7) -> Dict[str, str]:
    """
    Build predicate→category mapping from SVG-Relations JSONL.

    Parameters
    ----------
    min_dominance : float
        Minimum fraction of occurrences a category must have for a predicate
        to be assigned *exclusively* to that category.
        Example: if "wearing" is 90% interactional and 10% functional,
        it will be labeled "interactional" when min_dominance <= 0.9.
        Predicates that don't reach this threshold are labeled "ambiguous".

    Returns
    -------
    mapping : dict
        Predicate string → category label.
    """

    print("Downloading SVG-Relations file from HuggingFace (if not cached)...")
    jsonl_path = hf_hub_download(
        repo_id="jamepark3922/svg",
        repo_type="dataset",
        filename=(
            "relations/"
            "train_coco_relation_category_interaction_sam_seem_regions_150_"
            "verified_qwen_llava_rule.jsonl"
        ),
    )
    print(f"✓ JSONL path: {jsonl_path}")

    pred_counts: Dict[str, Counter] = defaultdict(Counter)
    total_relations = 0
    total_lines = 0

    print("\nScanning predicates and categories (this may take a few minutes)...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Lines", mininterval=1.0):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            rel_cat = entry.get("relations_category")
            if rel_cat is None:
                continue

            tr, _ = update_predicate_counts(rel_cat, pred_counts)
            total_relations += tr
            total_lines += 1

    print("\nSummary of raw counts:")
    print(f"  Lines processed:      {total_lines}")
    print(f"  Total relations seen: {total_relations}")
    print(f"  Unique predicates:    {len(pred_counts)}")

    # Decide final category per predicate
    mapping: Dict[str, str] = {}

    for predicate, counts in pred_counts.items():
        total = sum(counts.values())
        if not total:
            continue

        cat, freq = counts.most_common(1)[0]
        frac = freq / total

        if frac >= min_dominance:
            mapping[predicate] = cat
        else:
            mapping[predicate] = "ambiguous"

    # Sort predicates by total frequency for display
    pred_frequencies = [
        (pred, sum(counts.values())) for pred, counts in pred_counts.items()
    ]
    pred_frequencies.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 25 most common predicates:")
    total_all_relations = sum(freq for _, freq in pred_frequencies)
    for i, (pred, freq) in enumerate(pred_frequencies[:25], 1):
        category = mapping.get(pred, "unknown")
        pct = 100 * freq / total_all_relations

        # Show category distribution if multi-category
        counts = pred_counts[pred]
        if len(counts) > 1:
            cat_dist = ", ".join([f"{cat}:{cnt}" for cat, cnt in counts.most_common()])
            print(
                f"  {i:2d}. {pred!r:20s} → {category:15s} | {freq:7,d} uses ({pct:5.2f}%) | [{cat_dist}]"
            )
        else:
            print(
                f"  {i:2d}. {pred!r:20s} → {category:15s} | {freq:7,d} uses ({pct:5.2f}%)"
            )

    if len(pred_frequencies) > 25:
        print("  ...")

    return mapping


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Build the mapping and save it under data/predicate_to_category.json."""

    mapping = build_predicate_category_map(min_dominance=0.7)

    if not mapping:
        print("\n❌ No predicates were mapped – aborting.")
        return

    out_path = Path("data") / "predicate_to_category.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"✓ Saved predicate→category mapping to: {out_path.resolve()}")
    print(f"✓ Total predicates in mapping: {len(mapping)}")
    print("=" * 60)

    # Quick sanity check for a few common predicates
    for p in ["on", "above", "wearing", "holding", "riding", "eating", "kissing"]:
        print(f"  {p!r} → {mapping.get(p, 'NOT FOUND')}")
    print("(If these look sensible, you’re good to go.)")


if __name__ == "__main__":
    main()
