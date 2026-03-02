"""
inspect_stimuli.py
==================
Diagnostic script to understand the structure of stimuli_dataset.json
before writing Module 3.

Prints:
  1. Top-level structure (keys, number of entries)
  2. Full first entry with long lists truncated to [first, second, ..., last]
  3. Coordinate range analysis across all polygons
  4. Relation type inventory
  5. Object count distribution

Usage:
    python inspect_stimuli.py --metadata data_metadata/stimuli_dataset.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def truncate_list(lst, head=2, tail=1):
    """Replace middle elements of a long list with '...' for readability."""
    if len(lst) <= head + tail + 1:
        return lst
    return lst[:head] + ["..."] + lst[-tail:]


def truncate_value(val, depth=0):
    """Recursively truncate long lists within nested structures."""
    if isinstance(val, list):
        truncated = truncate_list(val) if depth >= 1 else val
        return [truncate_value(v, depth + 1) for v in truncated]
    elif isinstance(val, dict):
        return {k: truncate_value(v, depth + 1) for k, v in val.items()}
    return val


def analyse_polygons(entries):
    all_x, all_y = [], []
    poly_lengths = []
    poly_key_found = None

    for entry in entries:
        for key in ["polygons", "polygon", "segmentation", "regions", "objects"]:
            if key in entry:
                poly_key_found = key
                polys = entry[key]
                if isinstance(polys, list):
                    for poly in polys:
                        coords = None
                        if isinstance(poly, dict):
                            for sub_key in [
                                "polygon",
                                "segmentation",
                                "points",
                                "coords",
                            ]:
                                if sub_key in poly:
                                    coords = poly[sub_key]
                                    break
                        elif isinstance(poly, list):
                            coords = poly

                        if coords:
                            flat = []
                            for c in coords:
                                if isinstance(c, (list, tuple)):
                                    flat.extend(c)
                                elif isinstance(c, (int, float)):
                                    flat.append(c)
                            xs = flat[0::2]
                            ys = flat[1::2]
                            all_x.extend(xs)
                            all_y.extend(ys)
                            poly_lengths.append(len(xs))
                break

    result = {"polygon_key": poly_key_found}
    if all_x:
        result["x_range"] = [min(all_x), max(all_x)]
        result["y_range"] = [min(all_y), max(all_y)]
        result["n_polygons"] = len(poly_lengths)
        result["points_per_polygon"] = {
            "min": min(poly_lengths),
            "max": max(poly_lengths),
            "mean": round(sum(poly_lengths) / len(poly_lengths), 1),
        }
    return result


def analyse_relations(entries):
    rel_key_found = None
    predicate_counts = Counter()
    rel_counts = []

    for entry in entries:
        for key in ["relations", "relationships", "relation"]:
            if key in entry:
                rel_key_found = key
                rels = entry[key]
                if isinstance(rels, list):
                    rel_counts.append(len(rels))
                    for rel in rels:
                        if isinstance(rel, dict):
                            for pred_key in ["predicate", "relation", "type", "label"]:
                                if pred_key in rel:
                                    predicate_counts[rel[pred_key]] += 1
                                    break
                break

    return {
        "relation_key": rel_key_found,
        "predicate_counts": dict(predicate_counts.most_common(30)),
        "relations_per_entry": {
            "min": min(rel_counts) if rel_counts else 0,
            "max": max(rel_counts) if rel_counts else 0,
            "mean": round(sum(rel_counts) / len(rel_counts), 1) if rel_counts else 0,
        },
    }


def analyse_objects(entries):
    obj_key_found = None
    obj_counts = []
    obj_names = Counter()

    for entry in entries:
        for key in ["objects", "regions", "annotations"]:
            if key in entry:
                obj_key_found = key
                objs = entry[key]
                if isinstance(objs, list):
                    obj_counts.append(len(objs))
                    for obj in objs:
                        if isinstance(obj, dict):
                            for name_key in ["name", "label", "category", "class"]:
                                if name_key in obj:
                                    obj_names[obj[name_key]] += 1
                                    break
                break

    return {
        "object_key": obj_key_found,
        "objects_per_entry": {
            "min": min(obj_counts) if obj_counts else 0,
            "max": max(obj_counts) if obj_counts else 0,
            "mean": round(sum(obj_counts) / len(obj_counts), 1) if obj_counts else 0,
        },
        "top_object_names": dict(obj_names.most_common(20)),
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect stimuli_dataset.json")
    parser.add_argument(
        "--metadata",
        default="data_metadata/stimuli_dataset.json",
        help="Path to stimuli_dataset.json",
    )
    args = parser.parse_args()

    path = Path(args.metadata)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 70)
    print("STIMULI DATASET INSPECTION")
    print("=" * 70)

    print(f"\n[TOP-LEVEL TYPE]: {type(data).__name__}")

    if isinstance(data, dict):
        print(f"[TOP-LEVEL KEYS]: {list(data.keys())}")
        entries = None
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0:
                entries = val
                print(f"[ENTRIES KEY]: '{key}'  ({len(val)} entries)")
                break
        if entries is None:
            entries = [data]
    elif isinstance(data, list):
        entries = data
        print(f"[ENTRIES]: list of {len(data)} items")
    else:
        print(f"Unexpected top-level type: {type(data)}")
        return

    if entries:
        sample_entry = entries[0]
        print(
            f"\n[ENTRY KEYS]: {list(sample_entry.keys()) if isinstance(sample_entry, dict) else 'not a dict'}"
        )

    # First entry (truncated)
    print("\n" + "=" * 70)
    print("FIRST ENTRY (long lists truncated to [first, second, ..., last])")
    print("=" * 70)
    print(json.dumps(truncate_value(entries[0], depth=0), indent=2, default=str))

    # Second entry (truncated)
    if len(entries) > 1:
        print("\n" + "=" * 70)
        print("SECOND ENTRY (truncated)")
        print("=" * 70)
        print(json.dumps(truncate_value(entries[1], depth=0), indent=2, default=str))

    # Polygon analysis
    print("\n" + "=" * 70)
    print("POLYGON / COORDINATE ANALYSIS (across all entries)")
    print("=" * 70)
    print(json.dumps(analyse_polygons(entries), indent=2))

    # Relation analysis
    print("\n" + "=" * 70)
    print("RELATION ANALYSIS")
    print("=" * 70)
    print(json.dumps(analyse_relations(entries), indent=2))

    # Object analysis
    print("\n" + "=" * 70)
    print("OBJECT ANALYSIS")
    print("=" * 70)
    print(json.dumps(analyse_objects(entries), indent=2))

    # Per-entry summary
    print("\n" + "=" * 70)
    print("PER-ENTRY SUMMARY (image_id, n_objects, n_relations)")
    print("=" * 70)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        img_id = (
            entry.get("image_id")
            or entry.get("id")
            or entry.get("filename")
            or entry.get("coco_id")
            or "?"
        )
        n_obj = len(
            entry.get("objects", entry.get("regions", entry.get("annotations", [])))
        )
        n_rel = len(entry.get("relations", entry.get("relationships", [])))
        print(f"  {str(img_id):>12}  objects={n_obj:3d}  relations={n_rel:3d}")


if __name__ == "__main__":
    main()
