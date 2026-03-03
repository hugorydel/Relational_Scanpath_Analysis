"""
inspect_aoi_coverage.py
=======================
Diagnostic: what proportion of fixations land inside at least one AOI polygon?

Applies the screen → image coordinate transform:
    img_x = gaze_x * (1024 / 1920)
    img_y = gaze_y * (768  / 1080)

Then tests each fixation against all AOI polygons for that image using
ray-casting point-in-polygon. Reports:
  1. Overall miss rate (encoding vs decoding)
  2. Distribution of miss distances to nearest polygon centroid
  3. Per-image miss rates

Usage:
    python inspect_aoi_coverage.py \\
        --fixations output/eyetracking/Encode-Decode_Experiment-1-1_fixations.csv \\
        --metadata  data_metadata/stimuli_dataset.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import config
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------


def screen_to_image(gaze_x, gaze_y):
    return (
        gaze_x * (config.IMAGE_W / config.DISPLAY_WIDTH_PX),
        gaze_y * (config.IMAGE_H / config.DISPLAY_HEIGHT_PX),
    )


# ---------------------------------------------------------------------------
# Polygon utilities
# ---------------------------------------------------------------------------


def parse_polygon(flat_coords):
    """
    Convert a flat [x0, y0, x1, y1, ...] list to an (N, 2) numpy array.
    Filters out any non-numeric values (e.g. '...' from truncation).
    """
    nums = [c for c in flat_coords if isinstance(c, (int, float))]
    if len(nums) < 6:
        return None
    xs = nums[0::2]
    ys = nums[1::2]
    n = min(len(xs), len(ys))
    return np.array(list(zip(xs[:n], ys[:n])), dtype=float)


def point_in_polygon(px, py, polygon):
    """
    Ray-casting algorithm. Returns True if (px, py) is inside the polygon.
    polygon: (N, 2) numpy array of vertices.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def polygon_centroid(polygon):
    """Simple centroid of polygon vertices."""
    return polygon[:, 0].mean(), polygon[:, 1].mean()


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ---------------------------------------------------------------------------
# Load and index metadata
# ---------------------------------------------------------------------------


def load_metadata(metadata_path):
    """
    Returns a dict: image_id (str) -> list of parsed polygon arrays.
    Only includes objects that survived the >1% size filter
    (we use the actual objects array, not n_objects).
    """
    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("images", data) if isinstance(data, dict) else data
    index = {}

    for entry in entries:
        img_id = str(entry["image_id"])
        polygons = []
        for obj in entry.get("objects", []):
            poly = parse_polygon(obj.get("polygon", []))
            if poly is not None:
                polygons.append(poly)
        index[img_id] = polygons

    return index


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------


def run_diagnostic(fixations_path, metadata_path):
    fixations = pd.read_csv(fixations_path, dtype={"StimID": str})
    poly_index = load_metadata(metadata_path)

    print(f"Fixations loaded : {len(fixations)}")
    print(f"Images in metadata: {len(poly_index)}")
    print(f"Unique StimIDs in fixations: {fixations['StimID'].nunique()}")

    # Results collectors
    results = []

    for _, row in fixations.iterrows():
        stim_id = str(row["StimID"])
        gx, gy = row["GazeX"], row["GazeY"]
        phase = row["Phase"]
        trial_index = row["TrialIndex"]

        # Transform to image space
        ix, iy = screen_to_image(gx, gy)

        polygons = poly_index.get(stim_id, [])

        # Point-in-polygon test
        hit = False
        for poly in polygons:
            if point_in_polygon(ix, iy, poly):
                hit = True
                break

        # If miss: distance to nearest centroid
        nearest_dist = None
        if not hit and polygons:
            centroids = [polygon_centroid(p) for p in polygons]
            nearest_dist = min(distance(ix, iy, cx, cy) for cx, cy in centroids)

        results.append(
            {
                "StimID": stim_id,
                "Phase": phase,
                "TrialIndex": trial_index,
                "img_x": round(ix, 2),
                "img_y": round(iy, 2),
                "hit": hit,
                "nearest_dist": nearest_dist,
            }
        )

    df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # 1. Overall miss rate by phase
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. OVERALL MISS RATE BY PHASE")
    print("=" * 60)
    for phase in ["encoding", "decoding", "all"]:
        subset = df if phase == "all" else df[df["Phase"] == phase]
        n_total = len(subset)
        n_hit = subset["hit"].sum()
        n_miss = n_total - n_hit
        pct_hit = 100 * n_hit / n_total if n_total else 0
        pct_miss = 100 * n_miss / n_total if n_total else 0
        print(
            f"  {phase:10s}: {n_total:5d} fixations | "
            f"hit={n_hit} ({pct_hit:.1f}%) | "
            f"miss={n_miss} ({pct_miss:.1f}%)"
        )

    # ------------------------------------------------------------------
    # 2. Miss distance distribution
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. MISS DISTANCE TO NEAREST CENTROID (image pixels)")
    print("=" * 60)
    misses = df[~df["hit"] & df["nearest_dist"].notna()]

    if len(misses) == 0:
        print("  No misses — nothing to report.")
    else:
        dists = misses["nearest_dist"]
        print(f"  Total misses with distance: {len(dists)}")
        print(f"  Min    : {dists.min():.1f} px")
        print(f"  Median : {dists.median():.1f} px")
        print(f"  Mean   : {dists.mean():.1f} px")
        print(f"  Max    : {dists.max():.1f} px")
        print(f"  Std    : {dists.std():.1f} px")

        # Histogram buckets
        thresholds = [25, 50, 75, 100, 150, 200, float("inf")]
        labels = ["<25", "25-50", "50-75", "75-100", "100-150", "150-200", ">200"]
        prev = 0
        print(f"\n  Distance histogram:")
        for label, thresh in zip(labels, thresholds):
            count = ((dists >= prev) & (dists < thresh)).sum()
            pct = 100 * count / len(dists)
            print(f"    {label:>10} px : {count:4d}  ({pct:.1f}%)")
            prev = thresh

        # By phase
        print(f"\n  Miss distances by phase:")
        for phase in ["encoding", "decoding"]:
            subset = misses[misses["Phase"] == phase]["nearest_dist"]
            if len(subset) > 0:
                print(
                    f"    {phase:10s}: median={subset.median():.1f}  "
                    f"mean={subset.mean():.1f}  max={subset.max():.1f}  n={len(subset)}"
                )

    # ------------------------------------------------------------------
    # 3. Per-image miss rates
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. PER-IMAGE MISS RATES")
    print("=" * 60)
    per_image = (
        df.groupby("StimID")
        .agg(
            n_total=("hit", "count"),
            n_hit=("hit", "sum"),
            n_polygons=("StimID", lambda x: len(poly_index.get(str(x.iloc[0]), []))),
        )
        .assign(
            miss_rate=lambda d: 1 - d["n_hit"] / d["n_total"],
            pct_miss=lambda d: (100 * (1 - d["n_hit"] / d["n_total"])).round(1),
        )
        .sort_values("miss_rate", ascending=False)
    )
    print(
        f"  {'StimID':>12}  {'n_fix':>6}  {'n_hit':>6}  {'pct_miss':>9}  {'n_polygons':>10}"
    )
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*10}")
    for stim_id, row in per_image.iterrows():
        print(
            f"  {str(stim_id):>12}  {int(row['n_total']):>6}  "
            f"{int(row['n_hit']):>6}  {row['pct_miss']:>8.1f}%  "
            f"{int(row['n_polygons']):>10}"
        )

    # ------------------------------------------------------------------
    # 4. Sanity check: fixation coordinate ranges
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. TRANSFORMED COORDINATE RANGES (sanity check)")
    print("=" * 60)
    print(
        f"  img_x : [{df['img_x'].min():.1f}, {df['img_x'].max():.1f}]  "
        f"(expected 0–{config.IMAGE_W})"
    )
    print(
        f"  img_y : [{df['img_y'].min():.1f}, {df['img_y'].max():.1f}]  "
        f"(expected 0–{config.IMAGE_H})"
    )
    out_of_bounds = df[
        (df["img_x"] < 0)
        | (df["img_x"] > config.IMAGE_W)
        | (df["img_y"] < 0)
        | (df["img_y"] > config.IMAGE_H)
    ]
    print(
        f"  Out-of-bounds fixations: {len(out_of_bounds)} "
        f"({100*len(out_of_bounds)/len(df):.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser(description="AOI Coverage Diagnostic")
    parser.add_argument(
        "--fixations", required=True, help="Path to _fixations.csv from Module 2"
    )
    parser.add_argument(
        "--metadata", required=True, help="Path to stimuli_dataset.json"
    )
    args = parser.parse_args()

    run_diagnostic(
        fixations_path=Path(args.fixations),
        metadata_path=Path(args.metadata),
    )


if __name__ == "__main__":
    main()
