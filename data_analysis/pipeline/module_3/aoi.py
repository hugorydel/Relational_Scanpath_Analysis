"""
module_3/aoi.py
============
Step 3 of Module 3: AOI assignment for fixations.

Takes the fixation CSV from Module 2, assigns each fixation to an object
polygon (or None), and appends saliency and meaning values. Writes an enriched
_fixations_aoi.csv per participant.

Pipeline
--------
1. Transform screen coordinates → image coordinates
2. Point-in-polygon test (ray-casting) against all polygons for that image
3. Proximity fallback if no polygon hit — assign to nearest centroid within
   AOI_PROXIMITY_THRESHOLD_ENCODING_PX (encoding) or
   AOI_PROXIMITY_THRESHOLD_DECODING_PX (decoding)
4. Sample saliency map and meaning map at fixation location (bilinear
   interpolation)
5. Write enriched fixations CSV

Output columns (appended to all Module 2 columns)
--------------------------------------------------
ImgX                : float  — fixation x in image space (0–1024)
ImgY                : float  — fixation y in image space (0–768)
ObjectID            : int or NaN — assigned object ID
ObjectName          : str or NaN — object label
AssignmentMethod    : "polygon" | "proximity" | "none"
ProximityDist_px    : float or NaN — centroid distance for proximity hits
SalienceAtFixation  : float — saliency map value at (ImgX, ImgY)
MeaningAtFixation   : float — meaning map value at (ImgX, ImgY); NaN if
                      meaning maps have not yet been computed

Usage (from Module 3):
    from pipeline.module_3.aoi import assign_aoi

    fixations_aoi = assign_aoi(
        fixations_df,
        polygon_index,
        graph_index,
        saliency_maps,
    )
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import config
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.module_3.scene_graph import build_polygon_index
from pipeline.salience.saliency import get_saliency_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------


def screen_to_image(gaze_x: float, gaze_y: float) -> tuple[float, float]:
    """
    Transform EyeLink screen coordinates to image pixel coordinates.

    Screen space : 1920 × 1080 px (full display)
    Image space  : 1024 × 768  px (stimulus image, fills full screen)

    Parameters
    ----------
    gaze_x, gaze_y : float  — screen coordinates in pixels

    Returns
    -------
    (img_x, img_y) : float  — image coordinates in pixels
    """
    img_x = gaze_x * (config.IMAGE_W / config.DISPLAY_WIDTH_PX)
    img_y = gaze_y * (config.IMAGE_H / config.DISPLAY_HEIGHT_PX)
    return img_x, img_y


# ---------------------------------------------------------------------------
# Point-in-polygon (ray-casting)
# ---------------------------------------------------------------------------


def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """
    Ray-casting test: is point (x, y) inside polygon?

    Works correctly with concave polygons. Handles edge cases via the
    standard half-open interval convention (y in [y_i, y_j)).

    Parameters
    ----------
    x, y    : float       — query point in image pixels
    polygon : np.ndarray  — (N, 2) array of (x, y) vertices

    Returns
    -------
    bool
    """
    n = len(polygon)
    inside = False
    px, py = polygon[:, 0], polygon[:, 1]

    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]

        # Check if edge (j→i) crosses the horizontal ray from (x, y) rightward
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def _centroid_distance(x: float, y: float, centroid: tuple) -> float:
    """Euclidean distance from (x, y) to centroid (cx, cy) in image pixels."""
    return float(np.sqrt((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2))


# ---------------------------------------------------------------------------
# Map sampling  (bilinear interpolation — works for saliency or meaning maps)
# ---------------------------------------------------------------------------


def _sample_map(
    img_x: float,
    img_y: float,
    map_2d: np.ndarray,
) -> float:
    """
    Bilinear interpolation of a 2-D float map at sub-pixel location (img_x, img_y).

    Used for both saliency maps and meaning maps.

    Parameters
    ----------
    img_x, img_y : float        — image coordinates (0–1024, 0–768)
    map_2d       : np.ndarray   — (IMAGE_H, IMAGE_W) float32 map

    Returns
    -------
    float — interpolated map value
    """
    h, w = map_2d.shape

    # Clamp to valid range
    x = float(np.clip(img_x, 0, w - 1))
    y = float(np.clip(img_y, 0, h - 1))

    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)

    dx, dy = x - x0, y - y0

    val = (
        map_2d[y0, x0] * (1 - dx) * (1 - dy)
        + map_2d[y0, x1] * dx * (1 - dy)
        + map_2d[y1, x0] * (1 - dx) * dy
        + map_2d[y1, x1] * dx * dy
    )
    return float(val)


# Keep the old name as an alias so any external code that imports it directly
# continues to work without modification.
_sample_saliency = _sample_map


# ---------------------------------------------------------------------------
# Single fixation assignment
# ---------------------------------------------------------------------------


def _assign_single_fixation(
    img_x: float,
    img_y: float,
    phase: str,
    poly_list: list,
) -> dict:
    """
    Assign one fixation to an AOI.

    Returns a dict with keys:
        ObjectID, ObjectName, AssignmentMethod, ProximityDist_px
    """
    # --- Stage 1: point-in-polygon ---
    hits = []
    for obj in poly_list:
        if _point_in_polygon(img_x, img_y, obj["polygon"]):
            hits.append(obj)

    if len(hits) == 1:
        obj = hits[0]
        return {
            "ObjectID": obj["object_id"],
            "ObjectName": obj["name"],
            "AssignmentMethod": "polygon",
            "ProximityDist_px": np.nan,
        }

    if len(hits) > 1:
        # Overlapping polygons — assign to nearest centroid
        obj = min(hits, key=lambda o: _centroid_distance(img_x, img_y, o["centroid"]))
        return {
            "ObjectID": obj["object_id"],
            "ObjectName": obj["name"],
            "AssignmentMethod": "polygon",
            "ProximityDist_px": np.nan,
        }

    # --- Stage 2: proximity fallback ---
    threshold = (
        config.AOI_PROXIMITY_THRESHOLD_ENCODING_PX
        if phase == "encoding"
        else config.AOI_PROXIMITY_THRESHOLD_DECODING_PX
    )

    if poly_list:
        nearest = min(
            poly_list, key=lambda o: _centroid_distance(img_x, img_y, o["centroid"])
        )
        dist = _centroid_distance(img_x, img_y, nearest["centroid"])

        if dist <= threshold:
            return {
                "ObjectID": nearest["object_id"],
                "ObjectName": nearest["name"],
                "AssignmentMethod": "proximity",
                "ProximityDist_px": round(dist, 2),
            }

    # --- Stage 3: no assignment ---
    return {
        "ObjectID": np.nan,
        "ObjectName": np.nan,
        "AssignmentMethod": "none",
        "ProximityDist_px": np.nan,
    }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def assign_aoi(
    fixations_df: pd.DataFrame,
    image_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    meaning_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Assign AOI labels, saliency values, and meaning values to all fixations.

    Parameters
    ----------
    fixations_df : pd.DataFrame
        Module 2 fixations output. Must contain columns:
        StimID, Phase, GazeX, GazeY.
    image_dir : Path, optional
        Directory of stimulus images (for saliency map loading).
    cache_dir : Path, optional
        Directory of cached saliency .npy files.
    meaning_dir : Path, optional
        Directory of cached meaning map .npy files. If None, defaults to
        config.OUTPUT_DIR / "meaning_maps". MeaningAtFixation is set to NaN
        for any stimulus whose meaning map file is absent (e.g. before
        pipeline.meaning.meaning has been run).

    Returns
    -------
    pd.DataFrame
        Original fixations_df with additional columns:
        ImgX, ImgY, ObjectID, ObjectName, AssignmentMethod,
        ProximityDist_px, SalienceAtFixation, MeaningAtFixation.
    """
    polygon_index = build_polygon_index()

    image_dir = Path(image_dir) if image_dir else config.DATA_METADATA_DIR / "images"
    cache_dir = Path(cache_dir) if cache_dir else config.OUTPUT_DIR / "saliency_maps"
    meaning_dir = (
        Path(meaning_dir) if meaning_dir else config.OUTPUT_DIR / "meaning_maps"
    )

    stim_ids = fixations_df["StimID"].unique()

    # --- pre-load saliency maps ---
    sal_maps = {}
    for stim_id in stim_ids:
        try:
            sal_maps[stim_id] = get_saliency_map(
                stim_id, image_dir=image_dir, cache_dir=cache_dir
            )
        except FileNotFoundError:
            logger.warning(f"  Saliency map not found for {stim_id} — will use NaN.")
            sal_maps[stim_id] = None

    # --- pre-load meaning maps (gracefully absent before meaning pipeline runs) ---
    try:
        from pipeline.meaning.meaning import get_meaning_map as _get_meaning_map

        _meaning_available = True
    except ImportError:
        logger.warning(
            "  open-clip-torch not installed — MeaningAtFixation will be NaN. "
            "Install with: pip install open-clip-torch"
        )
        _meaning_available = False

    meaning_maps = {}
    if _meaning_available:
        for stim_id in stim_ids:
            try:
                meaning_maps[stim_id] = _get_meaning_map(
                    stim_id, image_dir=image_dir, cache_dir=meaning_dir
                )
            except FileNotFoundError:
                logger.warning(
                    f"  Meaning map not found for {stim_id} — will use NaN. "
                    "Run `python -m pipeline.meaning.meaning` to pre-compute."
                )
                meaning_maps[stim_id] = None

    # --- build result rows ---
    results = []

    for _, row in fixations_df.iterrows():
        stim_id = str(row["StimID"])
        phase = str(row["Phase"])

        img_x, img_y = screen_to_image(float(row["GazeX"]), float(row["GazeY"]))

        poly_list = polygon_index.get(stim_id, [])
        assignment = _assign_single_fixation(img_x, img_y, phase, poly_list)

        # Saliency
        sal_map = sal_maps.get(stim_id)
        salience = _sample_map(img_x, img_y, sal_map) if sal_map is not None else np.nan

        # Meaning
        meaning_map = meaning_maps.get(stim_id) if _meaning_available else None
        meaning = (
            _sample_map(img_x, img_y, meaning_map)
            if meaning_map is not None
            else np.nan
        )

        results.append(
            {
                "ImgX": round(img_x, 2),
                "ImgY": round(img_y, 2),
                "ObjectID": assignment["ObjectID"],
                "ObjectName": assignment["ObjectName"],
                "AssignmentMethod": assignment["AssignmentMethod"],
                "ProximityDist_px": assignment["ProximityDist_px"],
                "SalienceAtFixation": (
                    round(salience, 6) if not np.isnan(salience) else np.nan
                ),
                "MeaningAtFixation": (
                    round(meaning, 6) if not np.isnan(meaning) else np.nan
                ),
            }
        )

    result_df = pd.DataFrame(results, index=fixations_df.index)
    enriched = pd.concat([fixations_df, result_df], axis=1)

    # --- log assignment summary ---
    n_total = len(enriched)
    n_polygon = (enriched["AssignmentMethod"] == "polygon").sum()
    n_proximity = (enriched["AssignmentMethod"] == "proximity").sum()
    n_none = (enriched["AssignmentMethod"] == "none").sum()
    n_meaning_nan = enriched["MeaningAtFixation"].isna().sum()
    logger.info(
        f"  AOI assignment: {n_polygon} polygon ({100*n_polygon/n_total:.1f}%), "
        f"{n_proximity} proximity ({100*n_proximity/n_total:.1f}%), "
        f"{n_none} none ({100*n_none/n_total:.1f}%)  [n={n_total}]"
    )
    if n_meaning_nan > 0:
        logger.info(
            f"  MeaningAtFixation: {n_meaning_nan}/{n_total} NaN "
            "(meaning maps not yet computed for some stimuli)"
        )

    return enriched


# ---------------------------------------------------------------------------
# Participant-level runner (called from module3_features.py)
# ---------------------------------------------------------------------------


def run_aoi_assignment(
    subject_id: str,
    fixations_path: Path,
    output_path: Path,
    image_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    meaning_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load fixations for one participant, assign AOIs, write enriched CSV.

    Parameters
    ----------
    subject_id    : str
    fixations_path: Path  — {SubjectID}_fixations.csv from Module 2
    output_path   : Path  — where to write {SubjectID}_fixations_aoi.csv
    image_dir     : Path, optional
    cache_dir     : Path, optional   — saliency map cache directory
    meaning_dir   : Path, optional   — meaning map cache directory

    Returns
    -------
    pd.DataFrame — enriched fixations (also written to output_path)
    """
    logger.info(f"[{subject_id}] AOI assignment ...")

    if not fixations_path.exists():
        raise FileNotFoundError(f"Fixations file not found: {fixations_path}")

    fixations_df = pd.read_csv(fixations_path, dtype={"StimID": str})
    logger.info(f"  Loaded {len(fixations_df)} fixations.")

    enriched = assign_aoi(
        fixations_df,
        image_dir=image_dir,
        cache_dir=cache_dir,
        meaning_dir=meaning_dir,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    logger.info(f"  Written → {output_path.name}")

    return enriched
