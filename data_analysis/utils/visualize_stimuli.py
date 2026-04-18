"""
visualize_stimuli.py
====================
Generates Figure 1 building-block images for all 30 stimuli.

For each stimulus, creates a subfolder under:
    data_analysis/output/visualized_stimuli/{stim_id}/

containing three files:
    raw_image.jpg                — original letterboxed stimulus image
    segmentations_labeled.png    — polygon overlays with object name labels
    relational_graph_overlay.png — centroid nodes + relation edges on image

Data sources (all via existing pipeline infrastructure):
    Images   : config.DATA_METADATA_IMAGES_DIR / {stim_id}.jpg  (or .png)
    Polygons : pipeline/module_3/scene_graph.build_polygon_index()
    Edges    : pipeline/module_3/scene_graph.build_graph_index()["all"]

Usage
-----
    python visualize_stimuli.py
    python visualize_stimuli.py --stim 2383555        # single stim
    python visualize_stimuli.py --output-dir path/to/out
    python visualize_stimuli.py --dpi 300             # default 150
    python visualize_stimuli.py --force               # reprocess existing
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import config
from pipeline.module_3.scene_graph import build_graph_index, build_polygon_index

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
)
logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


# ---------------------------------------------------------------------------
# Colour palette — deterministic, visually distinct
# ---------------------------------------------------------------------------


def _make_palette(n: int) -> np.ndarray:
    """Return (n, 3) RGB float array with maximally distinct hues."""
    import colorsys

    colours = []
    for i in range(max(n, 1)):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.78, 0.90)
        colours.append((r, g, b))
    return np.array(colours)


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------


def _load_image(stim_id: str) -> np.ndarray:
    """Load stimulus image as (H, W, 3) uint8 RGB array."""
    import cv2

    for ext in _IMAGE_EXTENSIONS:
        path = config.DATA_METADATA_IMAGES_DIR / f"{stim_id}{ext}"
        if path.exists():
            img_bgr = cv2.imread(str(path))
            if img_bgr is not None:
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    raise FileNotFoundError(
        f"No image found for stim_id={stim_id} " f"in {config.DATA_METADATA_IMAGES_DIR}"
    )


# ---------------------------------------------------------------------------
# Panel 1: raw image
# ---------------------------------------------------------------------------


def _save_raw(stim_id: str, out_dir: Path) -> None:
    """Copy original image file directly — no re-rendering overhead."""
    for ext in _IMAGE_EXTENSIONS:
        src = config.DATA_METADATA_IMAGES_DIR / f"{stim_id}{ext}"
        if src.exists():
            shutil.copy2(src, out_dir / f"raw_image{src.suffix}")
            return
    logger.warning(f"  [{stim_id}] raw image not found — skipping raw panel")


# ---------------------------------------------------------------------------
# Label anchor: pole of inaccessibility
# ---------------------------------------------------------------------------


def _label_anchor(poly: np.ndarray, img_h: int, img_w: int) -> tuple[float, float]:
    """
    Return (x, y) image coordinates for the label anchor of a polygon.

    Uses the pole of inaccessibility — the point inside the polygon that
    is furthest from all edges — computed via distance transform on a
    rasterised mask.  This guarantees the label sits deep inside the
    actual filled region regardless of polygon concavity.

    Falls back to the centroid if the mask is degenerate (< 4 px area).
    """
    import cv2

    pts = poly.astype(np.int32).reshape((-1, 1, 2))
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)

    if mask.sum() < 4:
        return float(poly[:, 0].mean()), float(poly[:, 1].mean())

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    return float(max_loc[0]), float(max_loc[1])


# ---------------------------------------------------------------------------
# Label text: truncate to 1-3 meaningful words
# ---------------------------------------------------------------------------

# Words that carry no useful information when they appear at the end of a label
_FUNCTION_WORDS = {
    "a",
    "an",
    "the",
    "in",
    "on",
    "at",
    "by",
    "with",
    "behind",
    "above",
    "below",
    "near",
    "over",
    "under",
    "through",
    "between",
    "among",
    "from",
    "to",
    "of",
    "for",
    "about",
    "around",
    "into",
    "onto",
    "upon",
    "within",
    "and",
    "or",
    "but",
    "it",
    "its",
    "them",
    "they",
    "he",
    "she",
    "his",
    "her",
    "their",
    "is",
    "are",
    "was",
    "were",
    "being",
    "been",
    "this",
    "that",
    "these",
    "those",
    "where",
    "which",
    "who",
}


def _short_label(name: str, max_words: int = 5, min_words: int = 2) -> str:
    """
    Return a 2–5 word informative label from a full object name.

    1. Strip trailing function/preposition/pronoun words.
    2. Keep between min_words and max_words words.
    3. Strip trailing function words again after capping.
    4. Append "..." if anything was dropped.
    """
    name = name.rstrip(".").strip()
    if not name:
        return ""

    words = name.split()

    # Strip trailing function words before capping
    while len(words) > min_words and words[-1].lower() in _FUNCTION_WORDS:
        words = words[:-1]

    truncated = len(words) > max_words
    words = words[:max_words]

    # Strip trailing function words again after capping
    while len(words) > min_words and words[-1].lower() in _FUNCTION_WORDS:
        words = words[:-1]
        truncated = True

    return " ".join(words) + ("..." if truncated else "")


def _font_size_for_label(label: str) -> float:
    """
    Return font size in points based on label word count.
    Shorter labels get larger text; 4-5 word labels are smaller
    so they fit without pushing into other labels.
    """
    n_words = len(label.rstrip("...").split())
    if n_words <= 3:
        return 6.5
    return 5.5  # 4-5 words: slightly smaller, allowed to overflow polygon edge


# ---------------------------------------------------------------------------
# Collision detection and label placement
# ---------------------------------------------------------------------------

# Empirical estimates of label box dimensions in image-pixel space.
# Calibrated for fig_w=10in, img_w=1024px, dpi=150.
_CHAR_W_PX = 4.8  # image pixels per character at font size 6.5pt
_LINE_H_PX = 13.0  # image pixels per line at font size 6.5pt
_BOX_PAD = 5.0  # extra padding (image px) on each side


def _label_box(
    cx: float, cy: float, text: str, fs: float
) -> tuple[float, float, float, float]:
    """Estimate label bounding box (x0, y0, x1, y1) in image pixel space."""
    scale = fs / 6.5
    w = len(text) * _CHAR_W_PX * scale + _BOX_PAD * 2
    h = _LINE_H_PX * scale + _BOX_PAD * 1.4
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _boxes_overlap(a: tuple, b: tuple) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _clamp_to_image(
    cx: float, cy: float, text: str, fs: float, img_h: int, img_w: int
) -> tuple[float, float]:
    """Shift (cx, cy) so the estimated label box stays within image bounds."""
    x0, y0, x1, y1 = _label_box(cx, cy, text, fs)
    margin = 2
    if x0 < margin:
        cx += margin - x0
    if x1 > img_w - margin:
        cx -= x1 - (img_w - margin)
    if y0 < margin:
        cy += margin - y0
    if y1 > img_h - margin:
        cy -= y1 - (img_h - margin)
    return cx, cy


def _resolve_label_positions(
    placements: list,  # [{'cx', 'cy', 'text', 'fs', 'area'}, ...]
    img_h: int,
    img_w: int,
    n_iter: int = 60,
) -> list[tuple[float, float]]:
    """
    Iteratively push overlapping labels apart then clamp to image bounds.

    Labels belonging to larger polygons (more area) are given priority —
    they move less. Smaller-polygon labels are nudged away first.
    """
    # Work in numpy for speed
    positions = np.array([[p["cx"], p["cy"]] for p in placements], dtype=float)
    # Priority order: largest area first (index 0 = most important)
    priority = sorted(range(len(placements)), key=lambda i: -placements[i]["area"])

    for _ in range(n_iter):
        moved = False
        for rank_i, i in enumerate(priority):
            ti, fi = placements[i]["text"], placements[i]["fs"]
            # Clamp first
            positions[i, 0], positions[i, 1] = _clamp_to_image(
                positions[i, 0], positions[i, 1], ti, fi, img_h, img_w
            )
            bi = _label_box(positions[i, 0], positions[i, 1], ti, fi)

            for rank_j in range(rank_i):
                j = priority[rank_j]
                tj, fj = placements[j]["text"], placements[j]["fs"]
                bj = _label_box(positions[j, 0], positions[j, 1], tj, fj)

                if not _boxes_overlap(bi, bj):
                    continue

                # Overlap amounts on each axis
                ow = min(bi[2], bj[2]) - max(bi[0], bj[0])
                oh = min(bi[3], bj[3]) - max(bi[1], bj[1])

                # Push i away from j along the axis of lesser overlap
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]

                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    dx, dy = 1.0, 0.0  # break ties

                dist = np.sqrt(dx**2 + dy**2)
                push = max(ow, oh) + 2.0
                positions[i, 0] += (dx / dist) * push
                positions[i, 1] += (dy / dist) * push
                # Re-clamp after push
                positions[i, 0], positions[i, 1] = _clamp_to_image(
                    positions[i, 0], positions[i, 1], ti, fi, img_h, img_w
                )
                bi = _label_box(positions[i, 0], positions[i, 1], ti, fi)
                moved = True

        if not moved:
            break

    return [
        (float(positions[i, 0]), float(positions[i, 1])) for i in range(len(placements))
    ]


# ---------------------------------------------------------------------------
# Panel 2: segmentation polygons with labels
# ---------------------------------------------------------------------------


def _save_segmentations(
    img_rgb: np.ndarray,
    poly_list: list,
    out_path: Path,
    dpi: int,
) -> tuple[list, list]:
    """
    Segmentation polygons + labels overlaid on the stimulus image.
    Returns (placements, adjusted) so transparent variants can reuse the layout.
    """
    colours = _make_palette(len(poly_list))
    h, w = img_rgb.shape[:2]

    fig_w = 10.0
    fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(img_rgb, extent=[0, w, h, 0])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    _draw_segmentation_patches(ax, poly_list, colours)

    # ── Build label placement list ───────────────────────────────────────
    placements = []
    for idx, obj in enumerate(poly_list):
        poly = obj["polygon"]
        label = _short_label(obj.get("name", ""))
        if not label:
            continue
        fs = _font_size_for_label(label)
        cx, cy = _label_anchor(poly, h, w)
        cx, cy = _clamp_to_image(cx, cy, label, fs, h, w)
        area = int(
            0.5
            * abs(
                np.dot(poly[:, 0], np.roll(poly[:, 1], 1))
                - np.dot(poly[:, 1], np.roll(poly[:, 0], 1))
            )
        )
        placements.append(
            {
                "cx": cx,
                "cy": cy,
                "text": label,
                "fs": fs,
                "area": area,
                "colour": colours[idx % len(colours)],
            }
        )

    # ── Resolve overlaps ─────────────────────────────────────────────────
    adjusted = _resolve_label_positions(placements, h, w)

    # ── Draw labels ──────────────────────────────────────────────────────
    for p, (cx, cy) in zip(placements, adjusted):
        ax.text(
            cx,
            cy,
            p["text"],
            fontsize=p["fs"],
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=4,
            clip_on=True,
            bbox=dict(
                boxstyle="round,pad=0.15", facecolor="black", alpha=0.52, linewidth=0
            ),
        )

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return placements, adjusted


# ---------------------------------------------------------------------------
# Panel 3: relational graph overlaid on image
# ---------------------------------------------------------------------------


def _make_transparent_axes(img_h: int, img_w: int, dpi: int):
    """Return (fig, ax) sized to img dimensions with transparent background."""
    fig_w = 10.0
    fig_h = fig_w * (img_h / img_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def _draw_graph_elements(ax, poly_list, edges, colours, linewidth: float = 1.8):
    """Draw relation edges and centroid nodes onto ax (shared by overlay and transparent)."""
    id_to_idx = {obj["object_id"]: i for i, obj in enumerate(poly_list)}
    centroids = {obj["object_id"]: obj["centroid"] for obj in poly_list}
    # Node size scales slightly with line weight for visual consistency
    markersize = max(4.0, min(12.0, linewidth * 3.2))

    for edge in edges:
        ids = list(edge)
        if len(ids) != 2:
            continue
        a, b = ids
        if a not in centroids or b not in centroids:
            continue
        x0, y0 = centroids[a]
        x1, y1 = centroids[b]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="#e31a1c",
            linewidth=linewidth,
            alpha=0.70,
            zorder=2,
            solid_capstyle="round",
        )

    for obj_id, (cx, cy) in centroids.items():
        idx = id_to_idx.get(obj_id, 0)
        colour = colours[idx % len(colours)]
        ax.plot(
            cx,
            cy,
            "o",
            color=colour,
            markersize=markersize,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )


def _draw_segmentation_patches(ax, poly_list, colours, alpha_fill=0.22):
    """Draw polygon overlays onto ax (shared by overlay and transparent variants)."""
    for idx, obj in enumerate(poly_list):
        colour = colours[idx % len(colours)]
        poly = obj["polygon"]
        patch = mpatches.Polygon(
            poly,
            closed=True,
            linewidth=1.6,
            edgecolor=colour,
            facecolor=(*colour, alpha_fill),
            zorder=2,
        )
        ax.add_patch(patch)


def _save_relational_graph(
    img_rgb: np.ndarray,
    poly_list: list,
    edges: set,
    out_path: Path,
    dpi: int,
    linewidth: float = 1.8,
) -> None:
    """Relational graph overlaid on the stimulus image."""
    colours = _make_palette(len(poly_list))
    h, w = img_rgb.shape[:2]

    fig_w = 10.0
    fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(img_rgb, extent=[0, w, h, 0])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    _draw_graph_elements(ax, poly_list, edges, colours, linewidth=linewidth)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Transparent variants (panels 4, 5, 6, 7)
# ---------------------------------------------------------------------------


def _save_relational_graph_transparent(
    img_rgb: np.ndarray,
    poly_list: list,
    edges: set,
    out_path: Path,
    dpi: int,
    linewidth: float = 1.8,
) -> None:
    """Relational graph (nodes + edges) on a transparent background, no image."""
    colours = _make_palette(len(poly_list))
    h, w = img_rgb.shape[:2]
    fig, ax = _make_transparent_axes(h, w, dpi)
    _draw_graph_elements(ax, poly_list, edges, colours, linewidth=linewidth)
    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)


def _save_segmentations_labeled_transparent(
    img_rgb: np.ndarray,
    poly_list: list,
    adjusted: list,
    placements: list,
    out_path: Path,
    dpi: int,
) -> None:
    """Segmentation polygons + labels on a transparent background, no image."""
    colours = _make_palette(len(poly_list))
    h, w = img_rgb.shape[:2]
    fig, ax = _make_transparent_axes(h, w, dpi)

    _draw_segmentation_patches(ax, poly_list, colours)

    for p, (cx, cy) in zip(placements, adjusted):
        ax.text(
            cx,
            cy,
            p["text"],
            fontsize=p["fs"],
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=4,
            clip_on=True,
            bbox=dict(
                boxstyle="round,pad=0.15", facecolor="black", alpha=0.52, linewidth=0
            ),
        )

    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)


def _save_segmentations_only_transparent(
    img_rgb: np.ndarray,
    poly_list: list,
    out_path: Path,
    dpi: int,
) -> None:
    """Segmentation polygons only (no labels, no image) on transparent background."""
    colours = _make_palette(len(poly_list))
    h, w = img_rgb.shape[:2]
    fig, ax = _make_transparent_axes(h, w, dpi)
    _draw_segmentation_patches(ax, poly_list, colours)
    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)


def _save_labels_only_transparent(
    img_rgb: np.ndarray,
    adjusted: list,
    placements: list,
    out_path: Path,
    dpi: int,
) -> None:
    """Labels only (no polygons, no image) on transparent background.
    Positions are identical to the other label outputs."""
    h, w = img_rgb.shape[:2]
    fig, ax = _make_transparent_axes(h, w, dpi)

    for p, (cx, cy) in zip(placements, adjusted):
        ax.text(
            cx,
            cy,
            p["text"],
            fontsize=p["fs"],
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=4,
            clip_on=True,
            bbox=dict(
                boxstyle="round,pad=0.15", facecolor="black", alpha=0.52, linewidth=0
            ),
        )

    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-stimulus orchestrator
# ---------------------------------------------------------------------------


def process_stim(
    stim_id: str,
    poly_index: dict,
    graph_all: dict,
    output_root: Path,
    dpi: int,
    force: bool,
) -> bool:
    stim_dir = output_root / stim_id
    done_flag = stim_dir / ".done"

    if done_flag.exists() and not force:
        logger.info(f"  [{stim_id}] already done — skipping (--force to redo)")
        return True

    stim_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    try:
        img_rgb = _load_image(stim_id)
    except FileNotFoundError as exc:
        logger.warning(f"  [{stim_id}] {exc} — skipping")
        return False

    poly_list = poly_index.get(stim_id, [])
    edges = graph_all.get(stim_id, set())

    if not poly_list:
        logger.warning(f"  [{stim_id}] no polygon data — panels 2 & 3 will be empty")

    logger.info(
        f"  [{stim_id}]  {len(poly_list)} objects  " f"{len(edges)} relational edges"
    )

    # Panel 1 — raw image
    _save_raw(stim_id, stim_dir)

    # Panel 2 — segmentations + labels on image (also returns layout for reuse)
    placements, adjusted = _save_segmentations(
        img_rgb,
        poly_list,
        stim_dir / "segmentations_labeled.png",
        dpi,
    )

    # Panels 3a/b/c — relational graph on image, three line weights
    for label, lw in [("thin", 0.7), ("moderate", 1.8), ("thick", 4.0)]:
        _save_relational_graph(
            img_rgb,
            poly_list,
            edges,
            stim_dir / f"relational_graph_{label}.png",
            dpi,
            linewidth=lw,
        )

    # Panels 4a/b/c — relational graph on transparent background, three line weights
    for label, lw in [("thin", 0.7), ("moderate", 1.8), ("thick", 4.0)]:
        _save_relational_graph_transparent(
            img_rgb,
            poly_list,
            edges,
            stim_dir / f"relational_graph_transparent_{label}.png",
            dpi,
            linewidth=lw,
        )

    # Panel 5 — segmentations + labels on transparent background
    _save_segmentations_labeled_transparent(
        img_rgb,
        poly_list,
        adjusted,
        placements,
        stim_dir / "segmentations_labeled_transparent.png",
        dpi,
    )

    # Panel 6 — segmentations only (no labels) on transparent background
    _save_segmentations_only_transparent(
        img_rgb,
        poly_list,
        stim_dir / "segmentations_only_transparent.png",
        dpi,
    )

    # Panel 7 — labels only on transparent background (same positions as panel 5)
    _save_labels_only_transparent(
        img_rgb,
        adjusted,
        placements,
        stim_dir / "labels_only_transparent.png",
        dpi,
    )

    done_flag.touch()
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 1 building-block images for all stimuli."
    )
    parser.add_argument(
        "--stims",
        nargs="+",
        default=None,
        metavar="STIM_ID",
        help=(
            "One or more StimIDs to process (e.g. --stims 2367132 2348899 2347025). "
            "If omitted, all stimuli are processed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.OUTPUT_DIR / "visualized_stimuli"),
        help="Root output directory (default: output/visualized_stimuli/).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default 150; use 300 for print-quality output).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess stims even if output already exists.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Stimulus visualizer — Figure 1 building blocks")
    logger.info(f"  Output dir : {output_root}")
    logger.info(f"  DPI        : {args.dpi}")
    logger.info("=" * 60)

    logger.info("\nLoading polygon index ...")
    poly_index = build_polygon_index()

    logger.info("Loading graph index ...")
    graph_all = build_graph_index()["all"]

    if args.stims:
        unknown = [s for s in args.stims if s not in poly_index]
        if unknown:
            logger.warning(f"Unknown StimID(s) — will be skipped: {unknown}")
        stim_ids = args.stims
    else:
        stim_ids = sorted(poly_index.keys())
    logger.info(f"\nProcessing {len(stim_ids)} stim(s) ...\n")

    n_ok = n_fail = 0
    for stim_id in stim_ids:
        ok = process_stim(
            stim_id,
            poly_index,
            graph_all,
            output_root,
            dpi=args.dpi,
            force=args.force,
        )
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Done.  ok={n_ok}  failed={n_fail}")
    logger.info(f"Output → {output_root}")
    logger.info(f"{'='*60}\n")

    # Print folder summary
    dirs = sorted(d for d in output_root.iterdir() if d.is_dir())
    print("Output layout:")
    for d in dirs:
        files = [f.name for f in sorted(d.iterdir()) if not f.name.startswith(".")]
        print(f"  {d.name}/  →  {', '.join(files)}")


if __name__ == "__main__":
    main()
