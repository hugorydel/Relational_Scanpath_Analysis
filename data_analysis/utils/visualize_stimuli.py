"""
visualize_stimuli.py
====================
Generates Figure 1 building-block images for all 30 stimuli.

For each stimulus, creates a subfolder under:
    data_analysis/output/visualized_stimuli/{stim_id}/

containing:
    raw_image.jpg                            — original stimulus image
    segmentations_labeled.png                — polygon overlays + labels on image
    relational_graph_{thin|moderate|thick}.png
    relational_graph_transparent_{thin|moderate|thick}.png
    segmentations_labeled_transparent.png
    segmentations_only_transparent.png
    labels_only_transparent.png

Usage
-----
    python visualize_stimuli.py --stims 2367132 2348899 2347025
    python visualize_stimuli.py                  # all stimuli
    python visualize_stimuli.py --dpi 300        # print quality
    python visualize_stimuli.py --force          # reprocess existing
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
# Colour palette
# ---------------------------------------------------------------------------


def _make_palette(n: int) -> np.ndarray:
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
    import cv2

    for ext in _IMAGE_EXTENSIONS:
        path = config.DATA_METADATA_IMAGES_DIR / f"{stim_id}{ext}"
        if path.exists():
            img_bgr = cv2.imread(str(path))
            if img_bgr is not None:
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    raise FileNotFoundError(
        f"No image found for stim_id={stim_id} in {config.DATA_METADATA_IMAGES_DIR}"
    )


# ---------------------------------------------------------------------------
# Panel 1: raw image
# ---------------------------------------------------------------------------


def _save_raw(stim_id: str, out_dir: Path) -> None:
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
# Label text: truncate to 2-5 meaningful words
# ---------------------------------------------------------------------------

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


# Phrases that, when encountered mid-label, indicate the start of a
# relational/prepositional clause we don't need.  Everything from the
# trigger word(s) onwards is dropped.  Checked left-to-right; the first
# match wins.  At least one word is always kept before the trigger.
_MID_TRIGGERS = [
    ("to", "a"),  # "connected to a Wii..."
    ("of", "a"),  # "side of a table..."
    ("in", "a"),  # "sitting in a chair..."
    ("on", "a"),  # "standing on a..."
    ("being",),  # "shirt being worn by..."
    ("worn",),  # "shirt worn by..."
    ("held",),  # "controller held by..."
    ("in",),  # "man in gray shirt..."
]


def _truncate_at_triggers(words: list) -> tuple[list, bool]:
    """
    Scan words left-to-right (starting at index 1 so at least one word is
    kept) and truncate at the first matching trigger phrase.
    Returns (truncated_words, was_truncated).
    """
    for i in range(1, len(words)):
        w = words[i].lower()
        for trigger in _MID_TRIGGERS:
            if len(trigger) == 1:
                if w == trigger[0]:
                    return words[:i], True
            else:  # bigram
                if (
                    i + 1 < len(words)
                    and w == trigger[0]
                    and words[i + 1].lower() == trigger[1]
                ):
                    return words[:i], True
    return words, False


def _short_label(name: str, max_words: int = 5, min_words: int = 2) -> str:
    name = name.rstrip(".").strip()
    if not name:
        return ""

    words = name.split()

    # 1. Mid-label truncation at relational/prepositional triggers
    words, mid_truncated = _truncate_at_triggers(words)

    # 2. Strip trailing function words
    while len(words) > 1 and words[-1].lower() in _FUNCTION_WORDS:
        words = words[:-1]
        mid_truncated = True

    # 3. Cap at max_words
    trailing_truncated = len(words) > max_words
    words = words[:max_words]

    # 4. Strip trailing function words again after cap
    while len(words) > 1 and words[-1].lower() in _FUNCTION_WORDS:
        words = words[:-1]
        trailing_truncated = True

    return " ".join(words) + ("..." if (mid_truncated or trailing_truncated) else "")


def _font_size_for_label(label: str) -> float:
    """8.5pt for short labels (≤3 words), 7.5pt for longer ones (4-5 words)."""
    n_words = len(label.rstrip("...").split())
    if n_words <= 3:
        return 8.5
    return 7.5


# ---------------------------------------------------------------------------
# Collision detection and label placement
# ---------------------------------------------------------------------------

_CHAR_W_PX = 4.8
_LINE_H_PX = 13.0
_BOX_PAD = 5.0


def _label_box(cx, cy, text, fs):
    scale = fs / 6.5
    w = len(text) * _CHAR_W_PX * scale + _BOX_PAD * 2
    h = _LINE_H_PX * scale + _BOX_PAD * 1.4
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _boxes_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _clamp_to_image(cx, cy, text, fs, img_h, img_w):
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


def _resolve_label_positions(placements, img_h, img_w, n_iter=60):
    positions = np.array([[p["cx"], p["cy"]] for p in placements], dtype=float)
    priority = sorted(range(len(placements)), key=lambda i: -placements[i]["area"])

    for _ in range(n_iter):
        moved = False
        for rank_i, i in enumerate(priority):
            ti, fi = placements[i]["text"], placements[i]["fs"]
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
                ow = min(bi[2], bj[2]) - max(bi[0], bj[0])
                oh = min(bi[3], bj[3]) - max(bi[1], bj[1])
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    dx, dy = 1.0, 0.0
                dist = np.sqrt(dx**2 + dy**2)
                push = max(ow, oh) + 2.0
                positions[i, 0] += (dx / dist) * push
                positions[i, 1] += (dy / dist) * push
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
# Shared drawing helpers
# ---------------------------------------------------------------------------


def _draw_segmentation_patches(ax, poly_list, colours, alpha_fill=0.22):
    for idx, obj in enumerate(poly_list):
        colour = colours[idx % len(colours)]
        patch = mpatches.Polygon(
            obj["polygon"],
            closed=True,
            linewidth=1.6,
            edgecolor=colour,
            facecolor=(*colour, alpha_fill),
            zorder=2,
        )
        ax.add_patch(patch)


def _draw_labels(ax, placements, adjusted):
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


def _make_transparent_axes(img_h, img_w, dpi):
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


def _draw_graph_elements(ax, poly_list, edges, colours, linewidth=1.8):
    id_to_idx = {obj["object_id"]: i for i, obj in enumerate(poly_list)}
    centroids = {obj["object_id"]: obj["centroid"] for obj in poly_list}
    markersize = max(7.0, min(18.0, linewidth * 4.5))

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


def _build_placements(poly_list, colours, img_h, img_w):
    # ── First pass: build raw placements ────────────────────────────────
    raw = []
    for idx, obj in enumerate(poly_list):
        poly = obj["polygon"]
        label = _short_label(obj.get("name", ""))
        if not label:
            continue
        fs = _font_size_for_label(label)
        cx, cy = _label_anchor(poly, img_h, img_w)
        cx, cy = _clamp_to_image(cx, cy, label, fs, img_h, img_w)
        area = int(
            0.5
            * abs(
                np.dot(poly[:, 0], np.roll(poly[:, 1], 1))
                - np.dot(poly[:, 1], np.roll(poly[:, 0], 1))
            )
        )
        raw.append(
            {
                "cx": cx,
                "cy": cy,
                "text": label,
                "fs": fs,
                "area": area,
                "colour": colours[idx % len(colours)],
            }
        )

    # ── Second pass: deduplicate by (label, proximity) ───────────────────
    # If two entries share the same label text and their anchors are within
    # DEDUP_RADIUS pixels of each other, keep only the larger-polygon one.
    # This prevents the same truncated label (e.g. "man...") from appearing
    # multiple times for overlapping dataset objects of the same person.
    DEDUP_RADIUS = 80  # image pixels
    kept = []
    for entry in sorted(raw, key=lambda e: -e["area"]):  # largest first
        duplicate = False
        for k in kept:
            if k["text"] == entry["text"]:
                dist = np.sqrt(
                    (k["cx"] - entry["cx"]) ** 2 + (k["cy"] - entry["cy"]) ** 2
                )
                if dist < DEDUP_RADIUS:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(entry)

    return kept


# ---------------------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------------------


def _save_segmentations(img_rgb, poly_list, out_path, dpi):
    colours = _make_palette(len(poly_list))
    h, w = img_rgb.shape[:2]
    fig_w = 10.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * h / w), dpi=dpi)
    ax.imshow(img_rgb, extent=[0, w, h, 0])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _draw_segmentation_patches(ax, poly_list, colours)
    placements = _build_placements(poly_list, colours, h, w)
    adjusted = _resolve_label_positions(placements, h, w)
    _draw_labels(ax, placements, adjusted)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return placements, adjusted


def _save_relational_graph(img_rgb, poly_list, edges, out_path, dpi, linewidth=1.8):
    colours = _make_palette(len(poly_list))
    h, w = img_rgb.shape[:2]
    fig_w = 10.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * h / w), dpi=dpi)
    ax.imshow(img_rgb, extent=[0, w, h, 0])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _draw_graph_elements(ax, poly_list, edges, colours, linewidth=linewidth)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_relational_graph_transparent(
    img_rgb, poly_list, edges, out_path, dpi, linewidth=1.8
):
    colours = _make_palette(len(poly_list))
    h, w = img_rgb.shape[:2]
    fig, ax = _make_transparent_axes(h, w, dpi)
    _draw_graph_elements(ax, poly_list, edges, colours, linewidth=linewidth)
    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)


def _save_segmentations_labeled_transparent(
    img_rgb, poly_list, adjusted, placements, out_path, dpi
):
    colours = _make_palette(len(poly_list))
    h, w = img_rgb.shape[:2]
    fig, ax = _make_transparent_axes(h, w, dpi)
    _draw_segmentation_patches(ax, poly_list, colours)
    _draw_labels(ax, placements, adjusted)
    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)


def _save_segmentations_only_transparent(img_rgb, poly_list, out_path, dpi):
    colours = _make_palette(len(poly_list))
    h, w = img_rgb.shape[:2]
    fig, ax = _make_transparent_axes(h, w, dpi)
    _draw_segmentation_patches(ax, poly_list, colours)
    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)


def _save_labels_only_transparent(img_rgb, adjusted, placements, out_path, dpi):
    h, w = img_rgb.shape[:2]
    fig, ax = _make_transparent_axes(h, w, dpi)
    _draw_labels(ax, placements, adjusted)
    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-stimulus orchestrator
# ---------------------------------------------------------------------------


def process_stim(stim_id, poly_index, graph_all, output_root, dpi, force):
    stim_dir = output_root / stim_id
    done_flag = stim_dir / ".done"

    if done_flag.exists() and not force:
        logger.info(f"  [{stim_id}] already done — skipping (--force to redo)")
        return True

    stim_dir.mkdir(parents=True, exist_ok=True)

    try:
        img_rgb = _load_image(stim_id)
    except FileNotFoundError as exc:
        logger.warning(f"  [{stim_id}] {exc} — skipping")
        return False

    poly_list = poly_index.get(stim_id, [])
    edges = graph_all.get(stim_id, set())

    if not poly_list:
        logger.warning(f"  [{stim_id}] no polygon data")

    logger.info(f"  [{stim_id}]  {len(poly_list)} objects  {len(edges)} edges")

    _save_raw(stim_id, stim_dir)

    placements, adjusted = _save_segmentations(
        img_rgb, poly_list, stim_dir / "segmentations_labeled.png", dpi
    )

    for label, lw in [("thin", 0.7), ("moderate", 3.0), ("thick", 5.5)]:
        _save_relational_graph(
            img_rgb,
            poly_list,
            edges,
            stim_dir / f"relational_graph_{label}.png",
            dpi,
            linewidth=lw,
        )

    for label, lw in [("thin", 0.7), ("moderate", 3.0), ("thick", 5.5)]:
        _save_relational_graph_transparent(
            img_rgb,
            poly_list,
            edges,
            stim_dir / f"relational_graph_transparent_{label}.png",
            dpi,
            linewidth=lw,
        )

    _save_segmentations_labeled_transparent(
        img_rgb,
        poly_list,
        adjusted,
        placements,
        stim_dir / "segmentations_labeled_transparent.png",
        dpi,
    )

    _save_segmentations_only_transparent(
        img_rgb, poly_list, stim_dir / "segmentations_only_transparent.png", dpi
    )

    _save_labels_only_transparent(
        img_rgb, adjusted, placements, stim_dir / "labels_only_transparent.png", dpi
    )

    done_flag.touch()
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 1 building-block images for all stimuli."
    )
    parser.add_argument(
        "--stims",
        nargs="+",
        default=None,
        metavar="STIM_ID",
        help="StimIDs to process (e.g. --stims 2367132 2348899). Omit for all.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.OUTPUT_DIR / "visualized_stimuli"),
        help="Root output directory.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Stimulus visualizer — Figure 1 building blocks")
    logger.info(f"  Output dir : {output_root}")
    logger.info(f"  DPI        : {args.dpi}")
    logger.info("=" * 60)

    poly_index = build_polygon_index()
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
            stim_id, poly_index, graph_all, output_root, dpi=args.dpi, force=args.force
        )
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Done.  ok={n_ok}  failed={n_fail}")
    logger.info(f"Output → {output_root}")
    logger.info(f"{'='*60}\n")

    dirs = sorted(d for d in output_root.iterdir() if d.is_dir())
    print("Output layout:")
    for d in dirs:
        files = [f.name for f in sorted(d.iterdir()) if not f.name.startswith(".")]
        print(f"  {d.name}/  →  {', '.join(files)}")


if __name__ == "__main__":
    main()
