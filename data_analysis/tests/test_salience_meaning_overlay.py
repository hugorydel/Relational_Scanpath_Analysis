"""
test_salience_meaning_overlay.py
================================
Diagnostic: what do the salience and meaning maps actually look like on the
stimuli, and are they on comparable spatial scales?

The two maps are built by different code paths with different smoothing:

    salience  pipeline/salience/saliency.py    spectral residual -> resize
                                               -> min-max.  NO Gaussian blur.
    meaning   pipeline/meaning/ingest_deepmeaning.py
                                               DeepMeaning .npz -> Gaussian
                                               blur (sigma = 48 px) -> min-max.

That asymmetry is easy to assert and hard to picture, so this script renders a
3 x 3 panel grid (one row per stimulus) and prints the summary statistics that
quantify how peaked each map is.

Output:
    output/analysis/supplementary/salience_meaning_overlay.png
    console table of per-map statistics

Usage:
    python tests/test_salience_meaning_overlay.py
    python tests/test_salience_meaning_overlay.py --stim-ids 150472 2374670 2383555
    python tests/test_salience_meaning_overlay.py --n 5 --dpi 200
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import config
from pipeline.salience.saliency import get_saliency_map

logger = logging.getLogger(__name__)

_MEANING_DIR = config.OUTPUT_DIR / "meaning_maps"
_IMAGE_DIR = config.DATA_METADATA_IMAGES_DIR


def _load_image(stim_id: str, image_dir: Path) -> np.ndarray | None:
    """Read a stimulus image as RGB at the display resolution."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        p = image_dir / f"{stim_id}{ext}"
        if p.exists():
            bgr = cv2.imread(str(p))
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return cv2.resize(
                rgb, (config.IMAGE_W, config.IMAGE_H), interpolation=cv2.INTER_LINEAR
            )
    return None


def _load_meaning(stim_id: str) -> np.ndarray | None:
    """Load a cached DeepMeaning map, if the ingest step has been run."""
    p = _MEANING_DIR / f"{stim_id}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def _stats(m: np.ndarray) -> dict:
    """
    Summarise how concentrated a normalised map is.

    top1_mass / top10_mass are the share of total map mass falling in the
    brightest 1% / 10% of pixels: a blurred map spreads its mass out and gives
    lower values, a peaky map concentrates it.
    """
    flat = np.sort(m.ravel())[::-1]
    total = flat.sum()
    n = flat.size
    return {
        "mean": float(m.mean()),
        "sd": float(m.std()),
        "top1_mass": float(flat[: max(1, n // 100)].sum() / total) if total > 0 else np.nan,
        "top10_mass": float(flat[: max(1, n // 10)].sum() / total) if total > 0 else np.nan,
    }


def _pick_stim_ids(n: int) -> list[str]:
    """Choose stimuli that have both a meaning map and a readable image."""
    if not _MEANING_DIR.exists():
        return []
    ids = sorted(p.stem for p in _MEANING_DIR.glob("*.npy"))
    return ids[:n]


def build_overlay_figure(stim_ids: list[str], image_dir: Path, out_path: Path, dpi: int):
    rows = []
    for sid in stim_ids:
        img = _load_image(sid, image_dir)
        if img is None:
            logger.warning(f"  {sid}: image not found in {image_dir} — skipping.")
            continue
        try:
            sal = get_saliency_map(sid, image_dir=image_dir)
        except Exception as e:
            logger.warning(f"  {sid}: salience failed ({e}) — skipping.")
            continue
        mean_map = _load_meaning(sid)
        if mean_map is None:
            logger.warning(
                f"  {sid}: no cached meaning map in {_MEANING_DIR}. "
                "Run `python -m pipeline.meaning.ingest_deepmeaning` first — skipping."
            )
            continue
        rows.append((sid, img, sal, mean_map))

    if not rows:
        logger.error("No stimuli had both a salience and a meaning map. Nothing to plot.")
        return None

    fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 3.6 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])

    print()
    print(f"{'StimID':<12}{'map':<10}{'mean':>8}{'SD':>8}{'top1%':>9}{'top10%':>9}")
    print("-" * 56)

    for r, (sid, img, sal, mean_map) in enumerate(rows):
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{sid} — stimulus", fontsize=11)

        axes[r, 1].imshow(img)
        axes[r, 1].imshow(sal, cmap="inferno", alpha=0.55)
        axes[r, 1].set_title("salience (no smoothing)", fontsize=11)

        axes[r, 2].imshow(img)
        axes[r, 2].imshow(mean_map, cmap="plasma", alpha=0.55)
        axes[r, 2].set_title("meaning (Gaussian, σ = 48 px)", fontsize=11)

        for c in range(3):
            axes[r, c].axis("off")

        for label, m in (("salience", sal), ("meaning", mean_map)):
            s = _stats(m)
            print(
                f"{sid:<12}{label:<10}{s['mean']:>8.3f}{s['sd']:>8.3f}"
                f"{s['top1_mass']:>9.3f}{s['top10_mass']:>9.3f}"
            )

    print("-" * 56)
    print("top1% / top10% = share of total map mass in the brightest 1% / 10% of")
    print("pixels. Higher means more spatially concentrated.")
    print()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Written → {out_path}")
    return out_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    parser = argparse.ArgumentParser(
        description="Overlay salience and meaning maps on example stimuli."
    )
    parser.add_argument(
        "--stim-ids", nargs="+", default=None, help="Specific StimIDs to render."
    )
    parser.add_argument(
        "--n", type=int, default=3, help="How many stimuli to render (default 3)."
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Stimulus image directory (defaults to the configured one).",
    )
    parser.add_argument(
        "--out",
        default=str(
            config.OUTPUT_DIR / "analysis" / "supplementary" / "salience_meaning_overlay.png"
        ),
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    image_dir = (
        Path(args.image_dir)
        if args.image_dir
        else _IMAGE_DIR
    )
    stim_ids = args.stim_ids or _pick_stim_ids(args.n)
    if not stim_ids:
        logger.error(
            f"No stimuli to render. Looked for cached meaning maps in {_MEANING_DIR}."
        )
        sys.exit(1)

    logger.info(f"Rendering {len(stim_ids)} stimulus/stimuli from {image_dir}")
    result = build_overlay_figure(stim_ids, image_dir, Path(args.out), args.dpi)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
