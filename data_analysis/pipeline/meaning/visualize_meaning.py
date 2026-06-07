"""
pipeline/meaning/visualize_meaning.py
======================================
Visualize the CLIP-based meaning map for a given StimID.

Generates and visualizes the meaning map for a given StimID, utilizing the
cached .npy files if available. Layout mirrors visualize_saliency.py for
direct side-by-side comparison.

Usage:
    # Use cached map (or compute and cache if missing)
    python -m pipeline.meaning.visualize_meaning 2383555

    # Force recomputation and overwrite cache
    python -m pipeline.meaning.visualize_meaning 2383555 --force
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pipeline.meaning.meaning import get_meaning_map

logger = logging.getLogger(__name__)


def visualize_meaning(stim_id: str, image_dir: Path, cache_dir: Path, force: bool):
    """
    Loads (or computes) and plots the meaning map for a specific stimulus.
    """
    logger.info(f"Fetching meaning map for {stim_id}...")
    try:
        meaning_map = get_meaning_map(
            stim_id, image_dir=image_dir, cache_dir=cache_dir, force_recompute=force
        )
    except Exception as e:
        logger.error(f"Failed to fetch or generate meaning map: {e}")
        return

    # Locate the original image for the underlay
    img_path = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        candidate = image_dir / f"{stim_id}{ext}"
        if candidate.exists():
            img_path = candidate
            break

    if not img_path:
        logger.error(f"Could not find original image for {stim_id} in {image_dir}")
        return

    # Read and convert for matplotlib (BGR → RGB)
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(
        img_rgb, (config.IMAGE_W, config.IMAGE_H), interpolation=cv2.INTER_LINEAR
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # A. Original image
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original Image: {stim_id}", fontsize=14)
    axes[0].axis("off")

    # B. Meaning map heatmap
    im_m = axes[1].imshow(meaning_map, cmap="plasma")
    axes[1].set_title(f"CLIP Meaning Map (Mean: {meaning_map.mean():.4f})", fontsize=14)
    axes[1].axis("off")
    fig.colorbar(im_m, ax=axes[1], fraction=0.046, pad=0.04)

    # C. Overlay
    axes[2].imshow(img_rgb)
    axes[2].imshow(meaning_map, cmap="plasma", alpha=0.55)
    axes[2].set_title("Meaning Overlay", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(
        description="Visualize the CLIP meaning map for a StimID."
    )
    parser.add_argument(
        "stim_id", type=str, help="The StimID to visualize (e.g., 2383555)"
    )
    parser.add_argument(
        "--image-dir",
        default=str(config.DATA_METADATA_DIR / "images"),
        help="Path to the directory containing stimulus images",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(config.OUTPUT_DIR / "meaning_maps"),
        help="Path to the directory containing cached .npy meaning map files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of the map even if a cached file exists",
    )

    args = parser.parse_args()

    visualize_meaning(
        stim_id=args.stim_id,
        image_dir=Path(args.image_dir),
        cache_dir=Path(args.cache_dir),
        force=args.force,
    )


if __name__ == "__main__":
    main()
