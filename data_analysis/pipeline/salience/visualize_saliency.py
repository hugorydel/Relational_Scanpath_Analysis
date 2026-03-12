"""
salience/visualize_saliency.py
===========================
Step 2 of Module 3: Visualize the Spectral Residual saliency map for a given StimID.

Generates and visualizes the Spectral Residual saliency map for a given StimID,
utilizing the cached .npy files if available.

Usage:
    # Use cached map (or compute and cache if missing)
    python -m salience.visualize_saliency 2383555

    # Force recomputation and overwrite cache
    python -m salience.visualize_saliency 2383555 --force
"""

import argparse
import logging
from pathlib import Path

import config
import cv2
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.pipeline.salience.saliency import get_saliency_map

logger = logging.getLogger(__name__)


def visualize_saliency(stim_id: str, image_dir: Path, cache_dir: Path, force: bool):
    """
    Loads (or computes) and plots the saliency map for a specific stimulus.
    """
    # 1. Fetch the saliency map (uses cache if available and not forced)
    logger.info(f"Fetching saliency map for {stim_id}...")
    try:
        sal_map = get_saliency_map(
            stim_id, image_dir=image_dir, cache_dir=cache_dir, force_recompute=force
        )
    except Exception as e:
        logger.error(f"Failed to fetch or generate saliency map: {e}")
        return

    # 2. Locate the original image for the underlay
    img_path = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        candidate = image_dir / f"{stim_id}{ext}"
        if candidate.exists():
            img_path = candidate
            break

    if not img_path:
        logger.error(f"Could not find original image for {stim_id} in {image_dir}")
        return

    # 3. Read and format the original image for matplotlib (BGR -> RGB)
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Ensure the image matches the standard W x H output of our saliency map
    img_rgb = cv2.resize(
        img_rgb, (config.IMAGE_W, config.IMAGE_H), interpolation=cv2.INTER_LINEAR
    )

    # 4. Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # A. Original Image
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original Image: {stim_id}", fontsize=14)
    axes[0].axis("off")

    # B. Saliency Heatmap
    im_sal = axes[1].imshow(sal_map, cmap="inferno")
    axes[1].set_title(f"Spectral Residual (Mean: {sal_map.mean():.4f})", fontsize=14)
    axes[1].axis("off")
    fig.colorbar(im_sal, ax=axes[1], fraction=0.046, pad=0.04)

    # C. Overlay
    axes[2].imshow(img_rgb)
    axes[2].imshow(sal_map, cmap="inferno", alpha=0.55)
    axes[2].set_title("Saliency Overlay", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(
        description="Visualize the Spectral Residual map for a StimID."
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
        default=str(config.OUTPUT_FEATURES_DIR / "saliency_maps"),
        help="Path to the directory containing cached .npy files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of the map even if a cached file exists",
    )

    args = parser.parse_args()

    visualize_saliency(
        stim_id=args.stim_id,
        image_dir=Path(args.image_dir),
        cache_dir=Path(args.cache_dir),
        force=args.force,
    )


if __name__ == "__main__":
    main()
