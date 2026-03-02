"""
utils/visualize_saliency.py
===========================
Visualise spectral residual saliency maps overlaid on stimulus images.

Produces a grid of (image | heatmap overlay) pairs for inspection.

Usage:
    # All 30 experiment stimuli, 5 per row
    python -m pipeline.utils.visualize_saliency

    # Specific StimIDs
    python -m pipeline.utils.visualize_saliency --stim-ids 2383555 2386442 150472

    # Save to file instead of showing interactively
    python -m pipeline.utils.visualize_saliency --output saliency_grid.png
"""

import argparse
import logging
import sys
from pathlib import Path

import config
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pipeline.utils.saliency import get_saliency_map
from pipeline.utils.scene_graph import load_stimulus_metadata

logger = logging.getLogger(__name__)

IMAGE_W = 1024
IMAGE_H = 768


def visualize_saliency_maps(
    stim_ids: list[str],
    image_dir: Path,
    cache_dir: Path,
    ncols: int = 5,
    alpha: float = 0.55,
    output_path: Path | None = None,
):
    """
    Plot a grid with two columns per stimulus: original image and saliency overlay.

    Parameters
    ----------
    stim_ids : list of str
    image_dir : Path
    cache_dir : Path
    ncols : int
        Number of stimuli per row (each stimulus takes 2 subplot columns).
    alpha : float
        Opacity of the heatmap overlay (0=invisible, 1=fully opaque).
    output_path : Path or None
        If given, save figure to this path instead of showing interactively.
    """
    n = len(stim_ids)
    nrows = (n + ncols - 1) // ncols
    fig_w = ncols * 4
    fig_h = nrows * 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    for i, stim_id in enumerate(stim_ids):
        ax = axes[i]

        # Load original image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = image_dir / f"{stim_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            ax.text(
                0.5,
                0.5,
                f"{stim_id}\n(image not found)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            continue

        img = np.array(
            Image.open(img_path)
            .convert("RGB")
            .resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
        )

        # Load saliency map
        try:
            sal = get_saliency_map(stim_id, image_dir=image_dir, cache_dir=cache_dir)
        except FileNotFoundError:
            ax.imshow(img)
            ax.set_title(f"{stim_id}\n(saliency not found)", fontsize=7)
            ax.axis("off")
            continue

        # Show image with heatmap overlaid
        ax.imshow(img)
        ax.imshow(sal, cmap="hot", alpha=alpha, vmin=0, vmax=1)
        ax.set_title(stim_id, fontsize=7)
        ax.axis("off")

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Spectral Residual Saliency Maps (Hou & Zhang, 2007)\n"
        "Heatmap overlay: red = high saliency, black = low saliency",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Visualise saliency maps")
    parser.add_argument(
        "--stim-ids",
        nargs="*",
        help="StimIDs to visualise (default: all experiment stimuli)",
    )
    parser.add_argument("--image-dir", default=str(config.DATA_METADATA_DIR / "images"))
    parser.add_argument(
        "--cache-dir", default=str(config.OUTPUT_FEATURES_DIR / "saliency_maps")
    )
    parser.add_argument(
        "--ncols", type=int, default=5, help="Stimuli per row (default: 5)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.55, help="Heatmap opacity 0-1 (default: 0.55)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save to file instead of showing (e.g. saliency_grid.png)",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    cache_dir = Path(args.cache_dir)

    if args.stim_ids:
        stim_ids = args.stim_ids
    else:
        # Use only experiment stimuli (those in fixations output)
        metadata = load_stimulus_metadata()
        stim_ids = sorted(metadata.keys())

    visualize_saliency_maps(
        stim_ids=stim_ids,
        image_dir=image_dir,
        cache_dir=cache_dir,
        ncols=args.ncols,
        alpha=args.alpha,
        output_path=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
