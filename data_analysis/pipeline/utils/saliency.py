"""
utils/saliency.py
=================
Spectral Residual saliency maps (Hou & Zhang, CVPR 2007) via OpenCV.

Computes one saliency map per stimulus image and caches the result to
output/features/saliency_maps/{StimID}.npy so it is only computed once.

Algorithm:
    Utilizes cv2.saliency.StaticSaliencySpectralResidual_create() which
    natively handles the FFT, log-amplitude smoothing, residual subtraction,
    and inverse FFT reconstruction.

    The resulting map is resized to IMAGE_W x IMAGE_H and normalized
    to a probability map (sum = 1.0) for regression comparability.

Usage (standalone):
    # All stimuli in stimuli_dataset.json
    python -m pipeline.utils.saliency

    # Specific subset
    python -m pipeline.utils.saliency --stim-ids 2383555 2386442

    # Force recompute even if cache exists
    python -m pipeline.utils.saliency --force

Usage (from Module 3):
    from pipeline.utils.saliency import get_saliency_map
    sal_map = get_saliency_map("2383555")   # np.ndarray (768, 1024), float32, sums to 1
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import config
import cv2
import numpy as np
from pipeline.utils.scene_graph import load_stimulus_metadata

logger = logging.getLogger(__name__)

# Output image dimensions (all stimuli standardised to this)
IMAGE_W = 1024
IMAGE_H = 768


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def compute_saliency_map(image_path: Path) -> np.ndarray:
    """
    Compute a spectral residual saliency map using OpenCV.

    Parameters
    ----------
    image_path : Path
        Path to the stimulus image.

    Returns
    -------
    np.ndarray
        Float32 array of shape (IMAGE_H, IMAGE_W) = (768, 1024), values in [0, 1].
        Row-major: index as [y, x].
    """
    # 1. Read image via OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(
            f"OpenCV could not read image at {image_path}. Check file format."
        )

    # 2. Initialize the Spectral Residual algorithm
    saliency_algorithm = cv2.saliency.StaticSaliencySpectralResidual_create()

    # 3. Compute the map
    success, saliency_map = saliency_algorithm.computeSaliency(img)
    if not success:
        raise RuntimeError(f"Saliency computation failed for {image_path}")

    # 4. Resize to standard experimental dimensions
    sal_map_resized = cv2.resize(
        saliency_map, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_LINEAR
    )

    # 5. Min-Max Normalization (Scales from 0.0 to 1.0)
    sal_map_float = sal_map_resized.astype(np.float32)
    sal_min = sal_map_float.min()
    sal_max = sal_map_float.max()

    if sal_max - sal_min > 0:
        sal_map_float = (sal_map_float - sal_min) / (sal_max - sal_min)
    else:
        # Fallback if the map is completely uniform/blank
        sal_map_float[:] = 0.0

    return sal_map_float


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def _cache_path(stim_id: str, cache_dir: Path) -> Path:
    return cache_dir / f"{stim_id}.npy"


def get_saliency_map(
    stim_id: str,
    image_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Return the saliency map for a stimulus, loading from cache if available.
    """
    image_dir = Path(image_dir) if image_dir else config.DATA_METADATA_DIR / "images"
    cache_dir = (
        Path(cache_dir) if cache_dir else config.OUTPUT_FEATURES_DIR / "saliency_maps"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached = _cache_path(stim_id, cache_dir)

    if cached.exists() and not force_recompute:
        return np.load(cached)

    # Find image file — support common extensions
    img_path = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        candidate = image_dir / f"{stim_id}{ext}"
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        raise FileNotFoundError(
            f"No image found for StimID {stim_id} in {image_dir}. "
            f"Tried extensions: .jpg, .jpeg, .png, .bmp, .tiff"
        )

    logger.debug(f"  Computing saliency map for {stim_id} ...")
    sal_map = compute_saliency_map(img_path)
    np.save(cached, sal_map)
    return sal_map


# ---------------------------------------------------------------------------
# Batch computation (standalone / pre-cache)
# ---------------------------------------------------------------------------


def compute_all_saliency_maps(
    image_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    stim_ids: Optional[list[str]] = None,
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute and cache saliency maps for all (or a subset of) stimuli.
    """
    image_dir = Path(image_dir) if image_dir else config.DATA_METADATA_DIR / "images"
    cache_dir = (
        Path(cache_dir) if cache_dir else config.OUTPUT_FEATURES_DIR / "saliency_maps"
    )

    if stim_ids is None:
        metadata = load_stimulus_metadata()
        stim_ids = list(metadata.keys())

    logger.info(f"Computing saliency maps for {len(stim_ids)} stimuli ...")
    logger.info(f"  Image dir : {image_dir}")
    logger.info(f"  Cache dir : {cache_dir}")

    results = {}
    n_cached = 0
    n_new = 0
    errors = []

    for stim_id in stim_ids:
        cached = _cache_path(stim_id, cache_dir)
        if cached.exists() and not force_recompute:
            results[stim_id] = np.load(cached)
            n_cached += 1
            continue
        try:
            sal_map = get_saliency_map(
                stim_id,
                image_dir=image_dir,
                cache_dir=cache_dir,
                force_recompute=force_recompute,
            )
            results[stim_id] = sal_map
            n_new += 1
        except Exception as e:
            logger.warning(f"  Failed on {stim_id}: {e}")
            errors.append(stim_id)

    logger.info(
        f"Saliency maps ready: {n_new} computed, {n_cached} loaded from cache, "
        f"{len(errors)} failed."
    )
    if errors:
        logger.warning(f"  Failed StimIDs: {errors}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Pre-compute spectral residual saliency maps for all stimuli."
    )
    parser.add_argument(
        "--image-dir",
        default=str(config.DATA_METADATA_DIR / "images"),
        help="Directory containing stimulus images",
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.OUTPUT_FEATURES_DIR / "saliency_maps"),
        help="Directory to write cached .npy saliency maps",
    )
    parser.add_argument(
        "--stim-ids",
        nargs="*",
        help="Subset of StimIDs to process (default: all in stimuli_dataset.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if cached files already exist",
    )
    args = parser.parse_args()

    maps = compute_all_saliency_maps(
        image_dir=Path(args.image_dir),
        cache_dir=Path(args.output_dir),
        stim_ids=args.stim_ids,
        force_recompute=args.force,
    )

    # Quick sanity check on computed maps
    print(f"\nSanity check on {len(maps)} maps:")
    for stim_id, sal in list(maps.items())[:5]:
        print(
            f"  {stim_id}: shape={sal.shape}  "
            f"min={sal.min():.4f}  max={sal.max():.4f}  "
            f"mean={sal.mean():.4f}  dtype={sal.dtype}"
        )
    if len(maps) > 5:
        print(f"  ... ({len(maps) - 5} more)")


if __name__ == "__main__":
    main()
