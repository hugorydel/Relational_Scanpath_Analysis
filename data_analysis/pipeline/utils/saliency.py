"""
utils/saliency.py
=================
Spectral Residual saliency maps (Hou & Zhang, CVPR 2007).

Computes one saliency map per stimulus image and caches the result to
output/features/saliency_maps/{StimID}.npy so it is only computed once.

Algorithm (equations from the paper):
    1. Downsample image to 64×64
    2. Per RGB channel:
       a. Compute log amplitude spectrum:  L(f) = log|FFT(channel)|
       b. Smooth with 3×3 average filter:  A(f) = h₃ * L(f)
       c. Spectral residual:               R(f) = L(f) - A(f)
       d. Reconstruct:  S_c(x) = |IFFT(exp(R(f) + i·P(f)))|²
          where P(f) is the preserved phase spectrum
    3. Average S_c across channels
    4. Smooth with Gaussian σ=8 (config.SALIENCE_SMOOTHING_SIGMA)
    5. Upsample back to IMAGE_W × IMAGE_H (1024×768)
    6. Normalise to [0, 1]

Usage (standalone):
    python -m pipeline.utils.saliency \\
        --image-dir  data_metadata/images/ \\
        --output-dir output/features/saliency_maps/ \\
        --stim-ids   2383555 2386442        # optional: subset

Usage (from Module 3):
    from pipeline.utils.saliency import get_saliency_map
    sal_map = get_saliency_map("2383555")   # np.ndarray (768, 1024), float32, [0,1]
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import config
import numpy as np
from PIL import Image
from pipeline.utils.scene_graph import load_stimulus_metadata
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# Output image dimensions (all stimuli standardised to this)
IMAGE_W = 1024
IMAGE_H = 768

# FFT computation size (paper recommendation)
FFT_SIZE = 64

# Average filter size for spectral residual (paper uses 3×3)
AVG_FILTER_SIZE = 3


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def _spectral_residual_channel(channel: np.ndarray) -> np.ndarray:
    """
    Compute saliency map for a single grayscale channel (H×W, float).

    Returns a saliency map of the same spatial size, unnormalised.
    """
    # Downsample to FFT_SIZE × FFT_SIZE
    img_pil = Image.fromarray((channel * 255).astype(np.uint8))
    img_small = img_pil.resize((FFT_SIZE, FFT_SIZE), Image.BILINEAR)
    arr = np.array(img_small, dtype=np.float64) / 255.0

    # FFT
    fft = np.fft.fft2(arr)
    amp = np.abs(fft)
    phase = np.angle(fft)  # P(f) — preserved throughout

    # Log amplitude spectrum — add small epsilon to avoid log(0)
    log_amp = np.log(amp + 1e-12)  # L(f)

    # Smooth with AVG_FILTER_SIZE × AVG_FILTER_SIZE mean filter
    from scipy.ndimage import uniform_filter

    avg_amp = uniform_filter(log_amp, size=AVG_FILTER_SIZE)  # A(f)

    # Spectral residual
    residual = log_amp - avg_amp  # R(f)

    # Reconstruct in spatial domain: exp(R(f) + i·P(f))
    reconstructed = np.exp(residual) * np.exp(1j * phase)
    saliency_small = np.abs(np.fft.ifft2(reconstructed)) ** 2  # squared as per paper

    return saliency_small


def compute_saliency_map(image_path: Path) -> np.ndarray:
    """
    Compute a spectral residual saliency map for one image.

    Parameters
    ----------
    image_path : Path
        Path to the stimulus image (any PIL-readable format).

    Returns
    -------
    np.ndarray
        Float32 array of shape (IMAGE_H, IMAGE_W) = (768, 1024), values in [0, 1].
        Row-major: index as [y, x].
    """
    # Load and convert to RGB float [0, 1]
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)

    # Per-channel spectral residual
    channels = []
    for c in range(3):
        sal_c = _spectral_residual_channel(img_arr[:, :, c])
        channels.append(sal_c)

    # Average across channels → (FFT_SIZE, FFT_SIZE)
    sal_map = np.mean(channels, axis=0).astype(np.float32)

    # Smooth with Gaussian σ=8 at the 64×64 FFT scale.
    # The paper applies σ=8 at the 64px stage before upsampling — no scaling needed.
    sal_map = gaussian_filter(sal_map, sigma=config.SALIENCE_SMOOTHING_SIGMA).astype(
        np.float32
    )

    # Upsample to IMAGE_H × IMAGE_W
    sal_pil = Image.fromarray(sal_map)
    sal_up = sal_pil.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
    sal_map = np.array(sal_up, dtype=np.float32)

    # Normalise to [0, 1]
    sal_min, sal_max = sal_map.min(), sal_map.max()
    if sal_max > sal_min:
        sal_map = (sal_map - sal_min) / (sal_max - sal_min)
    else:
        sal_map[:] = 0.0

    return sal_map


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

    Parameters
    ----------
    stim_id : str
        Image ID (e.g. "2383555").
    image_dir : Path, optional
        Directory containing stimulus images. Defaults to
        config.DATA_METADATA_DIR / "images".
    cache_dir : Path, optional
        Directory for cached .npy files. Defaults to
        config.OUTPUT_FEATURES_DIR / "saliency_maps".
    force_recompute : bool
        If True, recompute even if a cached file exists.

    Returns
    -------
    np.ndarray
        Float32 (768, 1024) saliency map, values in [0, 1].

    Raises
    ------
    FileNotFoundError
        If no image file with a supported extension is found for stim_id.
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

    Parameters
    ----------
    image_dir : Path, optional
    cache_dir : Path, optional
    stim_ids : list of str, optional
        If None, uses all StimIDs found in stimuli_dataset.json.
    force_recompute : bool
        Recompute even if cached files already exist.

    Returns
    -------
    dict : {stim_id: np.ndarray}
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
        except FileNotFoundError as e:
            logger.warning(f"  {e}")
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
