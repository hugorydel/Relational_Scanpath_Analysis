"""
pipeline/meaning/meaning.py
===========================
CLIP-based meaning maps (cf. Henderson & Hayes 2017; Hayes & Henderson 2025).

Computes one meaning map per stimulus image and caches the result to
output/meaning_maps/{StimID}.npy so it is only computed once.

Algorithm:
    Following the meaning-map framework of Henderson & Hayes (2017) and the
    DeepMeaning model of Hayes & Henderson (2025), each image is divided into
    overlapping patches at two spatial scales. Each patch is encoded with a
    CLIP ViT-L/14 vision encoder and scored by cosine similarity to a text
    embedding of a semantic meaningfulness prompt. Patch scores are accumulated
    over overlapping regions, averaged, Gaussian-smoothed, and min-max
    normalised to [0, 1].

    CLIP ViT-L/14 weights are downloaded automatically via open-clip-torch on
    first run (~1.7 GB, cached to ~/.cache/huggingface/). On CPU-only machines
    computation is ~1–3 min/image; pre-compute all maps once and rely on the
    .npy cache thereafter.

Installation:
    pip install open-clip-torch Pillow scipy

Usage (standalone):
    # All stimuli in stimuli_dataset.json
    python -m pipeline.meaning.meaning

    # Specific subset
    python -m pipeline.meaning.meaning --stim-ids 2383555 2386442

    # Force recompute even if cache exists
    python -m pipeline.meaning.meaning --force

Usage (from Module 3):
    from pipeline.meaning.meaning import get_meaning_map
    meaning_map = get_meaning_map("2383555")  # np.ndarray (768, 1024), float32, in [0, 1]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
import numpy as np
from pipeline.module_3.scene_graph import load_stimulus_metadata
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Two-scale patch analysis — (patch_size_px, stride_px) in original image space.
# Following Henderson & Hayes (2017): two scales capture both fine-grained
# object detail (fine scale) and coarser region-level meaning (coarse scale).
_PATCH_SCALES = [
    (128, 32),  # fine scale  — ~9×7 grid on a 1024×768 image with full coverage
    (256, 64),  # coarse scale — broader semantic context per patch
]

# Maximum patches per forward pass (reduce if GPU OOM).
_ENCODE_BATCH_SIZE = 64

# Text prompt that defines "meaningfulness" for the CLIP scorer.
# Cosine similarity between each patch's visual embedding and this text
# embedding serves as the patch-level meaning score.
_MEANING_PROMPT = "a meaningful and recognizable region in a scene"

# Gaussian smoothing sigma applied after averaging patch scores.
# Reuses the salience config value so both maps are spatially comparable.
_SMOOTH_SIGMA = config.SALIENCE_SMOOTHING_SIGMA


# ---------------------------------------------------------------------------
# Model loader  (module-level singleton — loaded once per process)
# ---------------------------------------------------------------------------

_model = None
_preprocess = None
_text_features = None
_device = None


def _load_clip():
    """
    Load CLIP ViT-L/14 and precompute text features for _MEANING_PROMPT.
    Subsequent calls return the cached objects immediately.
    """
    global _model, _preprocess, _text_features, _device

    if _model is not None:
        return _model, _preprocess, _text_features

    try:
        import open_clip
        import torch
    except ImportError:
        raise ImportError(
            "open-clip-torch is required for meaning maps.\n"
            "Install with:  pip install open-clip-torch"
        )

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"Loading CLIP ViT-L/14 (device={_device}). " "First run downloads ~1.7 GB ..."
    )

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    _model = _model.to(_device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    with torch.no_grad():
        tokens = tokenizer([_MEANING_PROMPT]).to(_device)
        feats = _model.encode_text(tokens)
        _text_features = (feats / feats.norm(dim=-1, keepdim=True)).cpu()

    logger.info("  CLIP model ready.")
    return _model, _preprocess, _text_features


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def compute_meaning_map(image_path: Path) -> np.ndarray:
    """
    Compute a CLIP-based meaning map for one stimulus image.

    Parameters
    ----------
    image_path : Path
        Path to the stimulus image (any format readable by Pillow).

    Returns
    -------
    np.ndarray
        Float32 array of shape (IMAGE_H, IMAGE_W) = (768, 1024), values in [0, 1].
        Row-major: index as [y, x].
    """
    try:
        import torch
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow and PyTorch are required.\n"
            "Install with:  pip install Pillow torch"
        )

    model, preprocess, text_features = _load_clip()
    text_features = text_features.to(_device)

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    meaning_map = np.zeros((config.IMAGE_H, config.IMAGE_W), dtype=np.float64)
    count_map = np.zeros((config.IMAGE_H, config.IMAGE_W), dtype=np.float64)

    for patch_size, stride in _PATCH_SCALES:
        # --- collect all patch crops and their top-left positions ---
        patches = []
        positions = []  # (x, y) in original image pixels

        for y in range(0, orig_h - patch_size + 1, stride):
            for x in range(0, orig_w - patch_size + 1, stride):
                patch = img.crop((x, y, x + patch_size, y + patch_size))
                patches.append(preprocess(patch))
                positions.append((x, y))

        if not patches:
            continue

        # --- batch encode ---
        scores = []
        for i in range(0, len(patches), _ENCODE_BATCH_SIZE):
            batch_tensors = torch.stack(patches[i : i + _ENCODE_BATCH_SIZE]).to(_device)
            with torch.no_grad():
                img_feats = model.encode_image(batch_tensors)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                batch_scores = (img_feats @ text_features.T).squeeze(1)
            scores.extend(batch_scores.cpu().tolist())

        # --- splat scores onto the output map (IMAGE_W × IMAGE_H space) ---
        sx = config.IMAGE_W / orig_w
        sy = config.IMAGE_H / orig_h

        for (x, y), score in zip(positions, scores):
            x0 = int(x * sx)
            y0 = int(y * sy)
            x1 = min(int((x + patch_size) * sx), config.IMAGE_W)
            y1 = min(int((y + patch_size) * sy), config.IMAGE_H)
            meaning_map[y0:y1, x0:x1] += score
            count_map[y0:y1, x0:x1] += 1.0

    # --- average overlapping patch scores ---
    valid = count_map > 0
    meaning_map[valid] /= count_map[valid]

    # --- Gaussian smooth ---
    meaning_map = gaussian_filter(meaning_map.astype(np.float32), sigma=_SMOOTH_SIGMA)

    # --- min-max normalise to [0, 1] ---
    m_min, m_max = float(meaning_map.min()), float(meaning_map.max())
    if m_max - m_min > 0:
        meaning_map = (meaning_map - m_min) / (m_max - m_min)
    else:
        meaning_map[:] = 0.0

    return meaning_map.astype(np.float32)


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def _cache_path(stim_id: str, cache_dir: Path) -> Path:
    return cache_dir / f"{stim_id}.npy"


def get_meaning_map(
    stim_id: str,
    image_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Return the meaning map for a stimulus, loading from cache if available.

    Parameters
    ----------
    stim_id         : str   — stimulus image ID
    image_dir       : Path  — directory containing stimulus images
    cache_dir       : Path  — directory for cached .npy meaning maps
    force_recompute : bool  — recompute even if a cached file exists

    Returns
    -------
    np.ndarray — float32 (IMAGE_H, IMAGE_W) meaning map, values in [0, 1]
    """
    image_dir = Path(image_dir) if image_dir else config.DATA_METADATA_DIR / "images"
    cache_dir = Path(cache_dir) if cache_dir else config.OUTPUT_DIR / "meaning_maps"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached = _cache_path(stim_id, cache_dir)
    if cached.exists() and not force_recompute:
        return np.load(cached)

    img_path = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        candidate = image_dir / f"{stim_id}{ext}"
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        raise FileNotFoundError(
            f"No image found for StimID {stim_id} in {image_dir}. "
            "Tried: .jpg, .jpeg, .png, .bmp, .tiff"
        )

    meaning_map = compute_meaning_map(img_path)
    np.save(cached, meaning_map)
    return meaning_map


# ---------------------------------------------------------------------------
# Batch computation  (standalone pre-cache)
# ---------------------------------------------------------------------------


def compute_all_meaning_maps(
    image_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    stim_ids: Optional[list[str]] = None,
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute and cache meaning maps for all (or a subset of) stimuli.

    Returns
    -------
    dict[str, np.ndarray] — {stim_id: meaning_map}
    """
    image_dir = Path(image_dir) if image_dir else config.DATA_METADATA_DIR / "images"
    cache_dir = Path(cache_dir) if cache_dir else config.OUTPUT_DIR / "meaning_maps"

    if stim_ids is None:
        metadata = load_stimulus_metadata()
        stim_ids = list(metadata.keys())

    n_total = len(stim_ids)
    logger.info(f"Computing meaning maps for {n_total} stimuli ...")
    logger.info(f"  Image dir  : {image_dir}")
    logger.info(f"  Cache dir  : {cache_dir}")

    results = {}
    n_cached, n_new = 0, 0
    errors = []

    for i, stim_id in enumerate(stim_ids, 1):
        cached = _cache_path(stim_id, cache_dir)
        if cached.exists() and not force_recompute:
            results[stim_id] = np.load(cached)
            n_cached += 1
            logger.info(f"  [{i}/{n_total}] {stim_id}: loaded from cache")
            continue

        logger.info(f"  [{i}/{n_total}] {stim_id}: computing ...")
        try:
            meaning_map = get_meaning_map(
                stim_id,
                image_dir=image_dir,
                cache_dir=cache_dir,
                force_recompute=force_recompute,
            )
            results[stim_id] = meaning_map
            n_new += 1
            logger.info(
                f"  [{i}/{n_total}] {stim_id}: done  "
                f"(mean={meaning_map.mean():.4f})"
            )
        except Exception as e:
            logger.warning(f"  [{i}/{n_total}] {stim_id}: FAILED — {e}")
            errors.append(stim_id)

    logger.info(
        f"Meaning maps ready: {n_new} computed, {n_cached} loaded from cache, "
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
        description="Pre-compute CLIP-based meaning maps for all stimuli."
    )
    parser.add_argument(
        "--image-dir",
        default=str(config.DATA_METADATA_DIR / "images"),
        help="Directory containing stimulus images",
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.OUTPUT_DIR / "meaning_maps"),
        help="Directory to write cached .npy meaning maps",
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

    maps = compute_all_meaning_maps(
        image_dir=Path(args.image_dir),
        cache_dir=Path(args.output_dir),
        stim_ids=args.stim_ids,
        force_recompute=args.force,
    )

    print(f"\nSanity check on {len(maps)} maps:")
    for stim_id, m in list(maps.items())[:5]:
        print(
            f"  {stim_id}: shape={m.shape}  "
            f"min={m.min():.4f}  max={m.max():.4f}  "
            f"mean={m.mean():.4f}  dtype={m.dtype}"
        )
    if len(maps) > 5:
        print(f"  ... ({len(maps) - 5} more)")


if __name__ == "__main__":
    main()
