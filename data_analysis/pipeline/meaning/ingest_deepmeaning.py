"""
pipeline/meaning/ingest_deepmeaning.py
=======================================
One-time conversion script: ingests DeepMeaning .npz output maps into the
pipeline's meaning map cache as .npy files.

Run this AFTER batch_deep_meaning.py has finished generating maps for all
stimuli. Reads from
    third_party/deepmeaning/data/create_meaning_maps/output/{indoor,outdoor}/coca/
(the 'coca' subfolder is DeepMeaning's own convention for organising output
by embedding model — see predict_attention.py for the same pattern), applies
Gaussian smoothing and min-max normalisation (matching the processing in
meaning.py), and writes to output/meaning_maps/{StimID}.npy.

Once complete, the pipeline's get_meaning_map() will find the cached files
and load them directly without needing to run the CLIP-based fallback.

Usage:
    python -m pipeline.meaning.ingest_deepmeaning
    python -m pipeline.meaning.ingest_deepmeaning --force   # overwrite existing
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# Must match _SMOOTH_SIGMA in meaning.py so maps are on the same scale
# as anything the CLIP fallback would have produced.
_SMOOTH_SIGMA = 48

# data_analysis/ — this file lives at:
#   data_analysis/pipeline/meaning/ingest_deepmeaning.py
# data_analysis/ is three levels up. third_party/ lives inside data_analysis/.
_DATA_ANALYSIS_ROOT = Path(__file__).resolve().parent.parent.parent

_DEEPMEANING_OUTPUT = (
    _DATA_ANALYSIS_ROOT
    / "third_party"
    / "deepmeaning"
    / "data"
    / "create_meaning_maps"
    / "output"
)

_PIPELINE_CACHE = config.OUTPUT_DIR / "meaning_maps"


def _process_npz(npz_path: Path, force: bool) -> tuple[str, bool]:
    """
    Load one DeepMeaning .npz, smooth, normalise, write .npy.

    Returns (stim_id, was_written).
    """
    stim_id = npz_path.stem
    out_path = _PIPELINE_CACHE / f"{stim_id}.npy"

    if out_path.exists() and not force:
        logger.info(
            f"  {stim_id}: already cached — skipping (use --force to overwrite)"
        )
        return stim_id, False

    raw = np.load(npz_path)["array"].astype(np.float32)

    # Gaussian smooth — DeepMeaning outputs are unsmoothed by default
    smoothed = gaussian_filter(raw, sigma=_SMOOTH_SIGMA)

    # Min-max normalise to [0, 1]
    m_min, m_max = float(smoothed.min()), float(smoothed.max())
    if m_max - m_min > 0:
        normalised = (smoothed - m_min) / (m_max - m_min)
    else:
        normalised = np.zeros_like(smoothed)

    np.save(out_path, normalised.astype(np.float32))
    logger.info(
        f"  {stim_id}: written  "
        f"(mean={normalised.mean():.4f}, shape={normalised.shape})"
    )
    return stim_id, True


def ingest(force: bool = False) -> dict[str, np.ndarray]:
    """
    Process all .npz files in the DeepMeaning output folder (indoor + outdoor).

    Looks in the 'coca' subfolder of each category — DeepMeaning's own
    convention for organising output by embedding model (see
    predict_attention.py's CAT_deep_dir + category + '/coca/' pattern).

    Returns
    -------
    dict[str, np.ndarray]  — {stim_id: meaning_map} for all successfully processed maps
    """
    _PIPELINE_CACHE.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(
        list(_DEEPMEANING_OUTPUT.glob("indoor/coca/*.npz"))
        + list(_DEEPMEANING_OUTPUT.glob("outdoor/coca/*.npz"))
    )

    if not npz_files:
        logger.error(
            f"No .npz files found in {_DEEPMEANING_OUTPUT}/{{indoor,outdoor}}/coca/.\n"
            "Run batch_deep_meaning.py first, then re-run this script."
        )
        return {}

    logger.info(f"Found {len(npz_files)} DeepMeaning maps to ingest.")
    logger.info(f"  Source : {_DEEPMEANING_OUTPUT}")
    logger.info(f"  Target : {_PIPELINE_CACHE}")

    results = {}
    n_written, n_skipped, errors = 0, 0, []

    for npz_path in npz_files:
        try:
            stim_id, written = _process_npz(npz_path, force)
            results[stim_id] = np.load(_PIPELINE_CACHE / f"{stim_id}.npy")
            if written:
                n_written += 1
            else:
                n_skipped += 1
        except Exception as e:
            logger.warning(f"  {npz_path.stem}: FAILED — {e}")
            errors.append(npz_path.stem)

    logger.info(
        f"\nIngestion complete: {n_written} written, {n_skipped} skipped, "
        f"{len(errors)} failed."
    )
    if errors:
        logger.warning(f"  Failed StimIDs: {errors}")

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Ingest DeepMeaning .npz maps into the pipeline cache as .npy files. "
            "Run after batch_deep_meaning.py has finished."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .npy files in the cache.",
    )
    args = parser.parse_args()

    maps = ingest(force=args.force)

    if maps:
        print(f"\nSanity check ({len(maps)} maps ingested):")
        for stim_id, m in list(maps.items())[:5]:
            print(
                f"  {stim_id}: shape={m.shape}  "
                f"min={m.min():.4f}  max={m.max():.4f}  "
                f"mean={m.mean():.4f}"
            )
        if len(maps) > 5:
            print(f"  ... ({len(maps) - 5} more)")


if __name__ == "__main__":
    main()
