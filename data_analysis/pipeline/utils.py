"""
utils.py
========
Shared utilities used across pipeline modules.

Contents:
  - Logging setup
  - Subject ID discovery
  - Stimulus metadata loader (cached — loads stimuli_dataset.json once)
  - Output directory initialisation
"""

import functools
import json
import logging
from pathlib import Path
from typing import Optional

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(level: Optional[str] = None, logfile: Optional[Path] = None):
    """
    Configure root logger. Call once from run_pipeline.py or a test.

    Parameters
    ----------
    level : str, optional
        Override config.LOG_LEVEL (e.g. 'DEBUG' during development).
    logfile : Path, optional
        If provided, also write logs to this file in addition to stdout.
    """
    level = level or config.LOG_LEVEL

    handlers = [logging.StreamHandler()]
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATEFMT,
        handlers=handlers,
        force=True,  # re-apply if called more than once (e.g. in tests)
    )


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------


def get_subject_ids(
    input_dir: Optional[Path] = None,
    extension: str = ".txt",
) -> list[str]:
    """
    Return a sorted list of subject IDs found in input_dir by scanning for
    files with the given extension. Subject ID = filename stem.

    Parameters
    ----------
    input_dir : Path, optional
        Defaults to config.DATA_BEHAVIORAL_DIR.
    extension : str
        File extension to scan for (default '.txt').
    """
    input_dir = input_dir or config.DATA_BEHAVIORAL_DIR
    ids = sorted(p.stem for p in Path(input_dir).glob(f"*{extension}"))
    return ids


# ---------------------------------------------------------------------------
# Stimulus metadata loader
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def load_stimulus_metadata(metadata_file: Optional[Path] = None) -> dict:
    """
    Load stimuli_dataset.json and return a dict keyed by image_id (string).

    The result is cached after the first call so all modules share one copy
    in memory rather than each loading the file independently.

    Returns
    -------
    dict
        {
            "2383555": {
                "image_id": "2383555",
                "objects": [...],
                "relations": [...],
                ...
            },
            ...
        }
    """
    metadata_file = metadata_file or config.METADATA_FILE
    metadata_file = Path(metadata_file)

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    logger = logging.getLogger(__name__)
    logger.info(f"Loading stimulus metadata from {metadata_file.name} ...")

    with metadata_file.open(encoding="utf-8") as f:
        raw = json.load(f)

    # Re-key by image_id for O(1) lookups downstream
    images = raw.get("images", [])
    keyed = {str(img["image_id"]): img for img in images}

    logger.info(f"  Loaded metadata for {len(keyed)} stimuli.")
    return keyed


# ---------------------------------------------------------------------------
# Output directory initialisation
# ---------------------------------------------------------------------------


def init_output_dirs():
    """
    Create all expected output subdirectories if they don't exist.
    Called once at pipeline start.
    """
    for d in [
        config.OUTPUT_DIR,
        config.OUTPUT_BEHAVIORAL_DIR,
        config.OUTPUT_EYETRACKING_DIR,
        config.OUTPUT_FEATURES_DIR,
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Convenience: safe CSV read with StimID preserved as string
# ---------------------------------------------------------------------------

import pandas as pd


def read_csv_with_stim_id(filepath: Path, **kwargs) -> pd.DataFrame:
    """
    pd.read_csv wrapper that always reads StimID as string, preventing
    pandas from silently coercing image IDs to integers.
    """
    dtype = kwargs.pop("dtype", {})
    dtype["StimID"] = str
    return pd.read_csv(filepath, dtype=dtype, **kwargs)
