"""
misc.py
=======
Miscellaneous shared utilities used across pipeline modules.

Contents:
  - Logging setup
  - Subject ID discovery
  - Output directory initialisation
  - Safe CSV reader (StimID preserved as string)
"""

import logging
from pathlib import Path
from typing import Optional

import config
import pandas as pd

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
        force=True,
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
    """
    input_dir = input_dir or config.DATA_BEHAVIORAL_DIR
    ids = sorted(p.stem for p in Path(input_dir).glob(f"*{extension}"))
    return ids


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


def read_csv_with_stim_id(filepath: Path, **kwargs) -> pd.DataFrame:
    """
    pd.read_csv wrapper that always reads StimID as string, preventing
    pandas from silently coercing image IDs to integers.
    """
    dtype = kwargs.pop("dtype", {})
    dtype["StimID"] = str
    return pd.read_csv(filepath, dtype=dtype, **kwargs)
