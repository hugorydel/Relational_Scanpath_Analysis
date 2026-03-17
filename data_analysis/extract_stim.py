"""
extract_stim.py
===============
Extract per-stimulus data for manual codebook review.

Given a StimID, writes two files to output/extract/:

  (1) {StimID}_recalled.csv
      Rows from recall_scores.csv where StimID matches and recalled == 1.

  (2) {StimID}_encoding.csv
      Rows from trial_features_all.csv where StimID matches and Phase == encoding,
      trimmed to: SubjectID, StimID, Phase, svg_z, svg_obs, svg_null_mean, svg_null_std

Usage
-----
    python extract_stim.py 150472
    python extract_stim.py 150472 --recall path/to/recall_scores.csv
    python extract_stim.py 150472 --features path/to/trial_features_all.csv
    python extract_stim.py 150472 --output-dir path/to/output/
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths — adjust to match your project layout
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config

    DEFAULT_RECALL_PATH = config.OUTPUT_DIR / "scoring" / "recall_scores.csv"
    DEFAULT_FEATURES_PATH = config.OUTPUT_FEATURES_DIR / "trial_features_all.csv"
    DEFAULT_OUTPUT_DIR = config.OUTPUT_DIR / "extract"
except Exception:
    DEFAULT_RECALL_PATH = Path("output/scoring/recall_scores.csv")
    DEFAULT_FEATURES_PATH = Path("output/features/trial_features_all.csv")
    DEFAULT_OUTPUT_DIR = Path("output/extract")

try:
    DEFAULT_BEHAVIORAL_DIR = config.OUTPUT_DIR / "behavioral"
except Exception:
    DEFAULT_BEHAVIORAL_DIR = Path("output/behavioral")

ENCODING_COLS = [
    "SubjectID",
    "StimID",
    "Phase",
    "svg_z",
    "svg_obs",
    "svg_null_mean",
    "svg_null_std",
]


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def extract_recalled(recall_path: Path, stim_id: str) -> "pd.DataFrame":
    import pandas as pd

    df = pd.read_csv(recall_path, dtype={"StimID": str, "SubjectID": str})
    subset = df[(df["StimID"] == stim_id) & (df["recalled"] == 1)].copy()
    logger.info(
        f"  recall_scores:      {len(df)} total rows → {len(subset)} recalled=1 for StimID={stim_id}"
    )
    return subset


def extract_encoding(features_path: Path, stim_id: str) -> "pd.DataFrame":
    import pandas as pd

    df = pd.read_csv(features_path, dtype={"StimID": str, "SubjectID": str})
    subset = df[(df["StimID"] == stim_id) & (df["Phase"] == "encoding")].copy()

    # Keep only requested columns (skip any that are missing with a warning)
    missing = [c for c in ENCODING_COLS if c not in subset.columns]
    if missing:
        logger.warning(f"  Columns not found in features file, skipping: {missing}")
    keep = [c for c in ENCODING_COLS if c in subset.columns]
    subset = subset[keep]

    logger.info(
        f"  trial_features_all: {len(df)} total rows → {len(subset)} encoding rows for StimID={stim_id}"
    )
    return subset


def extract_responses(behavioral_dir: Path, stim_id: str) -> "pd.DataFrame":
    import pandas as pd

    pattern = "Encode-Decode_Experiment-*-1_decoding.csv"
    files = sorted(behavioral_dir.glob(pattern))
    if not files:
        logger.warning(
            f"  No decoding CSVs found in {behavioral_dir} matching {pattern}"
        )
        return pd.DataFrame()

    chunks = []
    for f in files:
        df = pd.read_csv(f, dtype={"StimID": str, "SubjectID": str})
        subset = df[df["StimID"] == stim_id].copy()
        if not subset.empty:
            # Attach participant file source in case SubjectID isn't in the CSV
            if "source_file" not in subset.columns:
                subset.insert(0, "source_file", f.name)
            chunks.append(subset)

    if not chunks:
        logger.warning(f"  No decoding responses found for StimID={stim_id}")
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True)
    logger.info(
        f"  behavioral CSVs:    {len(files)} files scanned → "
        f"{len(combined)} response rows for StimID={stim_id}"
    )
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(
        description="Extract per-stimulus recall and encoding data."
    )
    parser.add_argument("stim_id", help="StimID to extract (e.g. 150472).")
    parser.add_argument(
        "--recall",
        default=str(DEFAULT_RECALL_PATH),
        help="Path to recall_scores.csv.",
    )
    parser.add_argument(
        "--features",
        default=str(DEFAULT_FEATURES_PATH),
        help="Path to trial_features_all.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--behavioral",
        default=str(DEFAULT_BEHAVIORAL_DIR),
        help="Directory containing per-participant decoding CSVs.",
    )
    args = parser.parse_args()

    stim_id = str(args.stim_id)
    recall_path = Path(args.recall)
    feat_path = Path(args.features)
    behavioral_dir = Path(args.behavioral)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"StimID: {stim_id}")

    # (1) Recalled nodes
    if not recall_path.exists():
        logger.error(f"recall_scores.csv not found: {recall_path}")
        sys.exit(1)
    recalled_df = extract_recalled(recall_path, stim_id)
    out_recall = output_dir / f"{stim_id}_recalled.csv"
    recalled_df.to_csv(out_recall, index=False)
    logger.info(f"  Written → {out_recall}")

    # (2) Encoding SVG features
    if not feat_path.exists():
        logger.error(f"trial_features_all.csv not found: {feat_path}")
        sys.exit(1)
    encoding_df = extract_encoding(feat_path, stim_id)
    out_enc = output_dir / f"{stim_id}_encoding.csv"
    encoding_df.to_csv(out_enc, index=False)
    logger.info(f"  Written → {out_enc}")

    # (3) Free-text decoding responses
    responses_df = extract_responses(behavioral_dir, stim_id)
    out_resp = output_dir / f"{stim_id}_responses.csv"
    responses_df.to_csv(out_resp, index=False)
    logger.info(f"  Written → {out_resp}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
