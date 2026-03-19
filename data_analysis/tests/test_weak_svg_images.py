"""
test_weak_svg_images.py
=======================
Identifies images with the weakest within-image SVG → memory correlations.

For each StimID, computes Pearson r between svg_z_enc_within and each
proportion DV across participants. Ranks images from weakest to strongest
and writes a sorted CSV + console summary.

Useful for diagnosing:
  - Images where relational scanning doesn't drive memory (low relational density?)
  - Images that may be pulling down the pooled LMM effect
  - Candidates for qualitative follow-up

Usage
-----
    python tests/test_weak_svg_images.py
    python tests/test_weak_svg_images.py --input path/to/analysis_enc.csv
    python tests/test_weak_svg_images.py --dv prop_total         # focus on one DV
    python tests/test_weak_svg_images.py --top 10                # show N weakest
    python tests/test_weak_svg_images.py --min-n 8               # minimum participants per image
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent      # tests/
_ROOT = _HERE.parent                         # data_analysis/
sys.path.insert(0, str(_ROOT))

try:
    import config
    DEFAULT_INPUT  = config.OUTPUT_DIR / "analysis" / "figure_data" / "analysis_enc.csv"
    DEFAULT_OUTPUT = config.OUTPUT_DIR / "analysis"
except Exception:
    DEFAULT_INPUT  = _ROOT / "output" / "analysis" / "figure_data" / "analysis_enc.csv"
    DEFAULT_OUTPUT = _ROOT / "output" / "analysis"

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

DVS = {
    "prop_total":     "Total recall",
    "prop_relations": "Relational recall",
    "prop_objects":   "Object recall",
}

PREDICTOR = "svg_z_enc_within"


def compute_per_image_corrs(
    df: pd.DataFrame,
    dv_cols: list[str],
    min_n: int,
) -> pd.DataFrame:
    records = []
    for stim_id, grp in df.groupby("StimID"):
        row = {"StimID": stim_id, "n_participants": grp[PREDICTOR].notna().sum()}
        for dv_col in dv_cols:
            valid = grp[[PREDICTOR, dv_col]].dropna()
            n = len(valid)
            if n < min_n:
                row[f"r_{dv_col}"] = np.nan
                row[f"p_{dv_col}"] = np.nan
                row[f"n_{dv_col}"] = n
            else:
                r, p = stats.pearsonr(valid[PREDICTOR], valid[dv_col])
                row[f"r_{dv_col}"] = round(r, 4)
                row[f"p_{dv_col}"] = round(p, 4)
                row[f"n_{dv_col}"] = n
        records.append(row)

    return pd.DataFrame(records)


def rank_by_weakness(corr_df: pd.DataFrame, dv_col: str) -> pd.DataFrame:
    """Sort by r ascending (weakest/most negative first)."""
    r_col = f"r_{dv_col}"
    return (
        corr_df.dropna(subset=[r_col])
        .sort_values(r_col, ascending=True)
        .reset_index(drop=True)
    )


def print_summary(corr_df: pd.DataFrame, dv_cols: list[str], top_n: int) -> None:
    print(f"\n{'='*65}")
    print(f"  Weakest SVG → memory correlations  (top {top_n} per DV)")
    print(f"{'='*65}")

    for dv_col in dv_cols:
        label = DVS.get(dv_col, dv_col)
        ranked = rank_by_weakness(corr_df, dv_col)
        n_negative = (ranked[f"r_{dv_col}"] < 0).sum()
        mean_r = ranked[f"r_{dv_col}"].mean()

        print(f"\n  {label}  (mean r = {mean_r:+.3f}, {n_negative}/{len(ranked)} negative)")
        print(f"  {'StimID':<12} {'r':>7}  {'p':>7}  {'n':>4}")
        print(f"  {'-'*36}")

        for _, row in ranked.head(top_n).iterrows():
            r   = row[f"r_{dv_col}"]
            p   = row[f"p_{dv_col}"]
            n   = int(row[f"n_{dv_col}"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {str(row['StimID']):<12} {r:>+7.3f}  {sig:>7}  {n:>4}")

    print(f"\n{'='*65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank images by weakest SVG → memory correlation."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to analysis_enc.csv (Module 4 output).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Directory for output CSV.",
    )
    parser.add_argument(
        "--dv",
        default=None,
        choices=list(DVS.keys()),
        help="Focus on one DV (default: all three).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of weakest images to show per DV (default: 10).",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=6,
        help="Minimum participants per image to include (default: 6).",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output_dir)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path, dtype={"StimID": str, "SubjectID": str})

    # Exclude low_n trials if present
    if "low_n_enc" in df.columns:
        before = len(df)
        df = df[~df["low_n_enc"]].copy()
        logger.info(f"Excluded {before - len(df)} low_n trials.")

    if PREDICTOR not in df.columns:
        logger.error(
            f"Column '{PREDICTOR}' not found. "
            "Run Module 4 with the within-image decomposition first."
        )
        sys.exit(1)

    dv_cols = [args.dv] if args.dv else list(DVS.keys())
    missing = [c for c in dv_cols if c not in df.columns]
    if missing:
        logger.error(f"DV columns not found in input: {missing}")
        sys.exit(1)

    logger.info(
        f"Loaded {len(df)} rows, "
        f"{df['SubjectID'].nunique()} participants, "
        f"{df['StimID'].nunique()} images."
    )

    corr_df = compute_per_image_corrs(df, dv_cols, min_n=args.min_n)

    print_summary(corr_df, dv_cols, top_n=args.top)

    # Write full ranked output for each DV
    output_dir.mkdir(parents=True, exist_ok=True)
    for dv_col in dv_cols:
        ranked = rank_by_weakness(corr_df, dv_col)
        out_path = output_dir / f"weak_images_{dv_col}.csv"
        ranked.to_csv(out_path, index=False)
        logger.info(f"Written → {out_path.name}")


if __name__ == "__main__":
    main()