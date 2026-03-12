"""
diag_salience_svg_correlation.py
=================================
Diagnostic: does svg_obs correlate with mean_salience_relational or
mean_salience_nonrelational, by phase?

If svg_obs ~ mean_salience_relational is substantial (|r| > 0.2), the
targeted salience covariate is doing work that the global mean_salience
covariate was missing. If weak, the existing covariate was sufficient.

Also reports svg_obs ~ mean_salience (global) for comparison.

Output: console table only. No figure needed for a diagnostic.

Usage:
    python diag_salience_svg_correlation.py
    python diag_salience_svg_correlation.py --features path/to/trial_features_all.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import config

    _DEFAULT_FEATURES = config.OUTPUT_FEATURES_DIR / "trial_features_all.csv"
except Exception:
    _DEFAULT_FEATURES = _PROJECT_ROOT / "output" / "features" / "trial_features_all.csv"

PAIRS = [
    (
        "svg_obs",
        "mean_salience_relational",
        "SVG obs ~ salience at relational fixations",
    ),
    (
        "svg_obs",
        "mean_salience_nonrelational",
        "SVG obs ~ salience at non-relational fixations",
    ),
    ("svg_obs", "mean_salience", "SVG obs ~ mean salience (global, for reference)"),
]


def _corr(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    n = mask.sum()
    if n < 5:
        return np.nan, np.nan, n
    r, p = pearsonr(x[mask], y[mask])
    return r, p, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(_DEFAULT_FEATURES))
    args = parser.parse_args()

    df = pd.read_csv(args.features, dtype={"StimID": str, "SubjectID": str})

    print("\n" + "=" * 72)
    print("SALIENCE × SVG CORRELATION DIAGNOSTIC")
    print("=" * 72)
    print(f"  {'Phase':<10} {'Comparison':<50} {'r':>7} {'p':>8} {'n':>5}  note")
    print("-" * 72)

    for phase in ["encoding", "decoding"]:
        sub = df[df["Phase"] == phase]
        # exclude low_n trials
        if "low_n" in sub.columns:
            sub = sub[~sub["low_n"]]

        for x_col, y_col, label in PAIRS:
            if x_col not in sub.columns or y_col not in sub.columns:
                print(f"  {phase:<10} {label:<50}  [column missing]")
                continue
            r, p, n = _corr(sub[x_col].values, sub[y_col].values)
            if np.isnan(r):
                print(f"  {phase:<10} {label:<50}  [insufficient data]")
                continue
            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            note = "⚠ substantial" if abs(r) > 0.2 else ""
            print(
                f"  {phase:<10} {label:<50} {r:>+7.3f} {p:>8.4f} {n:>5}  {sig}  {note}"
            )

        print()

    print("=" * 72)
    print("  Interpretation:")
    print("  |r| > 0.2 → targeted salience covariate is doing real work")
    print("  |r| < 0.2 → global mean_salience covariate was likely sufficient")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
