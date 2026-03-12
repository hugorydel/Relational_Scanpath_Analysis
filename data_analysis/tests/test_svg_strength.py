"""
tests/test_svg_strength.py
===========================
Standalone diagnostic: mean relational scanpath strength (SVG z-score) with
95% bootstrap CI, compared across encoding and decoding phases.

Four estimates:
  - Encoding SVG interactional
  - Encoding SVG all-edges
  - Decoding SVG interactional
  - Decoding SVG all-edges

Bootstrap CI accounts for the repeated-measures structure by resampling
participants (cluster bootstrap) rather than individual trials.

Output
------
  Console table of mean, SD, 95% CI per condition.
  output/analysis/svg_strength.png — point + CI plot, encoding vs decoding.

Usage
-----
    python tests/test_svg_strength.py
    python tests/test_svg_strength.py --features path/to/trial_features_all.csv
    python tests/test_svg_strength.py --output   path/to/output/analysis
    python tests/test_svg_strength.py --n-boot   5000
    python tests/test_svg_strength.py --no-plot
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import config

    _DEFAULT_FEATURES = config.OUTPUT_FEATURES_DIR / "trial_features_all.csv"
    _DEFAULT_OUTPUT = config.OUTPUT_DIR / "analysis"
except Exception:
    _DEFAULT_FEATURES = _PROJECT_ROOT / "output" / "features" / "trial_features_all.csv"
    _DEFAULT_OUTPUT = _PROJECT_ROOT / "output" / "analysis"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONDITIONS = [
    ("encoding", "svg_z", "Encoding\nSVG (core)", "#2166ac"),
    ("decoding", "svg_z", "Decoding\nSVG (core)", "#d6604d"),
]

# ---------------------------------------------------------------------------
# Bootstrap CI (cluster bootstrap over participants)
# ---------------------------------------------------------------------------


def _cluster_bootstrap_ci(
    df: pd.DataFrame,
    col: str,
    n_boot: int = 5000,
    ci: float = 95.0,
    rng: np.random.Generator = None,
) -> tuple[float, float]:
    """
    Resample participants with replacement, compute mean of `col` each time.
    Returns (ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    subjects = df["SubjectID"].unique()
    boot_means = np.empty(n_boot)

    for i in range(n_boot):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        boot_df = pd.concat(
            [df[df["SubjectID"] == s] for s in sampled_subjects],
            ignore_index=True,
        )
        boot_means[i] = boot_df[col].mean()

    alpha = (100.0 - ci) / 2.0
    return (
        float(np.percentile(boot_means, alpha)),
        float(np.percentile(boot_means, 100.0 - alpha)),
    )


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute(features_path: Path, n_boot: int = 5000) -> list[dict]:
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")

    df = pd.read_csv(features_path, dtype={"StimID": str, "SubjectID": str})
    rng = np.random.default_rng(42)
    records = []

    for phase, col, label, colour in CONDITIONS:
        sub = df[(df["Phase"] == phase) & df[col].notna()].copy()

        if "low_n" in sub.columns:
            sub = sub[~sub["low_n"]]

        vals = sub[col].values
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        n = len(vals)
        n_subj = sub["SubjectID"].nunique()

        ci_lo, ci_hi = _cluster_bootstrap_ci(sub, col, n_boot=n_boot, rng=rng)

        records.append(
            {
                "phase": phase,
                "col": col,
                "label": label,
                "colour": colour,
                "mean": mean,
                "sd": sd,
                "n_trials": n,
                "n_subj": n_subj,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
        )

    return records


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(records: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("SVG RELATIONALITY STRENGTH — mean z-score with 95% bootstrap CI")
    print("(cluster bootstrap over participants; low-n trials excluded)")
    print("=" * 72)
    print(
        f"  {'Condition':<26} {'n_trials':>8} {'n_subj':>7} "
        f"{'mean':>8} {'SD':>7} {'95% CI':>18}"
    )
    print("-" * 72)
    for r in records:
        ci_str = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        print(
            f"  {r['label'].replace(chr(10),' '):<26} {r['n_trials']:>8} "
            f"{r['n_subj']:>7} {r['mean']:>+8.3f} {r['sd']:>7.3f} {ci_str:>18}"
        )
    print("=" * 72)

    # Flag which conditions have CIs entirely above zero
    print()
    for r in records:
        above = r["ci_lo"] > 0
        below = r["ci_hi"] < 0
        status = (
            "CI entirely above 0 ✓"
            if above
            else ("CI entirely below 0" if below else "CI spans 0")
        )
        print(f"  {r['label'].replace(chr(10),' '):<26} {status}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(records: list[dict], output_path: Path) -> None:
    enc_recs = [r for r in records if r["phase"] == "encoding"]
    dec_recs = [r for r in records if r["phase"] == "decoding"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=False)

    for ax, recs, phase_label in zip(
        axes, [enc_recs, dec_recs], ["Encoding phase", "Decoding phase"]
    ):
        x_positions = np.arange(len(recs))

        for x, r in zip(x_positions, recs):
            # CI line
            ax.plot(
                [x, x],
                [r["ci_lo"], r["ci_hi"]],
                color=r["colour"],
                linewidth=2.5,
                zorder=2,
            )
            # Caps
            for y_cap in (r["ci_lo"], r["ci_hi"]):
                ax.plot(
                    x,
                    y_cap,
                    "_",
                    color=r["colour"],
                    markersize=12,
                    markeredgewidth=2,
                    zorder=3,
                )
            # Mean point
            ax.scatter(
                x,
                r["mean"],
                color=r["colour"],
                s=70,
                zorder=4,
                edgecolors="white",
                linewidths=0.8,
            )
            # Mean label
            ax.text(
                x,
                r["ci_hi"] + 0.04,
                f"{r['mean']:+.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=r["colour"],
            )

        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([r["label"] for r in recs], fontsize=9)
        ax.set_ylabel("Mean SVG z-score", fontsize=9)
        ax.set_title(phase_label, fontsize=10, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

        # Annotate n
        y_min = min(r["ci_lo"] for r in recs)
        for x, r in zip(x_positions, recs):
            ax.text(
                x,
                y_min - 0.12,
                f"n={r['n_trials']}",
                ha="center",
                va="top",
                fontsize=7,
                color="grey",
            )

    fig.suptitle(
        "Relational scanpath strength — encoding vs decoding\n"
        "Mean SVG z-score ± 95% bootstrap CI (cluster over participants)",
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Mean SVG z-score strength with 95% CI: encoding vs decoding."
    )
    parser.add_argument("--features", default=str(_DEFAULT_FEATURES))
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument(
        "--n-boot",
        type=int,
        default=5000,
        help="Number of bootstrap iterations (default 5000)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("SVG relationality strength: encoding vs decoding")
    print("=" * 60)

    records = compute(Path(args.features), n_boot=args.n_boot)
    _print_summary(records)

    if not args.no_plot:
        _plot(records, Path(args.output) / "svg_strength.png")


if __name__ == "__main__":
    main()
