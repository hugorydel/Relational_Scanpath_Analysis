"""
visualize_scanpath.py
=====================
Generates a per-trial scanpath overlay (transparent background, mono-red)
for a given stimulus.

Fixations are drawn as red bubbles whose radius scales with fixation
duration (area ≈ proportional to duration, absolute mapping 80–600 ms →
10–50 px so sizes are comparable across trials). Saccades are drawn as
red arrows connecting consecutive bubble edges. Only the first 10
fixations of the trial are shown. Bubbles are numbered to indicate
temporal order. No image background and no title — intended as a
figure-overlay PNG.

Output:
    output/visualized_scanpaths/{stim_id}/{sub}_{phase}_trial{trial_id}_transparent.png

Usage
-----
    # Random participant, first encoding viewing of stim 2367132
    python visualize_scanpath.py --stim 2367132

    # Specific participant
    python visualize_scanpath.py --stim 2367132 --subject sub05

    # Decoding (recall) trial instead of encoding
    python visualize_scanpath.py --stim 2367132 --phase decoding

    # Reproducible random selection
    python visualize_scanpath.py --stim 2367132 --seed 42

    # Every participant who has that stim+phase
    python visualize_scanpath.py --stim 2367132 --all
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import config

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bubble-radius mapping (absolute, so plots are comparable across trials)
# ---------------------------------------------------------------------------

_BUBBLE_RADIUS_MIN_PX = 10.0
_BUBBLE_RADIUS_MAX_PX = 50.0
_DURATION_MIN_MS = 80.0  # matches config.MIN_FIXATION_DURATION_MS
_DURATION_MAX_MS = 600.0  # clip cap — fixations longer than this look the same

# Visual style
_MAX_FIXATIONS = 10  # only the first N fixations are drawn
_RED_FILL = "#d62728"  # bubble fill + saccade line
_RED_EDGE = "#8b1a1a"  # bubble outline (darker red for definition)


def _duration_to_radius(durations: np.ndarray) -> np.ndarray:
    """
    Map fixation durations (ms) → bubble radii (image pixels).

    Uses sqrt scaling so that bubble *area* is approximately proportional to
    duration. Absolute (not per-trial) so radii are meaningful across plots.
    """
    if len(durations) == 0:
        return np.array([])
    d = np.clip(durations.astype(float), _DURATION_MIN_MS, _DURATION_MAX_MS)
    sq = np.sqrt(d - _DURATION_MIN_MS)
    sq_max = np.sqrt(_DURATION_MAX_MS - _DURATION_MIN_MS)
    return _BUBBLE_RADIUS_MIN_PX + (sq / sq_max) * (
        _BUBBLE_RADIUS_MAX_PX - _BUBBLE_RADIUS_MIN_PX
    )


# ---------------------------------------------------------------------------
# Screen → image coordinate transform
# ---------------------------------------------------------------------------

_SCALE_X = config.IMAGE_W / config.DISPLAY_WIDTH_PX
_SCALE_Y = config.IMAGE_H / config.DISPLAY_HEIGHT_PX


def _screen_to_image(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x * _SCALE_X, y * _SCALE_Y


# ---------------------------------------------------------------------------
# Saccade segment trimming
# ---------------------------------------------------------------------------


def _trim_segment(
    x1: float,
    y1: float,
    r1: float,
    x2: float,
    y2: float,
    r2: float,
) -> tuple[float, float, float, float] | None:
    """
    Trim line A→B so it starts on the edge of bubble A and ends on the edge
    of bubble B. Returns (sx, sy, ex, ey) in image pixels, or None if the
    two bubbles overlap (in which case no meaningful arrow can be drawn).
    """
    dx, dy = x2 - x1, y2 - y1
    d = float(np.hypot(dx, dy))
    if d <= r1 + r2:
        return None
    ux, uy = dx / d, dy / d
    return (x1 + r1 * ux, y1 + r1 * uy, x2 - r2 * ux, y2 - r2 * uy)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _discover_participants(eyetracking_dir: Path) -> list[str]:
    files = sorted(eyetracking_dir.glob("*_fixations.csv"))
    return [f.stem.replace("_fixations", "") for f in files]


def _load_subject_fixations(subject_id: str, eyetracking_dir: Path) -> pd.DataFrame:
    path = eyetracking_dir / f"{subject_id}_fixations.csv"
    return pd.read_csv(path, dtype={"StimID": str})


def _select_first_trial(
    fix_df: pd.DataFrame, stim_id: str, phase: str
) -> tuple[int, pd.DataFrame]:
    """
    Return (TrialID, fixations_for_that_trial) for the first-viewing trial.
    First viewing = lowest TrialID for this (StimID, Phase) combination.
    """
    sub = fix_df[(fix_df["StimID"] == stim_id) & (fix_df["Phase"] == phase)]
    if sub.empty:
        raise ValueError(f"no fixations for StimID={stim_id} Phase={phase}")
    trial_id = int(sub["TrialID"].min())
    trial_fix = (
        sub[sub["TrialID"] == trial_id]
        .sort_values("FixStart_ms")
        .reset_index(drop=True)
    )
    return trial_id, trial_fix


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot_scanpath(
    fix_df: pd.DataFrame,
    stim_id: str,
    subject_id: str,
    phase: str,
    trial_id: int,
    out_path: Path,
    dpi: int,
) -> None:
    # Cap to first N fixations (input is already time-sorted)
    fix_df = fix_df.head(_MAX_FIXATIONS)
    n_fix = len(fix_df)

    if n_fix == 0:
        logger.warning(f"  [{subject_id}] trial has 0 fixations — skipping")
        return

    # Canvas matches image space (so coordinates align if overlaid on the stim)
    w, h = config.IMAGE_W, config.IMAGE_H
    fig_w = 10.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * h / w), dpi=dpi)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    img_x, img_y = _screen_to_image(
        fix_df["GazeX"].to_numpy(), fix_df["GazeY"].to_numpy()
    )
    radii = _duration_to_radius(fix_df["Duration_ms"].to_numpy())

    # ── Saccades (below bubbles) ──────────────────────────────────────────
    for i in range(n_fix - 1):
        seg = _trim_segment(
            img_x[i],
            img_y[i],
            radii[i],
            img_x[i + 1],
            img_y[i + 1],
            radii[i + 1],
        )
        if seg is None:
            continue
        sx, sy, ex, ey = seg
        ax.annotate(
            "",
            xy=(ex, ey),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=_RED_FILL,
                lw=2.0,
                mutation_scale=16,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=3,
        )

    # ── Bubbles ───────────────────────────────────────────────────────────
    for i in range(n_fix):
        circ = mpatches.Circle(
            (img_x[i], img_y[i]),
            radius=radii[i],
            facecolor=_RED_FILL,
            edgecolor=_RED_EDGE,
            linewidth=1.4,
            alpha=0.92,
            zorder=5,
        )
        ax.add_patch(circ)

        # Order label inside bubble
        font_size = max(7.0, min(13.0, radii[i] * 0.36))
        ax.text(
            img_x[i],
            img_y[i],
            str(i + 1),
            ha="center",
            va="center",
            fontsize=font_size,
            color="white",
            fontweight="bold",
            zorder=6,
            path_effects=[pe.withStroke(linewidth=1.6, foreground="black")],
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Per-trial scanpath visualizer for a given stimulus."
    )
    parser.add_argument("--stim", required=True, help="StimID to visualise.")
    parser.add_argument(
        "--phase",
        choices=["encoding", "decoding"],
        default="encoding",
        help="Trial phase (default: encoding — first viewing).",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Specific subject ID. If omitted, picks a random participant.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render every participant who has that stim+phase (overrides --subject).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for random participant selection (reproducibility).",
    )
    parser.add_argument(
        "--eyetracking-dir",
        default=str(config.OUTPUT_EYETRACKING_DIR),
        help="Directory containing *_fixations.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.OUTPUT_DIR / "visualized_scanpaths"),
        help="Root output directory.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    eyetracking_dir = Path(args.eyetracking_dir)
    output_root = Path(args.output_dir) / args.stim

    logger.info("=" * 60)
    logger.info("Scanpath visualizer")
    logger.info(f"  StimID    : {args.stim}")
    logger.info(f"  Phase     : {args.phase}")
    logger.info(f"  Source    : {eyetracking_dir}")
    logger.info(f"  Output    : {output_root}")
    logger.info("=" * 60)

    # ── Participant selection ────────────────────────────────────────────
    all_participants = _discover_participants(eyetracking_dir)
    if not all_participants:
        logger.error(f"No *_fixations.csv files in {eyetracking_dir}")
        sys.exit(1)

    if args.all:
        participants = all_participants
    elif args.subject:
        if args.subject not in all_participants:
            logger.error(
                f"Subject {args.subject} not found in {eyetracking_dir} — "
                f"available: {all_participants}"
            )
            sys.exit(1)
        participants = [args.subject]
    else:
        # Filter to participants who actually have data for this stim+phase
        candidates = []
        for sid in all_participants:
            fix = _load_subject_fixations(sid, eyetracking_dir)
            mask = (fix["StimID"] == args.stim) & (fix["Phase"] == args.phase)
            if mask.any():
                candidates.append(sid)
        if not candidates:
            logger.error(
                f"No participant has fixations for StimID={args.stim} Phase={args.phase}"
            )
            sys.exit(1)
        rng = random.Random(args.seed)
        chosen = rng.choice(candidates)
        logger.info(f"  Randomly selected : {chosen} (of {len(candidates)} candidates)")
        participants = [chosen]

    # ── Render ────────────────────────────────────────────────────────────
    n_ok = n_skip = 0
    for sid in participants:
        try:
            fix_df = _load_subject_fixations(sid, eyetracking_dir)
            trial_id, trial_fix = _select_first_trial(fix_df, args.stim, args.phase)
        except ValueError as e:
            logger.warning(f"  [{sid}] {e}")
            n_skip += 1
            continue

        out_path = output_root / f"{sid}_{args.phase}_trial{trial_id}_transparent.png"
        _plot_scanpath(
            trial_fix,
            args.stim,
            sid,
            args.phase,
            trial_id,
            out_path,
            dpi=args.dpi,
        )
        logger.info(
            f"  [{sid}] trial {trial_id}: {min(len(trial_fix), _MAX_FIXATIONS)} fixations drawn "
            f"(of {len(trial_fix)} total) → {out_path.name}"
        )
        n_ok += 1

    logger.info("=" * 60)
    logger.info(f"Done.  ok={n_ok}  skipped={n_skip}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
