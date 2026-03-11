"""
pipeline/module4/pilot_diagnostics.py
=======================================
Pilot diagnostic plots for Module 4.

Produces a single multi-panel figure summarising all available pilot data,
regardless of whether any statistical model converges.  Call from summarise()
in output.py, or run standalone:

    python pilot_diagnostics.py \\
        --features output/features/trial_features_all.csv \\
        --scores   output/data_scoring/memory_scores.csv \\
        --out      output/analysis/pilot_diagnostics.png

Panels
------
A  Data coverage heatmap       — which (participant × stimulus) cells have
                                  enc SVG / dec SVG / memory scores
B  SVG z-score distributions   — encoding vs decoding, per participant
C  LCS (replay) distributions  — per participant
D  Memory score profile        — mean counts by content type × status,
                                  stacked bars, aggregated across participants
E  LCS → relational memory     — scatter + OLS line (best-coverage predictor)
F  Enc SVG → relational memory — scatter + OLS line (H2 signal check)
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── Colour scheme ────────────────────────────────────────────────────────────
# One colour per participant (up to 8); neutral palette that's print-safe.
_SUBJ_PALETTE = [
    "#2166ac",
    "#d6604d",
    "#4dac26",
    "#762a83",
    "#f4a582",
    "#92c5de",
    "#a6dba0",
    "#c2a5cf",
]

_CT_COLORS = {
    "object_identity": "#4393c3",
    "object_attribute": "#74add1",
    "action_relation": "#d73027",
    "spatial_relation": "#f46d43",
    "scene_gist": "#878787",
}
_STATUS_HATCHES = {
    "correct": "",
    "incorrect": "//",
    "inference": "..",
    "repeat": "xx",
}

_CONTENT_TYPES = [
    "object_identity",
    "object_attribute",
    "action_relation",
    "spatial_relation",
    "scene_gist",
]
_STATUSES = ["correct", "incorrect", "inference", "repeat"]

CT_LABELS = {
    "object_identity": "Obj. identity",
    "object_attribute": "Obj. attribute",
    "action_relation": "Action/rel.",
    "spatial_relation": "Spatial rel.",
    "scene_gist": "Scene gist",
}


# ── Data helpers ─────────────────────────────────────────────────────────────


def _derive_legacy(ms: pd.DataFrame) -> pd.DataFrame:
    """Add legacy aggregate columns to new-schema memory scores."""
    new_cols = [f"n_{ct}_{st}" for ct in _CONTENT_TYPES for st in _STATUSES]
    for c in new_cols:
        if c not in ms.columns:
            ms[c] = 0
    ms["n_relational_correct"] = (
        ms["n_action_relation_correct"] + ms["n_spatial_relation_correct"]
    )
    ms["n_relational_incorrect"] = (
        ms["n_action_relation_incorrect"] + ms["n_spatial_relation_incorrect"]
    )
    ms["n_objects_correct"] = (
        ms["n_object_identity_correct"] + ms["n_object_attribute_correct"]
    )
    ms["n_objects_incorrect"] = (
        ms["n_object_identity_incorrect"] + ms["n_object_attribute_incorrect"]
    )
    ms["n_total_correct"] = sum(ms[f"n_{ct}_correct"] for ct in _CONTENT_TYPES)
    return ms


def load_data(features_path: Path, scores_path: Path) -> tuple:
    tf = pd.read_csv(features_path, dtype={"StimID": str, "SubjectID": str})
    ms = pd.read_csv(scores_path, dtype={"StimID": str, "SubjectID": str})

    # Detect and handle new vs legacy schema
    is_new = all(
        f"n_{_CONTENT_TYPES[0]}_{_STATUSES[0]}" in ms.columns for _ in [1]
    )  # spot-check
    if is_new:
        ms = _derive_legacy(ms)

    subjects = sorted(tf["SubjectID"].unique())
    subj_colors = {
        s: _SUBJ_PALETTE[i % len(_SUBJ_PALETTE)] for i, s in enumerate(subjects)
    }

    enc = tf[tf["Phase"] == "encoding"].copy()
    dec = tf[tf["Phase"] == "decoding"].copy()
    dec_ms = dec.merge(ms, on=["SubjectID", "StimID"], how="left")

    return enc, dec, dec_ms, ms, subjects, subj_colors


# ── Panel A: data coverage heatmap ──────────────────────────────────────────


def _panel_coverage(ax, enc, dec, ms, subjects):
    all_stims = sorted(
        set(enc["StimID"].unique())
        | set(dec["StimID"].unique())
        | set(ms["StimID"].unique())
    )
    n_stim = len(all_stims)
    n_subj = len(subjects)
    stim_idx = {s: i for i, s in enumerate(all_stims)}
    subj_idx = {s: i for i, s in enumerate(subjects)}

    # Bitmask: 4 bits per cell
    # bit 0 = enc SVG, bit 1 = dec SVG, bit 2 = scored
    grid = np.zeros((n_subj, n_stim, 3), dtype=float)

    for _, row in enc.iterrows():
        if row["StimID"] in stim_idx and row["SubjectID"] in subj_idx:
            si, pi = stim_idx[row["StimID"]], subj_idx[row["SubjectID"]]
            if pd.notna(row.get("svg_z_inter", np.nan)):
                grid[pi, si, 0] = 1.0  # enc SVG present

    for _, row in dec.iterrows():
        if row["StimID"] in stim_idx and row["SubjectID"] in subj_idx:
            si, pi = stim_idx[row["StimID"]], subj_idx[row["SubjectID"]]
            if pd.notna(row.get("svg_z_inter", np.nan)):
                grid[pi, si, 1] = 1.0  # dec SVG present

    for _, row in ms.iterrows():
        if row["StimID"] in stim_idx and row["SubjectID"] in subj_idx:
            si, pi = stim_idx[row["StimID"]], subj_idx[row["SubjectID"]]
            grid[pi, si, 2] = 1.0  # memory scored

    # Combine layers into a single colour array
    # colour = blend of three translucent layers
    rgba = np.ones((n_subj, n_stim, 4)) * 0.93  # light grey background

    # enc SVG → blue tint
    enc_mask = grid[:, :, 0] == 1
    rgba[enc_mask, :3] = np.array([0.13, 0.47, 0.71])
    rgba[enc_mask, 3] = 0.5

    # dec SVG → overlay orange
    dec_mask = grid[:, :, 1] == 1
    rgba[dec_mask, :3] = np.array([0.84, 0.38, 0.12])
    rgba[dec_mask, 3] = 0.6

    # scored → overlay green
    scored_mask = grid[:, :, 2] == 1
    rgba[scored_mask, :3] = np.array([0.30, 0.68, 0.29])
    rgba[scored_mask, 3] = 0.75

    # all three → darker combined
    all3 = enc_mask & dec_mask & scored_mask
    rgba[all3, :3] = np.array([0.15, 0.35, 0.15])
    rgba[all3, 3] = 1.0

    ax.imshow(
        rgba,
        aspect="auto",
        interpolation="nearest",
        extent=[-0.5, n_stim - 0.5, -0.5, n_subj - 0.5],
    )

    short_labels = [s.split("-")[-2] + "-" + s.split("-")[-1] for s in subjects]
    ax.set_yticks(range(n_subj))
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Stimulus index", fontsize=8)
    ax.set_title("A  Data coverage", fontsize=9, fontweight="bold", loc="left")

    legend_patches = [
        mpatches.Patch(facecolor="#2178b4", alpha=0.6, label="Enc SVG"),
        mpatches.Patch(facecolor="#d66020", alpha=0.7, label="Dec SVG"),
        mpatches.Patch(facecolor="#4dac47", alpha=0.85, label="Scored"),
        mpatches.Patch(facecolor="#254a25", label="All three"),
    ]
    ax.legend(handles=legend_patches, fontsize=6, loc="upper right", framealpha=0.8)


# ── Panel B: SVG z-score distributions ──────────────────────────────────────


def _panel_svg_dist(ax, enc, dec, subjects, subj_colors):
    data_enc = [
        enc[enc["SubjectID"] == s]["svg_z_inter"].dropna().values for s in subjects
    ]
    data_dec = [
        dec[dec["SubjectID"] == s]["svg_z_inter"].dropna().values for s in subjects
    ]

    n = len(subjects)
    positions_enc = np.arange(n) * 2.2
    positions_dec = positions_enc + 0.9

    for i, s in enumerate(subjects):
        col = subj_colors[s]
        for pos, vals, alpha, lw in [
            (positions_enc[i], data_enc[i], 0.55, 1.2),
            (positions_dec[i], data_dec[i], 0.85, 1.4),
        ]:
            if len(vals) == 0:
                continue
            parts = ax.violinplot(
                [vals],
                positions=[pos],
                widths=0.75,
                showmedians=True,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(col)
                pc.set_alpha(alpha)
                pc.set_edgecolor("white")
                pc.set_linewidth(0.4)
            parts["cmedians"].set_color(col)
            parts["cmedians"].set_linewidth(lw)
            # Jitter
            jx = pos + np.random.default_rng(42).uniform(-0.2, 0.2, len(vals))
            ax.scatter(jx, vals, s=8, color=col, alpha=0.55, zorder=3, linewidths=0)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    short = [s.split("-")[-2] + "-" + s.split("-")[-1] for s in subjects]
    tick_pos = positions_enc + 0.45
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(short, fontsize=7)
    ax.set_ylabel("SVG z-score", fontsize=8)
    ax.set_title(
        "B  SVG distributions (light=enc, dark=dec)",
        fontsize=9,
        fontweight="bold",
        loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)


# ── Panel C: LCS distributions ───────────────────────────────────────────────


def _panel_lcs(ax, dec, subjects, subj_colors):
    for i, s in enumerate(subjects):
        vals = dec[dec["SubjectID"] == s]["lcs_enc_dec"].dropna().values
        if len(vals) == 0:
            continue
        col = subj_colors[s]
        label = s.split("-")[-2] + "-" + s.split("-")[-1]
        ax.hist(
            vals,
            bins=12,
            range=(0, 1),
            alpha=0.55,
            color=col,
            label=label,
            density=True,
        )
        ax.axvline(np.mean(vals), color=col, linewidth=1.5, linestyle="--", alpha=0.85)

    ax.set_xlabel("LCS (enc–dec overlap)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(
        "C  Replay (LCS) distributions", fontsize=9, fontweight="bold", loc="left"
    )
    ax.legend(fontsize=6, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


# ── Panel D: Memory profile (stacked bar by content type) ────────────────────


def _panel_memory_profile(ax, ms):
    ct_order = _CONTENT_TYPES
    status_order = ["correct", "incorrect", "inference", "repeat"]

    means = {}
    for ct in ct_order:
        for st in status_order:
            col = f"n_{ct}_{st}"
            means[(ct, st)] = ms[col].mean() if col in ms.columns else 0.0

    x = np.arange(len(ct_order))
    width = 0.18
    offsets = np.linspace(-0.27, 0.27, len(status_order))

    for j, st in enumerate(status_order):
        heights = [means[(ct, st)] for ct in ct_order]
        bars = ax.bar(
            x + offsets[j],
            heights,
            width,
            label=st.capitalize(),
            hatch=_STATUS_HATCHES[st],
            color=[_CT_COLORS[ct] for ct in ct_order],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CT_LABELS[ct] for ct in ct_order], fontsize=7, rotation=20, ha="right"
    )
    ax.set_ylabel("Mean count per trial", fontsize=8)
    ax.set_title(
        "D  Memory score profile (mean across scored trials)",
        fontsize=9,
        fontweight="bold",
        loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)

    status_patches = [
        mpatches.Patch(
            facecolor="#aaaaaa",
            hatch=_STATUS_HATCHES[st],
            label=st.capitalize(),
            edgecolor="white",
        )
        for st in status_order
    ]
    ax.legend(handles=status_patches, fontsize=6, framealpha=0.7, loc="upper right")


# ── Panel E: LCS → relational memory ────────────────────────────────────────


def _panel_scatter_lcs(ax, dec_ms, subjects, subj_colors):
    df = dec_ms[
        ["SubjectID", "lcs_enc_dec", "n_relational_correct", "n_objects_correct"]
    ].dropna()

    for s in subjects:
        sub = df[df["SubjectID"] == s]
        if sub.empty:
            continue
        label = s.split("-")[-2] + "-" + s.split("-")[-1]
        ax.scatter(
            sub["lcs_enc_dec"],
            sub["n_relational_correct"],
            s=28,
            color=subj_colors[s],
            alpha=0.75,
            label=label,
            zorder=3,
            linewidths=0,
        )

    if len(df) >= 4:
        x, y = df["lcs_enc_dec"].values, df["n_relational_correct"].values
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xr = np.linspace(x.min(), x.max(), 80)
        ax.plot(
            xr,
            intercept + slope * xr,
            color="#333333",
            linewidth=1.5,
            linestyle="-",
            zorder=2,
        )
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(
            0.05,
            0.93,
            f"r = {r:.2f}, p = {p:.3f} {sig}",
            transform=ax.transAxes,
            fontsize=7.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
        )

    ax.set_xlabel("LCS (enc–dec scanpath overlap)", fontsize=8)
    ax.set_ylabel("n relational correct", fontsize=8)
    ax.set_title(
        "E  Replay → relational memory", fontsize=9, fontweight="bold", loc="left"
    )
    ax.legend(fontsize=6, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


# ── Panel F: Enc SVG → relational memory ────────────────────────────────────


def _panel_scatter_enc_svg(ax, enc, dec_ms, subjects, subj_colors):
    enc_svg = enc[["SubjectID", "StimID", "svg_z_inter"]].rename(
        columns={"svg_z_inter": "svg_z_inter_enc"}
    )
    df = (
        dec_ms[["SubjectID", "StimID", "n_relational_correct", "n_objects_correct"]]
        .merge(enc_svg, on=["SubjectID", "StimID"], how="inner")
        .dropna()
    )

    for s in subjects:
        sub = df[df["SubjectID"] == s]
        if sub.empty:
            continue
        label = s.split("-")[-2] + "-" + s.split("-")[-1]
        ax.scatter(
            sub["svg_z_inter_enc"],
            sub["n_relational_correct"],
            s=28,
            color=subj_colors[s],
            alpha=0.75,
            label=label,
            zorder=3,
            linewidths=0,
            marker="s",
        )
        # objects_correct as open circles
        ax.scatter(
            sub["svg_z_inter_enc"],
            sub["n_objects_correct"],
            s=28,
            facecolors="none",
            edgecolors=subj_colors[s],
            alpha=0.55,
            zorder=3,
            linewidths=0.8,
        )

    if len(df) >= 3:
        x, y = df["svg_z_inter_enc"].values, df["n_relational_correct"].values
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xr = np.linspace(x.min(), x.max(), 80)
        ax.plot(xr, intercept + slope * xr, color="#333333", linewidth=1.5, zorder=2)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(
            0.05,
            0.93,
            f"r = {r:.2f}, p = {p:.3f} {sig}\n"
            f"n = {len(df)} trials  "
            r"■ relational  ○ objects",
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
        )
    else:
        ax.text(
            0.5,
            0.5,
            f"n = {len(df)} trials\n(insufficient for regression)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#666666",
        )

    ax.set_xlabel("SVG z-score (encoding)", fontsize=8)
    ax.set_ylabel("Memory score (count)", fontsize=8)
    ax.set_title(
        "F  Enc SVG → relational / object memory",
        fontsize=9,
        fontweight="bold",
        loc="left",
    )
    ax.legend(fontsize=6, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


# ── Master function ──────────────────────────────────────────────────────────


def make_pilot_diagnostics(
    features_path: Path,
    scores_path: Path,
    output_path: Path,
) -> None:
    """Generate and save the 6-panel pilot diagnostics figure."""
    logger.info("  Generating pilot diagnostics figure ...")

    enc, dec, dec_ms, ms, subjects, subj_colors = load_data(features_path, scores_path)

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.45,
        wspace=0.32,
        left=0.07,
        right=0.97,
        top=0.93,
        bottom=0.07,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    _panel_coverage(ax_a, enc, dec, ms, subjects)
    _panel_svg_dist(ax_b, enc, dec, subjects, subj_colors)
    _panel_lcs(ax_c, dec, subjects, subj_colors)
    _panel_memory_profile(ax_d, ms)
    _panel_scatter_lcs(ax_e, dec_ms, subjects, subj_colors)
    _panel_scatter_enc_svg(ax_f, enc, dec_ms, subjects, subj_colors)

    n_scored = len(ms)
    n_subj = len(subjects)
    n_stim = ms["StimID"].nunique()
    fig.suptitle(
        f"Module 4 — Pilot diagnostics   "
        f"({n_subj} participants, {n_stim} stimuli scored, "
        f"{n_scored} total scored trials)",
        fontsize=10,
        fontweight="bold",
        y=0.975,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Written → {output_path.name}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    parser = argparse.ArgumentParser(
        description="Generate pilot diagnostic plots for Module 4."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    make_pilot_diagnostics(
        Path(args.features),
        Path(args.scores),
        Path(args.out),
    )


if __name__ == "__main__":
    main()
