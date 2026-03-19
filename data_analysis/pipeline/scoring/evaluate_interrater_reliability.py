"""
pipeline/scoring/interrater_reliability.py
===========================================
Computes inter-rater reliability across three raters:

    AI-1  : output/scoring/recall_scores.csv           (node-level)
             output/scoring/recall_by_category.csv      (category-level)
    AI-2  : output/scoring/second_AI_rated_responses/recall_scores.csv  (node-level)
             output/scoring/second_AI_rated_responses/recall_by_category.csv
    Human : output/scoring/human_rated_responses/memory_scores.csv      (category-level only)

Comparisons
-----------
AI-1 vs AI-2   — node-level Cohen's kappa + % agreement
                  (strongest comparison: binary recalled decisions per node_id)

AI-1 vs Human  — category-level Pearson r + ICC(2,1)
AI-2 vs Human  — category-level Pearson r + ICC(2,1)
                  (human data has no node IDs; comparison is on per-category
                   recall counts across shared (SubjectID, StimID) pairs)

Category mapping (Human → AI)
------------------------------
Human `n_{ct}_correct`  ↔  AI `n_{ct}_recalled`
Categories: object_identity, object_attribute, action_relation,
            spatial_relation, scene_gist

Human "inference" and "repeat" annotations are excluded from the _correct
count, matching the AI's binary recalled=1 criterion.

Outputs
-------
  output/scoring/reliability/
    reliability_summary.txt   — human-readable report
    reliability_metrics.csv   — tidy metrics table (one row per comparison × metric)
    node_level_agreement.csv  — per-(SubjectID, StimID) AI-1 vs AI-2 agreement stats
    category_level_data.csv   — merged category counts used for human comparisons

Usage
-----
    python pipeline/scoring/interrater_reliability.py
    python pipeline/scoring/interrater_reliability.py --output-dir path/to/reliability/
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_DA_DIR = _HERE.parent.parent
sys.path.insert(0, str(_DA_DIR))

import config

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCORING_DIR = config.OUTPUT_DIR / "scoring"
AI1_NODES_CSV = SCORING_DIR / "recall_scores.csv"
AI1_CAT_CSV = SCORING_DIR / "recall_by_category.csv"
AI2_DIR = SCORING_DIR / "second_AI_rated_responses"
AI2_NODES_CSV = AI2_DIR / "recall_scores.csv"
AI2_CAT_CSV = AI2_DIR / "recall_by_category.csv"
HUMAN_CAT_CSV = SCORING_DIR / "human_rated_responses" / "memory_scores.csv"

CONTENT_TYPES = [
    "object_identity",
    "object_attribute",
    "action_relation",
    "spatial_relation",
    "scene_gist",
]

# ---------------------------------------------------------------------------
# Cohen's kappa (binary)
# ---------------------------------------------------------------------------


def cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> dict:
    """
    Compute Cohen's kappa and % agreement for two binary arrays.
    Returns dict with keys: kappa, pct_agreement, n, n_agree,
    tp, fp, fn, tn.
    """
    assert len(y1) == len(y2), "Arrays must be same length"
    n = len(y1)
    tp = int(((y1 == 1) & (y2 == 1)).sum())
    tn = int(((y1 == 0) & (y2 == 0)).sum())
    fp = int(((y1 == 0) & (y2 == 1)).sum())
    fn = int(((y1 == 1) & (y2 == 0)).sum())
    n_agree = tp + tn
    pct_agreement = n_agree / n if n > 0 else np.nan

    # Expected agreement by chance
    p1_pos = (tp + fn) / n  # rater 1 positive rate
    p2_pos = (tp + fp) / n  # rater 2 positive rate
    p_e = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)

    kappa = (pct_agreement - p_e) / (1 - p_e) if (1 - p_e) > 0 else np.nan

    return dict(
        kappa=kappa,
        pct_agreement=pct_agreement,
        n=n,
        n_agree=n_agree,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


# ---------------------------------------------------------------------------
# ICC(2,1) — two-way random, single measures, consistency
# ---------------------------------------------------------------------------


def icc_2_1(x: np.ndarray, y: np.ndarray) -> float:
    """
    ICC(2,1): two-way random effects, single measures, absolute agreement.
    Treats each (SubjectID, StimID) pair as a subject, the two raters as
    conditions. Returns ICC value (nan on failure).
    """
    n = len(x)
    if n < 3:
        return np.nan
    data = np.stack([x, y], axis=1)  # (n, 2)
    k = 2  # number of raters

    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    ss_total = ((data - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_error = (n - 1) * (k - 1)

    ms_rows = ss_rows / df_rows if df_rows > 0 else np.nan
    ms_cols = ss_cols / df_cols if df_cols > 0 else np.nan
    ms_error = ss_error / df_error if df_error > 0 else np.nan

    if ms_rows is np.nan or ms_error is np.nan:
        return np.nan

    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    return float(icc)


# ---------------------------------------------------------------------------
# AI-1 vs AI-2: node-level kappa
# ---------------------------------------------------------------------------


def compare_ai_node_level(output_dir: Path) -> tuple[dict, pd.DataFrame]:
    """
    Join AI-1 and AI-2 on (SubjectID, StimID, node_id).
    Compute Cohen's kappa and % agreement over the shared node set.
    Also returns per-(SubjectID, StimID) agreement stats.
    """
    logger.info("Loading AI-1 node scores ...")
    ai1 = pd.read_csv(AI1_NODES_CSV, dtype={"SubjectID": str, "StimID": str})

    logger.info("Loading AI-2 node scores ...")
    ai2 = pd.read_csv(AI2_NODES_CSV, dtype={"SubjectID": str, "StimID": str})

    # Scope AI-1 to the pairs AI-2 covers
    ai2_pairs = set(zip(ai2["SubjectID"], ai2["StimID"]))
    ai1_scoped = ai1[
        ai1.apply(lambda r: (r["SubjectID"], r["StimID"]) in ai2_pairs, axis=1)
    ]

    merged = pd.merge(
        ai1_scoped[["SubjectID", "StimID", "node_id", "recalled"]].rename(
            columns={"recalled": "recalled_ai1"}
        ),
        ai2[["SubjectID", "StimID", "node_id", "recalled"]].rename(
            columns={"recalled": "recalled_ai2"}
        ),
        on=["SubjectID", "StimID", "node_id"],
        how="inner",
    )

    n_ai1_nodes = len(ai1_scoped)
    n_ai2_nodes = len(ai2)
    n_merged = len(merged)
    logger.info(
        f"AI-1 nodes (scoped): {n_ai1_nodes} | "
        f"AI-2 nodes: {n_ai2_nodes} | "
        f"merged on node_id: {n_merged}"
    )
    if n_ai1_nodes != n_merged or n_ai2_nodes != n_merged:
        logger.warning(
            "Node counts differ after merge — some node_ids may be present "
            "in one rater but not the other. Only matched nodes enter kappa."
        )

    y1 = merged["recalled_ai1"].values.astype(int)
    y2 = merged["recalled_ai2"].values.astype(int)
    overall = cohens_kappa(y1, y2)
    overall["n_pairs"] = len(ai2_pairs)
    overall["comparison"] = "AI-1 vs AI-2"
    overall["level"] = "node"

    # Per-(SubjectID, StimID) agreement
    per_pair_rows = []
    for (subj, stim), grp in merged.groupby(["SubjectID", "StimID"]):
        stats = cohens_kappa(
            grp["recalled_ai1"].values.astype(int),
            grp["recalled_ai2"].values.astype(int),
        )
        per_pair_rows.append({
            "SubjectID": subj,
            "StimID": stim,
            **stats,
        })
    per_pair_df = pd.DataFrame(per_pair_rows)

    return overall, per_pair_df


# ---------------------------------------------------------------------------
# AI vs Human: category-level correlation + ICC
# ---------------------------------------------------------------------------


def _align_human_to_ai_format(human: pd.DataFrame) -> pd.DataFrame:
    """
    Rename human `n_{ct}_correct` columns to `n_{ct}_recalled` so they
    align with the AI recall_by_category.csv format.
    Also compute n_correct_nodes_recalled as sum across content types.
    """
    rename = {}
    for ct in CONTENT_TYPES:
        human_col = f"n_{ct}_correct"
        ai_col = f"n_{ct}_recalled"
        if human_col in human.columns:
            rename[human_col] = ai_col

    human = human.rename(columns=rename)

    # Recompute total recalled from content-type columns
    recalled_cols = [f"n_{ct}_recalled" for ct in CONTENT_TYPES if f"n_{ct}_recalled" in human.columns]
    if recalled_cols:
        human["n_correct_nodes_recalled_human"] = human[recalled_cols].sum(axis=1)

    return human


def compare_ai_vs_human(
    rater_label: str,
    ai_cat_csv: Path,
) -> dict:
    """
    Compare one AI rater (category-level) against the human rater.
    Returns dict of metrics per category + overall.
    """
    logger.info(f"Loading {rater_label} category scores ...")
    ai = pd.read_csv(ai_cat_csv, dtype={"SubjectID": str, "StimID": str})
    human = pd.read_csv(HUMAN_CAT_CSV, dtype={"SubjectID": str, "StimID": str})
    human = _align_human_to_ai_format(human)

    # Restrict to shared pairs
    human_pairs = set(zip(human["SubjectID"], human["StimID"]))
    ai_scoped = ai[
        ai.apply(lambda r: (r["SubjectID"], r["StimID"]) in human_pairs, axis=1)
    ].copy()

    merged = pd.merge(
        ai_scoped,
        human[["SubjectID", "StimID"] + [f"n_{ct}_recalled" for ct in CONTENT_TYPES if f"n_{ct}_recalled" in human.columns] + ["n_correct_nodes_recalled_human"]],
        on=["SubjectID", "StimID"],
        how="inner",
        suffixes=("_ai", "_human"),
    )

    logger.info(
        f"  {rater_label} vs Human: {len(merged)} shared (SubjectID, StimID) pairs"
    )

    results = {"comparison": f"{rater_label} vs Human", "n_pairs": len(merged)}

    # Per content type
    ct_metrics = {}
    for ct in CONTENT_TYPES:
        ai_col = f"n_{ct}_recalled_ai"
        human_col = f"n_{ct}_recalled_human"
        if ai_col not in merged.columns or human_col not in merged.columns:
            continue
        x = merged[ai_col].fillna(0).values.astype(float)
        y = merged[human_col].fillna(0).values.astype(float)

        r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan
        icc = icc_2_1(x, y)
        mae = float(np.abs(x - y).mean())

        ct_metrics[ct] = {"pearson_r": r, "icc_2_1": icc, "mae": mae}

    results["by_content_type"] = ct_metrics

    # Overall: total correct nodes recalled
    ai_total_col = "n_correct_nodes_recalled"
    human_total_col = "n_correct_nodes_recalled_human"
    if ai_total_col in merged.columns and human_total_col in merged.columns:
        x = merged[ai_total_col].fillna(0).values.astype(float)
        y = merged[human_total_col].fillna(0).values.astype(float)
        results["overall_pearson_r"] = float(np.corrcoef(x, y)[0, 1])
        results["overall_icc_2_1"] = icc_2_1(x, y)
        results["overall_mae"] = float(np.abs(x - y).mean())

    return results, merged


# ---------------------------------------------------------------------------
# Format report
# ---------------------------------------------------------------------------


def _fmt_kappa(k: dict) -> str:
    lines = [
        f"  Level         : {k.get('level', 'node')}",
        f"  N pairs       : {k.get('n_pairs', '—')}",
        f"  N nodes       : {k.get('n', '—')}",
        f"  % agreement   : {k.get('pct_agreement', np.nan)*100:.1f}%",
        f"  Cohen's kappa : {k.get('kappa', np.nan):.4f}",
        f"  Confusion     : TP={k.get('tp')}, TN={k.get('tn')}, FP={k.get('fp')}, FN={k.get('fn')}",
    ]
    return "\n".join(lines)


def _fmt_cat(res: dict) -> str:
    lines = [
        f"  N pairs       : {res.get('n_pairs', '—')}",
        f"  Overall r     : {res.get('overall_pearson_r', np.nan):.4f}",
        f"  Overall ICC   : {res.get('overall_icc_2_1', np.nan):.4f}",
        f"  Overall MAE   : {res.get('overall_mae', np.nan):.3f} nodes",
        "",
        "  Per content type:",
    ]
    for ct, m in res.get("by_content_type", {}).items():
        lines.append(
            f"    {ct:<22}  r={m['pearson_r']:+.3f}  ICC={m['icc_2_1']:+.3f}  MAE={m['mae']:.2f}"
        )
    return "\n".join(lines)


def _build_metrics_csv(ai1_ai2: dict, ai1_human: dict, ai2_human: dict) -> pd.DataFrame:
    rows = []

    # AI-1 vs AI-2
    rows.append({
        "comparison": "AI-1 vs AI-2",
        "level": "node",
        "category": "overall",
        "metric": "cohens_kappa",
        "value": ai1_ai2.get("kappa"),
    })
    rows.append({
        "comparison": "AI-1 vs AI-2",
        "level": "node",
        "category": "overall",
        "metric": "pct_agreement",
        "value": ai1_ai2.get("pct_agreement"),
    })

    # AI vs Human
    for label, res in [("AI-1 vs Human", ai1_human), ("AI-2 vs Human", ai2_human)]:
        for metric in ("overall_pearson_r", "overall_icc_2_1", "overall_mae"):
            rows.append({
                "comparison": label,
                "level": "category",
                "category": "overall",
                "metric": metric.replace("overall_", ""),
                "value": res.get(metric),
            })
        for ct, m in res.get("by_content_type", {}).items():
            for metric, val in m.items():
                rows.append({
                    "comparison": label,
                    "level": "category",
                    "category": ct,
                    "metric": metric,
                    "value": val,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute AI-1 vs AI-2 vs Human inter-rater reliability."
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCORING_DIR / "reliability"),
        help="Directory for reliability outputs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Inter-rater reliability: AI-1 vs AI-2 vs Human")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Check inputs exist
    # ------------------------------------------------------------------
    missing = []
    for label, path in [
        ("AI-1 node scores", AI1_NODES_CSV),
        ("AI-1 category scores", AI1_CAT_CSV),
        ("AI-2 node scores", AI2_NODES_CSV),
        ("AI-2 category scores", AI2_CAT_CSV),
        ("Human category scores", HUMAN_CAT_CSV),
    ]:
        if not path.exists():
            missing.append(f"  {label}: {path}")
    if missing:
        logger.error("Missing required files:\n" + "\n".join(missing))
        sys.exit(1)

    # ------------------------------------------------------------------
    # AI-1 vs AI-2: node-level kappa
    # ------------------------------------------------------------------
    logger.info("\n[1/3] AI-1 vs AI-2 (node-level kappa) ...")
    ai1_ai2_metrics, per_pair_df = compare_ai_node_level(output_dir)

    per_pair_df.to_csv(output_dir / "node_level_agreement.csv", index=False)
    logger.info(f"  Written -> node_level_agreement.csv")

    # ------------------------------------------------------------------
    # AI-1 vs Human: category-level
    # ------------------------------------------------------------------
    logger.info("\n[2/3] AI-1 vs Human (category-level) ...")
    ai1_human_metrics, merged_ai1 = compare_ai_vs_human("AI-1", AI1_CAT_CSV)

    # ------------------------------------------------------------------
    # AI-2 vs Human: category-level
    # ------------------------------------------------------------------
    logger.info("\n[3/3] AI-2 vs Human (category-level) ...")
    ai2_human_metrics, merged_ai2 = compare_ai_vs_human("AI-2", AI2_CAT_CSV)

    # Save merged category data
    merged_ai1["rater"] = "AI-1"
    merged_ai2["rater"] = "AI-2"
    cat_data = pd.concat([merged_ai1, merged_ai2], ignore_index=True)
    cat_data.to_csv(output_dir / "category_level_data.csv", index=False)
    logger.info("  Written -> category_level_data.csv")

    # ------------------------------------------------------------------
    # Tidy metrics CSV
    # ------------------------------------------------------------------
    metrics_df = _build_metrics_csv(ai1_ai2_metrics, ai1_human_metrics, ai2_human_metrics)
    metrics_df.to_csv(output_dir / "reliability_metrics.csv", index=False)
    logger.info("  Written -> reliability_metrics.csv")

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------
    summary_lines = [
        "=" * 60,
        "Inter-rater Reliability Summary",
        "=" * 60,
        "",
        "Note on comparison levels:",
        "  AI-1 vs AI-2  : node-level (binary recalled per node_id) — strongest test",
        "  AI vs Human   : category-level counts only (human has no node IDs)",
        "",
        "=" * 60,
        "AI-1 vs AI-2 (node-level)",
        "=" * 60,
        _fmt_kappa(ai1_ai2_metrics),
        "",
        "=" * 60,
        "AI-1 vs Human (category-level)",
        "=" * 60,
        _fmt_cat(ai1_human_metrics),
        "",
        "=" * 60,
        "AI-2 vs Human (category-level)",
        "=" * 60,
        _fmt_cat(ai2_human_metrics),
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(output_dir / "reliability_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info(f"\n  Written -> {output_dir / 'reliability_summary.txt'}")

    logger.info("\n" + "=" * 60)
    logger.info("Reliability analysis complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()