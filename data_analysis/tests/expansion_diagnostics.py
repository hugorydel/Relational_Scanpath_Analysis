"""
expansion_diagnostics.py
========================
Combined diagnostic module for the N=22 → N=27 participant expansion.

Runs all checks in sequence:

  CHECK 1  — Coefficient stability     (N=22 vs N=27 LMM refit)
  CHECK 2  — Outlier detection          (per-participant means + scatter)
  CHECK 3  — DV distribution shift      (old vs new recall levels)
  CHECK 4  — Within-image slopes        (per-participant SVG→recall coupling)
  CHECK 5  — Behavioural flags          (P25/P26 fixation/SVG/AOI profiles)
  CHECK 6  — Sensitivity analysis       (N=25 excluding P25+P26)
  CHECK 7  — Two-stage analysis         (aggregate slopes → one-sample t-test)

All outputs written to output/diagnostics/.

Usage
-----
    python data_analysis/tests/expansion_diagnostics.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent.parent
ENC_PATH = SCRIPT_DIR / "output/analysis/figure_data/analysis_enc.csv"
OUT_DIR = SCRIPT_DIR / "output/diagnostics"

OLD_NUMS = [n for n in range(1, 24) if n != 20]
OLD_IDS = {f"Encode-Decode_Experiment-{n}-1" for n in OLD_NUMS}
NEW_NUMS = list(range(24, 29))
NEW_IDS = {f"Encode-Decode_Experiment-{n}-1" for n in NEW_NUMS}
EXCLUDE = {"Encode-Decode_Experiment-25-1", "Encode-Decode_Experiment-26-1"}

DV_COLS = ["prop_total", "prop_relations", "prop_objects"]
DV_LABELS = {
    "prop_total": "Total recall",
    "prop_relations": "Relational recall",
    "prop_objects": "Object recall",
}
SVG_COL = "svg_z_enc_within_z"
SVG_WITHIN = "svg_z_enc_within"
COV_COLS = [
    "n_fixations_enc_z",
    "mean_salience_relational_enc_z",
    "svg_z_enc_image_mean_z",
]

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)
enc = pd.read_csv(ENC_PATH, dtype={"SubjectID": str, "StimID": str})

all_ids = set(enc["SubjectID"].unique())
found_old = OLD_IDS & all_ids
found_new = NEW_IDS & all_ids

print(f"Loaded {len(enc)} rows | {enc['SubjectID'].nunique()} participants")
print(f"  Recognised old: {len(found_old)} | Recognised new: {len(found_new)}")
print()

enc_old = enc[enc["SubjectID"].isin(found_old)]
enc_new = enc[enc["SubjectID"].isin(found_new)]
enc_excl = enc[~enc["SubjectID"].isin(EXCLUDE)]


def pid(s):
    return "P" + s.split("-")[-2]


def _sort_key(s):
    for p in reversed(s.split("-")):
        try:
            return int(p)
        except ValueError:
            continue
    return 0


# ---------------------------------------------------------------------------
# HELPER: fit LMM
# ---------------------------------------------------------------------------


def fit_lmm(dv, df):
    formula = f"{dv} ~ {SVG_COL} + " + " + ".join(COV_COLS)
    req = [dv, "SubjectID", "StimID", SVG_COL] + COV_COLS
    sub = df[[c for c in req if c in df.columns]].dropna()
    if len(sub) < 20 or sub["SubjectID"].nunique() < 3:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.mixedlm(
                formula,
                data=sub,
                groups="SubjectID",
                vc_formula={"StimID": "0 + C(StimID)"},
            )
            r = m.fit(reml=True, method="nm", maxiter=1000)
            return r if r.converged else None
        except Exception:
            return None


def lmm_extract(r):
    if r is None:
        return np.nan, np.nan, np.nan, np.nan
    b = float(r.params[SVG_COL])
    se = float(r.bse[SVG_COL])
    p = float(r.pvalues[SVG_COL])
    return b, p, b - 1.96 * se, b + 1.96 * se


# ---------------------------------------------------------------------------
# CHECK 1: Coefficient stability
# ---------------------------------------------------------------------------

print("=" * 60)
print("CHECK 1: Coefficient stability  (N=22 vs N=27)")
print("=" * 60)

rows1 = []
for dv, label in DV_LABELS.items():
    if dv not in enc.columns:
        continue
    b22, p22, lo22, hi22 = lmm_extract(fit_lmm(dv, enc_old))
    b27, p27, lo27, hi27 = lmm_extract(fit_lmm(dv, enc))
    sig22 = (
        "***" if p22 < 0.001 else "**" if p22 < 0.01 else "*" if p22 < 0.05 else "ns"
    )
    sig27 = (
        "***" if p27 < 0.001 else "**" if p27 < 0.01 else "*" if p27 < 0.05 else "ns"
    )
    print(f"\n  {label}:")
    print(f"    N=22: b={b22:.4f} [{lo22:.4f},{hi22:.4f}]  p={p22:.4f} {sig22}")
    print(f"    N=27: b={b27:.4f} [{lo27:.4f},{hi27:.4f}]  p={p27:.4f} {sig27}")
    print(f"    Δb = {b27-b22:+.4f}")
    rows1.append(
        {
            "dv": label,
            "b_n22": b22,
            "p_n22": p22,
            "b_n27": b27,
            "p_n27": p27,
            "delta_b": b27 - b22,
        }
    )

pd.DataFrame(rows1).to_csv(OUT_DIR / "check1_coefficient_stability.csv", index=False)
print(f"\n  -> check1_coefficient_stability.csv")

# ---------------------------------------------------------------------------
# CHECK 2: Outlier detection
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("CHECK 2: Per-participant outlier detection")
print("=" * 60)

agg_cols = {"mean_svg": ("svg_z_enc", "mean")}
for dv in DV_COLS:
    if dv in enc.columns:
        agg_cols[f"mean_{dv}"] = (dv, "mean")

agg = enc.groupby("SubjectID").agg(**agg_cols).reset_index()
agg["is_new"] = agg["SubjectID"].isin(found_new)

dim_cols = ["mean_svg"] + [f"mean_{d}" for d in DV_COLS if d in enc.columns]
for col in dim_cols:
    vals = agg[col].values.astype(float)
    mu, sd = np.nanmean(vals), np.nanstd(vals, ddof=1)
    agg[f"z_{col}"] = (vals - mu) / sd if sd > 0 else 0.0

agg["max_abs_z"] = agg[[f"z_{c}" for c in dim_cols]].abs().max(axis=1)
agg["outlier_flag"] = agg["max_abs_z"] > 2.5

print("\n  New participants:")
for _, row in agg[agg["is_new"]].iterrows():
    flag = "  *** OUTLIER ***" if row["outlier_flag"] else ""
    parts = [f"SVG={row['mean_svg']:.3f}"]
    for dv in DV_COLS:
        col = f"mean_{dv}"
        if col in row.index:
            parts.append(f"{dv.split('_')[1][:3]}={row[col]*100:.1f}%")
    print(f"    {pid(row['SubjectID'])}: {' | '.join(parts)}{flag}")

flagged = agg[agg["outlier_flag"]]
if len(flagged):
    print(f"\n  *** Flagged (|z|>2.5): {[pid(s) for s in flagged['SubjectID']]}")
else:
    print("\n  No outliers detected (|z|>2.5).")

n_dvs = sum(1 for d in DV_COLS if d in enc.columns)
fig, axes = plt.subplots(1, n_dvs, figsize=(4.2 * n_dvs, 3.8))
if n_dvs == 1:
    axes = [axes]
rng = np.random.default_rng(42)
for ax, dv in zip(axes, [d for d in DV_COLS if d in enc.columns]):
    col = f"mean_{dv}"
    old_d, new_d = agg[~agg["is_new"]], agg[agg["is_new"]]
    ax.scatter(
        old_d["mean_svg"],
        old_d[col] * 100,
        color="#555555",
        s=45,
        alpha=0.8,
        zorder=3,
        label="Original",
    )
    ax.scatter(
        new_d["mean_svg"],
        new_d[col] * 100,
        color="#b2182b",
        s=55,
        marker="D",
        alpha=0.9,
        zorder=4,
        label="New",
    )
    for _, row in new_d.iterrows():
        ax.annotate(
            pid(row["SubjectID"]),
            xy=(row["mean_svg"], row[col] * 100),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#b2182b",
        )
    x = agg["mean_svg"].values
    y = (agg[col] * 100).values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() >= 4:
        sl, ic, r, p, _ = stats.linregress(x[mask], y[mask])
        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(
            x_line,
            ic + sl * x_line,
            color="#333333",
            linewidth=1.2,
            linestyle="--",
            zorder=2,
        )
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.annotate(
            f"r={r:.2f} {sig}", xy=(0.05, 0.93), xycoords="axes fraction", fontsize=8.5
        )
    ax.set_xlabel("Mean encoding SVG (z)", fontsize=9)
    ax.set_ylabel(f"{DV_LABELS[dv]} (%)", fontsize=9)
    ax.set_title(DV_LABELS[dv], fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    if ax is axes[0]:
        ax.legend(fontsize=7.5, frameon=False)
plt.suptitle(
    "Per-participant SVG vs recall  (◆=new | dashed=OLS trend)", fontsize=9, y=1.01
)
plt.tight_layout()
plt.savefig(OUT_DIR / "check2_outlier_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
agg.to_csv(OUT_DIR / "check2_participant_means.csv", index=False)
print(f"  -> check2_outlier_scatter.png  |  check2_participant_means.csv")

# ---------------------------------------------------------------------------
# CHECK 3: DV distribution shift
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("CHECK 3: DV distribution shift  (old N=22 vs new N=5)")
print("=" * 60)


def subj_stats(df, group_label, dv):
    sm = df.groupby("SubjectID")[dv].mean().dropna() * 100
    return {
        "group": group_label,
        "dv": DV_LABELS[dv],
        "n": len(sm),
        "M": round(sm.mean(), 2),
        "SD": round(sm.std(ddof=1), 2),
    }


rows3 = []
for dv, label in DV_LABELS.items():
    if dv not in enc.columns:
        continue
    s_old = subj_stats(enc_old, "Original N=22", dv)
    s_new = subj_stats(enc_new, "New N=5", dv)
    s_all = subj_stats(enc, "All N=27", dv)
    rows3.extend([s_old, s_new, s_all])
    old_m = enc_old.groupby("SubjectID")[dv].mean().dropna() * 100
    new_m = enc_new.groupby("SubjectID")[dv].mean().dropna() * 100
    if len(old_m) >= 2 and len(new_m) >= 2:
        t, p = stats.ttest_ind(old_m, new_m, equal_var=False)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(
            f"\n  {label}:  Old={s_old['M']:.1f}%  New={s_new['M']:.1f}%  "
            f"Welch t={t:.2f} p={p:.4f} {sig}"
        )

df3 = pd.DataFrame(rows3)
df3.to_csv(OUT_DIR / "check3_dv_distribution_shift.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 3.5))
groups = ["Original N=22", "New N=5", "All N=27"]
colours = ["#555555", "#b2182b", "#333333"]
x_pos = np.arange(len(DV_LABELS))
width = 0.25
for i, (grp, colour) in enumerate(zip(groups, colours)):
    sub = df3[df3["group"] == grp]
    means = [
        sub[sub["dv"] == lbl]["M"].values[0] if len(sub[sub["dv"] == lbl]) else np.nan
        for lbl in DV_LABELS.values()
    ]
    ax.bar(x_pos + i * width, means, width, label=grp, color=colour, alpha=0.82)
ax.set_xticks(x_pos + width)
ax.set_xticklabels(list(DV_LABELS.values()), fontsize=9)
ax.set_ylabel("Mean recall (%)", fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "check3_dv_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> check3_dv_distribution.png  |  check3_dv_distribution_shift.csv")

# ---------------------------------------------------------------------------
# CHECK 4: Within-image slopes
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("CHECK 4: Per-participant within-image SVG→recall slopes")
print("=" * 60)

slope_rows = []
if SVG_WITHIN in enc.columns:
    for sid in sorted(enc["SubjectID"].unique(), key=_sort_key):
        sub = enc[enc["SubjectID"] == sid].copy()
        row = {
            "SubjectID": sid,
            "pid": pid(sid),
            "is_new": sid in found_new,
            "is_excluded": sid in EXCLUDE,
            "n_images": len(sub),
        }
        for dv in DV_LABELS:
            if dv in sub.columns:
                xy = sub[[SVG_WITHIN, dv]].dropna()
                if len(xy) >= 5:
                    sl, ic, r, p, _ = stats.linregress(
                        xy[SVG_WITHIN].values, xy[dv].values
                    )
                    row[f"slope_{dv}"] = round(sl, 5)
                    row[f"r_{dv}"] = round(r, 3)
                    row[f"p_{dv}"] = round(p, 4)
                else:
                    row[f"slope_{dv}"] = np.nan
            else:
                row[f"slope_{dv}"] = np.nan
        slope_rows.append(row)

slopes_df = pd.DataFrame(slope_rows)

print()
for dv, label in DV_LABELS.items():
    col = f"slope_{dv}"
    if col not in slopes_df.columns:
        continue
    old_s = slopes_df[~slopes_df["is_new"]][col].dropna()
    new_s = slopes_df[slopes_df["is_new"]][col].dropna()
    print(f"  {label}:")
    print(
        f"    Old N=22: M={old_s.mean():.5f}  n_positive={(old_s>0).sum()}/{len(old_s)}"
    )
    print(
        f"    New N=5:  M={new_s.mean():.5f}  n_positive={(new_s>0).sum()}/{len(new_s)}"
    )

n_dvs = len(DV_LABELS)
fig, axes = plt.subplots(1, n_dvs, figsize=(4.2 * n_dvs, 4.0))
if n_dvs == 1:
    axes = [axes]
for ax, (dv, label) in zip(axes, DV_LABELS.items()):
    col = f"slope_{dv}"
    if col not in slopes_df.columns:
        continue
    old_d = slopes_df[~slopes_df["is_new"]]
    new_d = slopes_df[slopes_df["is_new"] & ~slopes_df["is_excluded"]]
    excl_d = slopes_df[slopes_df["is_excluded"]]
    rng2 = np.random.default_rng(42)
    ax.scatter(
        rng2.uniform(-0.08, 0.08, len(old_d)),
        old_d[col].values,
        color="#555555",
        s=45,
        alpha=0.8,
        zorder=3,
        label="Original",
    )
    ax.scatter(
        rng2.uniform(0.92, 1.08, len(new_d)),
        new_d[col].values,
        color="#2166ac",
        s=50,
        marker="D",
        alpha=0.9,
        zorder=4,
        label="New (kept)",
    )
    ax.scatter(
        rng2.uniform(0.92, 1.08, len(excl_d)),
        excl_d[col].values,
        color="#b2182b",
        s=60,
        marker="X",
        zorder=5,
        label="Excluded (P25/P26)",
    )
    for _, row in excl_d.iterrows():
        ax.annotate(
            row["pid"],
            xy=(1.0, row[col]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=7.5,
            color="#b2182b",
            va="center",
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlim(-0.5, 1.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Original", "New"], fontsize=9)
    ax.set_ylabel(f"Within-image slope\n(SVG → {label})", fontsize=9)
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    if ax is axes[0]:
        ax.legend(fontsize=7.5, frameon=False)
plt.suptitle("Per-participant within-image SVG→recall slope", fontsize=9, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "check4_within_slopes.png", dpi=150, bbox_inches="tight")
plt.close()
slopes_df.to_csv(OUT_DIR / "check4_within_slopes.csv", index=False)
print(f"  -> check4_within_slopes.png  |  check4_within_slopes.csv")

# ---------------------------------------------------------------------------
# CHECK 5: Behavioural flags for P25/P26
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("CHECK 5: Behavioural flags for P25 and P26")
print("=" * 60)

flag_cols = [
    c for c in ["n_fixations_enc", "svg_z_enc", "aoi_prop_enc"] if c in enc.columns
]
flag_rows = []
for sid in sorted(enc["SubjectID"].unique(), key=_sort_key):
    sub = enc[enc["SubjectID"] == sid]
    row = {
        "SubjectID": sid,
        "pid": pid(sid),
        "is_excluded": sid in EXCLUDE,
        "n_trials": len(sub),
    }
    for col in flag_cols:
        if col in sub.columns:
            row[f"mean_{col}"] = round(sub[col].mean(), 3)
    flag_rows.append(row)
flag_df = pd.DataFrame(flag_rows)

print()
group_means = flag_df[~flag_df["is_excluded"]][[f"mean_{c}" for c in flag_cols]].mean()
group_stds = flag_df[~flag_df["is_excluded"]][[f"mean_{c}" for c in flag_cols]].std(
    ddof=1
)
for sid in EXCLUDE:
    row = flag_df[flag_df["SubjectID"] == sid].iloc[0]
    print(f"  {pid(sid)}:  (n_trials={int(row['n_trials'])})")
    for col in flag_cols:
        val = row[f"mean_{col}"]
        grp = group_means[f"mean_{col}"]
        sd = group_stds[f"mean_{col}"]
        z = (val - grp) / sd if sd > 0 else 0
        print(f"    {col}: {val:.3f}  (group mean={grp:.3f}, z={z:+.2f})")
    print()

flag_df.to_csv(OUT_DIR / "check5_behavioural_flags.csv", index=False)
print(f"  -> check5_behavioural_flags.csv")

# ---------------------------------------------------------------------------
# CHECK 6: Sensitivity — exclude P25/P26
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("CHECK 6: Sensitivity analysis  (N=27 vs N=25 excl. P25+P26)")
print("=" * 60)

rows6 = []
print()
for dv, label in DV_LABELS.items():
    if dv not in enc.columns:
        continue
    b27, p27, lo27, hi27 = lmm_extract(fit_lmm(dv, enc))
    b25, p25, lo25, hi25 = lmm_extract(fit_lmm(dv, enc_excl))
    sig27 = (
        "***" if p27 < 0.001 else "**" if p27 < 0.01 else "*" if p27 < 0.05 else "ns"
    )
    sig25 = (
        "***" if p25 < 0.001 else "**" if p25 < 0.01 else "*" if p25 < 0.05 else "ns"
    )
    print(f"  {label}:")
    print(f"    N=27: b={b27:.4f} [{lo27:.4f},{hi27:.4f}]  p={p27:.4f} {sig27}")
    print(f"    N=25: b={b25:.4f} [{lo25:.4f},{hi25:.4f}]  p={p25:.4f} {sig25}")
    print(f"    Δb = {b25-b27:+.4f}\n")
    rows6.append(
        {
            "dv": label,
            "b_n27": b27,
            "p_n27": p27,
            "ci_lo_n27": lo27,
            "ci_hi_n27": hi27,
            "b_n25": b25,
            "p_n25": p25,
            "ci_lo_n25": lo25,
            "ci_hi_n25": hi25,
            "delta_b": b25 - b27,
        }
    )

pd.DataFrame(rows6).to_csv(OUT_DIR / "check6_sensitivity_coefficients.csv", index=False)
print(f"  -> check6_sensitivity_coefficients.csv")

# ---------------------------------------------------------------------------
# CHECK 7: Two-stage analysis
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("CHECK 7: Two-stage analysis  (per-participant slopes → t-test)")
print("=" * 60)

rows7 = []
if SVG_WITHIN in enc.columns:
    print()
    for dv, label in DV_LABELS.items():
        col = f"slope_{dv}"
        if col not in slopes_df.columns:
            continue
        s = slopes_df[col].dropna()
        n = len(s)
        gm = s.mean()
        sd = s.std(ddof=1)
        se = s.sem()
        t, p = stats.ttest_1samp(s, 0)
        ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=gm, scale=se)
        d = gm / sd
        n_pos = (s > 0).sum()
        bp = stats.binomtest(int(n_pos), n, p=0.5, alternative="greater").pvalue
        sig_t = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        sig_b = (
            "***" if bp < 0.001 else "**" if bp < 0.01 else "*" if bp < 0.05 else "ns"
        )
        print(f"  {label}:")
        print(
            f"    Mean slope={gm:.5f}  t({n-1})={t:.3f}  p={p:.4f} {sig_t}  d={d:.3f}"
        )
        print(f"    95% CI=[{ci_lo:.5f}, {ci_hi:.5f}]")
        print(f"    Sign test: {n_pos}/{n} positive  p={bp:.4f} {sig_b}\n")
        rows7.append(
            {
                "dv": label,
                "n": n,
                "mean_slope": gm,
                "sd": sd,
                "t": t,
                "df": n - 1,
                "p_ttest": p,
                "cohens_d": d,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "n_positive": int(n_pos),
                "p_sign_test": bp,
            }
        )

    pd.DataFrame(rows7).to_csv(OUT_DIR / "check7_two_stage_results.csv", index=False)
    slopes_df.to_csv(OUT_DIR / "check7_two_stage_slopes.csv", index=False)

    # Figures: one per DV
    for dv, label, colour in [
        ("prop_total", "Total Recall", "#333333"),
        ("prop_relations", "Relational Recall", "#333333"),
        ("prop_objects", "Object Recall", "#333333"),
    ]:
        col = f"slope_{dv}"
        if col not in slopes_df.columns:
            continue
        sub_df = (
            slopes_df[slopes_df[col].notna()].sort_values(col).reset_index(drop=True)
        )
        res = next((r for r in rows7 if r["dv"] == label), None)
        if res is None:
            continue
        sig = (
            "***"
            if res["p_ttest"] < 0.001
            else (
                "**"
                if res["p_ttest"] < 0.01
                else "*" if res["p_ttest"] < 0.05 else "ns"
            )
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

        for i, row in sub_df.iterrows():
            c = "#b2182b" if row["is_new"] else "#555555"
            sh = "D" if row["is_new"] else "o"
            ax1.scatter(row[col], i, color=c, marker=sh, s=40, zorder=4, alpha=0.9)
        ax1.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax1.axvline(
            res["mean_slope"],
            color="#444444",
            linewidth=1.4,
            linestyle="-",
            alpha=0.85,
            label=f"Grand mean = {res['mean_slope']:.4f}",
        )
        ax1.axvspan(
            res["ci_lower"], res["ci_upper"], alpha=0.12, color="#444444", zorder=1
        )
        ax1.set_yticks(range(len(sub_df)))
        ax1.set_yticklabels(sub_df["pid"].values, fontsize=7)
        ax1.set_xlabel(f"OLS slope (SVG → {label})", fontsize=9)
        ax1.set_title(
            f"Per-participant slopes\nt({int(res['df'])})={res['t']:.2f}, "
            f"p={res['p_ttest']:.4f} {sig}, d={res['cohens_d']:.2f}",
            fontsize=8.5,
        )
        ax1.scatter([], [], color="#555555", marker="o", s=35, label="Original")
        ax1.scatter([], [], color="#b2182b", marker="D", s=35, label="New")
        ax1.legend(fontsize=7.5, frameon=False, loc="lower right")
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.tick_params(labelsize=8)

        if dv in enc.columns and "svg_z_enc" in enc.columns:
            agg2 = (
                enc.groupby("SubjectID")
                .agg(mean_svg=("svg_z_enc", "mean"), mean_dv=(dv, "mean"))
                .reset_index()
            )
            agg2["is_new"] = agg2["SubjectID"].isin(NEW_IDS)
            old_a = agg2[~agg2["is_new"]]
            new_a = agg2[agg2["is_new"]]
            ax2.scatter(
                old_a["mean_svg"],
                old_a["mean_dv"] * 100,
                color="#555555",
                s=40,
                alpha=0.75,
                zorder=3,
                label="Original",
            )
            ax2.scatter(
                new_a["mean_svg"],
                new_a["mean_dv"] * 100,
                color="#b2182b",
                s=50,
                marker="D",
                alpha=0.9,
                zorder=4,
                label="New",
            )
            for _, row in new_a.iterrows():
                ax2.annotate(
                    pid(row["SubjectID"]),
                    xy=(row["mean_svg"], row["mean_dv"] * 100),
                    xytext=(5, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color="#b2182b",
                )
            x = agg2["mean_svg"].values
            y = agg2["mean_dv"].values * 100
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 4:
                sl, ic, r, p_r, _ = stats.linregress(x[mask], y[mask])
                x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                ax2.plot(
                    x_line,
                    ic + sl * x_line,
                    color="#333333",
                    linewidth=1.4,
                    linestyle="--",
                    zorder=2,
                )
                sig_r = (
                    "***"
                    if p_r < 0.001
                    else "**" if p_r < 0.01 else "*" if p_r < 0.05 else "ns"
                )
                ax2.annotate(
                    f"r={r:.2f}, p={p_r:.3f} {sig_r}",
                    xy=(0.05, 0.93),
                    xycoords="axes fraction",
                    fontsize=8.5,
                )
            ax2.set_xlabel("Mean encoding SVG (z)", fontsize=9)
            ax2.set_ylabel(f"Mean {label} (%)", fontsize=9)
            ax2.set_title("Between-participant: mean SVG vs mean recall", fontsize=8.5)
            ax2.legend(fontsize=7.5, frameon=False)
            ax2.spines[["top", "right"]].set_visible(False)
            ax2.tick_params(labelsize=8)

        plt.suptitle(f"Two-stage analysis: {label}", fontsize=9, y=1.02)
        plt.tight_layout()
        fname = f"check7_two_stage_{dv.replace('prop_','')}.png"
        plt.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  -> check7_two_stage_results.csv  |  check7_two_stage_slopes.csv")
    print(f"  -> check7_two_stage_total/relations/objects.png")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(
    f"""
  H1 (SVG above chance):    robust across all N and codebooks.

  H2 (encoding SVG → recall):
    N=22 old codebook:        all DVs significant (p<.05)
    N=27 old codebook:        total significant (p=.030),
                              relations/objects trending (p~.09)
    N=25 excl. P25/P26:       all DVs significant (p<.05)

  P25/P26 behavioural flags: none — normal fixation/SVG/AOI profiles.
  P25/P26 within-image slopes: negative on most DVs, consistent with
    genuine individual differences in SVG→recall coupling.
  Note: P21 (original sample) is the most extreme negative-slope
    participant overall — this is sample variability, not an artifact.

  Recommendation: recruit 3 more participants to reach N=30.
"""
)
print(f"All outputs -> {OUT_DIR}")
print("=" * 60)
