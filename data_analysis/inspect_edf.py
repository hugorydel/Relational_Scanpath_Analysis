"""
inspect_edf.py
==============
Standalone EDF inspection script — run this BEFORE writing Module 2.

Usage:
    python inspect_edf.py --edf data_eyetracking/Encode-Decode_Experiment-1-1.edf

Output is printed to console and saved to:
    output/edf_inspection_{subject_id}.txt
"""

import argparse
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EDF reading
# ---------------------------------------------------------------------------


def read_via_eyelinkio(edf_path: Path) -> dict | None:
    try:
        import eyelinkio
        import numpy as np
        import pandas as pd

        logger.info("eyelinkio found — reading EDF directly...")
        edf = eyelinkio.read_edf(str(edf_path))

        discrete = edf.discrete  # the main dict of events

        logger.info(f"edf.discrete keys: {list(discrete.keys())}")

        def to_df(key):
            """Convert a numpy structured array in discrete to a DataFrame."""
            if key not in discrete:
                return pd.DataFrame()
            arr = discrete[key]
            if arr is None or len(arr) == 0:
                return pd.DataFrame()
            # structured numpy array → DataFrame
            if hasattr(arr, "dtype") and arr.dtype.names:
                return pd.DataFrame(arr)
            return pd.DataFrame(arr)

        messages = to_df("messages")
        fixations = to_df("fixations")
        saccades = to_df("saccades")
        blinks = to_df("blinks")

        # Decode bytes columns in messages
        for col in messages.columns:
            if messages[col].dtype == object or str(messages[col].dtype) == "object":
                try:
                    messages[col] = messages[col].apply(
                        lambda x: (
                            x.decode("utf-8", errors="replace")
                            if isinstance(x, bytes)
                            else x
                        )
                    )
                except Exception:
                    pass

        # eyelinkio stores time in seconds — convert to ms for consistency
        for df in [messages, fixations, saccades, blinks]:
            for col in df.columns:
                if col in ("stime", "etime") or "time" in col.lower():
                    if df[col].abs().max() < 10000:  # likely in seconds
                        df[col] = (df[col] * 1000).round().astype(int)

        # Also expose raw times array (in seconds → ms)
        times_ms = (edf["times"] * 1000).round().astype(int)

        return {
            "source": "eyelinkio",
            "info": dict(edf.info),
            "messages": messages,
            "fixations": fixations,
            "saccades": saccades,
            "blinks": blinks,
            "times_ms": times_ms,
            "discrete_keys": list(discrete.keys()),
        }

    except ImportError:
        logger.warning("eyelinkio not installed. Run: pip install eyelinkio")
        return None
    except Exception as e:
        logger.error(f"eyelinkio failed: {e}", exc_info=True)
        return None


def find_edf2asc() -> Path | None:
    candidates = [
        Path(
            "C:/Program Files (x86)/SR Research/EyeLink/EDF_Access_API/Example/edf2asc.exe"
        ),
        Path("C:/Program Files/SR Research/EyeLink/EDF_Access_API/Example/edf2asc.exe"),
        Path("C:/Program Files (x86)/SR Research/EyeLink/bin/edf2asc.exe"),
        Path("C:/Program Files/SR Research/EyeLink/bin/edf2asc.exe"),
        Path("edf2asc.exe"),
        Path("edf2asc"),
    ]
    for p in candidates:
        try:
            subprocess.run([str(p)], capture_output=True, timeout=3)
            logger.info(f"Found edf2asc at: {p}")
            return p
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return None


def convert_and_read_asc(edf_path: Path) -> dict | None:
    import pandas as pd

    asc_path = edf_path.with_suffix(".asc")
    if not asc_path.exists():
        edf2asc = find_edf2asc()
        if edf2asc is None:
            logger.error("edf2asc not found and eyelinkio failed. Cannot read EDF.")
            return None
        logger.info(f"Converting via edf2asc...")
        subprocess.run([str(edf2asc), str(edf_path)], capture_output=True)

    if not asc_path.exists():
        logger.error("ASC conversion failed.")
        return None

    msgs, fixations, saccades = [], [], []
    starts, ends = [], []

    with asc_path.open(encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("MSG"):
                parts = line.split(None, 2)
                ts = int(parts[1]) if len(parts) > 1 else -1
                txt = parts[2].strip() if len(parts) > 2 else ""
                msgs.append({"stime": ts, "msg": txt})
            elif line.startswith("START"):
                starts.append(line)
            elif line.startswith("END"):
                ends.append(line)
            elif line.startswith("EFIX"):
                p = line.split()
                if len(p) >= 8:
                    try:
                        fixations.append(
                            {
                                "eye": p[1],
                                "stime": int(p[2]),
                                "etime": int(p[3]),
                                "dur": int(p[4]),
                                "gavx": float(p[5]),
                                "gavy": float(p[6]),
                            }
                        )
                    except ValueError:
                        pass
            elif line.startswith("ESACC"):
                p = line.split()
                if len(p) >= 10:
                    try:
                        saccades.append(
                            {
                                "eye": p[1],
                                "stime": int(p[2]),
                                "etime": int(p[3]),
                                "dur": int(p[4]),
                                "gstx": float(p[5]),
                                "gsty": float(p[6]),
                                "genx": float(p[7]),
                                "geny": float(p[8]),
                                "avel": float(p[9]),
                            }
                        )
                    except ValueError:
                        pass

    def _ts(lines):
        out = []
        for l in lines:
            p = l.split()
            if len(p) >= 2:
                try:
                    out.append(int(p[1]))
                except ValueError:
                    pass
        return out

    return {
        "source": "asc",
        "info": {"file": str(asc_path)},
        "messages": pd.DataFrame(msgs),
        "fixations": pd.DataFrame(fixations),
        "saccades": pd.DataFrame(saccades),
        "blinks": pd.DataFrame(),
        "start_times": _ts(starts),
        "end_times": _ts(ends),
        "discrete_keys": ["messages", "fixations", "saccades"],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(data: dict, subject_id: str, output_dir: Path):
    import pandas as pd

    out_path = output_dir / f"edf_inspection_{subject_id}.txt"
    lines_out = []

    def log(s=""):
        print(s)
        lines_out.append(str(s))

    log(f"EDF INSPECTION: {subject_id}  (source: {data['source']})")
    log("=" * 70)

    log("\n[RECORDING INFO]")
    for k, v in data["info"].items():
        if k != "calibrations":
            log(f"  {k}: {v}")

    msgs = data["messages"]
    fixes = data["fixations"]
    saccs = data["saccades"]

    log(f"\n  discrete keys  : {data.get('discrete_keys', 'n/a')}")
    log(f"  Messages       : {len(msgs)}")
    log(f"  Fixations      : {len(fixes)}")
    log(f"  Saccades       : {len(saccs)}")
    if "start_times" in data:
        log(f"  START markers  : {len(data['start_times'])}")
        log(f"  END markers    : {len(data['end_times'])}")
    log(f"\n  Expected epochs: 90  (60 encoding × ~5000ms + 30 decoding × ~7000ms)")
    log(
        f"  Total recording: {data['times_ms'][-1] - data['times_ms'][0]} ms"
        if "times_ms" in data and len(data["times_ms"]) > 0
        else ""
    )

    # --- All messages ---
    log("\n" + "=" * 70)
    log(f"ALL MSG EVENTS ({len(msgs)} total)")
    log("=" * 70)
    if len(msgs) > 0:
        log(f"  Columns: {list(msgs.columns)}")
        log("")
        for _, row in msgs.iterrows():
            log(f"  {dict(row)}")
    else:
        log("  (no messages found)")

    # --- Unique message strings ---
    if len(msgs) > 0:
        msg_col = "msg" if "msg" in msgs.columns else msgs.columns[-1]
        log("\n" + "=" * 70)
        log("UNIQUE MESSAGE STRINGS (frequency sorted)")
        log("=" * 70)
        counts = Counter(msgs[msg_col].astype(str).tolist())
        for txt, n in sorted(counts.items(), key=lambda x: -x[1]):
            log(f"  {n:4d}x  {txt!r}")

    # --- Epoch analysis (eyelinkio: derive from recording time + fixation gaps) ---
    if "times_ms" in data and len(data["times_ms"]) > 0:
        log("\n" + "=" * 70)
        log("RECORDING TIME ANALYSIS")
        log("=" * 70)
        t = data["times_ms"]
        log(f"  Sample count   : {len(t)}")
        log(f"  Start time     : {t[0]} ms")
        log(f"  End time       : {t[-1]} ms")
        log(f"  Total duration : {t[-1] - t[0]} ms  ({(t[-1]-t[0])/60000:.1f} min)")
        import numpy as np

        diffs = np.diff(t)
        gaps = diffs[diffs > 100]  # gaps > 100ms = recording was off
        log(f"  Gaps > 100ms   : {len(gaps)}  (= number of between-trial pauses)")
        log(f"  Expected gaps  : 90  (one per trial)")
        if len(gaps) > 0:
            log(
                f"  Gap durations  : min={gaps.min()}ms  max={gaps.max()}ms  mean={gaps.mean():.0f}ms"
            )
            log(f"\n  All gaps (should cluster around ~5000ms and ~7000ms):")
            for i, g in enumerate(sorted(gaps)):
                log(f"    gap {i+1:3d}: {g} ms")

    # --- Epoch analysis (ASC: from START/END markers) ---
    if "start_times" in data:
        st = data["start_times"]
        en = data["end_times"]
        if len(st) == len(en) and len(st) > 0:
            durations = [e - s for s, e in zip(st, en)]
            log("\n" + "=" * 70)
            log("EPOCH DURATION ANALYSIS (from START/END markers)")
            log("=" * 70)
            log(f"  Epoch count  : {len(durations)}")
            log(f"  Min duration : {min(durations)} ms")
            log(f"  Max duration : {max(durations)} ms")
            log(f"  Mean duration: {sum(durations)/len(durations):.0f} ms")
            log(f"\n  All epoch durations:")
            for i, (s, d) in enumerate(zip(st, durations)):
                log(f"    Epoch {i+1:3d}: start={s}  dur={d} ms")

    # --- Fixations ---
    if len(fixes) > 0:
        log("\n" + "=" * 70)
        log(f"FIXATION SUMMARY ({len(fixes)} total)")
        log("=" * 70)
        log(f"  Columns: {list(fixes.columns)}")
        dur_col = next((c for c in fixes.columns if "dur" in c.lower()), None)
        if dur_col:
            log(f"  Min dur : {fixes[dur_col].min():.0f} ms")
            log(f"  Max dur : {fixes[dur_col].max():.0f} ms")
            log(f"  Mean dur: {fixes[dur_col].mean():.0f} ms")
        log(f"\n  First 20 fixations:")
        log(fixes.head(20).to_string())

    # --- Saccades ---
    if len(saccs) > 0:
        log("\n" + "=" * 70)
        log(f"SACCADE SUMMARY ({len(saccs)} total)")
        log("=" * 70)
        log(f"  Columns: {list(saccs.columns)}")
        log(f"\n  First 20 saccades:")
        log(saccs.head(20).to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines_out), encoding="utf-8")
    logger.info(f"\nInspection report saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="EDF Inspection Script")
    parser.add_argument("--edf", required=True, help="Path to .edf file")
    parser.add_argument("--output", default="output/", help="Output directory")
    args = parser.parse_args()

    edf_path = Path(args.edf)
    if not edf_path.exists():
        logger.error(f"EDF file not found: {edf_path}")
        sys.exit(1)

    subject_id = edf_path.stem
    output_dir = Path(args.output)

    data = read_via_eyelinkio(edf_path)
    if data is None:
        data = convert_and_read_asc(edf_path)
    if data is None:
        logger.error("All reading methods failed.")
        sys.exit(1)

    report(data, subject_id, output_dir)


if __name__ == "__main__":
    main()
