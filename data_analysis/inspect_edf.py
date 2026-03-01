"""
inspect_edf.py
==============
Standalone EDF inspection script — run this BEFORE writing Module 2.

Attempts to read the EDF file using eyelinkio (pip install eyelinkio).
If that fails, falls back to locating edf2asc.exe on the system and
converting to ASC automatically, then parsing the ASC.

Usage:
    pip install eyelinkio
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
# EDF reading — eyelinkio primary, edf2asc subprocess fallback
# ---------------------------------------------------------------------------


def read_via_eyelinkio(edf_path: Path) -> dict | None:
    """
    Try to read EDF using eyelinkio.
    Returns a dict with keys: 'messages', 'fixations', 'saccades', 'info'
    or None if eyelinkio is unavailable or fails.
    """
    try:
        import eyelinkio

        logger.info("eyelinkio found — reading EDF directly...")
        edf = eyelinkio.read_edf(str(edf_path))

        import pandas as pd

        dfs = edf.to_pandas()

        return {
            "source": "eyelinkio",
            "info": dict(edf.info),
            "messages": dfs.get("messages", pd.DataFrame()),
            "fixations": dfs.get("fixations", pd.DataFrame()),
            "saccades": dfs.get("saccades", pd.DataFrame()),
            "blinks": dfs.get("blinks", pd.DataFrame()),
            "discrete": dfs,
        }

    except ImportError:
        logger.warning("eyelinkio not installed. Run: pip install eyelinkio")
        return None
    except Exception as e:
        logger.warning(f"eyelinkio failed: {e}")
        return None


def find_edf2asc() -> Path | None:
    """
    Search common install locations for edf2asc.exe on Windows.
    """
    candidates = [
        # Standard EyeLink Developer Kit locations
        Path(
            "C:/Program Files (x86)/SR Research/EyeLink/EDF_Access_API/Example/edf2asc.exe"
        ),
        Path("C:/Program Files/SR Research/EyeLink/EDF_Access_API/Example/edf2asc.exe"),
        Path("C:/Program Files (x86)/SR Research/EyeLink/bin/edf2asc.exe"),
        Path("C:/Program Files/SR Research/EyeLink/bin/edf2asc.exe"),
        # Sometimes on PATH
        Path("edf2asc.exe"),
        Path("edf2asc"),
    ]
    for p in candidates:
        try:
            result = subprocess.run([str(p)], capture_output=True, timeout=3)
            # edf2asc returns non-zero when called with no args but that's OK
            logger.info(f"Found edf2asc at: {p}")
            return p
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return None


def convert_edf_to_asc(edf_path: Path) -> Path | None:
    """
    Use edf2asc.exe to convert EDF → ASC in the same directory.
    Returns path to the .asc file, or None if conversion failed.
    """
    asc_path = edf_path.with_suffix(".asc")
    if asc_path.exists():
        logger.info(f"ASC file already exists: {asc_path}")
        return asc_path

    edf2asc = find_edf2asc()
    if edf2asc is None:
        logger.error(
            "edf2asc not found. Please install EyeLink Developer Kit or run:\n"
            "  pip install eyelinkio"
        )
        return None

    logger.info(f"Converting {edf_path.name} → ASC using {edf2asc}...")
    result = subprocess.run(
        [str(edf2asc), str(edf_path)], capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(f"edf2asc failed:\n{result.stderr}")
        return None

    if asc_path.exists():
        logger.info(f"Conversion successful: {asc_path}")
        return asc_path

    logger.error("edf2asc ran but .asc file not found.")
    return None


def read_via_asc(asc_path: Path) -> dict | None:
    """
    Parse a .asc file directly into the same dict structure.
    """
    import pandas as pd

    msgs = []  # (timestamp, text)
    starts = []  # raw START lines
    ends = []  # raw END lines
    fixations = []
    saccades = []

    logger.info(f"Parsing ASC file: {asc_path.name} ...")

    with asc_path.open(encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("MSG"):
                parts = line.split(None, 2)
                ts = int(parts[1]) if len(parts) > 1 else -1
                txt = parts[2].strip() if len(parts) > 2 else ""
                msgs.append({"time": ts, "text": txt})

            elif line.startswith("START"):
                starts.append(line)

            elif line.startswith("END"):
                ends.append(line)

            elif line.startswith("EFIX"):
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        fixations.append(
                            {
                                "eye": parts[1],
                                "start_ms": int(parts[2]),
                                "end_ms": int(parts[3]),
                                "dur_ms": int(parts[4]),
                                "x": float(parts[5]),
                                "y": float(parts[6]),
                                "pupil": float(parts[7]),
                            }
                        )
                    except ValueError:
                        pass

            elif line.startswith("ESACC"):
                parts = line.split()
                if len(parts) >= 11:
                    try:
                        saccades.append(
                            {
                                "eye": parts[1],
                                "start_ms": int(parts[2]),
                                "end_ms": int(parts[3]),
                                "dur_ms": int(parts[4]),
                                "start_x": float(parts[5]),
                                "start_y": float(parts[6]),
                                "end_x": float(parts[7]),
                                "end_y": float(parts[8]),
                                "amplitude": float(parts[9]),
                            }
                        )
                    except ValueError:
                        pass

    # Extract timestamps from START/END lines
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

    start_times = _ts(starts)
    end_times = _ts(ends)

    return {
        "source": "asc",
        "info": {"file": str(asc_path)},
        "messages": pd.DataFrame(msgs),
        "fixations": pd.DataFrame(fixations),
        "saccades": pd.DataFrame(saccades),
        "start_times": start_times,
        "end_times": end_times,
        "n_starts": len(starts),
        "n_ends": len(ends),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(data: dict, subject_id: str, output_dir: Path):
    out_path = output_dir / f"edf_inspection_{subject_id}.txt"
    lines_out = []

    def log(s=""):
        print(s)
        lines_out.append(str(s))

    log(f"EDF INSPECTION: {subject_id}  (source: {data['source']})")
    log("=" * 70)

    # --- Info ---
    log("\n[INFO]")
    for k, v in data["info"].items():
        log(f"  {k}: {v}")

    msgs = data["messages"]
    fixes = data["fixations"]
    saccs = data["saccades"]

    log(f"\nMessages  : {len(msgs)}")
    log(f"Fixations : {len(fixes)}")
    log(f"Saccades  : {len(saccs)}")

    if "n_starts" in data:
        log(f"START markers : {data['n_starts']}")
        log(f"END markers   : {data['n_ends']}")

    log(f"\nExpected epochs: 90  (60 encoding × 5s + 30 decoding × 7s)")

    # --- All messages ---
    log("\n" + "=" * 70)
    log(f"ALL MSG EVENTS ({len(msgs)} total)")
    log("=" * 70)
    if len(msgs) > 0:
        log(f"Columns: {list(msgs.columns)}")
        log("")
        for _, row in msgs.iterrows():
            log(f"  {dict(row)}")
    else:
        log("  (no messages found)")

    # --- Unique message texts ---
    log("\n" + "=" * 70)
    log("UNIQUE MESSAGE TEXT VALUES")
    log("=" * 70)
    if len(msgs) > 0 and "text" in msgs.columns:
        counts = Counter(msgs["text"].tolist())
        for txt, n in sorted(counts.items(), key=lambda x: -x[1]):
            log(f"  {n:4d}x  {txt!r}")

    # --- Epoch duration analysis (ASC source) ---
    if data["source"] == "asc" and data.get("start_times") and data.get("end_times"):
        st = data["start_times"]
        en = data["end_times"]
        if len(st) == len(en):
            durations = [e - s for s, e in zip(st, en)]
            log("\n" + "=" * 70)
            log("EPOCH DURATION ANALYSIS")
            log("=" * 70)
            log(f"  Epoch count  : {len(durations)}")
            log(f"  Min duration : {min(durations)} ms")
            log(f"  Max duration : {max(durations)} ms")
            log(f"  Mean duration: {sum(durations)/len(durations):.0f} ms")
            log(f"\n  Expected: ~5000 ms (encoding) and ~7000 ms (decoding)")
            log(f"\n  All epoch durations:")
            for i, (s, d) in enumerate(zip(st, durations)):
                log(f"    Epoch {i+1:3d}: start={s}  dur={d} ms")

    # --- Fixation summary ---
    if len(fixes) > 0:
        log("\n" + "=" * 70)
        log("FIXATION SUMMARY")
        log("=" * 70)
        log(f"  Columns  : {list(fixes.columns)}")
        dur_col = next((c for c in fixes.columns if "dur" in c.lower()), None)
        if dur_col:
            log(f"  Count    : {len(fixes)}")
            log(f"  Min dur  : {fixes[dur_col].min():.0f} ms")
            log(f"  Max dur  : {fixes[dur_col].max():.0f} ms")
            log(f"  Mean dur : {fixes[dur_col].mean():.0f} ms")
        log(f"\n  First 10 fixations:")
        log(fixes.head(10).to_string())

    # --- Saccade summary ---
    if len(saccs) > 0:
        log("\n" + "=" * 70)
        log("SACCADE SUMMARY")
        log("=" * 70)
        log(f"  Columns: {list(saccs.columns)}")
        log(f"  Count  : {len(saccs)}")
        log(f"\n  First 10 saccades:")
        log(saccs.head(10).to_string())

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines_out), encoding="utf-8")
    logger.info(f"\nInspection report saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="EDF Inspection Script")
    parser.add_argument("--edf", required=True, help="Path to .edf file")
    parser.add_argument(
        "--output", default="output/", help="Output directory for report"
    )
    args = parser.parse_args()

    edf_path = Path(args.edf)
    if not edf_path.exists():
        logger.error(f"EDF file not found: {edf_path}")
        sys.exit(1)

    subject_id = edf_path.stem
    output_dir = Path(args.output)

    # Try eyelinkio first
    data = read_via_eyelinkio(edf_path)

    # Fallback: convert to ASC and parse
    if data is None:
        logger.info("Falling back to edf2asc conversion...")
        asc_path = convert_edf_to_asc(edf_path)
        if asc_path is None:
            logger.error("All EDF reading methods failed. Cannot proceed.")
            sys.exit(1)
        data = read_via_asc(asc_path)

    if data is None:
        logger.error("Failed to read EDF data.")
        sys.exit(1)

    report(data, subject_id, output_dir)


if __name__ == "__main__":
    main()
