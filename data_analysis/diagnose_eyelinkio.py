"""
diagnose_eyelinkio.py
=====================
Dumps the raw structure of what eyelinkio returns so we can see
the actual key names and data shapes.

Usage:
    python diagnose_eyelinkio.py --edf data_eyetracking/Encode-Decode_Experiment-1-1.edf
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf", required=True)
    args = parser.parse_args()

    import eyelinkio
    import numpy as np

    print(f"eyelinkio version: {eyelinkio.__version__}")

    edf_path = Path(args.edf)
    edf = eyelinkio.read_edf(str(edf_path))

    # --- Top-level object ---
    print("\n=== edf object type ===")
    print(type(edf))

    print("\n=== edf attributes / keys ===")
    if hasattr(edf, "__dict__"):
        for k, v in edf.__dict__.items():
            print(f"  {k}: {type(v).__name__}  = {repr(v)[:120]}")
    if hasattr(edf, "keys"):
        for k in edf.keys():
            print(f"  key: {k!r}  -> {type(edf[k]).__name__}")

    # --- Try to_pandas() and inspect returned keys ---
    print("\n=== edf.to_pandas() keys and shapes ===")
    try:
        dfs = edf.to_pandas()
        print(f"  type: {type(dfs)}")
        if hasattr(dfs, "keys"):
            for k in dfs.keys():
                v = dfs[k]
                import pandas as pd

                if isinstance(v, pd.DataFrame):
                    print(f"  {k!r}: DataFrame shape={v.shape}  cols={list(v.columns)}")
                    if len(v) > 0:
                        print(f"    first row: {dict(v.iloc[0])}")
                else:
                    print(f"  {k!r}: {type(v).__name__} = {repr(v)[:80]}")
        elif isinstance(dfs, dict):
            for k, v in dfs.items():
                print(f"  {k!r}: {type(v).__name__}")
        else:
            print(f"  returned: {repr(dfs)[:200]}")
    except Exception as e:
        print(f"  to_pandas() failed: {e}")

    # --- Try accessing discrete events directly ---
    print("\n=== direct attribute access attempts ===")
    for attr in [
        "discrete",
        "events",
        "messages",
        "fixations",
        "saccades",
        "blinks",
        "samples",
        "recording_blocks",
        "recordings",
        "_data",
        "data",
    ]:
        if hasattr(edf, attr):
            val = getattr(edf, attr)
            print(f"  edf.{attr}: {type(val).__name__}  = {repr(val)[:120]}")

    # --- If it's dict-like, dump all keys recursively ---
    print("\n=== full repr (truncated) ===")
    r = repr(edf)
    print(r[:3000])


if __name__ == "__main__":
    main()
