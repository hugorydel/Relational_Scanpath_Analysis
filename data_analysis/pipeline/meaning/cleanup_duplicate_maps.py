"""
pipeline/meaning/cleanup_duplicate_maps.py
===========================================
One-time cleanup: removes spurious duplicate .npz files from
third_party/deepmeaning/data/create_meaning_maps/output/outdoor/coca/.

Background
----------
A leftover patch_temp/ directory from an earlier interrupted run caused
batch_deep_meaning.py's outdoor pass to embed all 30 scenes (12 indoor +
18 outdoor) instead of just the 18 outdoor scenes. All 30 were then scored
with the OUTDOOR ensemble model and written to output/outdoor/coca/.

The 12 indoor StimIDs already have CORRECT maps (scored with the indoor
ensemble) in output/indoor/coca/ from an earlier successful run. This script
deletes the incorrect duplicates — any file in outdoor/coca/ whose stem also
appears in indoor/coca/ — leaving outdoor/coca/ with exactly the 18 correct
outdoor maps.

Usage:
    python -m pipeline.meaning.cleanup_duplicate_maps
    python -m pipeline.meaning.cleanup_duplicate_maps --dry-run   # preview only
"""

import argparse
import sys
from pathlib import Path

# data_analysis/ — this file lives at:
#   data_analysis/pipeline/meaning/cleanup_duplicate_maps.py
# data_analysis/ is three levels up. third_party/ lives inside data_analysis/.
_DATA_ANALYSIS_ROOT = Path(__file__).resolve().parent.parent.parent

_OUTPUT = (
    _DATA_ANALYSIS_ROOT
    / "third_party"
    / "deepmeaning"
    / "data"
    / "create_meaning_maps"
    / "output"
)

_INDOOR_DIR = _OUTPUT / "indoor" / "coca"
_OUTDOOR_DIR = _OUTPUT / "outdoor" / "coca"


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate indoor StimID maps from outdoor/coca/."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without deleting them.",
    )
    args = parser.parse_args()

    if not _INDOOR_DIR.exists():
        sys.exit(f"ERROR: {_INDOOR_DIR} does not exist.")
    if not _OUTDOOR_DIR.exists():
        sys.exit(f"ERROR: {_OUTDOOR_DIR} does not exist.")

    indoor_stems = {f.stem for f in _INDOOR_DIR.glob("*.npz")}
    outdoor_files = list(_OUTDOOR_DIR.glob("*.npz"))
    outdoor_stems = {f.stem for f in outdoor_files}

    print(f"indoor/coca/  : {len(indoor_stems)} files")
    print(f"outdoor/coca/ : {len(outdoor_stems)} files")

    duplicates = [f for f in outdoor_files if f.stem in indoor_stems]

    if not duplicates:
        print("\nNo duplicates found — nothing to clean up.")
        return

    print(
        f"\nFound {len(duplicates)} duplicate(s) in outdoor/coca/ "
        f"(also present in indoor/coca/):"
    )
    for f in duplicates:
        print(f"  {f.name}")

    if args.dry_run:
        print("\n--dry-run: no files deleted.")
        return

    for f in duplicates:
        f.unlink()

    remaining = len(list(_OUTDOOR_DIR.glob("*.npz")))
    print(f"\nDeleted {len(duplicates)} file(s).")
    print(f"outdoor/coca/ now has {remaining} files (expected: 18).")


if __name__ == "__main__":
    main()
