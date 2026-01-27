#!/usr/bin/env python3
"""
Extract image IDs from existing results.jsonl for comparison experiment.

Usage:
    python extract_image_ids.py --input results.jsonl --output image_ids.txt
"""

import argparse
import json
from pathlib import Path


def extract_image_ids(input_path: Path, output_path: Path):
    """Extract all image IDs from results.jsonl."""
    image_ids = []

    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                image_ids.append(result["image_id"])

    # Write image IDs to file (one per line)
    with open(output_path, "w") as f:
        for img_id in image_ids:
            f.write(f"{img_id}\n")

    print(f"✓ Extracted {len(image_ids)} image IDs from {input_path}")
    print(f"✓ Saved to {output_path}")

    return image_ids


def main():
    parser = argparse.ArgumentParser(
        description="Extract image IDs from results.jsonl for comparison experiment"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/scored_images/results.jsonl",
        help="Input results file (default: results.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="image_ids.txt",
        help="Output file for image IDs (default: image_ids.txt)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    output_path = Path(args.output)
    extract_image_ids(input_path, output_path)

    return 0


if __name__ == "__main__":
    exit(main())
