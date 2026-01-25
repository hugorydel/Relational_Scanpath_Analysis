#!/usr/bin/env python3
"""
Download Selected Stimuli Images

This script copies all raw images that were marked as 'selected' during manual
annotation to a clean output directory for easy download/use in experiments.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List

from utils import ensure_jpg


class StimuliDownloader:
    """Downloads selected raw stimuli images."""

    def __init__(
        self,
        dataset_dir: str = "data/diverse_selection",
        stimuli_dir: str = "data/stimuli",
        annotations_file: str = "manual_annotations.json",
        output_dir: str = "data/selected_stimuli_raw",
    ):
        """
        Initialize the downloader.

        Args:
            dataset_dir: Path to diverse selection dataset directory (for images)
            stimuli_dir: Path to stimuli directory (for manual_annotations.json)
            annotations_file: Name of manual annotations file
            output_dir: Path to output directory for raw selected images
        """
        self.dataset_dir = Path(dataset_dir)
        self.stimuli_dir = Path(stimuli_dir)
        self.images_dir = self.dataset_dir / "images"
        self.annotations_path = self.stimuli_dir / "annotations" / annotations_file
        self.output_dir = Path(output_dir)

        # Validate paths
        if not self.annotations_path.exists():
            raise FileNotFoundError(
                f"Annotations file not found: {self.annotations_path}\n"
                f"Please run manual_image_annotation.py first."
            )

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}\n"
                f"Please check your dataset_dir path."
            )

        # Load annotations
        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)

    def get_selected_images(self) -> List[Dict]:
        """Get list of selected image metadata."""
        selected = []
        for img_id, annotation in self.annotations.items():
            if annotation.get("status") == "selected":
                selected.append({"image_id": img_id, **annotation})
        return selected

    def download_selected_images(self) -> None:
        """Copy selected raw images to output directory."""
        selected_images = self.get_selected_images()

        if len(selected_images) == 0:
            print("\n❌ No images marked as 'selected' found!")
            print("Please run manual_image_annotation.py to select images first.")
            return

        print("=" * 60)
        print("DOWNLOADING SELECTED RAW STIMULI")
        print("=" * 60)
        print(f"Found {len(selected_images)} selected images\n")

        # Create output directory
        if self.output_dir.exists():
            print(f"⚠️  Output directory already exists: {self.output_dir}")
            response = input("Overwrite? (y/n): ").lower()
            if response != "y":
                print("Cancelled.")
                return
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        copied_count = 0
        missing_count = 0

        print("Copying images...")
        for img_meta in selected_images:
            img_id = img_meta["image_id"]
            file_name = img_meta.get("file_name", ensure_jpg(img_id))

            src_path = self.images_dir / file_name
            dst_path = self.output_dir / file_name

            if src_path.exists():
                shutil.copy(src_path, dst_path)
                copied_count += 1
                print(f"  ✓ {file_name}")
            else:
                print(f"  ❌ Missing: {file_name}")
                missing_count += 1

        # Save metadata
        metadata = {
            "info": {
                "description": "Selected Raw Stimuli Images",
                "num_images": len(selected_images),
                "source": "Manual annotation selection",
            },
            "images": selected_images,
        }

        metadata_path = self.output_dir / "selected_stimuli_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"✓ Copied {copied_count} raw images")
        if missing_count > 0:
            print(f"❌ Missing {missing_count} images (not found in source)")
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Metadata saved: {metadata_path}")
        print("=" * 60)

        # Print image details
        print("\nSelected Images:")
        for img_meta in selected_images:
            img_id = img_meta["image_id"]
            print(
                f"  - {img_id}: {img_meta['n_objects']} objects, "
                f"{img_meta['n_relations']} relations, "
                f"{img_meta['coverage_percent']:.1f}% coverage, "
                f"mem={img_meta['memorability']:.3f}"
            )


def main():
    """Main entry point."""
    # Configuration
    DATASET_DIR = "data/diverse_selection"
    STIMULI_DIR = "data/stimuli"
    ANNOTATIONS_FILE = "manual_annotations.json"
    OUTPUT_DIR = "data/selected_stimuli_raw"

    print("\n" + "=" * 60)
    print("SELECTED STIMULI DOWNLOADER")
    print("=" * 60)

    try:
        downloader = StimuliDownloader(
            dataset_dir=DATASET_DIR,
            stimuli_dir=STIMULI_DIR,
            annotations_file=ANNOTATIONS_FILE,
            output_dir=OUTPUT_DIR,
        )

        downloader.download_selected_images()

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
