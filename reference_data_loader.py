"""
reference_data_loader.py - Simple loader for precomputed reference data

Replaces the old caching system with a straightforward JSON loader.
"""

import json
from pathlib import Path
from typing import Dict, Optional


class ReferenceDataLoader:
    """Loads and provides access to precomputed filtering statistics."""

    # Required fields for filtering
    REQUIRED_FIELDS = [
        "memorability",
        "coverage_percent",
        "n_objects",
        "n_relations",
        "n_interactional_relations",
        "img_width",
        "img_height",
    ]

    def __init__(self, json_path: Path):
        """
        Initialize loader.

        Args:
            json_path: Path to precomputed_stats.json
        """
        self.json_path = json_path
        self.metadata = None
        self.images = None
        self._load_data()

    def _load_data(self):
        """Load JSON data from disk."""
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"Reference data not found: {self.json_path}\n"
                f"Please ensure precomputed_stats.json exists in the data/ directory."
            )

        print(f"Loading reference data from {self.json_path}...")
        with open(self.json_path, "r") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        self.images = data.get("images", {})

        print(f"✓ Loaded {len(self.images)} images from reference data")
        print(f"  Version: {self.metadata.get('version', 'unknown')}")
        print(
            f"  Min mask area: {self.metadata.get('min_mask_area_percent', 'unknown')}%\n"
        )

    def get(self, img_id: str) -> Optional[Dict]:
        """
        Get statistics for an image.

        Args:
            img_id: Image ID (e.g., "2345678.jpg")

        Returns:
            Dict with statistics, or None if not found
        """
        return self.images.get(img_id)

    def validate_entry(self, img_id: str, stats: Dict) -> bool:
        """
        Validate that an entry has all required fields.

        Args:
            img_id: Image ID
            stats: Statistics dict

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(stats, dict):
            print(f"Warning: {img_id} has invalid data type")
            return False

        missing_fields = [field for field in self.REQUIRED_FIELDS if field not in stats]

        if missing_fields:
            print(f"Warning: {img_id} missing fields: {missing_fields}")
            return False

        # Basic value validation
        try:
            # Check ranges
            if not (0 <= stats["memorability"] <= 1):
                print(f"Warning: {img_id} has invalid memorability")
                return False

            if not (0 <= stats["coverage_percent"] <= 100):
                print(f"Warning: {img_id} has invalid coverage_percent")
                return False

            # Check types and non-negative
            for field in [
                "n_objects",
                "n_relations",
                "n_interactional_relations",
                "img_width",
                "img_height",
            ]:
                if not isinstance(stats[field], int) or stats[field] < 0:
                    print(f"Warning: {img_id} has invalid {field}")
                    return False

            # Logical check
            if stats["n_interactional_relations"] > stats["n_relations"]:
                print(f"Warning: {img_id} has more interactional than total relations")
                return False

        except (TypeError, KeyError) as e:
            print(f"Warning: {img_id} validation error: {e}")
            return False

        return True

    def items(self):
        """Iterate over (img_id, stats) pairs."""
        return self.images.items()

    def __len__(self):
        """Return number of images in reference data."""
        return len(self.images)

    def get_metadata(self) -> Dict:
        """Get metadata about the reference dataset."""
        return self.metadata.copy()
