# Fix OpenMP conflict BEFORE importing other libraries
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Use non-interactive matplotlib backend to avoid Qt threading issues
import matplotlib

matplotlib.use("Agg")

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import config


class MemorabilityCache:
    """Manages caching of memorability scores."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.modified = False

    def _load_cache(self) -> Dict[str, float]:
        """Load existing cache from disk."""
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}

    def get(self, image_id: str) -> Optional[float]:
        """Get cached memorability score."""
        return self.cache.get(image_id)

    def set(self, image_id: str, score: float):
        """Store memorability score in cache."""
        self.cache[image_id] = float(score)
        self.modified = True

    def save(self):
        """Save cache to disk if modified."""
        if self.modified:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=2)
            print(f"Saved memorability cache with {len(self.cache)} entries")


class SVGDatasetCache:
    """Manages caching of preprocessed SVG dataset."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path

    def exists(self) -> bool:
        """Check if cache exists."""
        return self.cache_path.exists()

    def load(self) -> List[Dict]:
        """Load preprocessed dataset from cache."""
        print(f"Loading SVG dataset from cache: {self.cache_path}")
        with open(self.cache_path, "rb") as f:
            dataset = pickle.load(f)
        print(f"Loaded {len(dataset)} scene graphs from cache")
        return dataset

    def save(self, dataset: List[Dict]):
        """Save preprocessed dataset to cache."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved SVG dataset cache with {len(dataset)} entries")


class PrecomputedStatsCache:
    """
    Manages caching of precomputed image statistics.
    Caches expensive computations: memorability, coverage, object counts.
    This avoids recomputing when only config thresholds change.
    """

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.modified = False
        self._key_suffix_cache = {}  # Cache formatted suffixes

    def _load_cache(self) -> Dict:
        """Load existing cache from disk."""
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                cache = pickle.load(f)
                print(f"Loaded precomputed stats cache with {len(cache)} entries")
                return cache
        return {}

    def _get_cache_key(self, img_id: str, min_mask_area: float) -> str:
        """Generate cache key including mask area threshold (affects object counts)."""
        # OPTIMIZED: Cache the formatted suffix to avoid repeated string formatting
        if min_mask_area not in self._key_suffix_cache:
            self._key_suffix_cache[min_mask_area] = f"_{min_mask_area}"
        return f"{img_id}{self._key_suffix_cache[min_mask_area]}"

    def get(self, img_id: str, min_mask_area: float) -> Optional[Dict]:
        """
        Get cached stats for an image.

        Returns dict with: memorability, coverage_percent, n_objects, n_relations,
                          valid_objects, img_width, img_height
        """
        key = self._get_cache_key(img_id, min_mask_area)
        return self.cache.get(key)

    def set(self, img_id: str, min_mask_area: float, stats: Dict):
        """Store precomputed stats in cache."""
        key = self._get_cache_key(img_id, min_mask_area)
        self.cache[key] = stats
        self.modified = True

    def save(self):
        """Save cache to disk if modified."""
        if self.modified:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)
            print(f"Saved precomputed stats cache with {len(self.cache)} entries")
            self.modified = False
