#!/usr/bin/env python3
"""
MS-COCO Naturalistic Stimuli Dataset Creator
"""

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm import tqdm

# Import configuration if available
try:
    import utils.config as config

    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


class COCONaturalisticDataset:
    """Filter and process MS-COCO for naturalistic stimuli."""

    def __init__(
        self,
        coco_root: str = None,
        output_dir: str = "./coco_naturalistic_stimuli",
        min_area_percent: float = 0.5,
        min_objects: int = 6,
        min_coverage_percent: float = 85.0,
        target_size: Tuple[int, int] = (1024, 768),
        target_count: int = 200,
    ):
        """
        Initialize the dataset creator.

        Args:
            coco_root: Path to COCO dataset root (if None, tries to load from config)
            output_dir: Where to save processed images and annotations
            min_area_percent: Minimum area % to count toward object threshold (0.5 = 0.5%)
            min_objects: Minimum number of objects (>= min_area_percent) required
            min_coverage_percent: Minimum % of image covered by AOIs (85 = 85%)
            target_size: Fixed canvas size (width, height) for letterboxing
            target_count: Target number of images to select
        """
        # Handle config
        if coco_root is None:
            if HAS_CONFIG:
                self.coco_root = Path(config.COCO_PATH)
            else:
                raise ValueError(
                    "coco_root must be provided or config.COCO_PATH must be set"
                )
        else:
            self.coco_root = Path(coco_root)

        self.output_dir = Path(output_dir)
        self.min_area_percent = min_area_percent
        self.min_objects = min_objects  # NEW
        self.min_coverage_percent = min_coverage_percent
        self.target_size = target_size
        self.target_count = target_count

        # Create output directories (delete existing contents if present)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        print(f"Created fresh output directory: {self.output_dir}")

        # Load COCO API
        ann_file = self.coco_root / "annotations" / "instances_train2017.json"
        if not ann_file.exists():
            ann_file = self.coco_root / "annotations" / "instances_val2017.json"

        print(f"Loading COCO annotations from {ann_file}...")
        self.coco = COCO(str(ann_file))

        # Color palette for visualization
        self.colors = self._generate_colors(len(self.coco.getCatIds()))

    def _generate_colors(self, n: int) -> np.ndarray:
        """Generate distinguishable colors for visualization."""
        np.random.seed(42)
        colors = sns.color_palette("husl", n)
        return np.array(colors)

    def _calculate_area_percent(self, ann: Dict, img_area: float) -> float:
        """Calculate annotation area as percentage of image."""
        if "area" in ann:
            return (ann["area"] / img_area) * 100
        return 0.0

    def _calculate_coverage_percent(
        self, anns: List[Dict], img_width: int, img_height: int
    ) -> float:
        """
        Calculate true coverage using segmentation masks to avoid double-counting overlaps.
        Returns percentage of image covered by union of all masks.
        """
        img_area = img_width * img_height

        # Create binary mask for the entire image
        coverage_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        for ann in anns:
            # Get segmentation mask
            if "segmentation" in ann:
                seg = ann["segmentation"]

                # Check if it's a list (polygon format) - most common in COCO
                if isinstance(seg, list) and len(seg) > 0:
                    # Check if it's a list of polygons (not RLE)
                    if isinstance(seg[0], list):
                        # Polygon format - convert to RLE then decode
                        try:
                            rle = mask_utils.frPyObjects(seg, img_height, img_width)
                            mask = mask_utils.decode(rle)
                            if len(mask.shape) == 3:
                                mask = mask.max(axis=2)  # Combine multiple polygons
                        except:
                            # Skip if conversion fails
                            continue
                    else:
                        # Might be RLE in list format, skip for now
                        continue
                elif isinstance(seg, dict) and "counts" in seg:
                    # RLE format as dict
                    try:
                        mask = mask_utils.decode(seg)
                    except:
                        # Skip if decode fails
                        continue
                else:
                    # Unknown format, skip
                    continue

                # Add to coverage mask (union)
                coverage_mask = np.maximum(coverage_mask, mask)

        # Calculate percentage
        covered_pixels = np.sum(coverage_mask > 0)
        coverage_percent = (covered_pixels / img_area) * 100

        return coverage_percent

    def _get_dominant_category(self, anns: List[Dict]) -> int:
        """
        Get the most frequent category in the image.
        Returns category_id of the dominant category.
        """
        if not anns:
            return -1

        # Count instances by category
        cat_counts = Counter(ann["category_id"] for ann in anns)
        # Return most common category
        return cat_counts.most_common(1)[0][0]

    def _calculate_quality_score(self, img_data: Dict) -> float:
        """
        Calculate quality score based on AOI count and sizes.
        Higher score = more AOIs and better size distribution.
        """
        counted_anns = img_data["annotations_filtered"]  # Objects >= min_area_percent
        img_area = img_data["image_info"]["width"] * img_data["image_info"]["height"]

        # Component 1: Number of counted AOIs (normalized to 0-1) - HIGHEST WEIGHT
        aoi_count = len(counted_anns)
        aoi_score = min(aoi_count / 30, 1.0)  # Cap at 30

        # Component 2: Average AOI size as % of image
        if len(counted_anns) > 0:
            avg_size = np.mean([ann["area"] / img_area * 100 for ann in counted_anns])
            size_score = min(avg_size / 10, 1.0)  # Cap at 10%
        else:
            size_score = 0.0

        # Component 3: Coverage (already filtered to be >min_coverage)
        coverage = img_data["coverage_percent"]
        coverage_score = min(coverage / 100, 1.0)

        # Weighted combination (prioritize object count)
        quality_score = 0.6 * aoi_score + 0.25 * coverage_score + 0.15 * size_score

        return quality_score

    def _letterbox_image(
        self, img: np.ndarray, target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Letterbox image to target size while maintaining aspect ratio.

        Returns:
            letterboxed_img: Padded image
            scale: Scale factor applied
            offset: (x_offset, y_offset) for coordinate transformation
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Calculate scale to fit image within target
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create letterboxed canvas
        letterboxed = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        # Calculate offsets for centering
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        # Place resized image on canvas
        letterboxed[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return letterboxed, scale, (x_offset, y_offset)

    def _transform_annotations(
        self, anns: List[Dict], scale: float, offset: Tuple[int, int]
    ) -> List[Dict]:
        """Transform annotation coordinates for letterboxed image."""
        x_off, y_off = offset

        transformed = []
        for ann in anns.copy():
            # Transform bbox
            x, y, w, h = ann["bbox"]
            ann["bbox"] = [
                x * scale + x_off,
                y * scale + y_off,
                w * scale,
                h * scale,
            ]

            # Transform segmentation if present
            if "segmentation" in ann and isinstance(ann["segmentation"], list):
                new_segs = []
                for seg in ann["segmentation"]:
                    # seg is [x1, y1, x2, y2, ...]
                    seg_array = np.array(seg).reshape(-1, 2)
                    seg_array = seg_array * scale + [x_off, y_off]
                    new_segs.append(seg_array.flatten().tolist())
                ann["segmentation"] = new_segs

            transformed.append(ann)

        return transformed

    def filter_images(self) -> List[Dict]:
        """
        Filter COCO images based on criteria.

        Returns:
            List of selected image metadata with annotations
        """
        print("\nFiltering COCO images...")
        print(
            f"Criteria:\n"
            f"  - Minimum {self.min_objects} objects (>= {self.min_area_percent}% area)\n"
            f"  - Coverage: >= {self.min_coverage_percent}%"
        )

        img_ids = self.coco.getImgIds()
        candidate_images = []

        for img_id in tqdm(img_ids, desc="Scanning images"):
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            if len(anns) == 0:
                continue

            # Calculate image area
            img_area = img_info["width"] * img_info["height"]

            # Keep ALL objects (including small ones)
            all_valid_anns = [ann for ann in anns if ann.get("area", 0) > 0]

            # Count only objects >= min_area_percent toward the threshold
            counted_anns = [
                ann
                for ann in all_valid_anns
                if self._calculate_area_percent(ann, img_area) >= self.min_area_percent
            ]

            # Check minimum object count
            if len(counted_anns) < self.min_objects:
                continue

            # Calculate coverage using ALL objects (including small ones)
            coverage = self._calculate_coverage_percent(
                all_valid_anns, img_info["width"], img_info["height"]
            )

            # Filter by coverage threshold
            if coverage >= self.min_coverage_percent:
                # Get dominant category for stratification
                dominant_cat = self._get_dominant_category(counted_anns)

                candidate_images.append(
                    {
                        "image_info": img_info,
                        "annotations_original": anns,
                        "annotations_all": all_valid_anns,  # All objects
                        "annotations_filtered": counted_anns,  # Objects >= threshold
                        "object_count": len(
                            counted_anns
                        ),  # Count of >= threshold objects
                        "coverage_percent": coverage,
                        "dominant_category": dominant_cat,
                    }
                )

        print(
            f"\nFound {len(candidate_images)} candidate images with "
            f"{self.min_coverage_percent}%+ coverage and {self.min_objects}+ objects"
        )

        if len(candidate_images) == 0:
            print("No images met the criteria!")
            print("\nTry adjusting parameters:")
            print(
                f"  - Lower min_coverage_percent (currently {self.min_coverage_percent}%)"
            )
            print(f"  - Lower min_objects (currently {self.min_objects})")
            print(f"  - Lower min_area_percent (currently {self.min_area_percent}%)")
            return []

        # Calculate quality scores
        print("\nCalculating quality scores...")
        for img_data in candidate_images:
            img_data["quality_score"] = self._calculate_quality_score(img_data)

        # Sort by quality score (descending)
        candidate_images.sort(key=lambda x: x["quality_score"], reverse=True)

        print(f"Top quality score: {candidate_images[0]['quality_score']:.3f}")
        print(
            f"Median quality score: {candidate_images[len(candidate_images)//2]['quality_score']:.3f}"
        )

        # Stratified sampling by dominant category
        print(f"\nPerforming stratified sampling for {self.target_count} images...")
        selected_images = self._stratified_sample(candidate_images, self.target_count)

        print(f"Selected {len(selected_images)} images")
        return selected_images

    def _stratified_sample(self, candidates: List[Dict], n_samples: int) -> List[Dict]:
        """
        Sample images using stratified sampling by dominant category.
        Ensures diversity in scene types while prioritizing quality.
        """
        # Group by dominant category
        category_groups = defaultdict(list)
        for img in candidates:
            cat = img["dominant_category"]
            category_groups[cat].append(img)

        # Count images per category
        cat_counts = {cat: len(imgs) for cat, imgs in category_groups.items()}
        print(f"\nCategory distribution in candidates:")
        sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
        for cat_id, count in sorted_cats[:10]:
            cat_name = self.coco.loadCats(cat_id)[0]["name"]
            print(f"  {cat_name}: {count} images ({count/len(candidates)*100:.1f}%)")
        if len(sorted_cats) > 10:
            print(f"  ... and {len(sorted_cats)-10} more categories")

        # Calculate samples per category (proportional to availability)
        total_candidates = len(candidates)
        samples_per_cat = {}
        for cat_id, imgs in category_groups.items():
            # Proportional allocation
            proportion = len(imgs) / total_candidates
            n_cat_samples = max(
                1, int(proportion * n_samples)
            )  # At least 1 per category
            samples_per_cat[cat_id] = min(n_cat_samples, len(imgs))

        # Adjust if we're over/under the target
        total_allocated = sum(samples_per_cat.values())
        if total_allocated > n_samples:
            # Remove excess from largest categories
            while total_allocated > n_samples:
                largest_cat = max(samples_per_cat, key=samples_per_cat.get)
                if samples_per_cat[largest_cat] > 1:
                    samples_per_cat[largest_cat] -= 1
                    total_allocated -= 1
                else:
                    break
        elif total_allocated < n_samples:
            # Add to categories that have more images available
            remaining = n_samples - total_allocated
            for cat_id in sorted(
                category_groups.keys(),
                key=lambda c: len(category_groups[c]),
                reverse=True,
            ):
                if remaining == 0:
                    break
                available = len(category_groups[cat_id])
                current = samples_per_cat[cat_id]
                can_add = min(remaining, available - current)
                if can_add > 0:
                    samples_per_cat[cat_id] += can_add
                    remaining -= can_add

        # Sample from each category (top quality images)
        selected = []
        for cat_id, n_cat in samples_per_cat.items():
            cat_images = category_groups[cat_id]
            # Already sorted by quality, take top n
            selected.extend(cat_images[:n_cat])

        print(f"\nFinal selection distribution:")
        final_cat_counts = Counter(img["dominant_category"] for img in selected)
        for cat_id, count in final_cat_counts.most_common(10):
            cat_name = self.coco.loadCats(cat_id)[0]["name"]
            print(f"  {cat_name}: {count} images ({count/len(selected)*100:.1f}%)")

        return selected

    def visualize_image(self, img_data: Dict, show_stages: bool = True) -> None:
        """
        Create visualization showing filtering pipeline with segmentation masks.

        Args:
            img_data: Image metadata from filter_images()
            show_stages: If True, show original and filtered stages
        """
        img_info = img_data["image_info"]
        img_id = img_info["id"]

        # Load image
        img_path = self.coco_root / "train2017" / img_info["file_name"]
        if not img_path.exists():
            img_path = self.coco_root / "val2017" / img_info["file_name"]

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            return

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if show_stages:
            # Create multi-stage visualization
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))

            # Get dominant category name
            dom_cat = img_data["dominant_category"]
            dom_cat_name = (
                self.coco.loadCats(dom_cat)[0]["name"] if dom_cat != -1 else "none"
            )

            fig.suptitle(
                f"Image {img_id}: {img_info['file_name']}\n"
                f"All objects: {len(img_data['annotations_all'])} → "
                f"Counted objects (>={self.min_area_percent}%): {len(img_data['annotations_filtered'])}\n"
                f"Coverage: {img_data['coverage_percent']:.1f}% | "
                f"Quality Score: {img_data['quality_score']:.3f} | "
                f"Dominant: {dom_cat_name}",
                fontsize=14,
                fontweight="bold",
            )

            # Stage 1: Original with all annotations
            ax = axes[0, 0]
            ax.imshow(img)
            self._draw_annotations(
                ax,
                img_data["annotations_original"],
                title="Stage 1: All Original Annotations",
                alpha=0.4,
            )

            # Stage 2: All valid objects (including small ones)
            ax = axes[0, 1]
            ax.imshow(img)
            self._draw_annotations(
                ax,
                img_data["annotations_all"],
                title=f"Stage 2: All Valid Objects (n={len(img_data['annotations_all'])})",
                alpha=0.5,
                use_masks=True,
            )

            # Stage 3: Coverage visualization
            ax = axes[1, 0]
            coverage_vis = self._create_coverage_visualization(
                img,
                img_data["annotations_all"],
                img_info["width"],
                img_info["height"],
            )
            ax.imshow(coverage_vis)
            ax.set_title(
                f"Stage 3: Coverage Map ({img_data['coverage_percent']:.1f}%)",
                fontsize=12,
                fontweight="bold",
            )
            ax.axis("off")

            # Stage 4: Letterboxed final
            letterboxed, scale, offset = self._letterbox_image(img, self.target_size)
            transformed_anns = self._transform_annotations(
                img_data["annotations_all"], scale, offset
            )

            ax = axes[1, 1]
            ax.imshow(letterboxed)
            self._draw_annotations(
                ax,
                transformed_anns,
                title=f"Stage 4: Letterboxed to {self.target_size[0]}×{self.target_size[1]}",
                alpha=0.5,
                use_masks=True,
            )

            plt.tight_layout()

        else:
            # Single final visualization
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            letterboxed, scale, offset = self._letterbox_image(img, self.target_size)
            transformed_anns = self._transform_annotations(
                img_data["annotations_all"], scale, offset
            )

            ax.imshow(letterboxed)
            self._draw_annotations(
                ax,
                transformed_anns,
                title=f"Final: {len(transformed_anns)} AOIs",
                alpha=0.5,
                use_masks=True,
            )

        # Save visualization
        output_path = self.output_dir / "visualizations" / f"{img_id:012d}_viz.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _create_coverage_visualization(
        self,
        img: np.ndarray,
        anns: List[Dict],
        img_width: int,
        img_height: int,
    ) -> np.ndarray:
        """Create visualization showing coverage with segmentation masks."""
        # Create overlay
        overlay = img.copy()

        # Create binary mask
        coverage_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        for ann in anns:
            if "segmentation" in ann:
                seg = ann["segmentation"]

                # Check if it's a list (polygon format) - most common in COCO
                if isinstance(seg, list) and len(seg) > 0:
                    # Check if it's a list of polygons (not RLE)
                    if isinstance(seg[0], list):
                        # Polygon format - convert to RLE then decode
                        try:
                            rle = mask_utils.frPyObjects(seg, img_height, img_width)
                            mask = mask_utils.decode(rle)
                            if len(mask.shape) == 3:
                                mask = mask.max(axis=2)
                        except:
                            # Skip if conversion fails
                            continue
                    else:
                        # Might be RLE in list format, skip for now
                        continue
                elif isinstance(seg, dict) and "counts" in seg:
                    # RLE format as dict
                    try:
                        mask = mask_utils.decode(seg)
                    except:
                        # Skip if decode fails
                        continue
                else:
                    # Unknown format, skip
                    continue

                coverage_mask = np.maximum(coverage_mask, mask)

        # Create green overlay for covered areas
        green_overlay = np.zeros_like(img)
        green_overlay[:, :, 1] = 255  # Green channel

        # Blend
        covered = coverage_mask > 0
        overlay[covered] = cv2.addWeighted(
            img[covered], 0.5, green_overlay[covered], 0.5, 0
        )

        return overlay

    def _draw_annotations(
        self,
        ax: plt.Axes,
        anns: List[Dict],
        title: str = "",
        alpha: float = 0.4,
        use_masks: bool = False,
    ) -> None:
        """Draw annotations with segmentation masks."""
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

        # Draw each annotation
        for ann in anns:
            cat_id = ann["category_id"]
            cat_info = self.coco.loadCats(cat_id)[0]
            color = self.colors[cat_id % len(self.colors)]

            if use_masks and "segmentation" in ann:
                seg = ann["segmentation"]

                # Draw segmentation mask
                if isinstance(seg, list) and len(seg) > 0:
                    # Check if it's polygon format
                    if isinstance(seg[0], list):
                        # Polygon format
                        for s in seg:
                            poly = np.array(s).reshape(-1, 2)
                            polygon = mpatches.Polygon(
                                poly,
                                closed=True,
                                linewidth=2,
                                edgecolor=color,
                                facecolor=(*color, alpha),
                            )
                            ax.add_patch(polygon)
                    else:
                        # Unknown list format, fall back to bbox
                        x, y, w, h = ann["bbox"]
                        rect = mpatches.Rectangle(
                            (x, y),
                            w,
                            h,
                            linewidth=2,
                            edgecolor=color,
                            facecolor=(*color, alpha),
                        )
                        ax.add_patch(rect)
                elif isinstance(seg, dict) and "counts" in seg:
                    # RLE format - decode and draw
                    try:
                        mask = mask_utils.decode(seg)
                        # Create colored overlay
                        colored_mask = np.zeros((*mask.shape, 4))
                        colored_mask[:, :, :3] = color
                        colored_mask[:, :, 3] = mask * alpha
                        ax.imshow(colored_mask)
                    except:
                        # Fall back to bbox if decode fails
                        x, y, w, h = ann["bbox"]
                        rect = mpatches.Rectangle(
                            (x, y),
                            w,
                            h,
                            linewidth=2,
                            edgecolor=color,
                            facecolor=(*color, alpha),
                        )
                        ax.add_patch(rect)
                else:
                    # Unknown format, fall back to bbox
                    x, y, w, h = ann["bbox"]
                    rect = mpatches.Rectangle(
                        (x, y),
                        w,
                        h,
                        linewidth=2,
                        edgecolor=color,
                        facecolor=(*color, alpha),
                    )
                    ax.add_patch(rect)
            else:
                # Fall back to bounding box
                x, y, w, h = ann["bbox"]
                rect = mpatches.Rectangle(
                    (x, y),
                    w,
                    h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=(*color, alpha),
                )
                ax.add_patch(rect)

            # Label at bbox position
            x, y, w, h = ann["bbox"]
            label = cat_info["name"]

            ax.text(
                x,
                y - 5,
                label,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                color="white",
                fontweight="bold",
            )

    def create_summary_statistics(self, selected_images: List[Dict]) -> None:
        """Create summary statistics visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Dataset Summary: {len(selected_images)} Images",
            fontsize=16,
            fontweight="bold",
        )

        # Object count distribution
        ax = axes[0, 0]
        counts = [img["object_count"] for img in selected_images]
        ax.hist(counts, bins=20, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Number of AOIs per Image")
        ax.set_ylabel("Frequency")
        ax.set_title("AOI Count Distribution")
        ax.axvline(
            np.mean(counts),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(counts):.1f}",
        )
        ax.legend()

        # Category distribution
        ax = axes[0, 1]
        all_cats = []
        for img in selected_images:
            all_cats.extend([ann["category_id"] for ann in img["annotations_filtered"]])

        cat_counts = defaultdict(int)
        for cat_id in all_cats:
            cat_counts[cat_id] += 1

        # Top 15 categories
        top_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        cat_names = [self.coco.loadCats(c[0])[0]["name"] for c in top_cats]
        cat_vals = [c[1] for c in top_cats]

        ax.barh(cat_names, cat_vals, alpha=0.7)
        ax.set_xlabel("Count")
        ax.set_title("Top 15 Object Categories")
        ax.invert_yaxis()

        # Coverage distribution
        ax = axes[0, 2]
        coverages = [img["coverage_percent"] for img in selected_images]
        ax.hist(coverages, bins=20, edgecolor="black", alpha=0.7, color="green")
        ax.set_xlabel("Coverage (%)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Coverage Distribution (min: {self.min_coverage_percent}%)")
        ax.axvline(
            self.min_coverage_percent,
            color="red",
            linestyle="--",
            label=f"Threshold: {self.min_coverage_percent}%",
        )
        ax.axvline(
            np.mean(coverages),
            color="blue",
            linestyle="--",
            label=f"Mean: {np.mean(coverages):.1f}%",
        )
        ax.legend()

        # Quality score distribution
        ax = axes[1, 0]
        scores = [img["quality_score"] for img in selected_images]
        ax.hist(scores, bins=20, edgecolor="black", alpha=0.7, color="purple")
        ax.set_xlabel("Quality Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Quality Score Distribution")
        ax.axvline(
            np.mean(scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(scores):.3f}",
        )
        ax.legend()

        # Dominant category distribution (stratification check)
        ax = axes[1, 1]
        dom_cats = [img["dominant_category"] for img in selected_images]
        dom_cat_counts = Counter(dom_cats)

        # Top 10 dominant categories
        top_dom = dom_cat_counts.most_common(10)
        dom_names = [self.coco.loadCats(c[0])[0]["name"] for c in top_dom]
        dom_vals = [c[1] for c in top_dom]

        ax.barh(dom_names, dom_vals, alpha=0.7, color="orange")
        ax.set_xlabel("Number of Images")
        ax.set_title("Dominant Categories (Stratification)")
        ax.invert_yaxis()

        # Image dimensions
        ax = axes[1, 2]
        widths = [img["image_info"]["width"] for img in selected_images]
        heights = [img["image_info"]["height"] for img in selected_images]

        ax.scatter(widths, heights, alpha=0.5, s=20)
        ax.set_xlabel("Width (pixels)")
        ax.set_ylabel("Height (pixels)")
        ax.set_title("Original Image Dimensions")
        ax.axhline(
            self.target_size[1],
            color="red",
            linestyle="--",
            label=f"Target: {self.target_size[0]}×{self.target_size[1]}",
        )
        ax.axvline(self.target_size[0], color="red", linestyle="--")
        ax.legend()

        plt.tight_layout()

        # Save
        output_path = (
            self.output_dir / "summary_statistics.png"
        )  # Changed from visualizations/
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\nSaved summary statistics: {output_path}")

    def process_dataset(self, visualize_samples: int = 10) -> None:
        """
        Main processing pipeline.

        Args:
            visualize_samples: Number of example visualizations to create
        """
        # Filter images
        selected_images = self.filter_images()

        if len(selected_images) == 0:
            print("No images met the criteria!")
            return

        # Create summary statistics
        print("\nGenerating summary statistics...")
        self.create_summary_statistics(selected_images)

        # Visualize sample images
        print(f"\nGenerating {visualize_samples} example visualizations...")
        sample_indices = np.linspace(
            0,
            len(selected_images) - 1,
            min(visualize_samples, len(selected_images)),
            dtype=int,
        )

        for idx in tqdm(sample_indices, desc="Creating visualizations"):
            self.visualize_image(selected_images[idx], show_stages=True)

        # Process and save all images
        print("\nProcessing all selected images...")
        dataset_metadata = []

        for img_data in tqdm(selected_images, desc="Processing images"):
            img_info = img_data["image_info"]
            img_id = img_info["id"]

            # Load image
            img_path = self.coco_root / "train2017" / img_info["file_name"]
            if not img_path.exists():
                img_path = self.coco_root / "val2017" / img_info["file_name"]

            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))

            # Letterbox
            letterboxed, scale, offset = self._letterbox_image(img, self.target_size)

            # Transform ALL annotations (including small ones)
            transformed_anns = self._transform_annotations(
                img_data["annotations_all"], scale, offset
            )

            # Save letterboxed image
            output_img_path = self.output_dir / "images" / f"{img_id:012d}.jpg"
            cv2.imwrite(str(output_img_path), letterboxed)

            # Save metadata
            metadata = {
                "image_id": img_id,
                "file_name": f"{img_id:012d}.jpg",
                "original_file": img_info["file_name"],
                "width": self.target_size[0],
                "height": self.target_size[1],
                "original_width": img_info["width"],
                "original_height": img_info["height"],
                "scale": scale,
                "offset": offset,
                "num_aois": len(transformed_anns),
                "num_counted_aois": len(
                    img_data["annotations_filtered"]
                ),  # >= threshold
                "coverage_percent": img_data["coverage_percent"],
                "quality_score": img_data["quality_score"],
                "dominant_category": img_data["dominant_category"],
                "dominant_category_name": (
                    self.coco.loadCats(img_data["dominant_category"])[0]["name"]
                    if img_data["dominant_category"] != -1
                    else "none"
                ),
                "annotations": transformed_anns,
            }
            dataset_metadata.append(metadata)

        # Save dataset metadata
        metadata_path = self.output_dir / "annotations" / "dataset.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "info": {
                        "description": "Naturalistic MS-COCO stimuli with segmentation masks",
                        "date_created": "2025-10-30",
                        "min_area_percent": self.min_area_percent,
                        "min_objects": self.min_objects,
                        "min_coverage_percent": self.min_coverage_percent,
                        "target_size": self.target_size,
                        "num_images": len(dataset_metadata),
                        "selection_method": "quality-sorted + stratified, prioritize object count",
                        "uses_segmentation_masks": True,
                    },
                    "images": dataset_metadata,
                    "categories": [
                        self.coco.loadCats(c)[0] for c in self.coco.getCatIds()
                    ],
                },
                f,
                indent=2,
            )

        print(f"\n✓ Processing complete!")
        print(f"  Images saved: {self.output_dir / 'images'}")
        print(f"  Visualizations: {self.output_dir / 'visualizations'}")
        print(f"  Metadata: {metadata_path}")
        print(f"\nKey features:")
        print(f"  ✓ Minimum {self.min_objects} objects per image")
        print(f"  ✓ No max size filter (keeps all objects)")
        print(f"  ✓ Small objects (<{self.min_area_percent}%) included but not counted")
        print(f"  ✓ Uses segmentation masks (precise boundaries)")


def main():
    """Main entry point."""

    # Configuration
    COCO_ROOT = None
    OUTPUT_DIR = "./coco_naturalistic_stimuli"

    # Initialize dataset creator
    creator = COCONaturalisticDataset(
        coco_root=COCO_ROOT,
        output_dir=OUTPUT_DIR,
        min_area_percent=0.5,  # Objects >= this count toward minimum
        min_objects=6,  # Minimum number of counted objects
        min_coverage_percent=85.0,  # Minimum 85% coverage
        target_size=(1024, 768),  # Fixed canvas size
        target_count=200,  # Target number of images
    )

    # Process dataset
    creator.process_dataset(visualize_samples=10)


if __name__ == "__main__":
    main()
