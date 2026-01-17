#!/usr/bin/env python3
"""
select_diverse_stimuli.py - Select diverse subset from filtered SVG images
"""

import os

from utils import ensure_jpg, validate_paths

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

import json
import shutil
from pathlib import Path

import numpy as np

import config
from diversity_selection import DiversitySelector


def load_filtered_images(dataset_path: Path):
    """Load filtered images from preprocessing output."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data["images"]


def create_diversity_output_dir(base_output_dir: Path) -> Path:
    """Create output directory for diverse selection."""
    output_dir = base_output_dir / "diverse_selection"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "annotations").mkdir(exist_ok=True)
    return output_dir


def save_selected_dataset(
    selected_images: list,
    diversity_metrics: dict,
    output_dir: Path,
    selection_config: dict,
):
    """Save selected images and metadata."""
    # Save metadata
    metadata = {
        "info": {
            "description": "Diverse SVG Relational Stimuli Dataset",
            "num_images": len(selected_images),
            "selection_method": selection_config["method"],
            "embedding_type": selection_config["embedding_type"],
            "diversity_metrics": diversity_metrics,
            "filtering_criteria": {
                "min_memorability": config.MIN_MEMORABILITY,
                "min_objects": config.MIN_OBJECTS,
                "max_objects": config.MAX_OBJECTS,
                "min_relations": config.MIN_RELATIONS,
                "min_coverage_percent": config.MIN_COVERAGE_PERCENT,
            },
        },
        "images": selected_images,
    }

    metadata_path = output_dir / "annotations" / "diverse_dataset.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved metadata to {metadata_path}")

    # Copy selected images and visualizations
    processed_dir = Path(config.OUTPUT_DIR)

    for img_data in selected_images:
        img_id = img_data["image_id"]

        # Copy image
        src_img = processed_dir / "images" / ensure_jpg(img_id)
        if src_img.exists():
            dst_img = output_dir / "images" / ensure_jpg(img_id)
            shutil.copy(src_img, dst_img)

        # Copy visualization if exists
        src_viz = processed_dir / "visualizations" / f"{img_id}_viz.png"
        if src_viz.exists():
            dst_viz = output_dir / "visualizations" / f"{img_id}_viz.png"
            shutil.copy(src_viz, dst_viz)

    print(f"✓ Copied {len(selected_images)} images and visualizations")


def main():
    """Main diversity selection pipeline."""
    # Validate paths before processing
    print("=" * 60)
    print("DIVERSITY SELECTION")
    print("=" * 60)
    print("Validating paths...\n")
    validate_paths(required_for_processing=False, required_for_diversity=True)
    print("✓ Path validation passed\n")

    print(f"  Method: {config.SELECTION_METHOD}")
    print(f"  Embedding: {config.EMBEDDING_TYPE}")
    print(f"  Target images: {config.N_FINAL_IMAGES}")
    print("=" * 60 + "\n")

    # Load filtered images
    dataset_path = Path(config.OUTPUT_DIR) / "annotations" / "dataset.json"

    print("Loading filtered images...")
    filtered_images = load_filtered_images(dataset_path)
    print(f"✓ Loaded {len(filtered_images)} filtered images\n")

    if len(filtered_images) <= config.N_FINAL_IMAGES:
        print(
            f"Warning: Only {len(filtered_images)} images available, "
            f"requested {config.N_FINAL_IMAGES}"
        )
        print("All images will be included in final set.")
        return

    # Prepare image data for diversity selector
    # Need to reconstruct paths since they're not in the metadata
    vg_image_root = Path(config.VG_IMAGE_ROOT)
    images_data = []

    for img_meta in filtered_images:
        img_id = img_meta["image_id"]
        img_path = vg_image_root / img_meta["scene_graph"]["image"]

        images_data.append(
            {
                "scene_graph": img_meta["scene_graph"],
                "image_path": img_path,
                "metadata": img_meta,
            }
        )

    # Initialize diversity selector
    selector = DiversitySelector(
        embedding_cache_path=Path(config.EMBEDDING_CACHE_PATH),
    )

    # Compute embeddings
    embeddings = selector.compute_embeddings(
        images_data,
        embedding_type=config.EMBEDDING_TYPE,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )

    # Select diverse subset
    if config.SELECTION_METHOD == "clustering":
        selected_indices = selector.select_diverse_clustering(
            embeddings,
            n_select=config.N_FINAL_IMAGES,
            n_clusters=config.N_CLUSTERS,
            random_state=config.RANDOM_SEED,
        )
    else:  # greedy
        selected_indices = selector.select_diverse_greedy(
            embeddings,
            n_select=config.N_FINAL_IMAGES,
            similarity_threshold=config.SIMILARITY_THRESHOLD,
            random_state=config.RANDOM_SEED,
        )

    # Compute diversity metrics
    diversity_metrics = selector.compute_diversity_metrics(embeddings, selected_indices)

    print("\n" + "=" * 60)
    print("DIVERSITY METRICS")
    print("=" * 60)
    for key, value in diversity_metrics.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    # Create output directory
    output_dir = create_diversity_output_dir(Path(config.OUTPUT_DIR).parent)

    # Visualize embedding space
    labels = [img["metadata"]["image_id"] for img in images_data]
    selector.visualize_embedding_space(
        embeddings,
        selected_indices,
        output_path=output_dir / "embedding_space_tsne.png",
        labels=labels,
        method="tsne",
    )

    selector.visualize_embedding_space(
        embeddings,
        selected_indices,
        output_path=output_dir / "embedding_space_pca.png",
        labels=labels,
        method="pca",
    )

    # Save selected dataset
    selected_images = [filtered_images[idx] for idx in selected_indices]

    save_selected_dataset(
        selected_images,
        diversity_metrics,
        output_dir,
        selection_config={
            "method": config.SELECTION_METHOD,
            "embedding_type": config.EMBEDDING_TYPE,
            "n_select": config.N_FINAL_IMAGES,
        },
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SELECTION COMPLETE")
    print("=" * 60)
    print(f"✓ Selected {len(selected_images)} diverse images")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Images: {output_dir / 'images'}")
    print(f"✓ Visualizations: {output_dir / 'visualizations'}")
    print(f"✓ Metadata: {output_dir / 'annotations' / 'diverse_dataset.json'}")
    print(
        f"✓ Embedding space: {output_dir / 'embedding_space_tsne.png'}, "
        f"{output_dir / 'embedding_space_pca.png'}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
