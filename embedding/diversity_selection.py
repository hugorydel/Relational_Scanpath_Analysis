#!/usr/bin/env python3
"""
Diversity selection for SVG relational dataset.
Reduces candidate images to a diverse subset using embeddings.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class DiversitySelector:
    """Select diverse subset of images using embeddings."""

    def __init__(
        self,
        embedding_cache_path: Path,
        device: str = None,
    ):
        """
        Initialize diversity selector.

        Args:
            embedding_cache_path: Path to cache embeddings
            device: Device for models (None = auto-detect)
        """
        self.embedding_cache_path = embedding_cache_path
        self.embeddings_cache = self._load_embedding_cache()
        self.cache_modified = False

        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load CLIP model (for image embeddings only)
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model.eval()
        print("✓ CLIP loaded")

        # Load SentenceTransformer (for text embeddings)
        print("Loading SentenceTransformer model...")
        self.text_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.text_model.to(self.device)
        print("✓ SentenceTransformer loaded\n")

    def _load_embedding_cache(self) -> Dict:
        """Load embedding cache from disk."""
        if self.embedding_cache_path.exists():
            with open(self.embedding_cache_path, "rb") as f:
                cache = pickle.load(f)
                print(f"Loaded embedding cache with {len(cache)} entries")
                return cache
        return {}

    def _save_embedding_cache(self):
        """Save embedding cache to disk."""
        if self.cache_modified:
            self.embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.embedding_cache_path, "wb") as f:
                pickle.dump(self.embeddings_cache, f)
            print(f"Saved embedding cache with {len(self.embeddings_cache)} entries")

    def _generate_scene_description(self, img_data: Dict) -> str:
        """
        Generate text description from scene graph.

        Args:
            img_data: Image data dictionary with scene_graph

        Returns:
            Natural language scene description
        """
        scene_graph = img_data["scene_graph"]
        objects = scene_graph.get("objects", [])
        relations = scene_graph.get("relationships", [])

        # Extract object names
        obj_names = [obj.get("name", "object") for obj in objects[:10]]  # Top 10

        # Extract key relationships
        rel_descriptions = []
        for rel in relations[:8]:  # Top 8 relations
            subj_id = rel.get("subject_id")
            obj_id = rel.get("object_id")
            predicate = rel.get("predicate", "")

            # Find object names
            subj_name = next(
                (o.get("name") for o in objects if o.get("object_id") == subj_id),
                "object",
            )
            obj_name = next(
                (o.get("name") for o in objects if o.get("object_id") == obj_id),
                "object",
            )

            rel_descriptions.append(f"{subj_name} {predicate} {obj_name}")

        # Construct description
        if obj_names:
            description = f"A scene containing {', '.join(obj_names[:5])}"
            if len(obj_names) > 5:
                description += f" and {len(obj_names) - 5} other objects"
        else:
            description = "A scene"

        if rel_descriptions:
            description += f", where {'; '.join(rel_descriptions[:3])}"

        return description

    def compute_embeddings(
        self,
        images_data: List[Dict],
        embedding_type: str = "text",
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute embeddings for images.

        Args:
            images_data: List of image data dictionaries
            embedding_type: 'image' or 'text'
            batch_size: Batch size for processing

        Returns:
            embeddings: (N, D) array of embeddings
        """
        print(f"\nComputing {embedding_type} embeddings...")

        embeddings_list = []
        to_compute = []
        to_compute_indices = []

        # Check cache first
        for idx, img_data in enumerate(images_data):
            img_id = img_data["scene_graph"].get("image_id", "unknown")
            cache_key = f"{img_id}_{embedding_type}"

            if cache_key in self.embeddings_cache:
                embeddings_list.append(self.embeddings_cache[cache_key])
            else:
                embeddings_list.append(None)
                to_compute.append(img_data)
                to_compute_indices.append(idx)

        print(
            f"Using {len(images_data) - len(to_compute)} cached, "
            f"computing {len(to_compute)} new embeddings"
        )

        if len(to_compute) == 0:
            return np.array(embeddings_list)

        # Compute missing embeddings
        if embedding_type == "image":
            new_embeddings = self._compute_image_embeddings(to_compute, batch_size)
        else:  # text
            new_embeddings = self._compute_text_embeddings(to_compute, batch_size)

        # Update cache and results
        for idx, emb in zip(to_compute_indices, new_embeddings):
            img_id = images_data[idx]["scene_graph"].get("image_id", "unknown")
            cache_key = f"{img_id}_{embedding_type}"
            self.embeddings_cache[cache_key] = emb
            embeddings_list[idx] = emb
            self.cache_modified = True

        self._save_embedding_cache()

        return np.array(embeddings_list)

    def _compute_image_embeddings(
        self, images_data: List[Dict], batch_size: int
    ) -> List[np.ndarray]:
        """Compute CLIP image embeddings."""
        embeddings = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(images_data), batch_size),
                desc="Image embeddings",
                mininterval=1.0,
            ):
                batch = images_data[i : i + batch_size]

                # Load images
                images = []
                for img_data in batch:
                    img = cv2.imread(str(img_data["image_path"]))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)

                # Process batch
                inputs = self.clip_processor(
                    images=images, return_tensors="pt", padding=True
                ).to(self.device)
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                embeddings.extend(image_features.cpu().numpy())

        return embeddings

    def _compute_text_embeddings(
        self, images_data: List[Dict], batch_size: int
    ) -> List[np.ndarray]:
        """Compute SentenceTransformer text embeddings from full scene descriptions."""

        # Generate full descriptions (no summarization)
        descriptions = [
            self._generate_scene_description(img_data) for img_data in images_data
        ]

        # Compute embeddings in batches
        embeddings = []
        for i in tqdm(
            range(0, len(descriptions), batch_size),
            desc="Text embeddings",
            mininterval=1.0,
        ):
            batch_texts = descriptions[i : i + batch_size]
            batch_embeddings = self.text_model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
            )
            embeddings.extend(batch_embeddings)

        return embeddings

    def select_diverse_clustering(
        self,
        embeddings: np.ndarray,
        n_select: int,
        n_clusters: int = None,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Select diverse images using K-means clustering.
        Pick the image closest to each cluster center.

        Args:
            embeddings: (N, D) embedding array
            n_select: Number of images to select
            n_clusters: Number of clusters (default: n_select)
            random_state: Random seed

        Returns:
            selected_indices: Indices of selected images
        """
        if n_clusters is None:
            n_clusters = n_select

        print(f"\nClustering into {n_clusters} clusters...")

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Select representative from each cluster (closest to center)
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]

            # Find closest to center
            center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]

            selected_indices.append(closest_idx)

        selected_indices = np.array(selected_indices)

        # If we need fewer than n_clusters, select most central clusters
        if n_select < len(selected_indices):
            # Rank clusters by their distance from overall centroid
            overall_center = embeddings.mean(axis=0)
            cluster_distances = np.linalg.norm(
                kmeans.cluster_centers_ - overall_center, axis=1
            )
            central_clusters = np.argsort(cluster_distances)[:n_select]
            selected_indices = selected_indices[central_clusters]

        print(f"✓ Selected {len(selected_indices)} diverse images via clustering")
        return selected_indices

    def select_diverse_greedy(
        self,
        embeddings: np.ndarray,
        n_select: int,
        similarity_threshold: float = 0.85,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Greedy diversity selection.
        Iteratively add images that are sufficiently different from already selected.

        Args:
            embeddings: (N, D) embedding array
            n_select: Number of images to select
            similarity_threshold: Maximum cosine similarity to already selected
            random_state: Random seed

        Returns:
            selected_indices: Indices of selected images
        """
        print(f"\nGreedy selection (similarity threshold: {similarity_threshold})...")

        np.random.seed(random_state)
        n_images = len(embeddings)

        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Start with random image
        selected_indices = [np.random.randint(n_images)]
        candidates = set(range(n_images)) - set(selected_indices)

        # Greedy selection
        pbar = tqdm(total=n_select, desc="Greedy selection", mininterval=0.5)
        pbar.update(1)

        while len(selected_indices) < n_select and candidates:
            selected_embeddings = embeddings_norm[selected_indices]

            # Compute similarity of all candidates to selected images
            max_similarities = []
            candidate_list = list(candidates)

            for idx in candidate_list:
                similarities = embeddings_norm[idx] @ selected_embeddings.T
                max_sim = similarities.max()
                max_similarities.append(max_sim)

            max_similarities = np.array(max_similarities)

            # Find candidate with lowest max similarity
            best_candidate_idx = candidate_list[np.argmin(max_similarities)]
            best_similarity = max_similarities[np.argmin(max_similarities)]

            # Check threshold
            if best_similarity > similarity_threshold:
                print(
                    f"\nWarning: Best candidate has similarity {best_similarity:.3f} "
                    f"(threshold: {similarity_threshold})"
                )
                # Continue anyway but warn user

            selected_indices.append(best_candidate_idx)
            candidates.remove(best_candidate_idx)
            pbar.update(1)

        pbar.close()

        print(f"✓ Selected {len(selected_indices)} diverse images via greedy sampling")
        return np.array(selected_indices)

    def select_diverse_score_weighted(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        n_select: int,
        similarity_threshold: float = 0.75,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Score-weighted greedy diversity selection.
        Balances diversity (similarity threshold) with quality (scores).

        Algorithm:
        1. Start with highest-scoring image
        2. Iteratively add highest-scoring image that is sufficiently
           different from ALL already-selected images
        3. Continue until n_select images or no viable candidates

        Args:
            embeddings: (N, D) embedding array
            scores: (N,) array of image scores
            n_select: Number of images to select
            similarity_threshold: Maximum cosine similarity to already selected
            random_state: Random seed for tie-breaking

        Returns:
            selected_indices: Indices of selected images
        """
        print(
            f"\nScore-weighted greedy selection (threshold: {similarity_threshold})..."
        )

        np.random.seed(random_state)
        n_images = len(embeddings)

        if len(scores) != n_images:
            raise ValueError(
                f"Scores length ({len(scores)}) != embeddings ({n_images})"
            )

        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Start with highest-scoring image
        seed_idx = int(np.argmax(scores))
        selected_indices = [seed_idx]
        candidates = set(range(n_images)) - {seed_idx}

        print(f"  Seed: image {seed_idx} (score: {scores[seed_idx]:.2f})")

        # Greedy selection
        pbar = tqdm(total=n_select, desc="Score-weighted selection", mininterval=0.5)
        pbar.update(1)

        while len(selected_indices) < n_select and candidates:
            selected_embeddings = embeddings_norm[selected_indices]

            # Find viable candidates (pass similarity threshold)
            viable_candidates = []
            candidate_scores = []

            for idx in candidates:
                # Compute max similarity to any selected image
                similarities = embeddings_norm[idx] @ selected_embeddings.T
                max_sim = similarities.max()

                # Check if passes threshold
                if max_sim <= similarity_threshold:
                    viable_candidates.append(idx)
                    candidate_scores.append(scores[idx])

            if not viable_candidates:
                # No candidates pass threshold - warn and take best remaining
                print(
                    f"\n  Warning: No candidates below threshold {similarity_threshold:.3f}"
                )
                print(
                    f"  Taking highest-scoring from remaining {len(candidates)} candidates"
                )

                # Get all remaining candidate scores
                remaining_scores = [(idx, scores[idx]) for idx in candidates]
                best_idx = max(remaining_scores, key=lambda x: x[1])[0]

                selected_indices.append(best_idx)
                candidates.remove(best_idx)
            else:
                # Select highest-scoring viable candidate
                best_idx = viable_candidates[np.argmax(candidate_scores)]
                selected_indices.append(best_idx)
                candidates.remove(best_idx)

            pbar.update(1)

        pbar.close()

        if len(selected_indices) < n_select:
            print(
                f"  Warning: Only selected {len(selected_indices)}/{n_select} images "
                f"(ran out of viable candidates)"
            )

        print(
            f"✓ Selected {len(selected_indices)} diverse images via score-weighted greedy"
        )

        # Print selection statistics
        selected_scores = scores[selected_indices]
        print(
            f"  Score range: [{selected_scores.min():.2f}, {selected_scores.max():.2f}]"
        )
        print(f"  Mean score: {selected_scores.mean():.2f}")

        return np.array(selected_indices)

    def visualize_embedding_space(
        self,
        embeddings: np.ndarray,
        selected_indices: np.ndarray,
        output_path: Path,
        labels: List[str] = None,
        method: str = "tsne",
    ):
        """
        Visualize embedding space with selected images highlighted.

        Args:
            embeddings: (N, D) embedding array
            selected_indices: Indices of selected images
            output_path: Where to save visualization
            labels: Optional labels for points
            method: 'tsne' or 'pca'
        """
        print(f"\nVisualizing embedding space using {method.upper()}...")

        # Reduce to 2D
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            coords_2d = reducer.fit_transform(embeddings)
        else:  # pca
            reducer = PCA(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(embeddings)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot all images
        mask_selected = np.zeros(len(embeddings), dtype=bool)
        mask_selected[selected_indices] = True

        ax.scatter(
            coords_2d[~mask_selected, 0],
            coords_2d[~mask_selected, 1],
            c="lightgray",
            s=50,
            alpha=0.5,
            label="Not selected",
        )

        ax.scatter(
            coords_2d[mask_selected, 0],
            coords_2d[mask_selected, 1],
            c="red",
            s=100,
            alpha=0.8,
            edgecolors="black",
            linewidths=2,
            label="Selected",
        )

        # Add labels if provided
        if labels is not None:
            for idx in selected_indices[:20]:  # Label first 20 to avoid clutter
                ax.annotate(
                    labels[idx],
                    (coords_2d[idx, 0], coords_2d[idx, 1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                )

        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_title(
            f"Embedding Space Visualization\n"
            f"{len(selected_indices)}/{len(embeddings)} images selected"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved visualization to {output_path}")

    def compute_diversity_metrics(
        self, embeddings: np.ndarray, selected_indices: np.ndarray
    ) -> Dict:
        """
        Compute diversity metrics for selected images.

        Args:
            embeddings: (N, D) embedding array
            selected_indices: Indices of selected images

        Returns:
            metrics: Dictionary of diversity metrics
        """
        selected_embeddings = embeddings[selected_indices]

        # Normalize
        selected_norm = selected_embeddings / np.linalg.norm(
            selected_embeddings, axis=1, keepdims=True
        )

        # Pairwise similarities
        similarity_matrix = selected_norm @ selected_norm.T

        # Remove diagonal
        n = len(selected_indices)
        mask = ~np.eye(n, dtype=bool)
        pairwise_similarities = similarity_matrix[mask]

        metrics = {
            "n_selected": len(selected_indices),
            "mean_pairwise_similarity": float(pairwise_similarities.mean()),
            "std_pairwise_similarity": float(pairwise_similarities.std()),
            "min_pairwise_similarity": float(pairwise_similarities.min()),
            "max_pairwise_similarity": float(pairwise_similarities.max()),
            "median_pairwise_similarity": float(np.median(pairwise_similarities)),
        }

        return metrics
