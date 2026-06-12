#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: coca.py
Author: T.R. Hayes
Version: 1.0.0
Description: Computes OpenCLIP Contrastive Captioner (CoCa) embeddings for
             input images.

## CoCa ##
Yu, I. et al. (2022) CoCa: Contrastive Captioners are Image-Text Foundation
Models. https://arxiv.org/abs/2205.01917

## OpenCLIP ##
I, Gabriel et al. OpenCLIP (2021). https://doi.org/10.5281/zenodo.5143773

Changelog:
- 1.0.0 (2024-11-20): TRH Wrote it
"""

# %% 010: Import packages

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import (
    ImageFolderInstance,
    get_image_paths,
    load_existing_embeddings,
    save_embeddings,
)

# %% 020: Define functions


def extract_embeddings(
    model: torch.nn.Module, dataloader: DataLoader, device: str, verbose: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """Extract embeddings specifically for CoCa model."""
    embeddings = []
    paths = []

    model.eval()
    with torch.no_grad():
        iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc="Processing",
            unit="batch",
            disable=not verbose,
        )

        for batch, batch_paths in iterator:
            batch = batch.to(device)
            batch_embeddings = model.encode_image(batch)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(
                dim=1, keepdim=True
            )
            embeddings.append(batch_embeddings.cpu().numpy())
            paths.extend(batch_paths)

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, paths


def get_embeddings(
    image_dir: str,
    save_dir: Optional[str] = None,
    save_names: Optional[List[str]] = None,
    batch_size: int = 32,
    device: str = "cuda",
    extension: str = "*.png",
    force_compute: bool = False,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Extract CoCa embeddings from images."""
    if save_names is None:
        save_names = ["coca_embeddings.npz", "coca_paths.npz"]

    if save_names and len(save_names) != 2:
        raise ValueError(
            "save_names must be a list of exactly two strings [embeddings_filename, paths_filename]"
        )

    if save_dir and not force_compute:
        embeddings, filename_mapping = load_existing_embeddings(save_dir, save_names)
        if embeddings is not None:
            print("Using existing embeddings")
            return embeddings, filename_mapping

    # Load CoCa model
    model, _, transform = open_clip.create_model_and_transforms(
        "coca_ViT-L-14", device=device, pretrained="mscoco_finetuned_laion2b_s13b_b90k"
    )

    # Setup data loading
    image_paths = get_image_paths(image_dir, extension)
    dataset = ImageFolderInstance(image_paths, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 0 required on Windows — worker processes hang with >0
        pin_memory=False,  # pin_memory only benefits CUDA; False for CPU
    )

    # Extract embeddings
    print("CoCa embeddings")
    embeddings, paths = extract_embeddings(model, dataloader, device)

    if save_dir:
        save_embeddings(embeddings, paths, save_dir, save_names)

    filename_mapping = {os.path.basename(path): i for i, path in enumerate(paths)}
    return embeddings, filename_mapping
