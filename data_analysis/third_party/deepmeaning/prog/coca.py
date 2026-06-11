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

#%% 010: Import packages

import os
import torch
import open_clip
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict
from tqdm.auto import tqdm
from utils import (
    ImageFolderInstance, 
    get_image_paths, 
    save_embeddings,
    load_existing_embeddings)

#%% 020: Define functions
 
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    verbose: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """Extract embeddings specifically for CoCa model.
    
    This function processes batches of images through the CoCa model to extract
    normalized image embeddings, while also tracking the corresponding image paths.
    
    Args:
        model (torch.nn.Module): The CoCa model used for extraction
        dataloader (DataLoader): PyTorch DataLoader providing batches of images and paths
        device (str): Device to run inference on ('cuda' or 'cpu')
        verbose (bool, optional): Whether to print progress information. Defaults to False.
    
    Returns:
        Tuple[np.ndarray, List[str]]: A tuple containing:
            - np.ndarray: Array of shape (n_samples, embedding_dim) with normalized image embeddings
            - List[str]: List of image paths corresponding to each embedding
    """
    embeddings = []
    paths = []
    
    model.eval()
    with torch.no_grad():
        # Use tqdm for progress tracking if verbose
        iterator = tqdm(
            dataloader, 
            total=len(dataloader),
            desc="Processing",
            unit="batch",
            disable=not verbose
        )
            
        for batch, batch_paths in iterator:
            batch = batch.to(device)
            batch_embeddings = model.encode_image(batch)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
            embeddings.append(batch_embeddings.cpu().numpy())
            paths.extend(batch_paths)
    
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, paths

def get_embeddings(
    image_dir: str,
    save_dir: Optional[str] = None,
    save_names: Optional[List[str]] = None,
    batch_size: int = 32,
    device: str = 'cuda',
    extension: str = '*.png',
    force_compute: bool = False
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Extract CoCa embeddings from images.
    
    This function:
    1. Attempts to load existing embeddings if available and not forced to recompute
    2. Otherwise loads the CoCa model and extracts embeddings for all images
    3. Optionally saves the embeddings to specified files
    4. Returns both the embeddings and a mapping from filenames to indices
    
    Args:
        image_dir (str): Directory containing images to process
        save_dir (Optional[str], optional): Directory to save/load embeddings. 
            If None, embeddings won't be saved. Defaults to None.
        save_names (Optional[List[str]], optional): List of [embeddings_filename, paths_filename].
            Defaults to ['coca_embeddings.npy', 'coca_paths.npy'].
        batch_size (int, optional): Batch size for processing images. Defaults to 256.
        device (str, optional): Device to run computation on ('cuda' or 'cpu'). Defaults to 'cuda'.
        extension (str, optional): Image file extension pattern to match. Defaults to '*.png'.
        force_compute (bool, optional): Whether to force computation even if embeddings exist.
            Defaults to False.
    
    Returns:
        Tuple[np.ndarray, Dict[str, int]]: A tuple containing:
            - np.ndarray: Array of shape (n_samples, embedding_dim) with normalized CoCa embeddings
            - Dict[str, int]: Mapping from filenames to embedding indices
    
    Raises:
        ValueError: If save_names is provided but doesn't contain exactly 2 filenames
    """
    if save_names is None:
        save_names = ['coca_embeddings.npz', 'coca_paths.npz']
    
    if save_names and len(save_names) != 2:
        raise ValueError("save_names must be a list of exactly two strings [embeddings_filename, paths_filename]")
    
    # Try to load existing embeddings if not force_compute
    if save_dir and not force_compute:
        embeddings, filename_mapping = load_existing_embeddings(save_dir, save_names)
        if embeddings is not None:
            print("Using existing embeddings")
            return embeddings, filename_mapping
    
    # Load CoCa model
    model, _, transform = open_clip.create_model_and_transforms(
        'coca_ViT-L-14',
        device=device,
        pretrained='mscoco_finetuned_laion2b_s13b_b90k'
    )
    
    # Setup data loading
    image_paths = get_image_paths(image_dir, extension)
    dataset = ImageFolderInstance(image_paths, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Extract embeddings
    print('CoCa embeddings')
    embeddings, paths = extract_embeddings(model, dataloader, device)
    
    # Save if directory provided
    if save_dir:
        save_embeddings(embeddings, paths, save_dir, save_names)
    
    filename_mapping = {os.path.basename(path): i for i, path in enumerate(paths)}
    return embeddings, filename_mapping
