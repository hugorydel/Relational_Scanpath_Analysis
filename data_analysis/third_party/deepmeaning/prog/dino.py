#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: dino.py
Author: T.R. Hayes
Version: 1.0.0
Description: Computes self-distillation with no labels (DINO) embeddings for 
             input images.

Caron, M. et al. (2021) Emerging Properties in Self-Supervised Vision 
Transformers Models. https://arxiv.org/abs/2104.14294

Pretrained ViT-B/16 weights downloaded from:
  https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth

See DINO page for more details:
  https://github.com/facebookresearch/dino

Changelog:
- 1.0.0 (2024-11-27): TRH Wrote it
"""

#%% 010: Import packages

import os
import torch
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import vit_b_16
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict
from tqdm.auto import tqdm
from utils import (
    ImageFolderInstance, 
    get_image_paths, 
    save_embeddings,
    load_existing_embeddings)

#%% 020: Define class/functions

class DINOEmbedding(torch.nn.Module):
    """DINO embedding extraction module."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.model.encoder.pos_embedding
        x = self.model.encoder.dropout(x)
        x = self.model.encoder.layers(x)
        x = self.model.encoder.ln(x)
        return x[:, 0]

def get_transform():
    """Get preprocessing transform for DINO."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    verbose: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """Extract embeddings for DINO model.
    
    Processes batches of images through the DINO model to extract embeddings,
    while also tracking the corresponding image paths.
    
    Args:
        model (torch.nn.Module): The DINO model used for extraction
        dataloader (DataLoader): PyTorch DataLoader providing batches of images and paths
        device (str): Device to run inference on ('cuda' or 'cpu')
        verbose (bool, optional): Whether to print progress information. Defaults to False.
    
    Returns:
        Tuple[np.ndarray, List[str]]: A tuple containing:
            - np.ndarray: Array of shape (n_samples, embedding_dim) with DINO embeddings
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
            batch_embeddings = model(batch)
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
    """Extract DINO embeddings from images.
    
    This function:
    1. Attempts to load existing embeddings if available and not forced to recompute
    2. Otherwise loads the DINO (ViT-B/16) model with pretrained weights
    3. Processes all images to extract embeddings
    4. Optionally saves the embeddings to specified files
    5. Returns both the embeddings and a mapping from filenames to indices
    
    Args:
        image_dir (str): Directory containing images to process
        save_dir (Optional[str], optional): Directory to save/load embeddings. 
            If None, embeddings won't be saved. Defaults to None.
        save_names (Optional[List[str]], optional): List of [embeddings_filename, paths_filename].
            Defaults to ['dino_embeddings.npy', 'dino_paths.npy'].
        batch_size (int, optional): Batch size for processing images. Defaults to 256.
        device (str, optional): Device to run computation on ('cuda' or 'cpu'). Defaults to 'cuda'.
        extension (str, optional): Image file extension pattern to match. Defaults to '*.png'.
        force_compute (bool, optional): Whether to force computation even if embeddings exist.
            Defaults to False.
    
    Returns:
        Tuple[np.ndarray, Dict[str, int]]: A tuple containing:
            - np.ndarray: Array of shape (n_samples, embedding_dim) with DINO embeddings
            - Dict[str, int]: Mapping from filenames to embedding indices
    
    Raises:
        ValueError: If save_names is provided but doesn't contain exactly 2 filenames
    """
    if save_names is None:
        save_names = ['dino_embeddings.npz', 'dino_paths.npz']
    
    if save_names and len(save_names) != 2:
        raise ValueError("save_names must be a list of exactly two strings [embeddings_filename, paths_filename]")
    
    # Try to load existing embeddings if not force_compute
    if save_dir and not force_compute:
        embeddings, filename_mapping = load_existing_embeddings(save_dir, save_names)
        if embeddings is not None:
            print("Using existing embeddings")
            return embeddings, filename_mapping
    
    # Load DINO model
    base_model = vit_b_16()
    url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    state_dict = torch.hub.load_state_dict_from_url(url, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(state_dict, strict=False)
    model = DINOEmbedding(base_model).to(device)
    
    # Setup data loading
    transform = get_transform()
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
    print("DINO embeddings")
    embeddings, paths = extract_embeddings(model, dataloader, device)
    
    # Save if directory provided
    if save_dir:
        save_embeddings(embeddings, paths, save_dir, save_names)
    
    filename_mapping = {os.path.basename(path): i for i, path in enumerate(paths)}
    return embeddings, filename_mapping
