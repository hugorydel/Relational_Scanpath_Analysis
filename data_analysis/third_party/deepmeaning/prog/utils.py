#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: utils.py
Author: T.R. Hayes
Version: 1.0.0
Description: Variety of utility functions used in DeepMeaning: Estimating
             and interpreting scene meaning for attention using a 
             vision-language transformer (Hayes & Henderson, 2025) codebase.
              
Changelog:
- 1.0.0 (2024-11-20): TRH Wrote it
"""
#%% 010: Import packages

import torch
from PIL import Image
import re
import os
import shutil
import glob
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import csv
import cv2
import math
from tqdm.auto import tqdm
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec
import textwrap
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter
from scipy.stats import iqr, gaussian_kde
from multiprocessing import Pool
from functools import partial

#%% 020: General data organization, save/load, and print functions

class ImageFolderInstance:
    """Dataset class for loading images from a folder.
    
    This class creates a PyTorch-compatible dataset for loading and 
    transforming images from a list of file paths.
    
    Args:
      image_paths (List[str]): List of full paths to image files
      transform: PyTorch transform operations to apply to each image
    
    Returns:
      Tuple[torch.Tensor, str]: Transformed image tensor and corresponding file path
    """
    def __init__(self, image_paths: List[str], transform):
        self.image_paths = image_paths
        self.transform = transform
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, path
    
    def __len__(self) -> int:
        return len(self.image_paths)

def get_image_paths(image_dir: str, extension: str = '*.png') -> List[str]:
    """Get all image paths from a directory matching a specific extension.
    
    Args:
        image_dir (str): Directory containing images
        extension (str, optional): File extension pattern to match. Defaults to '*.png'.
    
    Returns:
        List[str]: List of full paths to all matching image files
    """
    return glob.glob(os.path.join(image_dir, extension))
  
def print_ttest(stats):
  """Print formatted t-test results from a Pingouin stats object.
   
  Args:
     stats: Pingouin ttest result object containing test statistics
   
  Prints:
     Formatted string with t-value, degrees of freedom, p-value, and confidence interval
  """
  print(
      f"""t({int(stats['dof'].iloc[0])}) = {stats['T'].iloc[0]:.3f},
  p = {stats['p-val'].iloc[0]:.3f},
  95% CI [{stats['CI95%'].iloc[0][0]:.3f}, {stats['CI95%'].iloc[0][1]:.3f}]""")

def print_heading(message):
    """
    Print a heading surrounded by '#' characters with all letters capitalized.
    
    Args:
        message (str): The heading text to display
    """
    # Convert all characters to uppercase
    formatted_message = message.upper()
    
    # Create the line of # characters and print heading
    line = "#" * len(formatted_message)
    print()
    print(line)
    print(formatted_message)
    print(line)

def print_section(message, line_skip=1):
    """
    Print a section heading with the message in uppercase,
    flanked by '##' on both sides, with optional line skips before the heading.
    
    Args:
        message (str): The section text to display
        line_skip (int, optional): Number of line breaks to insert before the heading.
                                 Defaults to 1.
    """
    # Convert message to uppercase
    formatted_message = message.upper()
    
    # Create the line breaks based on line_skip parameter
    line_breaks = '\n' * line_skip
    
    # Print the line breaks followed by the message flanked by ##
    print(f"{line_breaks}## {formatted_message} ##")
   
def save_embeddings(
    embeddings: np.ndarray,
    paths: List[str],
    save_dir: str,
    save_names: List[str]
) -> None:
    """Save embeddings and path mappings to specified files.

    Args:
        embeddings (np.ndarray): Embedding vectors to save
        paths (List[str]): List of file paths corresponding to the embeddings
        save_dir (str): Directory where to save the files
        save_names (List[str]): List containing [embeddings_filename, paths_filename]
    """
    os.makedirs(save_dir, exist_ok=True)
    filename_mapping = {os.path.basename(path): i for i, path in enumerate(paths)}
    np.savez_compressed(os.path.join(save_dir, save_names[0]), embeddings)
    np.savez_compressed(os.path.join(save_dir, save_names[1]), filename_mapping)

def load_embeddings(folder_path: str, models: list=['coca', 'dino','mae']) -> Dict:
    """Load embeddings and paths for all specified models.
    
    Args:
        folder_path (str): Directory containing embeddings files
        models (list, optional): List of model names to load. Defaults to ['coca', 'dino','mae'].
    
    Returns:
        Dict: Dictionary with model names as keys, each containing 'embeddings' and 'paths' entries
    """
    data = {}
    
    for model in models:
        embeddings_path = os.path.join(folder_path, f'{model}_embeddings.npz')
        paths_path = os.path.join(folder_path, f'{model}_paths.npz')
        
        data[model] = {
            'embeddings': np.load(embeddings_path, allow_pickle=True)['arr_0'],
            'paths': np.load(paths_path, allow_pickle=True)['arr_0'].item()
        }
    print(f"Loaded embeddings: {folder_path}")
    return data
  
def load_existing_embeddings(
    save_dir: str,
    save_names: List[str]
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, int]]]:
    """Check if embeddings already exist and load them if they do.
    
    Args:
        save_dir (str): Directory where embeddings might be saved
        save_names (List[str]): List of [embeddings_filename, paths_filename]
    
    Returns:
        Tuple of (embeddings, filename_mapping) if found, else (None, None)
    """
    embeddings_path = os.path.join(save_dir, save_names[0])
    paths_path = os.path.join(save_dir, save_names[1])
    
    if os.path.exists(embeddings_path) and os.path.exists(paths_path):
        print(f"Loading existing embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        filename_mapping = np.load(paths_path, allow_pickle=True)['arr_0'].item()
        return embeddings, filename_mapping
    
    return None, None

def load_ratings(file_path: str) -> Dict:
    """Load meaning ratings data for indoor and outdoor scenes.
    
    Args:
        file_path (str): Path to CSV file containing meaning ratings
    
    Returns:
        Dict: Dictionary with 'indoor' and 'outdoor' categories, each containing 'ratings' data
    """
    rating_data = pd.read_csv(file_path)
    scene_categories = ['indoor', 'outdoor']
    data = {}
    
    for scene_category in scene_categories:
        category_data = rating_data[rating_data['category']==scene_category]
        data[scene_category] = {
            'ratings': category_data,
        }
    print(f"Loaded ratings: {file_path}")
    return data

def save_variables(filename, *variables):
    """Save multiple variables to a pickle file.
    
    Args:
        filename (str): Path to the output pickle file
        *variables: Variable number of objects to save
    """  
    with open(filename, 'wb') as f:
        pickle.dump(variables, f)
        
def load_variables(filename):
    """Load variables from a pickle file.
    
    Args:
        filename (str): Path to the pickle file to load
    
    Returns:
        tuple: Tuple containing all variables stored in the pickle file
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
      
def clear_directory(path, file_patterns=None):
    """Clear a directory by either removing all contents or specific file patterns.
    
    Args:
        path (str): Directory path to clear
        file_patterns (list, optional): List of file patterns to remove (e.g., ['*.pkl', '*.csv'])
    """
    if not os.path.exists(path):
        print(f"Directory does not exist: {path}")
        return
        
    if file_patterns:
        for pattern in file_patterns:
            files = glob.glob(os.path.join(path, pattern))
            for file in files:
                try:
                    os.remove(file)
                    print(f"Removed file: {file}")
                except Exception as e:
                    print(f"Error removing {file}: {e}")
    else:
        try:
            shutil.rmtree(path)
            os.makedirs(path)
            print(f"Cleared directory: {path}")
        except Exception as e:
            print(f"Error clearing {path}: {e}")

def clear_create_directory(base_dir='../data/create_meaning_maps/'):
    """Clear all files in subdirectories of create_meaning_maps without 
    deleting the subdirectories.
    
    
    Args:
        base_dir (str, optional): Base directory path for create_meaning_maps. 
                                 Defaults to '../data/create_meaning_maps/'.
    """
    if not os.path.exists(base_dir):
        print(f"Base directory does not exist: {base_dir}")
        return
        
    # Get all subdirectories
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        
        # Get all files in the subdirectory
        files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        
        for file in files:
            try:
                os.remove(os.path.join(subdir_path, file))
            except Exception as e:
                print(f"Error removing {os.path.join(subdir_path, file)}: {e}")
        
        # Find and clear subdirectories within this subdirectory
        nested_subdirs = [d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d))]
        for nested_subdir in nested_subdirs:
            nested_path = os.path.join(subdir_path, nested_subdir)
            files = [f for f in os.listdir(nested_path) if os.path.isfile(os.path.join(nested_path, f))]
            for file in files:
                try:
                    os.remove(os.path.join(nested_path, file))
                except Exception as e:
                    print(f"Error removing {os.path.join(nested_path, file)}: {e}")
    
    print(f"All files cleared from subdirectories in {base_dir}")

#%% 030: Patch/Grid/Map utlity functions
  
def save_grid(x_grid: np.ndarray, y_grid: np.ndarray, grid_name: str) -> None:
    """Save fuzzy patch x and y grids to a .npz file.
    
    Args:
        x_grid (np.ndarray): The x coordinates grid
        y_grid (np.ndarray): The y coordinates grid
        grid_name (str): Filename to save the grid (will be saved in '../data/grid/')
    
    Returns:
        None: Prints confirmation message
    """
    np.savez('../data/grid/'+grid_name, x_grid=x_grid, y_grid=y_grid)
    return print(f"{grid_name} saved to ../data/grid/")

def load_grid(folder_path: str) -> Dict:
    """Load fuzzy patch x and y grids from a .npz file.
    
    Args:
        folder_path (str): Path to the .npz file containing the grids
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: x_grid and y_grid coordinate arrays
    """
    grids = np.load(folder_path)
    x_grid = grids['x_grid']
    y_grid = grids['y_grid']
    print(f"Loaded grid: {folder_path}")
    return x_grid, y_grid

def adjust_coords(coords, gaps):
    """Adjust coordinates to maintain even spacing between grid points.
    
    This function iteratively adjusts coordinate positions to minimize 
    the difference between adjacent gap sizes.
    
    Args:
        coords (np.ndarray): Array of coordinates to adjust
        gaps (np.ndarray): Differences between adjacent coordinates
    
    Returns:
        np.ndarray: Adjusted coordinates with more even spacing
    """
    while np.any(np.abs(gaps - gaps[:, np.newaxis]) > 1):
        if gaps[-1] > np.mean(gaps):
            idx = np.argmax(gaps)
            coords[idx] += 1
        else:
            idx = np.argmin(gaps)
            coords[idx] -= 1
        gaps = np.diff(coords)
    return coords

def define_fuzzy_grid(width, height, tile_size, percent_overlap, show_grid=False):
    """Generate a grid of tile coordinates with fuzzy overlap adjustment.
    
    Creates a grid of coordinates for extracting overlapping image tiles,
    handling cases where width and height are not evenly divisible by tile_size.
    
    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels
        tile_size (int): Size of each square tile in pixels
        percent_overlap (float): Overlap between adjacent tiles (0-1)
        show_grid (bool, optional): Whether to display the resulting grid. Defaults to False.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of x and y coordinates for tile positions
    
    Raises:
        ValueError: If inputs are not numeric or percent_overlap is outside [0,1)
    """
    # Validate inputs
    if not all(isinstance(x, (int, float)) for x in [width, height, tile_size]):
        raise ValueError("Width, height, and tile_size must be numeric")
    if not 0 <= percent_overlap < 1:
        raise ValueError("percent_overlap must be between 0 and 1")
    
    def get_coord_array(size):
        overlap = round(tile_size * percent_overlap)
        coords = np.arange(tile_size - overlap, size - tile_size + overlap, tile_size - overlap)
        coords = np.insert(coords, 0, 0)
        coords[-1] = size - tile_size
        return adjust_coords(coords, np.diff(coords))
    
    x_coords = get_coord_array(width)
    y_coords = get_coord_array(height)
    
    if show_grid:
        fig, ax = plt.subplots()
        for x in x_coords:
            for y in y_coords:
                ax.plot([x, x + tile_size, x + tile_size, x, x],
                       [y, y, y + tile_size, y + tile_size, y], 'k')
                
    if show_grid:
        for x in x_coords:
            for y in y_coords:
                ax.plot(x + tile_size/2, y + tile_size/2, color='b', marker='.')
        
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        plt.show()
    
    return x_coords.astype(int), y_coords.astype(int)

def norm_range(x, a, b):
    """Normalize array values to a specific range [a,b].
    
    Args:
        x (np.ndarray): Input array to normalize
        a (float): Minimum value of output range
        b (float): Maximum value of output range
    
    Returns:
        np.ndarray: Normalized array with values mapped to range [a,b]
    """
    norm_x = (b-a)*((x-np.min(x))/(np.max(x)-np.min(x)))+a
    return norm_x
  
def smooth_average_map(average_map, sigma=15):
    """Apply Gaussian smoothing to a meaning map and preserve its original range.
    
    Args:
        average_map (np.ndarray): Input map to smooth
        sigma (int, optional): Standard deviation for Gaussian kernel. Defaults to 15.
    
    Returns:
        np.ndarray: Smoothed map with values in the same range as the input
    """
    # Smooth map to reduce grid artifacts 
    smooth_map = gaussian_filter(average_map, sigma=sigma)  
    # Return smooth map to original map range (i.e., avg_map range)
    final_map = norm_range(smooth_map, np.min(average_map), np.max(average_map))
    return final_map

def get_patches_and_ratings_scene(args):
    """Process a single scene to extract patches and calculate meaning ratings.
    
    Args:
        args: Tuple containing (scene_index, scene, scene_path, scene_indoor, patch_out, 
                               meaning_path, x_grid, y_grid, patch_size)
    
    Returns:
        List of rows for the CSV file in the format [category, image, scene, meaning_rating]
    """
    i, scene, scene_path, scene_indoor, patch_out, meaning_path, x_grid, y_grid, patch_size = args
    
    # Read image and convert
    scene_stem = os.path.splitext(scene)[0]
    image = cv2.imread(os.path.join(scene_path, scene))
    
    # Read average human rating map to get patch ratings
    meaning = np.load(os.path.join(meaning_path, 'average_rating_maps', f'{scene_stem}.npz'))['array']
    
    # Determine if scene is indoor or outdoor
    scene_type = 'indoor' if scene in scene_indoor else 'outdoor'
    
    # Extract patches and save each as numbered png, collect CSV rows
    csv_rows = []
    count = 0
    for y in y_grid:
        for x in x_grid:
            tile = image[y:y+patch_size, x:x+patch_size]
            tile_meaning = meaning[y:y+patch_size, x:x+patch_size].mean()
            image_name = f'{scene_stem}_{count}.png'
            cv2.imwrite(os.path.join(patch_out, image_name), tile)
            csv_rows.append([scene_type, image_name, scene_stem, tile_meaning])
            count += 1
    
    return csv_rows

def get_patches_and_ratings(x_grid, y_grid, patch_size, num_workers=None):
    """Extract patches from all internal scenes with human meaning maps and 
       save them with corresponding average human meaning ratings.
    
    This parallelized function:
    1. Extracts square patches based on the provided grid coordinates
    2. Calculates the mean meaning rating for each patch
    3. Saves each patch as an image
    4. Records the category, filename, scene, and meaning rating in a CSV file
    
    Args:
        x_grid (np.ndarray): Array of x-coordinates for patch extraction
        y_grid (np.ndarray): Array of y-coordinates for patch extraction
        patch_size (int): Size of each square patch in pixels
        num_workers (int, optional): Number of worker processes. Defaults to CPU count - 1.
    """
    # Get all scene images and indoor for classification
    scene_path = '../data/scenes/internal/all/'
    scenes = os.listdir(scene_path)
    scene_indoor = os.listdir('../data/scenes/internal/indoor/')
    
    # Define output directories
    patch_out = '../data/patches/'
    meaning_path = '../data/human_meaning/'
    csv_file_path = os.path.join(meaning_path, 'mean_patch_rating.csv')
    
    # Determine number of processes to use
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)  # Use all cores but one
    
    # Prepare arguments for each scene
    args_list = [
        (i, scene, scene_path, scene_indoor, patch_out, meaning_path, x_grid, y_grid, patch_size) 
        for i, scene in enumerate(scenes)
    ]
    
    # Process scenes in parallel
    all_csv_rows = []
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(get_patches_and_ratings_scene, args_list),
            total=len(scenes),
            desc="Extracting patches and calculating ratings:",
            unit="scene"
        ))
        
        # Flatten results
        for scene_rows in results:
            all_csv_rows.extend(scene_rows)
    
    # Write all rows to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write header
        csvwriter.writerow(['category', 'image', 'scene', 'meaning_rating'])
        
        # Write all data rows
        csvwriter.writerows(all_csv_rows)
    
    print(f"Extracted patches and saved ratings for {len(scenes)} scenes")

#%% 040: Predict_meaning.py utility functions

def prepare_scene_data(
    embeddings: np.ndarray,
    path_mapping: Dict[str, int],
    ratings_df: pd.DataFrame
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for scene-wise cross validation by aligning embeddings with ratings.
    
    Args:
        embeddings (np.ndarray): Array of embeddings for patches
        path_mapping (Dict[str, int]): Mapping from filenames to embedding indices
        ratings_df (pd.DataFrame): DataFrame containing meaning ratings for patches
    
    Returns:
        Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - unique_scenes: List of unique scene names
            - aligned_embeddings: Embeddings aligned with ratings
            - aligned_ratings: Meaning ratings
            - sample_scenes: Scene name for each sample
            - sample_images: Image filename for each sample
    """
    aligned_embeddings = []
    aligned_ratings = []
    scenes = []
    images = []
    print('Preparing scene data for cross validation...')
    
    for _, row in ratings_df.iterrows():
        filename = row['image']
        if filename in path_mapping:
            idx = path_mapping[filename]
            aligned_embeddings.append(embeddings[idx])
            aligned_ratings.append(row['meaning_rating'])
            scenes.append(get_scene_from_path(filename))
            images.append(filename)
    
    return (
        list(set(scenes)),  # unique scenes
        np.array(aligned_embeddings),
        np.array(aligned_ratings),
        np.array(scenes),
        np.array(images)
    )

def process_single_scene(
    test_scene: str,
    X: np.ndarray,
    y: np.ndarray,
    sample_scenes: np.ndarray,
    sample_images: np.ndarray,
    model_name: str,
    category: str
) -> Tuple[List[dict], dict]:
    """Process a single scene for cross-validation and return predictions and model weights.
    
    This function:
    1. Splits data into training (other scenes) and test (current scene) sets
    2. Trains a linear regression model on the training data
    3. Makes predictions on the test data
    4. Returns both prediction results and model weight information
    
    Args:
        test_scene (str): Name of the scene to use as test set
        X (np.ndarray): Feature matrix (embeddings)
        y (np.ndarray): Target values (meaning ratings)
        sample_scenes (np.ndarray): Array of scene names for each sample
        sample_images (np.ndarray): Array of image filenames for each sample
        model_name (str): Name of the model being evaluated
        category (str): Category of scenes ('indoor' or 'outdoor')
    
    Returns:
        Tuple[List[dict], dict]: 
            - pred_results: List of prediction results for each test sample
            - weight_info: Dictionary with model weights and performance metrics
    """
    # Split data by scene
    test_mask = (sample_scenes == test_scene)
    train_mask = ~test_mask
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Get test images for this scene
    test_images = sample_images[test_mask]
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Store prediction results
    pred_results = []
    for img, true_rating, pred_rating in zip(test_images, y_test, y_pred):
        pred_results.append({
            'model': model_name,
            'category': category,
            'scene': test_scene,
            'image': img,
            'rating': true_rating,
            'predicted': pred_rating
        })
    
    # Store weight information
    weight_info = {
        'weights': model.coef_,
        'intercept': model.intercept_,
        'train_r2': model.score(X_train, y_train),
        'test_r2': model.score(X_test, y_test),
        'scene': test_scene
    }
    
    return pred_results, weight_info

def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    scenes: List[str],
    sample_scenes: np.ndarray,
    sample_images: np.ndarray,
    model_name: str,
    category: str
) -> Tuple[pd.DataFrame, Dict]:
    """Evaluate linear model using parallel leave-one-scene-out cross-validation.
    
    This function performs leave-one-scene-out cross-validation by:
    1. Processing each scene in parallel as a test set
    2. Combining all prediction results
    3. Creating a final model with averaged weights across all folds
    
    Args:
        X (np.ndarray): Feature matrix (embeddings)
        y (np.ndarray): Target values (meaning ratings)
        scenes (List[str]): List of unique scene names
        sample_scenes (np.ndarray): Array of scene names for each sample
        sample_images (np.ndarray): Array of image filenames for each sample
        model_name (str): Name of the model being evaluated
        category (str): Category of scenes ('indoor' or 'outdoor')
    
    Returns:
        Tuple[pd.DataFrame, Dict]: 
            - predictions_df: DataFrame with all predictions
            - model_info: Dictionary with averaged model and fold information
    """
    # Create a partial function with fixed arguments
    process_scene = partial(
        process_single_scene,
        X=X,
        y=y,
        sample_scenes=sample_scenes,
        sample_images=sample_images,
        model_name=model_name,
        category=category
    )
    
    # Use all cores except one to prevent system sluggishness
    num_cores = max(1, os.cpu_count() - 2)  # Ensure at least 1 core is used
    
    # Process scenes in parallel
    with Pool(processes=num_cores) as pool:
      results = list(tqdm(
          pool.imap(process_scene, scenes),
          total=len(scenes),
          desc="Processing"
      ))
    
    # Unzip the results
    pred_results, weight_infos = zip(*results)
    
    # Flatten prediction results
    flat_preds = [item for sublist in pred_results for item in sublist]
    predictions_df = pd.DataFrame(flat_preds)
    
    # Average weights and create final model
    all_weights = np.array([info['weights'] for info in weight_infos])
    all_intercepts = np.array([info['intercept'] for info in weight_infos])
    
    # Create final model with averaged weights
    final_model = LinearRegression()
    final_model.coef_ = np.mean(all_weights, axis=0)
    final_model.intercept_ = np.mean(all_intercepts)
    
    # Compute weight statistics
    weight_std = np.std(all_weights, axis=0)
    
    # Store model info
    model_info = {
        'model': final_model,
        'mean_weights': final_model.coef_,
        'mean_intercept': final_model.intercept_,
        'weight_std': weight_std,
        'all_weights': all_weights,
        'all_intercepts': all_intercepts,
        'scene_info': weight_infos,  # Includes per-scene R² scores
        'feature_size': X.shape[1]
    }
    
    return predictions_df, model_info

def compute_single_map(
    args: Tuple[str, str],
    df: pd.DataFrame,
    scene_width: int,
    scene_height: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    tile: int,
    scene_dir: str
) -> None:
    """Process a single scene to compute its DeepMeaning map with minimal memory usage.

    This function constructs a meaning map for a single scene by:
    1. Accumulating patch-level predictions at their corresponding spatial locations
    2. Averaging values in overlapping regions
    3. Saving the resulting map as a compressed numpy array (.npz)
    
    The function is designed to be memory-efficient by using direct accumulation
    rather than pre-allocating the full map structure for all scenes.
    
    Args:
        args (Tuple[str, str]): Tuple containing (model_name, scene_name)
        df (pd.DataFrame): DataFrame with patch-level predictions
        scene_width (int): Width of the scene in pixels
        scene_height (int): Height of the scene in pixels
        x_grid (np.ndarray): Array of x-coordinates for patches
        y_grid (np.ndarray): Array of y-coordinates for patches
        tile (int): Size of each patch in pixels
        scene_dir (str): Directory where meaning maps will be saved
    
    Returns:
        None: Results are saved to disk as .npz files
        
    Note:
        This function is designed to be called via multiprocessing to 
        efficiently process multiple scenes in parallel.
    """
    model, scene = args
    
    # Determine output path
    if os.path.basename(scene_dir) in ['indoor', 'outdoor', 'diffeomorph', 'original']:
        file_path = os.path.join(scene_dir, f"{scene}.npz")
    else:
        file_path = os.path.join(scene_dir, model, f"{scene}.npz")
    
    # Create parent directory if needed
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Skip if already exists
    if os.path.exists(file_path):
        return
    
    # Initialize accumulation arrays, uint8 is fine for count
    sum_map = np.zeros((scene_height, scene_width), dtype=np.float64)
    count_map = np.zeros((scene_height, scene_width), dtype=np.uint8)
    
    # Process patches
    count = 0
    for y in y_grid:
        for x in x_grid:
            tile_name = f"{scene}_{count}.png"
            patch_row = df[df['image'] == tile_name]
            
            if not patch_row.empty:
                prediction = float(patch_row['predicted'].values[0])
                
                # Add prediction to the sum map and increment count
                sum_map[y:y+tile, x:x+tile] += prediction
                count_map[y:y+tile, x:x+tile] += 1
            
            count += 1
    
    # Compute average (safe division)
    avg_map = np.zeros_like(sum_map)
    mask = count_map > 0
    avg_map[mask] = sum_map[mask] / count_map[mask]
    
    # Save result
    np.savez_compressed(file_path, array=avg_map)
    
    # Explicitly clean up to help garbage collection
    del sum_map
    del count_map
    del avg_map

def compute_single_map_wrapper(args):
    """Wrapper function to unpack arguments for compute_single_map.
    This allows us to use imap with tqdm while avoiding lambda functions.
    """
    return compute_single_map(*args)

def compute_scene_maps(
    df: pd.DataFrame,
    scene_width: int,
    scene_height: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    tile: int,
    output_dir: Optional[str] = None,
    force_compute: bool = False
) -> None:
    """Compute DeepMeaning maps for all scenes using optimized parallel processing.
    
    This function:
    1. Processes all scenes from a DataFrame containing patch-level predictions
    2. Creates meaning maps by averaging predictions at each pixel location
    3. Uses parallel processing to efficiently handle large datasets
    4. Skips scenes that already have computed maps unless force_compute is True
    5. Organizes outputs by model type in the specified directory structure
    
    The function handles different directory structures depending on whether the 
    output_dir is specified and what type of directory structure it contains.
    
    Args:
        df (pd.DataFrame): DataFrame containing patch predictions with columns:
                          'model', 'category', 'scene', 'image', 'predicted'
        scene_width (int): Width of the scene images in pixels
        scene_height (int): Height of the scene images in pixels
        x_grid (np.ndarray): Array of x-coordinates for patch extraction
        y_grid (np.ndarray): Array of y-coordinates for patch extraction
        tile (int): Size of each square patch in pixels
        output_dir (Optional[str], optional): Directory to save meaning maps.
                                             If None, uses '../data/DeepMeaning/internal/'.
                                             Defaults to None.
        force_compute (bool, optional): Whether to recompute maps that already exist.
                                       Defaults to False.
    
    Returns:
        None: Results are saved as .npz files in the specified directory structure
        
    Note:
        This function automatically creates the necessary directory structure
        and organizes maps by model type. It also uses multiprocessing to
        parallelize computation across CPU cores for efficiency.
    """
    models = df['model'].unique()
    scenes = df['scene'].unique()
    scene_dir = '../data/DeepMeaning/internal/' if output_dir is None else output_dir
    
    # Control parallelism through number of cores
    num_cores = max(1, min(os.cpu_count() - 1, 12))
    
    for i, model in enumerate(models, 1):
        print(f"{model.upper()} embedding model:")
        
        # Determine the model directory
        if output_dir is None:
            model_dir = os.path.join(scene_dir, model)
            os.makedirs(model_dir, exist_ok=True)
            output_pattern = os.path.join(model_dir, "*.npz")
        else:
            if os.path.basename(output_dir) in ['indoor', 'outdoor', 'diffeomorph', 'original']:
                output_pattern = os.path.join(output_dir, "*.npz")
            else:
                model_dir = os.path.join(output_dir, model)
                os.makedirs(model_dir, exist_ok=True)
                output_pattern = os.path.join(model_dir, "*.npz")
        
        # Get list of already processed scenes
        existing_scene_maps = set()
        if not force_compute:
            existing_scene_maps = {os.path.splitext(os.path.basename(f))[0] 
                                 for f in glob.glob(output_pattern)}
        
        # Filter scenes that need to be processed
        scenes_to_process = [scene for scene in scenes if scene not in existing_scene_maps]
        
        if not scenes_to_process:
            print(f"All scenes for model '{model}' already exist. Skipping...")
            continue
            
        total = len(scenes_to_process)
        print(f"Processing {total} scenes (skipping {len(scenes) - total} existing scenes)")
        
        # Filter the global dataframe to just this model
        model_df = df[df['model'] == model]
        
        # Prepare all tasks
        args_list = []
        for scene in scenes_to_process:
            scene_df = model_df[model_df['scene'] == scene]
            args_list.append((
                (model, scene), 
                scene_df, 
                scene_width, 
                scene_height, 
                x_grid, 
                y_grid, 
                tile, 
                scene_dir
            ))
        
        # Process all scenes in parallel with tqdm
        with Pool(processes=num_cores) as pool:
            list(tqdm(
                pool.imap(compute_single_map_wrapper, args_list),
                total=len(args_list),
                desc="Constructing and saving maps:",
                unit="scene"
            ))
            
#%% 050: Meaning map creation utility functions

def create_patches_scene(args):
    """Process a single scene to extract patches.
    
    Args:
        args: Tuple containing (scene, folder_path, patch_out, x_grid, y_grid, patch_size)
    """
    scene, folder_path, patch_out, x_grid, y_grid, patch_size = args
    
    # Read image and convert
    scene_stem = os.path.splitext(scene)[0]
    image = cv2.imread(os.path.join(folder_path, scene))
    
    # Extract patches and save each as numbered png
    count = 0
    for y in y_grid:
        for x in x_grid:
            tile = image[y:y+patch_size, x:x+patch_size]
            image_name = f'{scene_stem}_{count}.png'
            cv2.imwrite(os.path.join(patch_out, image_name), tile)
            count += 1
    
    return scene_stem

def create_patches_folder(folder_path, x_grid, y_grid, patch_size=128, num_workers=None):
    """Perform batch patch extraction for all scenes in the specified path using parallel processing.
    
    Args:
        folder_path (str): Directory containing input scenes
        x_grid (np.ndarray): Array of x-coordinates for patches
        y_grid (np.ndarray): Array of y-coordinates for patches
        patch_size (int, optional): Size of each patch in pixels. Defaults to 128.
        num_workers (int, optional): Number of worker processes. Defaults to CPU count - 1.
    
    Saves:
        Extracted patches to ../data/create_meaning_maps/patch_temp/ directory
    """
    # Get all scene images for folder path
    scenes = os.listdir(folder_path)
    
    # Define output directories
    patch_out = '../data/create_meaning_maps/patch_temp/'
    
    # Determine number of processes to use
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)  # Use all cores but one
    
    # Prepare arguments for each scene
    args_list = [(scene, folder_path, patch_out, x_grid, y_grid, patch_size) for scene in scenes]
    
    # Process scenes in parallel
    with Pool(processes=num_workers) as pool:
        _ = list(tqdm(
            pool.imap(create_patches_scene, args_list),
            total=len(scenes),
            desc="Extracting patches using specified grid:", 
            unit="scene"
        ))
    
#%% 060: Figure plotting utility functions

def draw_brace(ax, xspan, yy, text):
    """Draw a curly brace annotation on a matplotlib axis.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to draw on
        xspan (Tuple[float, float]): (xmin, xmax) span of the brace
        yy (float): Vertical position of the brace
        text (str): Text to display below the brace
    """
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, -y, color='black', lw=1, clip_on=False)

    ax.text((xmax+xmin)/2., -yy-.17*yspan, text, ha='center', va='bottom')

def plot_distribution(data, ax=None, y_position=0, color='skyblue', 
                     density_to_box_gap=0.25, box_to_strip_gap=0.25,
                     strip_width=0.1, density_scale=1.0, density_alpha=.9,
                     marker_size=20, marker_alpha=1,
                     figsize=(10, 6), bandwidth=0.15,
                     normalize_density=True):
    """Create a comprehensive distribution visualization combining KDE, boxplot, and stripplot.
    
    This function creates a detailed visualization of a distribution that includes:
    1. A kernel density estimate (KDE) curve
    2. Boxplot elements showing quartiles and whiskers
    3. Individual data points as a strip plot with controlled jitter
    
    Args:
        data (array-like): Data to visualize
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
        y_position (float, optional): Vertical position of the plot. Defaults to 0.
        color (str, optional): Color for the distribution. Defaults to 'skyblue'.
        density_to_box_gap (float, optional): Gap between density curve and boxplot. Defaults to 0.25.
        box_to_strip_gap (float, optional): Gap between boxplot and stripplot. Defaults to 0.25.
        strip_width (float, optional): Width of the strip plot band. Defaults to 0.1.
        density_scale (float, optional): Scaling factor for density curve height. Defaults to 1.0.
        density_alpha (float, optional): Alpha transparency for density curve. Defaults to 0.9.
        marker_size (float, optional): Size of strip plot markers. Defaults to 20.
        marker_alpha (float, optional): Alpha transparency for markers. Defaults to 1.
        figsize (tuple, optional): Figure size if creating new figure. Defaults to (10, 6).
        bandwidth (float, optional): Bandwidth for KDE. Defaults to 0.15.
        normalize_density (bool, optional): Whether to normalize density peak to 1. Defaults to True.
    
    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Figure and axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    data_clean = np.array(data).astype(float)
    kde = gaussian_kde(data_clean, bw_method=bandwidth)
    x_grid = np.linspace(min(data_clean)-min(data_clean)*.10, max(data_clean)+max(data_clean)*.10, 200)
    density = kde(x_grid)
    
    if normalize_density:
        # Normalize density to maintain consistent area under curve
        density = density / np.max(density)
        
    density_scaled = density * density_scale
    max_scaled_density = np.max(density_scaled)
    
    ax.fill_between(x_grid, y_position, density_scaled + y_position, 
                   color=color, alpha=density_alpha)
    ax.plot(x_grid, density_scaled + y_position, color='black', alpha=.6, linewidth=1.5)
    
    def draw_box_lines(ax, input_linewidth, data, center):
        """Draw boxplot information at center of the density."""
        # Compute the boxplot statistics
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        whisker_lim = 1.5 * iqr(data)
        h1 = np.min(data[data >= (q25 - whisker_lim)])
        h2 = np.max(data[data <= (q75 + whisker_lim)])
        
        # Draw whiskers
        ax.plot([h1, h2], [center, center],
                linewidth=input_linewidth,
                color='black')
        
        # Draw box
        ax.plot([q25, q75], [center, center],
                linewidth=input_linewidth * 3,
                color='black')
        
        # Draw median point
        ax.scatter(q50, center,
                  zorder=3,
                  color="white",
                  edgecolor='black',
                  s=np.square(input_linewidth * 2))
    
    # Calculate positions with consistent scaling
    unit_height = max_scaled_density  # Use this for consistent scaling
    
    # Calculate positions
    box_position = y_position - (unit_height * density_to_box_gap)
    strip_top = box_position - (unit_height * box_to_strip_gap)
    strip_bottom = strip_top - (unit_height * strip_width)
    
    # Draw box plot elements
    draw_box_lines(ax, 1.5, data_clean, box_position)
    
    # Add strip plot with controlled jitter
    n_points = len(data_clean)
    # Generate uniform jitter within the strip width band
    jitter_values = np.random.uniform(strip_bottom, strip_top, size=n_points)
    ax.scatter(data_clean, 
              jitter_values,
              alpha=marker_alpha,
              color=color,
              s=marker_size,
              linewidth=marker_size/10)
    
    # Adjust which splines are visible default to all
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_yticks([])
    
    return fig, ax

#%% 070: Interpretation analysis utility functions

def get_scene_from_path(path: str) -> str:
    """Extract scene name from a patch filename.
    
    Args:
        path (str): Patch filename (e.g., 'scene_123.png')
    
    Returns:
        str: Scene name (e.g., 'scene')
    """
    return re.split(r'_\d+\.png', path)[0]

def get_contrastive_criterion_embedding(
    pos_prompt: str,
    neg_prompt: str,
    tokenizer,
    coca_model,
    device='cpu'
) -> np.ndarray:
    """Get CoCa text embedding for a semantic criterion using positive and negative prompts.
    
    This function:
    1. Encodes both positive and negative text prompts using CoCa
    2. Computes the difference between the positive and negative embeddings
    3. Returns this difference as a semantic direction vector
    
    Args:
        pos_prompt (str): Positive description of the criterion (e.g. "many objects")
        neg_prompt (str): Negative description of the criterion (e.g. "no objects")
        tokenizer: CoCa tokenizer
        coca_model: CoCa model
        device (str, optional): Device to run model on. Defaults to 'cpu'.
        
    Returns:
        np.ndarray: Embedding representing the semantic direction
    """
    tokens = tokenizer([pos_prompt, neg_prompt])
    with torch.no_grad():
        embeddings = coca_model.encode_text(tokens.to(device))
    
    # Use difference between positive and negative embeddings as direction
    criterion_direction = embeddings[0] - embeddings[1]
    return criterion_direction.cpu().numpy()

def get_diverse_scene_indices(sorted_indices, image_paths, n_examples):
    """Select indices ensuring each comes from a different scene for diverse sampling.
    
    Args:
        sorted_indices (np.ndarray): Array of indices, typically sorted by some criterion
        image_paths (List[str]): List of image paths corresponding to the indices
        n_examples (int): Number of examples to select
    
    Returns:
        np.ndarray: Array of selected indices from diverse scenes
    """
    selected_indices = []
    seen_scenes = set()
    
    for idx in sorted_indices:
        scene = get_scene_from_path(image_paths[idx])
        if scene not in seen_scenes and len(selected_indices) < n_examples:
            selected_indices.append(idx)
            seen_scenes.add(scene)
            
        if len(selected_indices) == n_examples:
            break
            
    return np.array(selected_indices)

def visualize_projection_analysis(
    coca_model,
    coca_tokenizer,
    coca_image_embeddings: np.ndarray,
    attention: np.ndarray,
    scene_name: np.ndarray,
    fold_models,
    meaning_ratings: np.ndarray,
    image_paths: list,
    pos_prompt: str,
    neg_prompt: str,
    prompt_title: str,
    save_title: str,
    letters: list,
    color: str,
    output_dir: str,
    n_examples: int = 6
):
    """Create a visualization of projections along a semantic dimension with examples.
    
    This function:
    1. Computes a semantic direction from contrastive prompts
    2. Projects image embeddings onto this direction
    3. Analyzes correlations with meaning (human and DeepMeaning) and attention
    4. Creates a visualization with scatter plots and example patches
    
    Args:
        coca_model: CoCa model
        coca_tokenizer: CoCa tokenizer
        coca_image_embeddings (np.ndarray): Image embeddings (n_samples, 768 features)
        attention (np.ndarray): Patch fixation density values
        scene_name (np.ndarray): Array of scene names for each sample
        fold_models (dict): Dictionary containing DeepMeaning cross-validated folds
        meaning_ratings (np.ndarray): Ground truth meaning ratings
        image_paths (list): List of paths to image patches
        pos_prompt (str): Positive description of criterion
        neg_prompt (str): Negative description of criterion
        prompt_title (str): Title for the semantic dimension
        save_title (str): Filename to save the visualization
        letters (list): List of labels for figure panels
        color (str): Colormap to use for visualization
        output_dir (str): Directory to save the visualization
        n_examples (int, optional): Number of examples to show per category. Defaults to 6.
        
    Returns:
        np.ndarray: Array of projection values for each patch
    """
    
    # Use leave-one-scene-out weights for analysis for best estimate
    # of DeepMeaning generalization
    fold_scenes = [item['scene'] for item in fold_models['scene_info']]
    fold_intercepts = fold_models['all_intercepts']
    fold_weights = fold_models['all_weights']
    
    # Compute normalized DeepMeaning patch predictions (using LOSOCV fold values)
    DM = np.zeros(coca_image_embeddings.shape[0])
    for i, scene in enumerate(fold_scenes):
      current_idxs = scene_name == scene
      scene_model = LinearRegression()
      scene_model.coef_ = fold_weights[i]
      scene_model.intercept_ = fold_intercepts[i]
      DM[current_idxs] = scene_model.predict(coca_image_embeddings[current_idxs,:])
    DM_normalized = (DM - DM.min()) / (DM.max() - DM.min())
    
    # Normalize human meaning patch ratings
    meaning_normalized = (meaning_ratings - meaning_ratings.min()) / (meaning_ratings.max() - meaning_ratings.min())
    
    # Normalize embeddings
    coca_image_embeddings = coca_image_embeddings / np.linalg.norm(coca_image_embeddings, axis=1, keepdims=True)
    criterion_embedding = get_contrastive_criterion_embedding(pos_prompt, neg_prompt, coca_tokenizer, coca_model)
    criterion_direction = criterion_embedding / np.linalg.norm(criterion_embedding)
    
    # Calculate projections and normalize 0-1
    projections = coca_image_embeddings @ criterion_direction.T
    projections = projections.flatten()  # Ensure 1D array
    proj_normalized = (projections - projections.min()) / (projections.max() - projections.min())
    
    # Index all variables using attention nan values
    fix_idx = ~np.isnan(attention)
    DM_normalized = DM_normalized[fix_idx]
    meaning_normalized = meaning_normalized[fix_idx]
    proj_normalized = proj_normalized[fix_idx]
    image_paths = np.array(image_paths)[fix_idx]
    attention_normalized = attention[fix_idx]
    
    # Compute correlations
    DM_R = np.corrcoef(proj_normalized, DM_normalized)[0,1]
    meaning_R = np.corrcoef(proj_normalized, meaning_normalized)[0,1]
    attention_R = np.corrcoef(proj_normalized, attention_normalized)[0,1]
    
    # Sort indices for example patches
    # sorted_indices = np.argsort(proj_normalized)
    # low_criterion_idx = sorted_indices[:n_examples*2]
    # high_criterion_idx = sorted_indices[-n_examples*2:]
    # Sort indices for example patches
    sorted_indices = np.argsort(proj_normalized)
    n_total = len(sorted_indices)
    median_start = n_total//2 - n_examples*10  # Start looking around the middle
    median_end = n_total//2 + n_examples*10    # End looking around the middle
    low_criterion_idx = get_diverse_scene_indices(sorted_indices[:n_examples*20], image_paths, n_examples)
    median_criterion_idx = get_diverse_scene_indices(sorted_indices[median_start:median_end], image_paths, n_examples)
    high_criterion_idx = get_diverse_scene_indices(sorted_indices[-n_examples*20:][::-1], image_paths, n_examples)

    # Font sizes
    numfs = 13
    sfs = 11
    pfs = 11

    # Create figure and gridspec
    fig = plt.figure(figsize=(17,4))
    gs_main = gridspec.GridSpec(3, 7, width_ratios=[1, .2, 1, .001, .666, .666, .666], wspace=0.09, hspace=0)
     
    # Panel 1: Prompts (spans all rows)
    ax1 = fig.add_subplot(gs_main[:,0])
    ax1.axis('on')

    # Wrap text
    wrapped_pos = textwrap.fill(f"'{pos_prompt}'", width=33)
    wrapped_neg = textwrap.fill(f"'{neg_prompt}'", width=33)

    # Add text with wrapping and left alignment
    ax1.text(0,1.018, letters[0], fontsize = numfs, weight='bold')
    ax1.text(0.06, 0.72, wrapped_pos, ha='left', va='center', fontsize=pfs)
    ax1.text(0.06, 0.23, wrapped_neg, ha='left', va='center', fontsize=pfs)

    ax1.text(0.5, 0.95, "Positive Prompt:", ha='center', va='center', fontsize=pfs, weight='bold')
    ax1.text(0.5, 0.40, "Negative Prompt", ha='center', va='center', fontsize=pfs, weight='bold')

    # Set title
    ax1.set_xlabel(f"{prompt_title}\n Semantic Dimension Prompts", fontsize=sfs, labelpad=10)

    # Add plot box
    for spine in ax1.spines.values():
        spine.set_visible(True)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Panel 2: Scatter plot (spans all columns)
    ax2 = fig.add_subplot(gs_main[:, 2])
    ax2.text(-.05,1.068, letters[1], fontsize = numfs, weight='bold')
    scatter = ax2.scatter(meaning_normalized, DM_normalized, c=proj_normalized, cmap=color, s=0.3)
    ax2.text(.358, 0, r"Human Meaning $R^{2}$=%.2f" % meaning_R**2, fontsize = 10)
    ax2.text(.41, .08, r"DeepMeaning $R_{cv}^{2}$=%.2f" % DM_R**2, fontsize = 10)
    ax2.text(.545, .16, r"Attention $R^{2}$=%.2f" % attention_R**2, fontsize = 10)
    ax2.set_xlabel('Normalized Meaning Rating', fontsize=sfs)
    ax2.set_ylabel('Normalized DeepMeaning Rating', fontsize=sfs)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.grid(alpha=.4)
    ax2.set_xlim(-.05, 1.05)
    ax2.set_ylim(-.05, 1.05)
    ax2.plot([-2, 2], [-2, 2], color='black', linestyle='-', linewidth=1)

    # Get legend elements and modify them
    legend_elements = scatter.legend_elements()
    handles = legend_elements[0][1:]  # Skip first handle
    labels = legend_elements[1][1:]   # Skip first label
    legend = ax2.legend(handles, labels, loc="upper left",
                      title=f"{prompt_title}", ncol=2,
                      fontsize=6.5,  # Decrease the size of legend text
                      title_fontsize=7.5)  # Decrease the size of legend title)
    ax2.add_artist(legend)


    # Add titles for high and low criterion examples
    gs_3 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs_main[:, 4], wspace=0.04, hspace=0)
    gs_4 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs_main[:, 5], wspace=0.04, hspace=0)
    gs_5 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs_main[:, 6], wspace=0.04, hspace=0)
    ax_title_3 = fig.add_subplot(gs_3[:, :])
    ax_title_3.text(0.5, -.09, f"High {prompt_title}\n Score Patches", fontsize=sfs, ha='center', va='center')
    ax_title_3.set_axis_off()
    ax_title_3.text(0,1.018, letters[2], fontsize = numfs, weight='bold')

    ax_title_4 = fig.add_subplot(gs_4[:, :])
    ax_title_4.text(0.5, -.09, f"Median {prompt_title}\n Score Patches", fontsize=sfs, ha='center', va='center')
    ax_title_4.set_axis_off()
  

    ax_title_5 = fig.add_subplot(gs_5[:, :])
    ax_title_5.text(0.5, -.09, f"Low {prompt_title}\n Score Patches", fontsize=sfs, ha='center', va='center')
    ax_title_5.set_axis_off()

    # Panel 3: Highest criterion examples
    for i, idx in enumerate(high_criterion_idx[::-1][:6]):
        ax = fig.add_subplot(gs_3[i // 2, i % 2])
        img = Image.open(image_paths[idx])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(False)

    # Panel 4: Median criterion examples
    for i, idx in enumerate(median_criterion_idx[:6]):
        ax = fig.add_subplot(gs_4[i // 2, i % 2])
        img = Image.open(image_paths[idx])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(False)
        
    # Panel 5: Lowest criterion examples
    for i, idx in enumerate(low_criterion_idx[:6]):
        ax = fig.add_subplot(gs_5[i // 2, i % 2])
        img = Image.open(image_paths[idx])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(False)
        
    plt.savefig(os.path.join(output_dir, save_title +'.png'), dpi=100, bbox_inches='tight')
    plt.close()
    
    return projections
  
def analyze_semantic_prediction(
    coca_model,
    coca_tokenizer,
    coca_image_embeddings: np.ndarray,
    attention: np.ndarray,
    scene_name: np.ndarray,
    fold_models: dict,
    meaning_ratings: np.ndarray,
    prompt_pairs: List[Tuple[str, str]],
    device='cpu'
) -> Dict:
    """Analyze semantic dimensions' contribution to meaning using dominance analysis.
    
    This function:
    1. Computes projections for multiple semantic directions
    2. Performs dominance analysis to determine the relative importance of each dimension
    3. Calculates correlations with meaning and attention
    
    Args:
        coca_model: CoCa model
        coca_tokenizer: CoCa tokenizer
        coca_image_embeddings (np.ndarray): Image embeddings
        attention (np.ndarray): Patch fixation density values
        scene_name (np.ndarray): Array of scene names for each sample
        fold_models (dict): Dictionary containing DeepMeaning cross-validated folds
        meaning_ratings (np.ndarray): Ground truth meaning ratings
        prompt_pairs (List[Tuple[str, str]]): List of (positive, negative) prompt pairs
        device (str, optional): Device to run model on. Defaults to 'cpu'.
        
    Returns:
        Dict: Dictionary with analysis results including correlations and variance proportions
    """

    # Use leave-one-scene-out folds for analysis 
    fold_scenes = [item['scene'] for item in fold_models['scene_info']]
    fold_intercepts = fold_models['all_intercepts']
    fold_weights = fold_models['all_weights']
    
    # Compute normalized DeepMeaning patch predictions (using LOSOCV weights)
    DM = np.zeros(coca_image_embeddings.shape[0])
    for i, scene in enumerate(fold_scenes):
      current_idxs = scene_name == scene
      scene_model = LinearRegression()
      scene_model.coef_ = fold_weights[i]
      scene_model.intercept_ = fold_intercepts[i]
      DM[current_idxs] = scene_model.predict(coca_image_embeddings[current_idxs,:])
    DM_normalized = (DM - DM.min()) / (DM.max() - DM.min())
    
    # Normalize human meaning patch ratings
    meaning_normalized = (meaning_ratings - meaning_ratings.min()) / (meaning_ratings.max() - meaning_ratings.min())
    
    # Index all variables with using attention nan values to select only
    # the values that correspond with scenes with attention values
    fix_idx = ~np.isnan(attention)
    DM_normalized = DM_normalized[fix_idx]
    M_normalized = meaning_normalized[fix_idx]
    A_normalized = attention[fix_idx]
    
    # Compute projections for all contrastive prompt pairs
    all_projections = []
    for pos_prompt, neg_prompt in prompt_pairs:
        tokens = coca_tokenizer([pos_prompt, neg_prompt])
        with torch.no_grad():
            embeddings = coca_model.encode_text(tokens.to(device))
        direction = embeddings[0] - embeddings[1]
        direction = direction.cpu().numpy()
        projections = coca_image_embeddings @ direction.T
        projections = projections.flatten()
        proj_normalized = (projections - projections.min()) / (projections.max() - projections.min())
        proj_normalized = proj_normalized[fix_idx]
        all_projections.append(proj_normalized)
    X = np.vstack(all_projections).T
    n_dims = len(prompt_pairs)
    
    # Compute R² for all possible subsets, 
    dominance_averages_DM = np.zeros(n_dims)  # DeepMeaning 
    dominance_averages_M = np.zeros(n_dims)   # Human meaning
    dominance_averages_A = np.zeros(n_dims)   # Attention
    from itertools import combinations
    
    # For each subset size
    for k in range(1, n_dims + 1):
        # For each subset of size k
        for subset in combinations(range(n_dims), k):
            X_subset = X[:, list(subset)]
            DM_subset_r2 = LinearRegression().fit(X_subset, DM_normalized).score(X_subset, DM_normalized)
            M_subset_r2 = LinearRegression().fit(X_subset, M_normalized).score(X_subset, M_normalized)
            A_subset_r2 = LinearRegression().fit(X_subset, A_normalized).score(X_subset, A_normalized)
            
            # Add R² contribution to each variable in subset
            DM_subset_contribution = DM_subset_r2 / k  # Equal division 
            M_subset_contribution = M_subset_r2 / k    # Equal division
            A_subset_contribution = A_subset_r2 / k              # Equal division
            for idx in subset:
                dominance_averages_DM[idx] += DM_subset_contribution / math.comb(n_dims - 1, k - 1)
                dominance_averages_M[idx] += M_subset_contribution / math.comb(n_dims - 1, k - 1)
                dominance_averages_A[idx] += A_subset_contribution / math.comb(n_dims - 1, k - 1)
    
    # Normalize to get proportions of total variance for each projection
    # DeepMeaning
    DM_total_r2 = LinearRegression().fit(X, DM_normalized).score(X, DM_normalized)
    DM_variance_proportions = dominance_averages_DM / np.sum(dominance_averages_DM)
    # Human meaning
    M_total_r2 = LinearRegression().fit(X, M_normalized).score(X, M_normalized)
    M_variance_proportions = dominance_averages_M / np.sum(dominance_averages_M)
    # Attention
    A_total_r2 = LinearRegression().fit(X, A_normalized).score(X, A_normalized)
    A_variance_proportions = dominance_averages_A / np.sum(dominance_averages_A)
    
    return {
        'DeepMeaning_total_r2': DM_total_r2,
        'HumanMeaning_total_r2': M_total_r2,
        'Attention_total_r2': A_total_r2,
        'DeepMeaning_correlations': [pearsonr(proj, DM_normalized)[0] for proj in all_projections],
        'HumanMeaning_correlations': [pearsonr(proj, M_normalized)[0] for proj in all_projections],
        'Attention_correlations': [pearsonr(proj, A_normalized)[0] for proj in all_projections],
        'DeepMeaning_variance_proportions': DM_variance_proportions,
        'HumanMeaning_variance_proportions': M_variance_proportions,
        'Attention_variance_proportions': A_variance_proportions
    }

def print_semantic_results(results, prompt_pairs, prompt_ids):
    """Print formatted results from semantic analysis for DeepMeaning, human meaning, and attention.
    
    Args:
        results (Dict): Dictionary with analysis results from analyze_semantic_prediction
        prompt_pairs (List[Tuple[str, str]]): List of (positive, negative) prompt pairs
        prompt_ids (List[str]): List of names/ids for each semantic dimension
    
    Prints:
        Formatted output showing correlations and variance contributions for each dimension
    """
    for measure in ['DeepMeaning', 'HumanMeaning', 'Attention']:
        print(f"\n{measure} Semantic Direction Analysis:")
        print(f"Total variance explained (R²): {results[f'{measure}_total_r2']:.3f}")
        
        print("Per-dimension results:")
        for i, ((pos, neg), corr, var_prop) in enumerate(zip(
            prompt_pairs,
            results[f'{measure}_correlations'],
            results[f'{measure}_variance_proportions']
        )):
            print(f"\nDimension {i+1}: {prompt_ids[i]}")
            print(f"Pos: '{pos}'")
            print(f"Neg: '{neg}'")
            print(f"Correlation: {corr:.3f}")
            print(f"R²: {corr**2:.3f}")
            print('Dominance analysis:')
            print(f"Dimension contribution to total R²: {var_prop:.3f}")
