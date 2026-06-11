#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: batch_deep_meaning.py
Author: T.R. Hayes
Version: 1.0.0
Description: Script that will batch generate meaning maps for scenes stored in 
             the create_meaning_maps/input/ folder.

INSTRUCTIONS FOR USERS
1. Create conda environment with necessary packages (see environment.yml)
2. Define a grid for your scene size. Use one of the prexisting grids 
   (see DeepMeaning/data/grid/) or if your scenes are a different size create 
   a new grid (see define_fuzzy_grid.py in utils.py for details). 
   *NOTE Scenes must be a consistent size, if they are not pad them with zeros
   so that all scenes are a consistent size.
3. Place scenes in 'DeepMeaning/data/create_meaning_maps/input/' indoor and 
   outdoor folders.
4. From the terminal cd into DeepMeaning/prog folder and run script:
   python batch_deep_meaning.py --grid_file {'YOUR_DESIRED_GRID_FILE.npz'}
5. The respective indoor and outdoor DeepMeaning maps will be found in the
  'DeepMeaning/data/create_meaning/maps/output' indoor & outdoor folders.

Changelog:
- 1.0.0 (2024-01-02): TRH Wrote it
"""
#%% 010: Import packages

import os
import shutil
import pickle
import argparse
import pandas as pd
from coca import get_embeddings as get_coca_embeddings
from utils import (create_patches_folder, load_embeddings, load_grid, 
                   compute_scene_maps)

#%% 020: Define input and output directories

input_dir = "../data/create_meaning_maps/input/"
output_dir = "../data/create_meaning_maps/output/"
patch_dir = "../data/create_meaning_maps/patch_temp/"

#%% 030: Get indoor and outdoor input images

indoor_scenes = os.listdir(input_dir + 'indoor')
outdoor_scenes = os.listdir(input_dir + 'outdoor')

#%% 040: Define patch grid to be used
# see 'define_fuzzy_grid' to create custom grid size
# Note script does assume all scenes are the same size for simplicity.
# If you have images that vary in size, pad them with zeros so all images are 
# a fixed size first.

parser = argparse.ArgumentParser()
parser.add_argument("--grid_file", type=str)
args = parser.parse_args()
grid_file = args.grid_file

# Print error if no grid file is specified
if grid_file is None:
    print("ERROR: No grid file specified. Please use the --grid_file argument.")
    print("Example: python batch_deep_meaning.py --grid_file 1024x768_128px_73over.npz")
    import sys
    sys.exit(1)  # Exit with error status code

if grid_file=='1024x768_128px_73over.npz':
  x_grid, y_grid = load_grid('../data/grid/1024x768_128px_73over.npz') # Internal
  img_w = 1024
  img_h = 768
  patch_size = 128
elif grid_file=='1920x1080_128px_73over.npz':
  x_grid, y_grid = load_grid('../data/grid/1920x1080_128px_73over.npz') # CAT
  img_w = 1920
  img_h = 1080
  patch_size = 128

#%% 050: Specify indoor and outdoor linear model weights

indoor_model_path = '../data/DeepMeaning/DeepMeaning_indoor_ensemble.pkl'
indoor_model = pickle.load(open(indoor_model_path, 'rb'))['model']

outdoor_model_path = '../data/DeepMeaning/DeepMeaning_outdoor_ensemble.pkl'
outdoor_model = pickle.load(open(outdoor_model_path, 'rb'))['model']

#%% 060: Process indoor input scenes

# Generate patches for indoor_input
print('Processing indoor images:')
create_patches_folder(input_dir + 'indoor/', x_grid, y_grid, patch_size)

# Compute embeddings
get_coca_embeddings(patch_dir, patch_dir)

# Load CoCa embeddings
embeddings_data = load_embeddings('../data/create_meaning_maps/patch_temp/', ['coca'])
predicted_meaning = indoor_model.predict(embeddings_data['coca']['embeddings'])

# Create DataFrame from predictions
indoor_df = []
for i, (path, prediction) in enumerate(zip(embeddings_data['coca']['paths'].keys(), predicted_meaning)):
    # Extract scene name and patch number
    scene_name = path.split('_')[0]
    indoor_df.append({
        'model': 'coca',
        'category': 'indoor',
        'scene': scene_name,
        'image': path,
        'predicted': prediction
    })
indoor_df = pd.DataFrame(indoor_df)

# Ensure output directory exists
os.makedirs(output_dir + 'indoor/', exist_ok=True)

# Compute meaning maps using compute_scene_maps
compute_scene_maps(
    indoor_df,
    img_w,
    img_h,
    x_grid,
    y_grid,
    patch_size,
    output_dir + 'indoor/'
)

# Clear patch_temp directory
shutil.rmtree(patch_dir); os.mkdir(patch_dir)

#%% 070: Process outdoor input scenes

# Generate patches for indoor_input
print('Processing outdoor images:')
create_patches_folder(input_dir + 'outdoor/', x_grid, y_grid, patch_size)

# Compute embeddings
get_coca_embeddings(patch_dir, patch_dir)

# Load embeddings
embeddings_data = load_embeddings('../data/create_meaning_maps/patch_temp/', ['coca'])
predicted_meaning = outdoor_model.predict(embeddings_data['coca']['embeddings'])

# Create DataFrame from predictions
outdoor_df = []
for i, (path, prediction) in enumerate(zip(embeddings_data['coca']['paths'].keys(), predicted_meaning)):
    # Extract scene name and patch number
    scene_name = path.split('_')[0]
    outdoor_df.append({
        'model': 'coca',
        'category': 'outdoor',
        'scene': scene_name,
        'image': path,
        'predicted': prediction
    })
outdoor_df = pd.DataFrame(outdoor_df)

# Compute meaning maps and save
compute_scene_maps(
    outdoor_df,
    img_w,
    img_h,
    x_grid,
    y_grid,
    patch_size,
    output_dir + 'outdoor/'
)

# Clear patch_temp directory
shutil.rmtree(patch_dir); os.mkdir(patch_dir)