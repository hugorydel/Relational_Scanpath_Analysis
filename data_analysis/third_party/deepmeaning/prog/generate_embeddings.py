#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: generate_embeddings.py
Author: T.R. Hayes
Version: 1.0.0
Description: Convenience script to compute and save CoCa, DINO, & MAE 
             embeddings for DeepMeaning: Estimating and interpreting scene 
             meaning for attention using a vision-language 
             transformer (Hayes & Henderson, 2025).

*Note all embeddings are included to save processing time
              
Changelog:
- 1.0.0 (2024-11-20): TRH Wrote it
"""

#%% 010: Import packages

import os
import shutil
from coca import get_embeddings as get_coca_embeddings
from dino import get_embeddings as get_dino_embeddings
from mae import get_embeddings as get_mae_embeddings
from utils import (load_grid, get_patches_and_ratings, print_section, 
                   create_patches_folder, clear_create_directory)

#%% 020: Generate embeddings for main internal scene dataset

print_section("Generate embeddings: internal dataset")

# Break scenes into patches using grid
x_grid, y_grid = load_grid('../data/grid/1024x768_128px_73over.npz')
get_patches_and_ratings(x_grid, y_grid, 128)

# Define where patches are estored and where embeddings should be saved
image_dir = "../data/patches/"
save_dir = "../data/embeddings/internal/"

# Compute internal CoCa embeddings once and save to file
get_coca_embeddings(image_dir,save_dir)
get_dino_embeddings(image_dir,save_dir)
get_mae_embeddings(image_dir, save_dir)

#%% 030: Generate embeddings for CAT external scene dataset
# Indoor/outdoor scenes share names,
# create separate indoor and outdoor embeddings

print_section("Generate embeddings: CAT dataset")

# Processing create_meaning_maps workspace (input, output, temp)
input_dir = "../data/create_meaning_maps/input/"
output_dir = "../data/create_meaning_maps/output/"
patch_dir = "../data/create_meaning_maps/patch_temp/"
save_dir = "../data/embeddings/CAT/"

# Load grid and parameters for CAT scenes
x_grid, y_grid = load_grid('../data/grid/1920x1080_128px_73over.npz') # CAT
img_w = 1920
img_h = 1080
patch_size = 128

# Copy CAT indoor scenes to create_meaning_maps input and get list of scenes
source = '../data/scenes/CAT/indoor'
dest = '../data/create_meaning_maps/input/indoor'
shutil.copytree(source, dest, dirs_exist_ok=True)
indoor_scenes = os.listdir(input_dir + 'indoor')

# Generate patches for indoor input
print_section('Computing CAT indoor patches')
create_patches_folder(input_dir + 'indoor/', x_grid, y_grid, patch_size)

# Compute CAT indoor CoCa embeddings once and save to file
get_coca_embeddings(patch_dir, save_dir+'indoor/')

# Clear patch_temp directory
shutil.rmtree(patch_dir); os.mkdir(patch_dir)

# Copy CAT outdoor scenes to create_meaning_maps input and get list of scenes
source = '../data/scenes/CAT/outdoor'
dest = '../data/create_meaning_maps/input/outdoor'
shutil.copytree(source, dest, dirs_exist_ok=True)
outdoor_scenes = os.listdir(input_dir + 'outdoor')

# Generate patches for outdoor input
print_section('Computing CAT outdoor patches')
create_patches_folder(input_dir + 'outdoor/', x_grid, y_grid, patch_size)

# Compute CAT outdoor CoCa embeddings once and save to file
get_coca_embeddings(patch_dir, save_dir+'outdoor/')

#%% 040: Generate embeddings for diffeomorph scene dataset

print_section("Generate embeddings: diffeomorph dataset") 
clear_create_directory()

# Load patch grid
x_grid, y_grid = load_grid('../data/grid/1024x768_128px_73over.npz') 
img_w = 1024
img_h = 768
patch_size = 128

# Break diffeomorph scenes into patches
print('Breaking diffeomorph scenes into patches:')
diffeo_dir = '../data/scenes/diffeomorph/'
create_patches_folder(diffeo_dir, x_grid, y_grid, patch_size)

# Compute patch embeddings
print('Computing embeddings of diffeomorph patches:')
patch_dir = '../data/create_meaning_maps/patch_temp/'
out_dir = '../data/embeddings/diffeomorph/'
get_coca_embeddings(patch_dir, out_dir)