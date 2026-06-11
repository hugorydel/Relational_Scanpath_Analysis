#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: recreate_results_from_scratch.py
Author: T.R. Hayes
Version: 1.0.0
Description: Main script that clears all directories and then reruns all
             all analyses from DeepMeaning: Estimating and interpreting scene 
             meaning for attention using a vision-language transformer 
             (Hayes & Henderson, 2025) codebase.
              
Changelog:
- 1.0.0 (2025-01-02): TRH Wrote it
"""

#%% 010: Import packages

import os
from utils import (load_grid, get_patches_and_ratings, 
                   clear_directory, clear_create_directory, print_heading)

#%% 020: Clear directories before regenerating all results

print_heading('Clear all result directories')

# Clear image patches and mean patch ratings
clear_directory('../data/patches')
print("Cleared patch directory")
mean_rating_file = '../data/human_meaning/mean_patch_rating.csv'
if os.path.exists(mean_rating_file):
    os.remove(mean_rating_file)
    print(f"Removed file: {mean_rating_file}")

# Clear DeepMeaning internal dataset results
deep_meaning_internal = '../data/DeepMeaning/internal'
for subdir in ['coca', 'dino', 'mae']:
    clear_directory(os.path.join(deep_meaning_internal, subdir))
deep_meaning_dir = '../data/DeepMeaning'
clear_directory(deep_meaning_dir, ['*.pkl', '*.csv']) 

# Clear DeepMeaning CAT dataset results
cat_dir = os.path.join(deep_meaning_dir, 'CAT')
for subdir in ['indoor', 'outdoor']:
    clear_directory(os.path.join(cat_dir, subdir))

# Clear DeepMeaning diffeomorph results
diffeomorph_dir = os.path.join(deep_meaning_dir, 'diffeomorph_test')
for subdir in ['diffeomorph', 'original']:
    clear_directory(os.path.join(diffeomorph_dir, subdir))
clear_directory(diffeomorph_dir, ['*.npy'])

# Clear create_meaning_maps folders
clear_create_directory()

# Clear figures folder
clear_directory('../figures')

#%% 030: Create patches for internal scenes

print_heading('Compute scene patches from grid')
x_grid, y_grid = load_grid('../data/grid/1024x768_128px_73over.npz')
get_patches_and_ratings(x_grid, y_grid, patch_size=128)

#%% 040: Compute patch- and scene-level maps, ensemble DeepMeaning weights

os.system("python predict_meaning.py")

#%% 050: Test DeepMeaning transfer to attention (internal and CAT datasets)

os.system("python predict_attention.py")

#%% 060: Perform diffeomorph semantic content test

os.system("python diffeomorph_test.py")

#%% 070: Perform semantic prompt map analysis

os.system("python interpret_meaning.py")

#%% 080: Generate all figures from paper

os.system("python generate_figures.py")
