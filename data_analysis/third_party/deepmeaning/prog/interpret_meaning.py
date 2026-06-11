#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: interpret_meaning.py
Author: T.R. Hayes
Version: 1.0.0
Description: Script that runs DeepMeaning interpretation analyses from 
             DeepMeaning: Estimating and interpreting scene meaning for 
             attention using a vision-language transformer 
             (Hayes & Henderson, 2025).

Changelog:
- 1.0.0 (2024-01-02): TRH Wrote it
"""
#%% 010: Import packages

import os
import pickle
import pandas as pd
import numpy as np
import open_clip
import re
from utils import (load_embeddings, load_ratings, load_grid, 
                   prepare_scene_data, visualize_projection_analysis, 
                   analyze_semantic_prediction, print_semantic_results,
                   print_section, print_heading)

print_heading('Perform semantic prompt interpretation analysis')

#%% 020: Define semantic criteria language prompts

# Short contrastive semantic prompts that target specific semantic dimensions
prompt_pairs = [
    ("An image with many distinct objects and items clearly visible",
      "An image showing empty space with no objects or items"),
    ("Manipulable objects that invite human interaction through their affordances, such as tools, utensils, handles, buttons, levers, furniture, appliances, and devices", 
     "Large static objects that people don't directly interact with like walls, ceilings, or distant natural features like the sky"),
    ("A close-up view of prominent objects in the foreground that appear large and detailed",
     "A distant view of background elements that appear small and far away"),
    ("Objects shown in their typical setting where you would expect to find them",
     "Objects appearing in unexpected places you would rarely find them")
]

prompt_ids = ['Object Density', 
              'Interaction',
              'Foreground/Background', 
              'Context/Natural Setting']

save_ids = ['Objects', 
            'Interaction',
            'Foreground_Background', 
            'Natural Setting']

prompt_ids = ['Object Density', 'Interaction Potential', 'Figure-Ground', 'Local Context']
save_ids = ['Objects', 'Interaction', 'Foreground', 'Context']
letter_list = [['a','b','c'], ['d','e','f'], ['g','h','i'], ['j','k','l']]

#%% 030: Define relative directories

image_patch_dir = '../data/patches/'
embedding_dir = '../data/embeddings/internal/'
output_dir = '../data/interpretation/'
models_dir = '../data/DeepMeaning/'

#%% 040: Define CoCa embedding model and tokenizer

print_section("Load CoCa OpenClip model and tokenizer")

# Create the model and transforms
coca_model, _, transform = open_clip.create_model_and_transforms(
    'coca_ViT-L-14',
    pretrained='mscoco_finetuned_laion2b_s13b_b90k'
)
print('Model loaded')

# Get the tokenizer for the model
coca_tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')
print('Tokenizer loaded')

#%% 050: Load all coca image embeddings and path mapping

print_section('Prepare for semantic prompt analysis')

# Load coca image embeddings and image paths
embeddings_data = load_embeddings(embedding_dir)
model_data = embeddings_data['coca']
embeddings = model_data['embeddings']
path_mapping = model_data['paths']

#%% 060: Prepare data for Indoor scene semantic contrastive prompt analysis

# Load ratings data for indoor scenes
ratings_data = load_ratings('../data/human_meaning/mean_patch_rating.csv')
ratings_df = ratings_data['indoor']['ratings']

# Align indoor image embeddings and ratings data
scenes, coca_image_embeddings, meaning_ratings, scene_name, image_name = prepare_scene_data(
    embeddings, path_mapping, ratings_df)
# scenes:                list of scenes
# coca_image_embeddings: numpy array of size (N, 768) = (images, coca features)
# meaning_ratings:       numpy array of size (N,) = N image rating values
# scene_name:            numpy array of str (N,) = N scene name
# image_name:            numpy array of str (N,) = N image patch name

# Define image path
full_image_paths = [f"{image_patch_dir}{path}" for path in image_name.tolist()]

# Load DeepMeaning linear ensemble weights
# Weights trained to predict human meaning ratings from CoCa image embeddings
DM_path = os.path.join(models_dir, 'DeepMeaning_indoor_ensemble.pkl')
fold_models = pickle.load(open(DM_path, 'rb'))
fold_scenes = [item['scene'] for item in fold_models['scene_info']]

# Define patch map for getting patch attention values
image_width = 1024
image_height = 768
patch_size = 128
x_grid, y_grid = load_grid('../data/grid/1024x768_128px_73over.npz')
patch_map = np.zeros((image_height, image_width,
                      len(x_grid)*len(y_grid)), dtype=np.bool_)
count = 0
for y in y_grid:
  for x in x_grid:
    patch_map[y:y+patch_size, x:x+patch_size, count] = True
    count += 1

# For the 49 indoor scenes with attention maps in the set of 139 indoor scenes
# compute attention values for each patch
eye_dir = '../data/attention/internal/indoor/'
attention_scenes = os.listdir(eye_dir)
attention_stem = [os.path.splitext(file)[0] for file in attention_scenes]

# Initialize array for attention values
attention = np.full(coca_image_embeddings.shape[0], np.nan)

# Track scenes that have attention data for better progress reporting
scenes_with_attention = [scene for scene in fold_scenes if scene in attention_stem]
n_scenes_with_attention = len(scenes_with_attention)

# Process each scene
for i, scene in enumerate(fold_scenes):
    if scene in attention_stem:
        print(f"\rComputing patch attention density: indoor scene" 
              f" {scenes_with_attention.index(scene)+1}/{n_scenes_with_attention} with attention data", 
              end="", flush=True)
        
        # Load and normalize fixation density map
        fix_density = np.load(eye_dir + scene + '.npy')
        norm_density = (fix_density - np.min(fix_density)) / (np.max(fix_density) - np.min(fix_density))
        
        # Identify patches belonging to current scene
        current_idxs = scene_name == scene
        current_patches = image_name[current_idxs]
        
        # Pre-allocate array for efficiency
        patch_attention = np.zeros(len(current_patches))
        
        # Extract patch numbers directly from filenames using more robust pattern
        # This extracts the numeric part regardless of scene name length
        for k, patch in enumerate(current_patches):
            position_id = int(re.search(r'_(\d+)\.png$', patch).group(1))
            patch_attention[k] = np.mean(norm_density[patch_map[:,:,position_id]])
        
        # Assign attention values to the correct indices
        attention[current_idxs] = patch_attention
print("\nCompleted patch attention for all available scenes.")

#%% 070: Perform semantic contrastive prompt analysis

print_section('Perform semantic prompt analysis')

# Compute semantic directions and plot results
prompt_scores = []
for i, (pos_prompt, neg_prompt) in enumerate(prompt_pairs):
  current_scores = visualize_projection_analysis(coca_model,
                                                 coca_tokenizer,
                                                 coca_image_embeddings,
                                                 attention,
                                                 scene_name,
                                                 fold_models,
                                                 meaning_ratings,                                           
                                                 full_image_paths,
                                                 pos_prompt,
                                                 neg_prompt,
                                                 prompt_ids[i],
                                                 'indoor_'+save_ids[i],
                                                 letter_list[i],
                                                 'Blues',
                                                 output_dir,
                                                 6)
  prompt_scores.append(current_scores)

# Compute correlations between semantic prompt scores for supplement
print('Pairwise R-squared between semantic dimension scores: Supplement')
r2_df = pd.DataFrame(index=prompt_ids, columns=prompt_ids)
# Compute pairwise R-squared values
for i, id1 in enumerate(prompt_ids):
    for j, id2 in enumerate(prompt_ids):
        # For the diagonal (comparing with itself), R² will always be 1.0
        if i == j:
            r2_df.loc[id1, id2] = 1.0
        else:
            # Calculate R² between the two arrays
            r2 = np.corrcoef(prompt_scores[i], prompt_scores[j])[0,1]**2
            r2_df.loc[id1, id2] = r2
formatted_df = r2_df.map(lambda x: f"{x:.3f}")
print(formatted_df)

# Perform semantic ensemble analysis using dominance analysis to estimate the
# unique contribution of each semantic feature defined by each prompt
indoor_results = analyze_semantic_prediction(coca_model,
                                             coca_tokenizer,
                                             coca_image_embeddings,
                                             attention,
                                             scene_name,
                                             fold_models,
                                             meaning_ratings,
                                             prompt_pairs)

# Print results      
print_semantic_results(indoor_results, prompt_pairs, prompt_ids)                
