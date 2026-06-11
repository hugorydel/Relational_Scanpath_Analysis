#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: predict_attention.py
Author: T.R. Hayes
Version: 1.0.0
Description: Performs comparison of how well DeepMeaning transfers to where 
             people attend relative to human meaning maps. Analysis is 
             performed on an internal dataset and external CAT dataset.

Changelog:
- 1.0.0 (2024-11-20): TRH Wrote it
"""
#%% 010: Import packages

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import pingouin as pg
from utils import (smooth_average_map, save_variables, load_embeddings,
                   print_ttest, print_section, compute_scene_maps,
                   load_grid, print_heading)

print_heading('Compute DeepMeaning transfer to attention')

#%% 020: Define relevant directories

human_dir = '../data/human_meaning/average_rating_maps/'
deep_dir = '../data/DeepMeaning/internal/coca/'
internal_indoor = '../data/attention/internal/indoor/'
internal_outdoor = '../data/attention/internal/outdoor/'
CAT_attention_dir = '../data/attention/CAT/'
CAT_deep_dir = '../data/DeepMeaning/CAT/'

#%% 030: R DeepMeaning, HumanMeaning, & Attention: Internal dataset (Figure 3)

# Print section 
print_section("Compute human & DeepMeaning attention R & Rcv: internal")

# Combine scenes into a list of tuples with their types
scene_info = (
    [(scene, 'indoor', internal_indoor) for scene in os.listdir(internal_indoor)] + 
    [(scene, 'outdoor', internal_outdoor) for scene in os.listdir(internal_outdoor)]
)

# Initialize results list
results = []
# For each scene
for k, (scene, scene_type, scene_dir) in enumerate(scene_info):
    print(f"\rComputing R(human,attention) & Rcv(DeepMeaning,attention): scene {k+1}/{len(scene_info)}", end="", flush=True)

    human_meaning = np.load(os.path.join(human_dir, Path(scene).stem + '.npz'))['array']
    human_meaning = smooth_average_map(human_meaning)
    
    deep_meaning = np.load(os.path.join(deep_dir, Path(scene).stem + '.npz'))['array']
    deep_meaning = smooth_average_map(deep_meaning)
    
    attention = np.load(os.path.join(scene_dir, scene))
    
    results.append({
        'scene': scene,
        'category': scene_type,
        'R_meaning': np.corrcoef(human_meaning.flatten(), attention.flatten())[0,1],
        'Rcv_deep': np.corrcoef(deep_meaning.flatten(), attention.flatten())[0,1]
    })

# Create DataFrame with all results
int_df = pd.DataFrame(results)

#%% 040: Generate DeepMeaning maps for CAT dataset

# Print section 
print_section("Compute DeepMeaning maps for CAT scenes", 2)

# Define CAT parameters
x_grid, y_grid = load_grid('../data/grid/1920x1080_128px_73over.npz') # CAT
img_w = 1920
img_h = 1080
patch_size = 128

# Import DeepMeaning indoor weights
indoor_model_path = '../data/DeepMeaning/DeepMeaning_indoor_ensemble.pkl'
indoor_model = pickle.load(open(indoor_model_path, 'rb'))['model']

# Load CAT indoor embeddings
indoor_embeddings = load_embeddings('../data/embeddings/CAT/indoor/', ['coca'])

# Create indoor DataFrame from predictions
predicted_meaning = indoor_model.predict(indoor_embeddings['coca']['embeddings'])
indoor_df = []
for i, (path, prediction) in enumerate(zip(indoor_embeddings['coca']['paths'].keys(), predicted_meaning)):
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
output_dir = '../data/DeepMeaning/CAT/indoor/'
os.makedirs(output_dir, exist_ok=True)

# Compute meaning maps using compute_scene_maps
compute_scene_maps(
    indoor_df,
    img_w,
    img_h,
    x_grid,
    y_grid,
    patch_size,
    output_dir)

# Import DeepMeaning outdoor weights
outdoor_model_path = '../data/DeepMeaning/DeepMeaning_outdoor_ensemble.pkl'
outdoor_model = pickle.load(open(outdoor_model_path, 'rb'))['model']

# Load CAT outdoor embeddings
outdoor_embeddings = load_embeddings('../data/embeddings/CAT/outdoor/', ['coca'])

# Create outdoor DataFrame from predictions
predicted_meaning = outdoor_model.predict(outdoor_embeddings['coca']['embeddings'])
outdoor_df = []
for i, (path, prediction) in enumerate(zip(outdoor_embeddings['coca']['paths'].keys(), predicted_meaning)):
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

# Ensure output directory exists
output_dir = '../data/DeepMeaning/CAT/outdoor/'
os.makedirs(output_dir, exist_ok=True)

# Compute meaning maps using compute_scene_maps
compute_scene_maps(
    outdoor_df,
    img_w,
    img_h,
    x_grid,
    y_grid,
    patch_size,
    output_dir)

#%% 050: R DeepMeaning & Attention: CAT dataset (Figure 3)

# Print section 
print_section("Compute DeepMeaning attention R: CAT dataset")

# Initialize list
CAT_R = []
# For each category
for category in ['indoor','outdoor']:

  # Get all scenes in category
  curr_scenes = sorted(os.listdir(CAT_attention_dir + category))
  
  # For each scene in category
  for k, scene in enumerate(curr_scenes):
    print(f"\rComputing R(DeepMeaning,attention): {category} scene {k+1}/{len(curr_scenes)}", end="", flush=True)

    # Get scene ID
    scene_stem = scene.split('.')[0] 
    
    # Read attention map and DeepMeaning map
    attention_map = np.array(Image.open(CAT_attention_dir + category + '/' + scene))
    DeepMeaning_map = np.load(CAT_deep_dir + category + '/coca/' + scene_stem + '.npz', allow_pickle=True)['array']
    DeepMeaning_map = smooth_average_map(DeepMeaning_map)
    
    #--Store results
    curr_R = np.corrcoef(attention_map.flatten(), DeepMeaning_map.flatten())[0,1]
    CAT_R.append({
        'category': category,
        'scene': scene_stem,
        'R': curr_R})
  print()
    
# Create dataframe with all results
CAT_df = pd.DataFrame(CAT_R)

#%% 060: Save plot values to file (Figure 3) 

# Compute interal dataset plot values
indoor_R_meaning = int_df[int_df['category'] == 'indoor']['R_meaning'].values
outdoor_R_meaning = int_df[int_df['category'] == 'outdoor']['R_meaning'].values
indoor_Rcv_deep = int_df[int_df['category'] == 'indoor']['Rcv_deep'].values
outdoor_Rcv_deep = int_df[int_df['category'] == 'outdoor']['Rcv_deep'].values

# Compute CAT dataset plot values
indoor_R_CAT = CAT_df[CAT_df['category'] == 'indoor']['R'].values
outdoor_R_CAT = CAT_df[CAT_df['category'] == 'outdoor']['R'].values

# Save results to figures folder
save_file =  '../figures/figure3_variables.pkl'
print(f"Saving figure 3 variables to '{save_file}'")
save_variables(save_file,
               indoor_R_meaning, outdoor_R_meaning,
               indoor_Rcv_deep, outdoor_Rcv_deep,
               indoor_R_CAT, outdoor_R_CAT)

#%% 070: Compute descriptives and statistics for manuscript: internal dataset

print_section("Descriptives & statistics: Internal dataset")

# Report scene-level mean R and 95% CI (t-distribution)
print("Attention and human meaning and DeepMeaning means and CIs")
categories = ['indoor', 'outdoor']
metrics = ['R_meaning', 'Rcv_deep']
for category in categories:
  for metric in metrics:
    values = int_df.loc[int_df['category'] == category, metric]
    mean_r = np.mean(values)
    CI95 = pg.compute_bootci(values, func='mean', n_boot=10000, confidence=.95)
    print(f"Internal {category} {metric}: {mean_r:.2f}, bootstrap 95% CI ({CI95[0]:.2f}, {CI95[1]:.2f})")

# Test whether DeepMeaning and Human meaning predict attention equally well
print("\nDeepMeaning and human meaning predict attention equally well: indoor")
internal_indoor_stats = pg.ttest(indoor_R_meaning, indoor_Rcv_deep, paired=False)
print_ttest(internal_indoor_stats)

print("\nDeepMeaning and human meaning predict attention equally well: outdoor")
internal_outdoor_stats = pg.ttest(outdoor_R_meaning, outdoor_Rcv_deep, paired=False)
print_ttest(internal_outdoor_stats)

# Compute correlation between meaning/attention and DeepMeaning/attention
# High correlation shows human meaning and DeepMeaning predict similarly on a
# scene-by-scene basis.
scene_R = np.corrcoef(int_df['R_meaning'], int_df['Rcv_deep'])[0,1]
R_test = pg.corr(int_df['R_meaning'], int_df['Rcv_deep'])
print(f"\nCorrelation between R(attention, human) & R(attention, DeepMeaning): {scene_R:.2f}")
print(f"\nPearson R test: r{R_test['r'].iloc[0]:.2f}, "
      f"p = {R_test['p-val'].iloc[0]:.4f}, "
      f"95% CI = [{R_test['CI95%'].iloc[0][0]:.2f}, {R_test['CI95%'].iloc[0][1]:.2f}]")

#%% 080: Compute statistics for CAT replication for manuscript: CAT dataset

print_section("Descriptives & statistics: CAT")

print("CAT dataset replicates transfer of DeepMeaning to attention")
print("Indoor scenes")
CAT_indoor_stats = pg.ttest(indoor_R_CAT, len(indoor_R_CAT)-1)
print_ttest(CAT_indoor_stats)

print("\nOutdoor scenes")
CAT_outdoor_stats = pg.ttest(outdoor_R_CAT, len(outdoor_R_CAT)-1)
print_ttest(CAT_outdoor_stats)