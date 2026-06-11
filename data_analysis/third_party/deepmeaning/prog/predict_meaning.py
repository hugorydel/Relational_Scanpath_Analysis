#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: predict_meaning.py
Author: T.R. Hayes
Version: 1.0.0
Description: Performs LOOCV prediction (patch- and scene-level) for CoCa, DINO,
             and MAE. Saves best performing linear model weights (DeepMeaning)
             for indoor and outdoor scenes. 

Changelog:
- 1.0.0 (2024-11-20): TRH Wrote it
"""
#%% 010: Import packages

import numpy as np
import os
import pandas as pd
import pickle
import pingouin as pg
from scipy import stats
from utils import (print_ttest, load_grid, load_embeddings, load_ratings, 
                   prepare_scene_data, evaluate_model, print_section, 
                   compute_scene_maps, smooth_average_map, save_variables,
                   print_heading)

print_heading('Compute LOOCV meaning prediction (patch- and scene-level)')

#%% 020: Import grid, model embeddings and average meaning rating patch data

image_width = 1024
image_height = 768
patch_size = 128

print_section("loading stored variables")
x_grid, y_grid = load_grid('../data/grid/1024x768_128px_73over.npz')
embeddings_data = load_embeddings('../data/embeddings/internal/')
meaning_data = load_ratings('../data/human_meaning/mean_patch_rating.csv')

#%% 030: Leave-one(scene)-out cross validation on internal scenes: patch-level
# Performs leave-one-scene-out cross validation on interal dataset.
# In addition to saving patch-level results, on each fold the linear model
# weights are saved to create an LOOCV ensemble model (which is the mean linear
# weights across leave-one-out sets). The LOOCV ensemble are the weights used
# to predict new, scenes outside training set (e.g., CAT and user scenes).

output_path = os.path.join('../data/DeepMeaning/LOSOCV_internal_patch-level.csv')
models_dir = '../data/DeepMeaning/'

if os.path.exists(output_path):
    print(f"Loading existing LOSOCV patch-level dataframe from {output_path}")
    patch_df = pd.read_csv(output_path)
else:
  all_results = []
  model_results = {}
  for scene_category in ['indoor', 'outdoor']:
      print_section(f"{scene_category} patch-level processing")

      ratings_df = meaning_data[scene_category]['ratings']
      model_results[scene_category] = {}
      
      for model_name in ['coca','dino','mae']:
          print(f"Evaluating {model_name.upper()} embeddings...")
          model_data = embeddings_data[model_name]
          embeddings = model_data['embeddings']
          path_mapping = model_data['paths']
          
          unique_scenes, X, y, sample_scenes, sample_images = prepare_scene_data(
              embeddings, path_mapping, ratings_df
          )
          print(f"Number of unique {scene_category} scenes: {len(unique_scenes)}")
          print(f"Number of unique patches: {len(y)}")
          
          # Get both predictions and model info
          results_df, model_info = evaluate_model(
              X, y, unique_scenes, sample_scenes, 
              sample_images, model_name, scene_category
          )
          
          all_results.append(results_df)
          model_results[scene_category][model_name] = model_info
          
          # Save model info
          if model_name=='coca':  # coca_embeddings are best, use for DeepMeaning weights
            model_path = os.path.join(models_dir, f'DeepMeaning_{scene_category}_ensemble.pkl')
          else:
            model_path = os.path.join(models_dir, f'{model_name}_{scene_category}_ensemble.pkl')
          with open(model_path, 'wb') as f:
              pickle.dump(model_info, f)
          print(f"{scene_category.capitalize()} model saved to {model_path}\n")
  
  # Save predictions
  patch_df = pd.concat(all_results, ignore_index=True)
  patch_df.to_csv(output_path, index=False)
  print(f"\nPatch-level results saved to {output_path}")

# Print summary statistics
print("Summary: Patch-level embeddings comparison")
for category in ['indoor', 'outdoor']:
    for model in ['coca','dino','mae']:
        mask = (patch_df['model'] == model) & (patch_df['category'] == category)
        corr = np.corrcoef(
            patch_df.loc[mask, 'rating'],
            patch_df.loc[mask, 'predicted']
        )[0, 1]
        print(f"{category.capitalize()} - {model.upper()}: Patch Rcv (human rating, DeepMeaning rating) = {corr:.3f}")

#%% 040: Reconstruct scene-maps from patch-level data

print_section('Constructing scene maps from patch-level data')
compute_scene_maps(patch_df, image_width, image_height,
                   x_grid, y_grid, patch_size)

#%% 050: Leave-one(scene)-out cross validation on internal scenes: scene-level

print_section('Compute Scene-level LOOCV')
output_path = os.path.join('../data/DeepMeaning/LOSOCV_internal_scene-level.csv')

if os.path.exists(output_path):
  print(f"Loading existing LOSOCV scene-level dataframe from {output_path}")
  scene_df = pd.read_csv(output_path)
else:
  # Compute correlation between human meaning maps and model meaning maps
  models = patch_df['model'].unique() 
  scenes =  patch_df[['scene', 'category']].drop_duplicates(subset='scene').values
  model_map_dir = '../data/DeepMeaning/internal/'
  human_map_dir = '../data/human_meaning/average_rating_maps/'
  scene_results = []
  for i, model in enumerate(models):
    print(f"Model ({i+1}/{len(models)}): {model} embeddings", flush=True)
    for k, (scene, category) in enumerate(scenes):
      print(f"\rComputing Rcv scene-level: {k+1}/{len(scenes)}", end="", flush=True)
      model_meaning = np.load(model_map_dir + model + '/' + scene + '.npz')['array']
      model_meaning = smooth_average_map(model_meaning)
      human_meaning = np.load(human_map_dir + scene + '.npz')['array']
      human_meaning = smooth_average_map(human_meaning)
      scene_R = np.corrcoef(model_meaning.ravel(), human_meaning.ravel())[0,1]
      scene_results.append({
          'model': model,
          'category': category,
          'scene': scene,
          'R': scene_R
      })
    print()  # New line after completion
  scene_df = pd.DataFrame(scene_results)
  scene_df.to_csv(output_path, index=False)
  print(f"\nScene-level results saved to {output_path}\n")

# Print model summary for CoCa, DINO, and MAE
print("Scene-level Summary:")
for category in ['indoor', 'outdoor']:
    for model in ['coca', 'dino', 'mae']:
        mask = (scene_df['model'] == model) & (scene_df['category'] == category)
        mean_R = np.mean(scene_df.loc[mask, 'R'])
        std_R = np.std(scene_df.loc[mask, 'R'])
        print(f"{category.capitalize()} - {model.upper()}: mean_Rcv = {mean_R:.3f}, mean_SD = {std_R:.3f}")

#%% 060: Report patch-level R, scene-level R and CIs, indoor vs outdoor scenes

print_section("DeepMeaning patch- and scene-level meaning results")

# Report patch-level correlation
for category in ['indoor','outdoor']:
  mask = (patch_df['model'] == 'coca') & (patch_df['category'] == category)
  R = np.corrcoef(patch_df.loc[mask, 'rating'],
                        patch_df.loc[mask, 'predicted']
                        )[0, 1]
  print(f"Patch-level {category} Rcv: {R:.2f}")

# Report scene-level mean correlation and 95% CI (bootstrap)
for category in ['indoor','outdoor']:
  mask = (scene_df['model'] == 'coca') & (scene_df['category'] == category)
  values = scene_df.loc[mask, 'R']
  mean_r = np.mean(values)
  CI95 = pg.compute_bootci(values, func='mean', n_boot=10000, confidence=.95)
  print(f"Scene-level Mean {category} Rcv: {mean_r:.2f}, bootstrap 95% CI ({CI95[0]:.2f}, {CI95[1]:.2f})")

# Report simple test to show difference between indoor and outdoor performance
# Indoor scenes are better predicted than outdoor scenes by DeepMeaning just like human raters
R_map_df = scene_df[scene_df['model']=='coca']
R_map_indoor = R_map_df[R_map_df['category']=='indoor']['R']
R_map_outdoor = R_map_df[R_map_df['category']=='outdoor']['R']

# Perform Welch's t-test (because variances are not equal, outdoor var is higher)
t_statistic, p_value = stats.ttest_ind(R_map_indoor, R_map_outdoor, equal_var=False)

# Calculate degrees of freedom
n1, n2 = len(R_map_indoor), len(R_map_outdoor)
v1, v2 = np.var(R_map_indoor, ddof=1), np.var(R_map_outdoor, ddof=1)
df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))

# Calculate standard error
se = np.sqrt(v1/n1 + v2/n2)

# Calculate mean difference
mean_diff = np.mean(R_map_indoor) - np.mean(R_map_outdoor)

# Calculate confidence interval
CI95 = stats.t.interval(confidence=0.95, df=df, loc=mean_diff, scale=se)
print("Welch's t-test showing difference between indoor and outdoor scenes:")
print(f"t-welch({df:.2f})={t_statistic:.2f}, p<.001, 95% CI ({CI95[0]:.2f}, {CI95[1]:.2f})")

#%% 070: Compare DeepMeaning indoor ensemble weights to outdoor ensemble weights
# Evidence that it is a good idea to model the mapping between CoCa features
# and patch ratings separately for indoor and outdoor scenes

print_section('Compare indoor and outdoor ensemble weights')

# Load models
indoor_model_path = '../data/DeepMeaning/DeepMeaning_indoor_ensemble.pkl'
indoor_model = pickle.load(open(indoor_model_path, 'rb'))['model']
outdoor_model_path = '../data/DeepMeaning/DeepMeaning_outdoor_ensemble.pkl'
outdoor_model = pickle.load(open(outdoor_model_path, 'rb'))['model']

# Extract weight vectors (768 numpy array)
indoor_weights = indoor_model.coef_
outdoor_weights = outdoor_model.coef_

# Compute R and R-Squared between indoor and outdoor weight vectors
correlation = np.corrcoef(indoor_weights, outdoor_weights)[0, 1]
R_squared = correlation**2
print(f'Indoor and outdoor weights shared variance is R-squared={R_squared:.2f}')
# The low amount of shared variance shows that different CoCa features are 
# important for predicting meaning in indoor vs outdoor scenes

# Test of absolute differences (tests if weight differences are greater than zero)
# Use two-sided test to get full CI estimate
abs_differences = np.abs(indoor_weights - outdoor_weights)
stats = pg.ttest(abs_differences, 0, alternative='two-sided')

# Print results
print("Paired t-test of absolute difference indoor and outdoor weights")
print_ttest(stats)

#%% 090: Store all variables needed for Figure 2 using save_variables()

# Select best model
model = 'coca'

# Import leave-one-scene-out patch-level data and get values for plots
df = pd.read_csv('../data/DeepMeaning/LOSOCV_internal_patch-level.csv')
coca_indoor = [df[(df['model'] == model) & (df['category'] == 'indoor')]['rating'],
                df[(df['model'] == model) & (df['category'] == 'indoor')]['predicted']]
coca_indoor_R = np.corrcoef(coca_indoor[0], coca_indoor[1])[0,1]
coca_outdoor = [df[(df['model'] == model) & (df['category'] == 'outdoor')]['rating'],
                df[(df['model'] == model) & (df['category'] == 'outdoor')]['predicted']]
coca_outdoor_R = np.corrcoef(coca_outdoor[0], coca_outdoor[1])[0,1]

# Import leave-one-scene-out scene-level data and get values for plots
dfS = pd.read_csv('../data/DeepMeaning/LOSOCV_internal_scene-level.csv')
indoor_R_map = dfS[(dfS['model']==model) & (dfS['category']=='indoor')]['R']
outdoor_R_map = dfS[(dfS['model']==model) & (dfS['category']=='outdoor')]['R']

# Save results to figures folder
save_variables('../figures/figure2_variables.pkl',
                coca_indoor, coca_indoor_R, 
                coca_outdoor, coca_outdoor_R, 
                indoor_R_map, outdoor_R_map)

#%% 090: Estimate noise ceiling: 2 sets of human raters rating same 40 scenes
# Human raters set 1 comes from Henderson & Hayes (2017)
# Human raters set 2 comes from Hayes & Henderson (2022)
# To estimate noise ceiling compare non-diffeomorphed regions
# Provides an estimate of the best DeepMeaning could potentially perform given
# the noise in human raters estimates of local meaning.

print_section("Estimating noise ceiling of human raters")

# Load noise ceiling dataset
D = np.load('../data/human_meaning/ceiling_data.npz')
meaning_17 = D['meaning_17']       # Henderson & Hayes (2017) meaning maps
meaning_22 = D['meaning_22']       # Hayes & Henderson (2022) meaning maps
diffeo_mask = D['diffeo_mask']     # Exclude diffeomorphed region
scenes = D['scenes']

# Compute correlation between maps excluding diffeomorphed region
R = []
for i, scene in enumerate(scenes):
  mask = diffeo_mask[:,:,i]
  meaning_17_mask = np.ma.masked_array(meaning_17[:,:,i], mask=mask)
  meaning_22_mask = np.ma.masked_array(meaning_22[:,:,i], mask=mask)
  R.append(np.ma.corrcoef(meaning_17_mask.ravel(), meaning_22_mask.ravel())[0, 1])
  
# Compute bootstrap CI since R in this case is not normally distributed
ci = pg.compute_bootci(R, func='mean', n_boot=10000, confidence=.95)
R_mean = np.mean(R)
print(f"Human meaning estimated noise ceiling with bootstrap(N=10000): R={R_mean:.2f}, 95% CI ({ci[0]:.2f}, {ci[1]:.2f})")