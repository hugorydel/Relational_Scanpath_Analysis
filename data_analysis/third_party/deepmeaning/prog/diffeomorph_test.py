#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: diffeomorph_test.py
Author: T.R. Hayes
Version: 1.0.0
Description: Performs comparison of how well human meaning ratings and 
             DeepMeaning account for changes in local semantic content by 
             comparing original scene region to diffeomorphed scene region.

# Human meaning map data from:
# Hayes, T.R., and Henderson, J.M. (2022) Meaning maps detect the removal of 
# local semantic scene content but deep saliency models do not. 
# Atten Percept Psychophys 84, 647–654.
# https://doi.org/10.3758/s13414-021-02395-x

Changelog:
- 1.0.0 (2024-12-20): TRH Wrote it
"""
#%% 010: Import packages

import os
import open_clip
import shutil
import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
from sklearn.linear_model import LinearRegression
from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import torch
import re
from utils import (smooth_average_map, save_variables, load_grid, 
                   load_embeddings, compute_scene_maps,
                   print_section, print_heading)

print_heading('Perform diffeomorph test for human and DeepMeaning data')

#%% 020: Define relevant directories

human_orig_dir = '../data/human_meaning/diffeomorph_test/original/'
human_diffeo_dir = '../data/human_meaning/diffeomorph_test/diffeomorph/'
deep_orig_dir = '../data/DeepMeaning/diffeomorph_test/original/'
deep_diffeo_dir = '../data/DeepMeaning/diffeomorph_test/diffeomorph'
os.makedirs(deep_diffeo_dir, exist_ok=True)

#%% 030: Define diffeomorph critical region in each scene

# Defines circular mask region
def get_mask(h,w,x,y,r):
    z1,z2 = np.ogrid[-y:h-y,-x:w-x]
    mask = z2*z2+z1*z1 <=r*r
    return mask

# Defines square mask region
def get_square_mask(h,w,x,y,r):
    mask = np.zeros((h,w))
    mask[y-r:y+r, x-r:x+r] = 1
    mask = mask != 0
    return mask

# Define diffeomorph patch locations
patch = {}
patch["img_size"] = [768, 1024]    # resolution of the input image
patch["diameter"] = 205            # patch diameter
patch["density"] = 108             # How many patches per scene

# Build same patch grid from Hayes & Henderson (2022) AP&P
freq = np.round(np.sqrt(patch["img_size"][0] * patch["img_size"][1]) / np.sqrt(patch["density"]))
[y, x] = np.meshgrid(np.arange(freq, patch["img_size"][1], freq),np.arange(freq, patch["img_size"][0], freq))
offset_x = freq - (patch["img_size"][0] - np.max(x) + freq) / 2
offset_y = freq - (patch["img_size"][1] - np.max(y) + freq) / 2
x_diffeo = x - offset_x
y_diffeo = y - offset_y
x_diffeo = x_diffeo.flatten()
y_diffeo = y_diffeo.flatten()

# Get grid number that correspond with diffeomorph patchs
diff_patch_data = pd.read_csv('../data/grid/diffeomorph_grid.csv')

#%% 040: Human meaning original vs diffeomorphed region (Figure 4)

print_section("Compute human original and diffeomorph patch means")

# Define scenes
scenes = os.listdir(human_orig_dir)

# Preallocate
results = []
human_orig_square = np.zeros((len(scenes), 250, 250))
human_diffeo_square = np.zeros((len(scenes), 250, 250))

# For each scene
for i, scene in enumerate(scenes):
  print(f"\rProcessing: scene {i+1}/{len(scenes)}", end="", flush=True)

  # Get scene name stem
  scene_stem = scene.split('.')[0] 
  
  # Define masks
  patch_idx = diff_patch_data['patch'][diff_patch_data['scene']==scene_stem]
  x_loc = x_diffeo[patch_idx-1][0]
  y_loc = y_diffeo[patch_idx-1][0]
  mask = get_mask(768,1024,int(y_loc),int(x_loc),102)
  square_mask = get_square_mask(768,1024,int(y_loc),int(x_loc),125)
  
  # Import human original and diffeomorph meaning maps
  original = loadmat(human_orig_dir + scene)['smooth_meaning']
  diffeomorph = loadmat(human_diffeo_dir + scene)['smooth_meaning']
  
  # Compute mean original and diffeomorph patch values
  results.append({
        'scene': scene_stem,
        'mean_original': np.mean(original[mask]),
        'mean_diffeomorph':np.mean(diffeomorph[mask])})
  
  # Extract square region for plotting mean spatial difference (Figure 4c)
  human_orig_square[i,:,:] = np.reshape(original[square_mask], (250,250))
  human_diffeo_square[i,:,:] = np.reshape(diffeomorph[square_mask], (250,250))

# Convert to dataframe
human_df = pd.DataFrame(results)
  
#%% 050: DeepMeaning: Original scenes already exist just copy them over

# Original LOSOCV scenes already exist from predict_meaning.py cell 040
print('Original LOSOCV scenes already exist from predict_meaning.py')
# Simply copy them over to diffeomorph folder
scene_npz = [os.path.splitext(scene)[0] + '.npz' for scene in scenes]
source_dir = '../data/DeepMeaning/internal/coca/'
# Copy files from indoor source
for i, npz_file in enumerate(scene_npz):
  file_path = os.path.join(source_dir, npz_file)
  if os.path.exists(file_path):
    print(f"\rCopy them over to '{deep_orig_dir}': scene {i+1}/{len(scene_npz)}", end="", flush=True)
    shutil.copy2(file_path, deep_orig_dir)
scenes = os.listdir(deep_orig_dir)

#%% 060: DeepMeaning: Compute diffeo scenes DeepMeaning maps 

print_section("Compute diffeomorph scene DeepMeaning maps", 2)

# Define diffeomorph data parameters
x_grid, y_grid = load_grid('../data/grid/1024x768_128px_73over.npz') 
img_w = 1024
img_h = 768
patch_size = 128

# To compute diffeomorph scenes use LOSOCV weights that exclude that scene
indoor_model_path = '../data/DeepMeaning/DeepMeaning_indoor_ensemble.pkl'
outdoor_model_path = '../data/DeepMeaning/DeepMeaning_outdoor_ensemble.pkl'
indoor_model = pickle.load(open(indoor_model_path, 'rb'))
outdoor_model = pickle.load(open(outdoor_model_path, 'rb'))

indoor_scenes = [item['scene'] for item in indoor_model['scene_info']]
indoor_intercepts = indoor_model['all_intercepts']
indoor_weights = indoor_model['all_weights']

outdoor_scenes = [item['scene'] for item in outdoor_model['scene_info']]
outdoor_intercepts = outdoor_model['all_intercepts']
outdoor_weights = outdoor_model['all_weights']

# Load diffeomorph embeddings
embeddings_data = load_embeddings('../data/embeddings/diffeomorph/', ['coca'])

# Create a dataframe to store all predictions
diffeomorph_df = []
# For each diffeomorph scene, compute DeepMeaning predictions
for i, scene in enumerate(scenes):
    # Get scene name stem
    scene_stem = scene.split('.')[0] 
    
    # Determine category (indoor or outdoor) and get appropriate model
    if scene_stem in indoor_scenes:
        scene_idx = indoor_scenes.index(scene_stem)
        scene_model = LinearRegression()
        scene_model.coef_ = indoor_weights[scene_idx]
        scene_model.intercept_ = indoor_intercepts[scene_idx]
        category = 'indoor'
    elif scene_stem in outdoor_scenes:
        scene_idx = outdoor_scenes.index(scene_stem)
        scene_model = LinearRegression()
        scene_model.coef_ = outdoor_weights[scene_idx]
        scene_model.intercept_ = outdoor_intercepts[scene_idx]
        category = 'outdoor'
    else:
        print(f"Warning: Scene {scene_stem} not found in model lists")
        continue
    
    # Filter embeddings for this scene
    scene_embeddings = []
    scene_paths = []
    for path, idx in embeddings_data['coca']['paths'].items():
        if scene_stem in path:
            scene_embeddings.append(embeddings_data['coca']['embeddings'][idx])
            scene_paths.append(path)
    
    # If scene has embeddings, predict meaning values
    if scene_embeddings:
        predicted_meaning = scene_model.predict(np.array(scene_embeddings))
        
        # Add predictions to dataframe
        for path, prediction in zip(scene_paths, predicted_meaning):
            diffeomorph_df.append({
                'model': 'coca',
                'category': category,
                'scene': scene_stem,
                'image': path,
                'predicted': prediction
            })
    
    print(f"\rComputing DeepMeaning map for diffeomorph scenes: {i+1}/{len(scenes)}", end="", flush=True)
print()

# Convert to pandas DataFrame
diffeomorph_df = pd.DataFrame(diffeomorph_df)

# Use compute_scene_maps to generate meaning maps
compute_scene_maps(
    diffeomorph_df,
    img_w,
    img_h,
    x_grid,
    y_grid,
    patch_size,
    deep_diffeo_dir
)

#%% 070: DeepMeaning original vs diffeomorphed region (Figure 4)

print_section("Compute DeepMeaning original and diffeomorph patch means")

# Preallocate
results = []
deep_orig_square = np.zeros((len(scenes), 250, 250))
deep_diffeo_square = np.zeros((len(scenes), 250, 250))
# For each scene
for i, scene in enumerate(scenes):
  print(f"\rProcessing: scene {i+1}/{len(scenes)}", end="", flush=True)

  # Get scene name stem
  scene_stem = scene.split('.')[0] 
  
  # Define masks
  patch_idx = diff_patch_data['patch'][diff_patch_data['scene']==scene_stem]
  x_loc = x_diffeo[patch_idx-1][0]
  y_loc = y_diffeo[patch_idx-1][0]
  mask = get_mask(768,1024,int(y_loc),int(x_loc),102)
  square_mask = get_square_mask(768,1024,int(y_loc),int(x_loc),125)
  
  # Import human original and diffeomorph meaning maps
  original = np.load(deep_orig_dir + scene_stem + '.npz')['array']
  original = smooth_average_map(original)
  diffeomorph = np.load(os.path.join(deep_diffeo_dir, scene_stem + '.npz'))['array']
  diffeomorph = smooth_average_map(diffeomorph)
  
  # Compute mean original and diffeomorph patch values
  results.append({
        'scene': scene_stem,
        'mean_original': np.mean(original[mask]),
        'mean_diffeomorph':np.mean(diffeomorph[mask])})
  
  # Extract square region for plotting mean spatial difference (Figure 4c)
  deep_orig_square[i,:,:] = np.reshape(original[square_mask], (250,250))
  deep_diffeo_square[i,:,:] = np.reshape(diffeomorph[square_mask], (250,250))

# Convert to dataframe
deep_df = pd.DataFrame(results)

#%% 050: Save plot values to file (Figure 4) 

# Example coffeemaker scene diffeo patch location
patch_idx = diff_patch_data['patch'][diff_patch_data['scene']=='coffeemaker']
example_x_loc = x_diffeo[patch_idx-1][0]
example_y_loc = y_diffeo[patch_idx-1][0]

# Get human and DeepMeaning original & diffeomorph scene values
human_orig = human_df['mean_original']
human_diffeo = human_df['mean_diffeomorph']
deep_orig = deep_df['mean_original']
deep_diffeo = deep_df['mean_diffeomorph']

# Compute difference (spatial mean centered on diffeomorph region)
human_difference = np.mean((human_diffeo_square - human_orig_square), axis=0)
deep_difference = np.mean((deep_diffeo_square - deep_orig_square), axis=0)

# Save results to figures folder
save_file =  '../figures/figure4_variables.pkl'
print(f"\nSaving figure 4 variables to '{save_file}'")
save_variables(save_file,
               example_x_loc, example_y_loc,
               human_orig, human_diffeo,
               deep_orig, deep_diffeo,
               human_difference, deep_difference)

#%% 060: Report values and statistics for diffeomorph test

print_section("Compute statistics for diffeomorph test")

# DeepMeaning (original vs diffeomorph)
# Use paired samples t-test (each scene shares same critical region)
# Verify normality of differences first using Shapiro-Wilk test
shapiro_stat, shapiro_p = stats.shapiro(deep_orig - deep_diffeo) # Differences are normal
result = pg.ttest(deep_orig, deep_diffeo, paired=True)

# Format the statistics for printing
df = int(result['dof'].iloc[0])  # degrees of freedom
t_val = result['T'].iloc[0]
p_val = result['p-val'].iloc[0]
ci_lower = result['CI95%'].iloc[0][0]  # lower bound of CI
ci_upper = result['CI95%'].iloc[0][1]  # upper bound of CI
cohens_d = result['cohen-d'].iloc[0]
if p_val < .001:
    p_str = "p < .001"
else:
    p_str = f"p = {p_val:.3f}"
formatted_result = f"t({df})={t_val:.2f}, {p_str}, 95%CI [{ci_lower:.2f}, {ci_upper:.2f}], d={cohens_d:.2f}"
print('DeepMeaning (original, diffeomorph) paired samples t-test:')
print(formatted_result)


# Human Meaning (original vs diffeomorph)
# Use paired samples t-test (each scene shares same critical region)
# Verify normality of differences first using Shapiro-Wilk test
shapiro_stat, shapiro_p = stats.shapiro(human_orig - human_diffeo) # Differences are normal
result = pg.ttest(human_orig, human_diffeo, paired=True)

# Format the statistics for printing
df = int(result['dof'].iloc[0])  # degrees of freedom
t_val = result['T'].iloc[0]
p_val = result['p-val'].iloc[0]
ci_lower = result['CI95%'].iloc[0][0]  # lower bound of CI
ci_upper = result['CI95%'].iloc[0][1]  # upper bound of CI
cohens_d = result['cohen-d'].iloc[0]
if p_val < .001:
    p_str = "p < .001"
else:
    p_str = f"p = {p_val:.3f}"
formatted_result = f"t({df})={t_val:.2f}, {p_str}, 95%CI [{ci_lower:.2f}, {ci_upper:.2f}], d={cohens_d:.2f}"
print('\nHuman meaning (original, diffeomorph) paired samples t-test:')
print(formatted_result)

#%% 070: Proof-of-concept: Define original and diffeomorph patch files & plot

print_section("Save png files of original and diffeomorph patches")

# Define patch dirs 
original_dir = '../data/scenes/proof_of_concept/original/'
original_files = os.listdir(original_dir)
diffeo_dir = '../data/scenes/proof_of_concept/diffeomorph/'
diffeo_files = os.listdir(diffeo_dir)

# Plot patches as sanity-check
# Original
plt.figure(figsize=(10,6))
for i,x in enumerate(original_files):
    im = Image.open(original_dir + x).convert("RGB")
    plt.subplot(5,8,i+1)
    plt.imshow(im)
    plt.axis('off')
    plt.text(0, -5, str(i+1), fontsize = 12)
plt.savefig('../figures/original_40patches.png', bbox_inches='tight', dpi=300)
plt.close()
print('Saved figure: ../figures/original_40patches.png')
# Diffeomorph
plt.figure(figsize=(10,6))
for i,x in enumerate(diffeo_files):
    im = Image.open(diffeo_dir + x).convert("RGB")
    plt.subplot(5,8,i+1)
    plt.imshow(im)
    plt.axis('off')
    plt.text(0, -5, str(i+1), fontsize = 12)
plt.savefig('../figures/diffeomorph_40patches.png', bbox_inches='tight', dpi=300) 
plt.close()
print('Saved figure: ../figures/diffeomorph_40patches.png')   

#%% 080: Proof-of-concept: Generate captions for original and diffeomorph patches

print_section("Generate captions for original and diffeomorph patches")

output_path = os.path.join('../data/DeepMeaning/original_diffeomorph_captions.csv')

# If original & diffeomorph captions dataframe already exists load it
if os.path.exists(output_path):
    print(f"Loading existing captions dataframe from {output_path}")
    caption_df = pd.read_csv(output_path)
else:
    # Load CoCa model and tokenizer
    model, _, transform = open_clip.create_model_and_transforms(
      model_name="coca_ViT-L-14",
      pretrained="mscoco_finetuned_laion2b_s13b_b90k")
    tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')

    # Original patches
    original_caption = []
    for i,x in enumerate(original_files):
        print(f"\rGenerating captions for original patches: patch {i+1}/{len(original_files)}", end="", flush=True)
        im = Image.open(original_dir + x).convert("RGB")
        im = transform(im).unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            generated = model.generate(im, generation_type='top_p',
                                       top_p=0.05, temperature=1.5, 
                                       repetition_penalty=2.0)
        original_caption.append(re.split('\.|<end_of_text>',open_clip.decode(generated[0]))[0].replace("<start_of_text>",""))
    print()
    
    # Diffeomorph patches
    diffeo_caption = []
    for i,x in enumerate(diffeo_files):
        print(f"\rGenerating captions for diffeomorph patches: patch {i+1}/{len(original_files)}", end="", flush=True)
        im = Image.open(diffeo_dir + x).convert("RGB")
        im = transform(im).unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            generated = model.generate(im, generation_type='top_p',
                                       top_p=0.05, temperature=1.5, 
                                       repetition_penalty=2.0)
        diffeo_caption.append(re.split('\.|-|<end_of_text>',open_clip.decode(generated[0]))[0].replace("<start_of_text>",""))
    
    # Save captions as csv file for dataframe 
    print(f"\nSaving original and diffeomorph captions to {output_path}")
    caption_df = pd.DataFrame({'original_caption':original_caption,'diffeo_caption':diffeo_caption})
    caption_df.to_csv(output_path)
  
#%% 120: Caption comparison as tool for explaining drop in meaning

print_section("Original & Diffeomorph Caption Comparison Results")

# Compare if original vs diffeomorph caption describes content in original patch
# Performed by simple visual comparison between caption and original patch by authors
# Simple measure of loss of semantic content caused by diffeomorph
# Also a measure of how successful diffeomorph is a degrading semantic content
# 0=not accurate 1=accurate
original_original = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
                     1, 0, 1, 1, 1, 1, 1, 1, 1, 1]   
print(f"Original captions accurately describe original patch in: {sum(original_original)}/{len(caption_df)}")  
# Original captions succeeded on 37/40 to describe original patch content
# Fails bust.png    : A close up of some wooden benches and tables
#       duck.png    : A shelf with many bottles of bear on it
#       computer.png: A kitchen with an electric mixer and some cups

original_diffeomorph = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print(f"Diffeomorph captions accurately describe original patch in: {sum(original_diffeomorph)}/{len(caption_df)}")  

# Diffeomorph captions failed on 37/40 to describe original patch content 
# its 3 passes were correct in a very general sense:
# Passes clock.png       : Circular image of the outside world
#        fire_hydrant.png: Circular image of the outside world
#        cap.png         : Circular image of grass and water

# Compare number of semantic objects identified in original vs diffeomorph captions
# Performed by simply counting objects in each caption
# How we defined an object as: a physical, tangible entity, that has specific physical boundaries/spatial dimensions, can be described, measured, and interacted with
# So a 'room' is an object (has physical boundaries, spatial dimensions, can be interacted with)
# A 'swirl of blue' is not an object, it is an abstract visual concept/pattern
# 'the outside world' is not an object, it is a conceptual type as it does not have specific physical boundaries
# Underspecified objects count if they are acted upon (e.g., something counts in the sentence,
# 'a little girl sitting on the ground eating something')
# Repeat items count if they are distinct from one another spatially
# (e.g., a lawn with two chairs and one chair in the background)
# Abstract meta-references to the image itself were not counted as objects
# (e.g., a circular image of red and purple colors)

# Original caption objects counted
original_objects = [['computer monitor', 'headphones'],
                    ['hats', 'wall'],
                    ['garden gnome', 'plants'],
                    ['globe', 'bookshelves'],
                    ['street', 'cars'],
                    ['box', 'boxes'],
                    ['computer mouse', 'office desk'],
                    ['room', 'clutter', 'bottles', 'dresser'],
                    ['girl', 'ground', 'something'],
                    ['computer monitor', 'screen'],
                    ['benches', 'tables'],
                    ['stereo', 'speakers', 'wires', 'floor'],
                    ['room', 'dvds', 'cds'],
                    ['trees', 'forest'],
                    ['bowl', 'brushes', 'tools', 'counter'],
                    ['knife', 'grapes', 'plate'],
                    ['room', 'books', 'papers'],
                    ['industrial setting', 'tables', 'chairs'],
                    ['room', 'television', 'candles'],
                    ['boat', 'water', 'boats'],
                    ['shelf', 'jars', 'food'],
                    ['counter top', 'plants', 'other items'],
                    ['chair', 'couch'],
                    ['side yard', 'trees'],
                    ['table', 'plates', 'silverware'],
                    ['woman', 'blood', 'shirt', 'pants'],
                    ['lawn', 'chairs', 'chair'],
                    ['shelf', 'bottles', 'beer'],
                    ['motorcycle', 'assembly line'],
                    ['wall', 'wrenches'],
                    ['room', 'flags', 'wall'],
                    ['kitchen', 'electric mixer', 'cups'],
                    ['chest', 'metal handles'],
                    ['chairs'],
                    ['wall', 'desk'],
                    ['chair'],
                    ['chair', 'chairs', 'building'],
                    ['kitchen', 'cabinets', 'applicances'],
                    ['computer', 'table'],
                    ['wall', 'shelves', 'office']]
original_obj_count =  sum(len(sublist) for sublist in original_objects)
print(f"\nOriginal captions contain {original_obj_count} objects")  


# Diffeomorph caption objects counted
diffeomorph_objects = [['earth'],
                       [],
                       [],
                       [],
                       [],
                       [],
                       [],
                       [],
                       ['people', 'water'],
                       [],
                       [],
                       ['water'],
                       ['liquid', 'it'],
                       [],
                       [],
                       [],
                       [],
                       [],
                       [],
                       ['water'],
                       [],
                       ['vases', 'table'],
                       [],
                       [],
                       [],
                       [],
                       ['grass', 'water'],
                       ['water'],
                       [],
                       [],
                       ['clouds'],
                       ['water', 'sand'],
                       ['flower'],
                       [],
                       [],
                       [],
                       [],
                       [],
                       [],
                       []]
diffeo_obj_count = sum(len(sublist) for sublist in diffeomorph_objects)
print(f"Diffeomorph captions contain {diffeo_obj_count} objects")  

# Proportion drop in caption objects  
prc_decrease = (original_obj_count - diffeo_obj_count)/original_obj_count*100
print(f"Percent decrease (original - diffeomorph) caption objects: {prc_decrease:.2f} decrease")