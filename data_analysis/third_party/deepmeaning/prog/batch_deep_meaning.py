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

# %% 010: Import packages

import argparse
import glob
import os
import pickle
import shutil

import pandas as pd
from coca import get_embeddings as get_coca_embeddings
from utils import compute_scene_maps, create_patches_folder, load_embeddings, load_grid

# %% 020: Main guard required on Windows — multiprocessing.Pool uses "spawn"
# (not "fork") which re-imports this script for each worker. Without this
# guard every worker would re-execute all the top-level code, causing the
# "bootstrapping phase" RuntimeError. On Linux (where this was developed)
# "fork" is used and the guard is unnecessary, which is why it was omitted.

if __name__ == "__main__":

    # %% 030: Define input and output directories

    input_dir = "../data/create_meaning_maps/input/"
    output_dir = "../data/create_meaning_maps/output/"
    patch_dir = "../data/create_meaning_maps/patch_temp/"

    # Ensure runtime directories exist — they are gitignored and may be absent
    # on a fresh clone. cv2.imwrite silently fails into non-existent directories.
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(output_dir + "indoor/", exist_ok=True)
    os.makedirs(output_dir + "outdoor/", exist_ok=True)

    # %% 040: Get indoor and outdoor input images

    indoor_scenes = os.listdir(input_dir + "indoor")
    outdoor_scenes = os.listdir(input_dir + "outdoor")

    # %% 050: Define patch grid to be used

    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_file", type=str)
    args = parser.parse_args()
    grid_file = args.grid_file

    if grid_file is None:
        print("ERROR: No grid file specified. Please use the --grid_file argument.")
        print(
            "Example: python batch_deep_meaning.py --grid_file 1024x768_128px_73over.npz"
        )
        import sys

        sys.exit(1)

    if grid_file == "1024x768_128px_73over.npz":
        x_grid, y_grid = load_grid("../data/grid/1024x768_128px_73over.npz")
        img_w = 1024
        img_h = 768
        patch_size = 128
    elif grid_file == "1920x1080_128px_73over.npz":
        x_grid, y_grid = load_grid("../data/grid/1920x1080_128px_73over.npz")
        img_w = 1920
        img_h = 1080
        patch_size = 128

    # %% 060: Specify indoor and outdoor linear model weights

    indoor_model_path = "../data/DeepMeaning/DeepMeaning_indoor_ensemble.pkl"
    indoor_model = pickle.load(open(indoor_model_path, "rb"))["model"]

    outdoor_model_path = "../data/DeepMeaning/DeepMeaning_outdoor_ensemble.pkl"
    outdoor_model = pickle.load(open(outdoor_model_path, "rb"))["model"]

    # %% 065: Helper — check whether all scenes for a category already have output maps
    #
    # NOTE on output location: compute_scene_maps() decides where to write
    # based on os.path.basename(output_dir). Because output_dir is passed
    # here WITH a trailing slash (e.g. ".../output/indoor/"),
    # os.path.basename() returns '' (not 'indoor'), so the 'indoor'/'outdoor'
    # special-case in compute_scene_maps is never hit. It falls through to
    # the else-branch and writes into a 'coca' subfolder:
    #     .../output/indoor/coca/{scene}.npz
    # This check must look in that same actual location.
    #
    # The CoCa embedding step itself has no skip-if-exists logic (only
    # compute_scene_maps does, per-scene). On interrupted runs this means a
    # fully-completed category would otherwise be re-embedded from scratch
    # (~70+ min) only to have compute_scene_maps immediately discard the
    # redone work. This check skips that category's embedding pass entirely.

    def _category_complete(scenes, category_output_dir, model="coca"):
        search_dir = os.path.join(category_output_dir, model)
        existing = {
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(search_dir, "*.npz"))
        }
        stems = {os.path.splitext(s)[0] for s in scenes}
        return stems.issubset(existing) and len(stems) > 0

    # %% 070: Process indoor input scenes

    if _category_complete(indoor_scenes, output_dir + "indoor/"):
        print(f"All {len(indoor_scenes)} indoor maps already exist — skipping indoor.")
    else:
        print("Processing indoor images:")
        create_patches_folder(input_dir + "indoor/", x_grid, y_grid, patch_size)

        # Compute embeddings — device='cpu' (no CUDA available)
        get_coca_embeddings(patch_dir, patch_dir, device="cpu")

        embeddings_data = load_embeddings(
            "../data/create_meaning_maps/patch_temp/", ["coca"]
        )
        predicted_meaning = indoor_model.predict(embeddings_data["coca"]["embeddings"])

        indoor_df = []
        for i, (path, prediction) in enumerate(
            zip(embeddings_data["coca"]["paths"].keys(), predicted_meaning)
        ):
            scene_name = path.split("_")[0]
            indoor_df.append(
                {
                    "model": "coca",
                    "category": "indoor",
                    "scene": scene_name,
                    "image": path,
                    "predicted": prediction,
                }
            )
        indoor_df = pd.DataFrame(indoor_df)

        compute_scene_maps(
            indoor_df, img_w, img_h, x_grid, y_grid, patch_size, output_dir + "indoor/"
        )

        shutil.rmtree(patch_dir)
        os.mkdir(patch_dir)

    # %% 080: Process outdoor input scenes

    if _category_complete(outdoor_scenes, output_dir + "outdoor/"):
        print(
            f"All {len(outdoor_scenes)} outdoor maps already exist — skipping outdoor."
        )
    else:
        print("Processing outdoor images:")
        create_patches_folder(input_dir + "outdoor/", x_grid, y_grid, patch_size)

        # Compute embeddings — device='cpu' (no CUDA available)
        get_coca_embeddings(patch_dir, patch_dir, device="cpu")

        embeddings_data = load_embeddings(
            "../data/create_meaning_maps/patch_temp/", ["coca"]
        )
        predicted_meaning = outdoor_model.predict(embeddings_data["coca"]["embeddings"])

        outdoor_df = []
        for i, (path, prediction) in enumerate(
            zip(embeddings_data["coca"]["paths"].keys(), predicted_meaning)
        ):
            scene_name = path.split("_")[0]
            outdoor_df.append(
                {
                    "model": "coca",
                    "category": "outdoor",
                    "scene": scene_name,
                    "image": path,
                    "predicted": prediction,
                }
            )
        outdoor_df = pd.DataFrame(outdoor_df)

        compute_scene_maps(
            outdoor_df,
            img_w,
            img_h,
            x_grid,
            y_grid,
            patch_size,
            output_dir + "outdoor/",
        )

        shutil.rmtree(patch_dir)
        os.mkdir(patch_dir)
