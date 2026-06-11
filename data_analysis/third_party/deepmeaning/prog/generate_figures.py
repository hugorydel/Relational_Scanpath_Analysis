#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: generate_manuscript_figures.py
Author: T.R. Hayes
Version: 1.0.0
Description: This script generates the figures for DeepMeaning: Estimating and
             interpreting scene meaning for attention using a vision-language 
             transformer (Hayes & Henderson, 2025).

Changelog:
- 1.0.0 (2025-01-05): TRH Wrote it
"""
#%% 010: Import packages

from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.image as mpimg
import colorcet as cc
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from utils import (load_grid, plot_distribution, draw_brace, 
                   smooth_average_map, load_variables, print_heading)

print_heading('Generating all figures from Hayes & Henderson (2025)')

#%% 020: Define global figure properties

# Font sizes
mfs = 11
mfs2 = 10.5
sfs = 13 
pad = 0.03

# Define common plot colors
Cplot1 = (20/255, 115/255, 175/255)
Cplot2 = (25/255, 158/255, 116/255)
C3 = (127/255, 127/255, 127/255)

# Create legend elements for indoor/outdoor patches
legend_elements = [Patch(facecolor=Cplot1, edgecolor=Cplot1, label='Indoor Scenes'),
                   Patch(facecolor=Cplot2, edgecolor=Cplot2, label='Outdoor Scenes')]

#%% 020: Generate Figure 1

# Define figure properties
Lmin, Lmax = 0.25, 5.75
fig = plt.figure(figsize=(17,4))

# Setup figure grid layout and define subplot axes
gs1 = GridSpec(1, 5, width_ratios=[1.55, 0.05, 0.75, 1.25, 0.955])
gs1.update(wspace=0.07)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[2])
ax3 = fig.add_subplot(gs1[3])
ax4 = fig.add_subplot(gs1[4])
axes = [ax1, ax2, ax3, ax4]

# Panel 1: Plot scene
scene_image = Image.open('../data/scenes/internal/indoor/coffeemaker.png')
axes[0].set_aspect(1)
axes[0].imshow(scene_image)
axes[0].set_yticks([])
axes[0].set_xticks([])
axes[0].set_axisbelow(True)
axes[0].set_xlabel("Real-world Scene", fontsize=11)
axes[0].text(0, -24, "a", fontsize=sfs, weight='bold')

# Panel 2: Plot human spatial grids
axes[1].set_yticks([])
axes[1].set_xticks([])
axes[1].set_xlim(1, 6)
axes[1].set_axisbelow(True)
axes[1].set_xlabel("Human Spatial Grids", fontsize=mfs)
axes[1].set_ylabel('Coarse                        Fine', fontsize=mfs)
plt.setp(axes[1].spines.values(), visible=False)
axes[1].text(1, 1.025, "b", fontsize=sfs, weight='bold')

# Load and process spatial grid overlay
orig_patch = Image.open('../data/accessory_images/human_spatial_grids.png')
orig = fig.add_axes([0.07575, 0.105, 0.776, 0.776], zorder=1)
rgba = orig_patch.convert("RGBA")
datas = rgba.getdata()

# Convert black pixels to transparent
newData = []
for item in datas:
    if item[0] == 0 and item[1] == 0 and item[2] == 0:
        newData.append((255, 255, 255, 0))  # Transparent
    else:
        newData.append(item)  # Keep original color

rgba.putdata(newData)
orig.imshow(rgba)
orig.axis('off')

# Panel 3: Example Meaning map with DeepMeaning sampling grid overlay
meaning_image = np.load('../data/human_meaning/average_rating_maps/coffeemaker.npz')['array']
meaning_image = smooth_average_map(meaning_image)
axes[2].set_aspect(1)
meaning_plt = axes[2].imshow(meaning_image, cmap=cc.m_fire, vmin=1, vmax=5)

# Setup grid parameters
glb_overlap = 0.35
glb_split = 128
img_sz = (768, 1024)

# Generate grid points
x_glb, y_glb = load_grid('../data/grid/1024x768_128px_73over.npz')

# Draw grid overlay
for y in y_glb:
    for x in x_glb:
        rect = patches.Rectangle((x,y), glb_split, glb_split, linewidth=0.5,
                               edgecolor=(1/255,1/255,1/255), facecolor="none")
        #axes[2].plot(x + glb_split/2, y + glb_split/2, 'k.', ms=2)
        axes[2].add_patch(rect)

# Configure meaning map panel
axes[2].autoscale(tight=True)
axes[2].set_yticks([])
axes[2].set_xticks([])
axes[2].set_xlabel("Low Meaning                            High Meaning", fontsize=10)
axes[2].text(0, -27, "c", fontsize=sfs, weight='bold')
axes[2].text(15, 1093, "Human Meaning Map with Square Grid", fontsize=mfs)
fig.colorbar(meaning_plt, ax=axes[2], orientation="horizontal", pad=0.1, shrink=0.9015)

# Panel 4: Human Patch rating distributions
patch_data = pd.read_csv('../data/human_meaning/mean_patch_rating.csv')
Idata = patch_data[patch_data['category']=='indoor']
Odata = patch_data[patch_data['category']=='outdoor']

axes[3].grid(alpha=0.4)

# Plot indoor distribution
plot_distribution(Idata['meaning_rating'], ax=axes[3], y_position=1.30, 
                 color=Cplot1, 
                 density_scale=0.25, strip_width=1, marker_size=.5, marker_alpha=0.1)

# Plot outdoor distribution
plot_distribution(Odata['meaning_rating'], ax=axes[3], y_position=0.45, 
                 color=Cplot2, 
                 density_scale=0.25, strip_width=1, marker_size=.5, marker_alpha=0.1)

# Configure distribution panel
axes[3].set_yticks([])
axes[3].set_xlim(0.5, 5.5)
axes[3].set_ylim(0, 2)
axes[3].set_axisbelow(True)
axes[3].set_xlabel("Mean Human Rating (Square Grid)", fontsize=mfs)
axes[3].text(0.5, 2.05, "d", fontsize=sfs, weight='bold')
axes[3].legend(handles=legend_elements, loc='upper left', handlelength=1)

# Adjust position of last panel
pos = axes[3].get_position()
pos.bounds = (pos.bounds[0], pos.bounds[1]+0.05, pos.bounds[2], pos.bounds[3]-0.05)
axes[3].set_position(pos, which='both')

# Save Figure 1
figure1_path = '../figures/figure1.jpg'
plt.savefig(figure1_path, bbox_inches='tight', pad_inches=pad, dpi=425)
plt.close()
print(f"Figure 1 saved to: {figure1_path}")

#%% 030: Generate Figure 2

# Load variables computed in 'predict_meaning.py' cell 080
(coca_indoor, coca_indoor_R, coca_outdoor, coca_outdoor_R,
indoor_R_map, outdoor_R_map)=load_variables('../figures/figure2_variables.pkl')

# Define figure properties
Lmin = 0.25 ; Lmax = 5.75
fig = plt.figure(figsize=(17,4.71))

# Setup figure grid layout and define subplot axes
gs1 = GridSpec(1, 6, width_ratios = [1.19,.05,.94,.05,.94,.94])
gs1.update(wspace=.13)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[2])
ax3 = fig.add_subplot(gs1[4])
ax4 = fig.add_subplot(gs1[5])
axes = [ax1,ax2,ax3,ax4]

# Panel 1: DeepMeaning flowchart
model_img = Image.open('../data/accessory_images/DeepMeaning_flowchart.png')
axes[0].imshow(model_img)
axes[0].set_aspect(1)
axes[0].set_axis_off()
axes[0].text(0, -30, "a", fontsize = sfs, weight='bold')
axes[0].text(28, 1800, "Contrastive Captioner\n   Feature Extraction", fontsize = 10)
axes[0].text(825, 1800, "Leave-one-scene-out\n   Cross-validation", fontsize = 10)
pos = axes[0].get_position()
new_pos = [pos.x0, pos.y0 - 0.071, pos.width, pos.height]
axes[0].set_position(new_pos)
draw_brace(axes[0], (8,749),-1545, '')
draw_brace(axes[0], (765,1569),-1545, '')

# Panel 2: Observed human ratings by DeepMeaning predictions: Indoor patches
axes[1].scatter(coca_indoor[0], coca_indoor[1], color=Cplot1, marker='.', s=.02, alpha=0.5)
axes[1].plot([Lmin, Lmax], [Lmin, Lmax], linestyle='-', color='k', linewidth=1)
axes[1].set_aspect(1)
axes[1].set_xlim(Lmin, Lmax)
axes[1].set_ylim(Lmin, Lmax)
axes[1].grid(alpha=.4)
axes[1].set_axisbelow(True)
axes[1].set_xlabel("Observed Human Patch Rating", fontsize = mfs2)
axes[1].set_ylabel("Predicted DeepMeaning Patch Rating", fontsize = mfs2)
axes[1].text(1.5, 4.73, r"$R_{cv}$=%.2f" % coca_indoor_R, fontsize = 10)
axes[1].text(.25, 5.89, "b", fontsize = sfs, weight='bold')
axes[1].legend(handles=[Patch(facecolor=Cplot1, edgecolor=Cplot1, label='Indoor Patches')], loc='upper left', handlelength=1)

# Panel 3: Observed human ratings by DeepMeaning predictions: Outdoor patches
axes[2].scatter(coca_outdoor[0], coca_outdoor[1], color=Cplot2, marker='.', s=.02, alpha=0.5)
axes[2].plot([Lmin, Lmax], [Lmin, Lmax], linestyle='-', color='k', linewidth=1)
axes[2].set_aspect(1)
axes[2].set_xlim(Lmin, Lmax)
axes[2].set_ylim(Lmin, Lmax)
axes[2].grid(alpha=.4)
axes[2].set_axisbelow(True)
axes[2].set_xlabel("Observed Human Patch Rating", fontsize = mfs)
axes[2].set_ylabel("Predicted DeepMeaning Patch Rating", fontsize = mfs2)
axes[2].text(1.5, 4.73, r"$R_{cv}$=%.2f" % coca_outdoor_R, fontsize = 10)
axes[2].text(.25, 5.89, "c", fontsize = sfs, weight='bold')
axes[2].legend(handles=[Patch(facecolor=Cplot2, edgecolor=Cplot1, label='Outdoor Patches')], loc='upper left', handlelength=1)

# Panel 4: Distributions plots of indoor and outdoor scene Rcv(human,DeepMeaning)
axes[3].grid(alpha=.4)
plot_distribution(indoor_R_map, ax=axes[3], y_position=0.65, 
                 color=Cplot1, density_to_box_gap=0.45, box_to_strip_gap=0.45,
                 density_scale=0.09, strip_width=1.4,
                 marker_size=5, marker_alpha=.7, bandwidth=0.35)
plot_distribution(outdoor_R_map, ax=axes[3], y_position=0.25, 
                 color=Cplot2, density_to_box_gap=0.45, box_to_strip_gap=0.45,
                 density_scale=0.09, strip_width=1.4,
                 marker_size=5, marker_alpha=.7, bandwidth=0.35)
axes[3].set_aspect(1)
axes[3].set_xlim(0, 1)
axes[3].set_ylim(0, 1)
axes[3].set_xlabel("DeepMeaning & Human Meaning ($R_{cv}$)", fontsize = mfs2)
axes[3].text(0, 1.026, "d", fontsize=sfs, weight='bold')
axes[3].legend(handles=legend_elements, loc='upper left', handlelength=1)

# Save Figure 2 
figure2_path = '../figures/figure2.jpg'
plt.savefig(figure2_path, bbox_inches='tight', pad_inches=pad, dpi=425)
plt.close()
print(f"Figure 2 saved to: {figure2_path}")

#%% 040: Generate Figure 3

# Load variables computed in 'predict_attention.py' cell 060
(indoor_R_meaning, outdoor_R_meaning, indoor_Rcv_deep, outdoor_Rcv_deep,
indoor_R_CAT, outdoor_R_CAT)=load_variables('../figures/figure3_variables.pkl')

# Define figure properties
Lmin = 0.25 ; Lmax = 5.75
fig = plt.figure(figsize=(17,4))

# Setup figure grid layout and define subplot axes
gs1 = GridSpec(1, 5, width_ratios = [1,1,.08,1,1])
gs1.update(wspace=.11)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])
ax3 = fig.add_subplot(gs1[3])
ax4 = fig.add_subplot(gs1[4])
axes = [ax1,ax2,ax3,ax4]

# Panel 1: DeepMeaning & attention Rcv for indoor and outdoor scenes (internal)
axes[0].grid(alpha=.4)
plot_distribution(indoor_Rcv_deep, ax=axes[0], y_position=0.65, 
                 color=Cplot1, density_to_box_gap=0.45, box_to_strip_gap=0.45,
                 density_scale=0.09, strip_width=1.4,
                 marker_size=5, marker_alpha=.7, bandwidth=0.35)
plot_distribution(outdoor_Rcv_deep, ax=axes[0], y_position=0.25, 
                 color=Cplot2, density_to_box_gap=0.45, box_to_strip_gap=0.45,
                 density_scale=0.09, strip_width=1.4,
                 marker_size=5, marker_alpha=.7, bandwidth=0.35)
axes[0].set_yticks([])
axes[0].set_aspect(1)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].set_axisbelow(True)
axes[0].set_xlabel(r"DeepMeaning and Attention ($R_{cv}$)",  fontsize = mfs)
axes[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axes[0].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
axes[0].text(0, 1.02, "a", fontsize=sfs, weight='bold')
axes[0].legend(handles=legend_elements, loc='upper left', handlelength=1)

# Panel 2: Human meaning and attention R for indoor and outdoor scenes
axes[1].grid(alpha=.4)
plot_distribution(indoor_R_meaning, ax=axes[1], y_position=0.65, 
                 color=Cplot1, density_to_box_gap=0.45, box_to_strip_gap=0.45,
                 density_scale=0.09, strip_width=1.4,
                 marker_size=5, marker_alpha=.7, bandwidth=0.35)
plot_distribution(outdoor_R_meaning, ax=axes[1], y_position=0.25, 
                 color=Cplot2, density_to_box_gap=0.45, box_to_strip_gap=0.45,
                 density_scale=0.09, strip_width=1.4,
                 marker_size=5, marker_alpha=.7, bandwidth=0.35)
axes[1].set_yticks([])
axes[1].set_aspect(1)
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].set_axisbelow(True)
axes[1].set_aspect('equal', 'box')
axes[1].set_xlabel("Human Meaning and Attention (R)", fontsize = mfs)
axes[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axes[1].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
axes[1].text(0, 1.02, "b", fontsize=sfs, weight='bold')
axes[1].legend(handles=legend_elements, loc='upper left', handlelength=1)

# Panel 3: Scene-by-scene comparison of attention R for humans vs DeepMeaning
axes[2].scatter(indoor_R_meaning, indoor_Rcv_deep, color=Cplot1, marker='o', s=20, alpha=.5)
axes[2].scatter(outdoor_R_meaning, outdoor_Rcv_deep, color=Cplot2, marker='x', s=20, alpha=.5)
axes[2].plot([0, 1], [0, 1], linestyle='-', color='k', linewidth=1)
axes[2].set_xlim(0, 1)
axes[2].set_ylim(0, 1)
axes[2].grid(alpha=.4)
axes[2].set_axisbelow(True)
axes[2].set_aspect('equal', 'box')
axes[2].set_xlabel("Human Meaning and Attention (R)", fontsize = mfs)
axes[2].set_ylabel(r"DeepMeaning and Attention ($R_{cv}$)", fontsize = mfs)
axes[2].text(0, 1.02, "c", fontsize = sfs, weight='bold')
axes[2].legend(handles=[Line2D([0], [0], marker='o', color=Cplot1, label='Indoor Scenes', 
                        alpha=1, markersize=9,linewidth=0),
                        Line2D([0], [0], marker='x', color=Cplot2, label='Outdoor Scenes', 
                        alpha=1, markersize=7.5,linewidth=0)], handlelength=1)

# Panel 4: DeepMeaning & attention Rcv for indoor and outdoor scenes (CAT)
axes[3].grid(alpha=.4)
# Indoor
plot_distribution(indoor_R_CAT, ax=axes[3], y_position=0.65, 
                 color=Cplot1, density_to_box_gap=0.45, box_to_strip_gap=0.45,
                 density_scale=0.09, strip_width=1.4,
                 marker_size=5, marker_alpha=.7, bandwidth=0.35)
# Outdoor
plot_distribution(outdoor_R_CAT, ax=axes[3], y_position=0.25, 
                 color=Cplot2, density_to_box_gap=0.45, box_to_strip_gap=0.45,
                 density_scale=0.09, strip_width=1.4,
                 marker_size=5, marker_alpha=.7, bandwidth=0.35)

axes[3].set_yticks([])
axes[3].set_aspect(1)
axes[3].set_xlim(0, 1)
axes[3].set_ylim(0, 1)
axes[3].set_axisbelow(True)
axes[3].set_aspect('equal', 'box')
axes[3].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axes[3].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
axes[3].set_xlabel(r"CAT: DeepMeaning and Attention ($R$)", fontsize = mfs)
axes[3].text(0, 1.02, "d", fontsize = sfs, weight='bold')
axes[3].legend(handles=legend_elements, loc='upper left', handlelength=1)

# Save Figure 3 
figure3_path = '../figures/figure3.jpg'
plt.savefig(figure3_path, bbox_inches='tight', pad_inches=pad, dpi=425)
plt.close()
print(f"Figure 3 saved to: {figure3_path}")

#%% 050: Generate figure 4

# Load variables computed in 'diffeomorph_test.py'
(example_x_loc, example_y_loc, human_orig, human_diffeo,
deep_orig, deep_diffeo, human_difference, deep_difference)=load_variables('../figures/figure4_variables.pkl')

# Define figure properties
Lmin = 0.25 ; Lmax = 5.75
fig = plt.figure(figsize=(17,4))

# Setup figure grid layout and define subplot axes
gs1 = GridSpec(1, 7, width_ratios = [1.55,.05,1.155,.05,.5,.05,1])
gs1.update(wspace=.13)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[2])
ax3 = fig.add_subplot(gs1[4])
ax4 = fig.add_subplot(gs1[6])
axes = [ax1,ax2,ax3,ax4]

# Panel 1: Plot diffeomorph example scene
diffeo_scene_image = Image.open('../data/scenes/diffeomorph/coffeemaker.png')
axes[0].set_aspect(1)
axes[0].imshow(diffeo_scene_image)
axes[0].set_yticks([])
axes[0].set_xticks([])
axes[0].set_axisbelow(True)
axes[0].set_xlabel("Diffeomorph Scene", fontsize=mfs, labelpad=7)
axes[0].text(0, -27, "a", fontsize = sfs, weight='bold')
circ = Circle((example_y_loc,example_x_loc),102,edgecolor='lime',fill=False, linewidth=2)
axes[0].add_patch(circ)

# Panel 2: Plot Original vs Diffeomorph for Human meaning and DeepMeaning
axes[1].grid(alpha=.4)
# Human original 
plot_distribution(human_orig, ax=axes[1], y_position=.95, 
                 color='mediumpurple', density_to_box_gap=0.3, box_to_strip_gap=0.3,
                 density_scale=0.05, strip_width=.4,
                 marker_size=5, marker_alpha=.7, bandwidth=0.35)
# DeepMeaning original 
plot_distribution(deep_orig, ax=axes[1], y_position=.82, 
                 color=C3, density_to_box_gap=0.3, box_to_strip_gap=0.3,
                 density_scale=0.05, strip_width=.4,
                 marker_size=5, marker_alpha=.7, bandwidth=0.35)
# Human diffeomorph 
plot_distribution(human_diffeo, ax=axes[1], y_position=.65, 
                 color='mediumpurple', density_to_box_gap=0.3, box_to_strip_gap=0.3,
                 density_scale=0.05, strip_width=.4,
                 marker_size=5, marker_alpha=.7, bandwidth=0.35)
# DeepMeaning diffeomorph 
plot_distribution(deep_diffeo, ax=axes[1], y_position=.52, 
                 color=C3, density_to_box_gap=0.3, box_to_strip_gap=0.3,
                 density_scale=0.05, strip_width=.4,
                 marker_size=5, marker_alpha=.7, bandwidth=0.35)
axes[1].set_yticks([])
axes[1].set_xlim(1.5, 5.5)
axes[1].set_axisbelow(True)
axes[1].set_xlabel("Mean Patch Rating", fontsize = mfs)
axes[1].text(1.5,1.045, "b", fontsize = sfs, weight='bold')
legend_elements = [Patch(facecolor='mediumpurple', edgecolor='mediumpurple', label='Human'),
                   Patch(facecolor=C3, edgecolor=C3, label='DeepMeaning'),]
axes[1].set_ylabel(' Diffeomorph                Original    ', fontsize=mfs)
axes[1].legend(handles=legend_elements, loc='lower right', handlelength=1)
pos = axes[1].get_position()
pos.bounds = (pos.bounds[0], pos.bounds[1]+.05, pos.bounds[2], pos.bounds[3]-.05)
axes[1].set_position(pos, which='both')


# Panel 3: Plot mean critical patch values (diffeomorph-original) across all 40 scenes
axes[2].set_yticks([])
axes[2].set_xticks([]) 
axes[2].set_axisbelow(False)
axes[2].text(0,1.025, "c", fontsize = sfs, weight='bold')
plt.setp(axes[2].spines.values(), visible=False)
axes[2].set_ylabel('  DeepMeaning             Human      ', fontsize=mfs)
axes[2].set_xlabel(" Mean Difference", fontsize = mfs, labelpad=21)
pos = axes[2].get_position()
pos.bounds = (pos.bounds[0], pos.bounds[1]+.05, pos.bounds[2], pos.bounds[3]-.05)
axes[2].set_position(pos, which='both')
human = fig.add_axes([0.492,0.531,0.35,0.35], zorder=1)
human.imshow(human_difference, cmap='seismic',vmin=-1.5,vmax=1.5)
circ = Circle((125,125),102,facecolor='lime',edgecolor='lime',fill=False, linewidth=3)
human.add_patch(circ)
human.axis('off')
deep = fig.add_axes([0.492,0.175,0.35,0.35], zorder=1)
deep.imshow(deep_difference, cmap='seismic',vmin=-1.5,vmax=1.5)
circ = Circle((125,125),102,facecolor='lime',edgecolor='lime',fill=False, linewidth=3)
deep.add_patch(circ)
cax = fig.add_axes([axes[2].get_position().x0,axes[2].get_position().y0,.082,.0158])
fig.colorbar(deep.images[0], orientation="horizontal", cax=cax) 
deep.axis('off')

# Panel 4: Show original and diffeomorph example with CoCa captions
axes[3].set_yticks([])
axes[3].set_xticks([])
axes[3].set_xlim(1, 6)
axes[3].set_axisbelow(True)
axes[3].set_xlabel("Example Patches & CoCa Captions", fontsize = mfs, labelpad=7)
axes[3].set_ylabel('Diffeomorph                  Original', fontsize=mfs)
#plt.setp(axes[1].spines.values(), visible=False)
axes[3].text(1,1.025, "d", fontsize = sfs, weight='bold')

orig_patch = Image.open('../data/accessory_images/coffeemaker_original.png')
orig = fig.add_axes([0.5935,0.496,0.383,0.383], zorder=1)
rgba = orig_patch.convert("RGBA")
datas = rgba.getdata()
newData = []
rgba.putdata(newData)
orig.imshow(rgba)
orig.axis('off')
axes[3].text(3.90,.68, "a shelf with\nmany jars of\nfood on it", fontsize = sfs-3)

diffeo_patch = Image.open('../data/accessory_images/coffeemaker_diffeomorph.png')
diffeo = fig.add_axes([0.5935,0.109,0.383,0.383], zorder=1)
rgba = diffeo_patch.convert("RGBA")
datas = rgba.getdata()
newData = []
rgba.putdata(newData)
diffeo.imshow(rgba)
diffeo.axis('off')
axes[3].text(3.90,.14, "a circular image\nof some sort\nwith different\ncolors", fontsize = sfs-3)

# Save Figure 4
figure4_path = '../figures/figure4.jpg'
plt.savefig(figure4_path, bbox_inches='tight', pad_inches=pad, dpi=425)
print(f"Figure 4 saved to: {figure4_path}")

#%% 060: Generate figure 5

# Create figure with appropriate size
fig, axs = plt.subplots(4, 1, figsize=(17, 13))  # Adjust figsize as needed

# Interpretation image directory
interp_dir = '../data/interpretation/'

# File names
image_files = [
    'indoor_Objects.png',
    'indoor_Interaction.png',
    'indoor_Foreground.png',
    'indoor_Context.png'
]

# Load and display each image
for idx, (ax, img_file) in enumerate(zip(axs, image_files)):
    # Read image
    img = mpimg.imread(interp_dir + img_file)
    
    # Display image
    ax.imshow(img)
    
    # Remove axes
    ax.axis('off')

# Adjust layout to minimize empty space between plots
plt.tight_layout()

# Save the combined figure
figure5_path = '../figures/figure5.jpg'
plt.savefig('../figures/figure5.jpg', bbox_inches='tight', pad_inches=0, dpi=425)
plt.close()
print(f"Figure 5 saved to: {figure5_path}")