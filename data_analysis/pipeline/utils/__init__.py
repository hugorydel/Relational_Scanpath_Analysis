"""
pipeline/utils/
===============
Utility subpackage for the relational replay pipeline.

Modules
-------
scene_graph  — AOI polygon index and relational graph index (Step 1, Module 3)
saliency     — Spectral residual saliency maps (Hou & Zhang, 2007)
"""

from .saliency import compute_all_saliency_maps, get_saliency_map
from .scene_graph import build_graph_index, build_polygon_index, load_stimulus_metadata

__all__ = [
    "load_stimulus_metadata",
    "build_polygon_index",
    "build_graph_index",
    "get_saliency_map",
    "compute_all_saliency_maps",
]
