#!/usr/bin/env python3

"""Encoding functions for multi-label edges

- `default`: encodes into binary format
- `RGB`: encodes into binary format that is compatible with image fs
"""

import numpy as np


def default_multilabel_encoding(edges: np.ndarray):
    """Encode multi-label edges to binary format

    For now we use uint32 as the base (PIL can save uint32).
    Therefore, we can save at most 32 classes.

    However, RGB encoding is efficient unless you need more classes.
    """
    num_classes, h, w = edges.shape

    cat_edge_map = np.zeros((h, w), dtype=np.uint32)
    for trainId in range(num_classes):
        edge_map = edges[trainId]
        cat_edge_map = cat_edge_map + (2**trainId) * edge_map

    return cat_edge_map


def rgb_multilabel_encoding(edges: np.ndarray):
    """Encode multi-label edges to RGB format

    Each channel is 8-bit, so the RGB format can encode the edges into 24-bit
    (maximum of 24 classes).

    This format is useful for training data where edges need to be transformed
    (compatible with various 3-channel augmentations).
    """

    num_classes, h, w = edges.shape

    cat_edge_b = np.zeros((h, w), dtype=np.uint8)
    cat_edge_g = np.zeros((h, w), dtype=np.uint8)
    cat_edge_r = np.zeros((h, w), dtype=np.uint8)
    cat_edge_png = np.zeros((h, w, 3), dtype=np.uint8)

    for trainId in range(num_classes):
        edge_map = edges[trainId]
        if trainId >= 0 and trainId < 8:
            cat_edge_b = cat_edge_b + (2**trainId) * edge_map
        elif trainId >= 8 and trainId < 16:
            cat_edge_g = cat_edge_g + (2 ** (trainId - 8)) * edge_map
        elif trainId >= 16 and trainId < 24:
            cat_edge_r = cat_edge_r + (2 ** (trainId - 16)) * edge_map
        else:
            raise ValueError()

    cat_edge_png[:, :, 0] = cat_edge_r
    cat_edge_png[:, :, 1] = cat_edge_g
    cat_edge_png[:, :, 2] = cat_edge_b

    return cat_edge_png
