#!/usr/bin/env python3

import numpy as np


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def multilabel_to_binary_edges(edges: np.ndarray) -> np.ndarray:
    """
    Converts multilabel edges to binary edge data
    """
    return (np.sum(edges, axis=0) > 0).astype(np.uint8)


def onehot_edges_to_multilabel_edges(edges: np.ndarray) -> np.ndarray:
    """
    Converts multilabel edges to encoded single channel edge data
    """
    labels, h, w = edges.shape

    edge_map = np.zeros((h, w), dtype=np.uint32)
    for l in range(labels):
        m = edges[l]
        edge_map = edge_map + (2**l) * m

    return edge_map
