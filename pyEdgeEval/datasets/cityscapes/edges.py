#!/usr/bin/env python3

from typing import List

import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def onehot_mask_to_multilabel_edges(
    mask: np.ndarray,
    radius: int = 2,
    num_classes: int = 34,
    ignore_classes: List[int] = [2, 3],
) -> np.ndarray:
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)
    """
    if radius < 1:
        return mask

    h, w = mask.shape[1:]

    # make ignore mask
    ignore_mask = np.zeros((h, w), dtype=np.uint8)
    for i in ignore_classes:
        ignore_mask += mask[i]

    edges = np.zeros_like(mask)
    for c in range(num_classes):
        if c in ignore_classes:
            continue

        # if there are no class labels in the mask
        if not np.count_nonzero(mask[c]):
            continue

        inner = distance_transform_edt((mask[c] + ignore_mask) > 0)
        outer = distance_transform_edt(1.0 - mask[c])
        dist = outer + inner

        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)

        edges[c] = dist

    return edges


def faster_onehot_mask_to_multilabel_edges(
    mask: np.ndarray,
    radius: int = 2,
    num_classes: int = 34,
    ignore_classes: List[int] = [2, 3],
    quality: int = 0,
) -> np.ndarray:
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)
    NOTE: quality doesn't really change speed
    """
    if radius < 1:
        return mask

    h, w = mask.shape[1:]

    # make ignore mask
    ignore_mask = np.zeros((h, w), dtype=np.uint8)
    for i in ignore_classes:
        ignore_mask += mask[i]

    edges = np.zeros_like(mask)
    for c in range(num_classes):
        if c in ignore_classes:
            continue

        # if there are no class labels in the mask
        if not np.count_nonzero(mask[c]):
            continue

        inner = cv2.distanceTransform(
            ((mask[c] + ignore_mask) > 0).astype(np.uint8), cv2.DIST_L2, quality
        )
        outer = cv2.distanceTransform(
            ((1.0 - mask[c]) > 0).astype(np.uint8), cv2.DIST_L2, quality
        )
        dist = outer + inner

        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)

        edges[c] = dist

    return edges


def multilabel_to_binary_edges(edges: np.ndarray) -> np.ndarray:
    return (np.sum(edges, axis=0) > 0).astype(np.uint8)


def onehot_edges_to_multilabel_edges(edges: np.ndarray) -> np.ndarray:
    labels, h, w = edges.shape

    edge_map = np.zeros((h, w), dtype=np.uint32)
    for l in range(labels):
        m = edges[l]
        edge_map = edge_map + (2**l) * m

    return edge_map
