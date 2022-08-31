#!/usr/bin/env python3

import numpy as np


def mask2onehot(mask, labels):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    c = mask.shape[0]
    assert (
        len(labels) > 0
    ), "`labels` should be a list with more than 1 elements"
    assert c >= len(
        labels
    ), "tried to convert into onehot with more labels than the original mask"
    _mask = [mask == i for i in labels]
    return np.array(_mask).astype(np.uint8)


def edge_multilabel2binary(edges: np.ndarray) -> np.ndarray:
    """
    Converts multilabel edge to binary edge data (collapse multi-label)
    """
    return (np.sum(edges, axis=0) > 0).astype(np.uint8)


def edge_onehot2multilabel(edges: np.ndarray) -> np.ndarray:
    """
    Converts multilabel edges to encoded single channel edge data
    while preserving multi-label
    """
    labels, h, w = edges.shape

    edge_map = np.zeros((h, w), dtype=np.uint32)
    for l in range(labels):
        m = edges[l]
        edge_map = edge_map + (2**l) * m

    return edge_map


def mask_label2trainId(mask: np.ndarray, label2trainId: dict) -> np.ndarray:
    """Python version of `labelid2trainid` function for segmentation data

    Args:
        mask: single channel image containing segmentation label

    Returns:
        np.ndarray
    """

    if len(mask.shape) == 2:
        h, w = mask.shape
    elif len(mask.shape) == 3:
        h, w, c = mask.shape
        assert c == 1, f"ERR: input label has {c} channels which should be 1"
    else:
        raise ValueError()

    # 1. create an array populated with 255 (background pixel)
    trainId_mask = 255 * np.ones((h, w), dtype=np.uint8)  # 8-bit array

    # 2. map all pixels from `label` to `trainId`
    for labelId, trainId in label2trainId.items():
        idx = mask == labelId
        trainId_mask[idx] = trainId
    return trainId_mask


def edge_label2trainId(edge: np.ndarray, label2trainId: dict) -> np.ndarray:
    assert (
        len(edge.shape) == 3
    ), f"ERR: should be 3 channel input but got {edge.shape}"
    _, h, w = edge.shape
    edges_trainIds = np.zeros((len(label2trainId), h, w), dtype=np.uint8)
    for labelId, trainId in label2trainId.items():
        edges_trainIds[trainId] = edge[labelId, ...]
    return edges_trainIds
