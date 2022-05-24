#!/usr/bin/env python3

import os

import numpy as np
from PIL import Image

from .edges import (
    faster_onehot_mask_to_multilabel_edges,
    mask_to_onehot,
    onehot_mask_to_multilabel_edges,
)
from .edge_encoding import default_encoding, rgb_encoding
from .label2trainId import label_mapping


def ids_to_trainIds(edges_ids: np.ndarray, trainIds: int = 19):
    _, h, w = edges_ids.shape
    edges_trainIds = np.zeros((trainIds, h, w), dtype=np.uint8)
    for labelId, trainId in label_mapping.items():
        edges_trainIds[trainId] = edges_ids[labelId, ...]
    return edges_trainIds


def label2edge(
    label_path: str,
    save_path: str,
    radius: int = 2,
    faster: bool = True,
) -> None:
    """Instance-insensitive edge"""
    assert os.path.exists(label_path)
    _, save_format = os.path.splitext(save_path)
    assert save_format in (".png", ".tif", ".bin")

    label_img = Image.open(label_path)
    mask = np.array(label_img)

    # NOTE: hard-coded
    num_ids = 34  # total number of classes including ones to ignore
    num_trainIds = 19  # final number of classes
    ignore_classes = [2, 3]
    h, w = mask.shape

    # create label mask
    m = mask_to_onehot(mask, num_ids)

    # create label edge maps
    if faster:
        edges_ids = faster_onehot_mask_to_multilabel_edges(
            mask=m,
            radius=radius,
            num_classes=num_ids,
            ignore_classes=ignore_classes,
            quality=0,
        )
    else:
        edges_ids = onehot_mask_to_multilabel_edges(
            mask=m,
            radius=radius,
            num_classes=num_ids,
            ignore_classes=ignore_classes,
        )

    # post-process labels
    edges_trainIds = ids_to_trainIds(edges_ids, num_trainIds)

    # encode and save
    if save_format == ".png":
        edges = rgb_encoding(edges_trainIds)
        edges_img = Image.fromarray(edges)
        edges_img.save(save_path)
    elif save_format == ".tif":
        edges = default_encoding(edges_trainIds)
        edges = edges.view(np.int32)
        edges_img = Image.fromarray(edges)
        edges_img.save(save_path)
    elif save_format == ".bin":
        edges = default_encoding(edges_trainIds)
        edges.tofile(
            save_path,
            dtype=np.uint32,
        )
    else:
        raise ValueError()
