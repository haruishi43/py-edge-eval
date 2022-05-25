#!/usr/bin/env python3

import os
from typing import Optional

import numpy as np
from PIL import Image

from .edge_encoding import default_encoding, rgb_encoding
from .labels import label_mapping
from .seg2edges import (
    instance_insensitive_seg2edges,
    instance_sensitive_seg2edges,
)
from pyEdgeEval.utils import mask_to_onehot


def ids2trainIds(edges_ids: np.ndarray, trainIds: int = 19):
    _, h, w = edges_ids.shape
    edges_trainIds = np.zeros((trainIds, h, w), dtype=np.uint8)
    for labelId, trainId in label_mapping.items():
        edges_trainIds[trainId] = edges_ids[labelId, ...]
    return edges_trainIds


def label2edge(
    label_path: str,
    save_path: str,
    radius: int = 2,
    inst_sensitive: bool = True,
    inst_path: Optional[str] = None,
    faster: bool = True,
    nproc: int = 1,
) -> None:
    """main function for converting label-file to multi-label edges"""
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
    if inst_sensitive:
        assert os.path.exists(inst_path)
        inst_img = Image.open(inst_path)
        inst_mask = np.array(inst_img)  # int32

        edges_ids = instance_sensitive_seg2edges(
            mask=m,
            inst_mask=inst_mask,
            radius=radius,
            num_classes=num_ids,
            ignore_classes=ignore_classes,
            nproc=nproc,
            use_cv2=faster,
            quality=0,
        )
    else:
        edges_ids = instance_insensitive_seg2edges(
            mask=m,
            radius=radius,
            num_classes=num_ids,
            ignore_classes=ignore_classes,
            nproc=nproc,
            use_cv2=faster,
            quality=0,
        )

    # post-process labels
    edges_trainIds = ids2trainIds(edges_ids, num_trainIds)

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
