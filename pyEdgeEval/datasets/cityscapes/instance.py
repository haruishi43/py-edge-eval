#!/usr/bin/env python3

from functools import partial
from typing import List

import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from pyEdgeEval.utils import (
    track_parallel_progress,
    track_progress,
)


inst_labelIds = [
    24,  # "person"
    25,  # "rider"
    26,  # "car"
    27,  # "truck"
    28,  # "bus"
    31,  # "train"
    32,  # "motorcycle"
    33,  # "bicycle"
]


def single_run(
    c,
    mask,
    inst_mask,
    ignore_classes,
    cls_inst,
    calc_dist,
):
    m = mask[c]

    if c in ignore_classes:
        return np.zeros_like(m)

    # if there are no class labels in the mask
    if not np.count_nonzero(mask[c]):
        return np.zeros_like(m)

    dist = calc_dist(m)

    if c in cls_inst.keys():
        instances = cls_inst[c]  # list
        for instance in instances:
            iid = int(str(c) + str(instance).zfill(3))
            imask = inst_mask == iid
            idist = calc_dist(imask)
            dist = dist | idist

    return dist


def scipy_calc_dist(m, ignore_mask, radius):
    inner = distance_transform_edt((m + ignore_mask) > 0)
    outer = distance_transform_edt(1.0 - m)
    dist = outer + inner

    dist[dist > radius] = 0
    dist = (dist > 0).astype(np.uint8)
    return dist


def cv2_calc_dist(m, ignore_mask, radius, quality):
    inner = cv2.distanceTransform(
        ((m + ignore_mask) > 0).astype(np.uint8), cv2.DIST_L2, quality
    )
    outer = cv2.distanceTransform(
        ((1.0 - m) > 0).astype(np.uint8), cv2.DIST_L2, quality
    )
    dist = outer + inner

    dist[dist > radius] = 0
    dist = (dist > 0).astype(np.uint8)
    return dist


def onehot_mask_to_instance_sensitive_multilabel_edges(
    mask: np.ndarray,
    inst_mask: np.ndarray,
    radius: int = 2,
    num_classes: int = 34,
    ignore_classes: List[int] = [2, 3],
    nproc: int = 1,
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

    cls_inst = {}
    _cand_insts = np.unique(inst_mask)
    for c in _cand_insts:
        if c < inst_labelIds[0] * 1000:  # 24000
            continue
        _c = int(str(c)[:2])
        _i = int(str(c)[2:])
        if _c not in cls_inst.keys():
            cls_inst[_c] = [_i]
        else:
            cls_inst[_c].append(_i)

    _calc_dist = partial(
        scipy_calc_dist,
        ignore_mask=ignore_mask,
        radius=radius,
    )

    _single_run = partial(
        single_run,
        mask=mask,
        inst_mask=inst_mask,
        ignore_classes=ignore_classes,
        cls_inst=cls_inst,
        calc_dist=_calc_dist,
    )

    if nproc > 1:
        results = track_parallel_progress(
            _single_run,
            list(range(num_classes)),
            nproc=nproc,
            keep_order=True,
            no_bar=True,
        )
    else:
        results = track_progress(
            _single_run,
            list(range(num_classes)),
            no_bar=True,
        )

    edges = np.zeros_like(mask)
    for c in range(num_classes):
        edges[c] = results[c]

    return edges


def faster_onehot_mask_to_instance_sensitive_multilabel_edges(
    mask: np.ndarray,
    inst_mask: np.ndarray,
    radius: int = 2,
    num_classes: int = 34,
    ignore_classes: List[int] = [2, 3],
    quality: int = 0,
    nproc: int = 1,
) -> np.ndarray:
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)

    NOTE: quality doesn't really change speed
    NOTE: nproc > 1 is slower

    FIXME: try to reduce the serialization/deserialization size for mp
    """
    if radius < 1:
        return mask

    h, w = mask.shape[1:]

    # make ignore mask
    ignore_mask = np.zeros((h, w), dtype=np.uint8)
    for i in ignore_classes:
        ignore_mask += mask[i]

    cls_inst = {}
    _cand_insts = np.unique(inst_mask)
    for c in _cand_insts:
        if c < inst_labelIds[0] * 1000:  # 24000
            continue
        _c = int(str(c)[:2])
        _i = int(str(c)[2:])
        if _c not in cls_inst.keys():
            cls_inst[_c] = [_i]
        else:
            cls_inst[_c].append(_i)

    _calc_dist = partial(
        cv2_calc_dist,
        ignore_mask=ignore_mask,
        radius=radius,
        quality=quality,
    )

    _single_run = partial(
        single_run,
        mask=mask,
        inst_mask=inst_mask,
        ignore_classes=ignore_classes,
        cls_inst=cls_inst,
        calc_dist=_calc_dist,
    )

    if nproc > 1:
        results = track_parallel_progress(
            _single_run,
            list(range(num_classes)),
            nproc=nproc,
            keep_order=True,
            no_bar=True,
        )
    else:
        results = track_progress(
            _single_run,
            list(range(num_classes)),
            no_bar=True,
        )

    edges = np.zeros_like(mask)
    for c in range(num_classes):
        edges[c] = results[c]

    return edges
