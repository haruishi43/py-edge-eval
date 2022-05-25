#!/usr/bin/env python3

from functools import partial
from typing import List

import numpy as np

from .distance_transforms import seg2bdry
from .labels import inst_labelIds

from pyEdgeEval.utils import (
    track_parallel_progress,
    track_progress,
)


def instance_sensitive_run(
    c,
    mask,
    inst_mask,
    ignore_classes,
    cls_inst,
    func_seg2bdry,
):
    m = mask[c]

    if c in ignore_classes:
        return np.zeros_like(m)

    # if there are no class labels in the mask
    if not np.count_nonzero(m):
        return np.zeros_like(m)

    # class-wise distances
    dist = func_seg2bdry(mask=m)

    # per instance boundaries
    if c in cls_inst.keys():
        instances = cls_inst[c]  # list
        for instance in instances:
            iid = int(str(c) + str(instance).zfill(3))
            dist = dist | func_seg2bdry(inst_mask == iid)

    return dist


def instance_insensitive_run(
    c,
    mask,
    ignore_classes,
    func_seg2bdry,
):
    m = mask[c]

    if c in ignore_classes:
        return np.zeros_like(m)

    # if there are no class labels in the mask
    if not np.count_nonzero(m):
        return np.zeros_like(m)

    # class-wise distances
    return func_seg2bdry(mask=m)


def instance_sensitive_seg2edges(
    mask: np.ndarray,
    inst_mask: np.ndarray,
    radius: int = 2,
    num_classes: int = 34,
    ignore_classes: List[int] = [2, 3],
    nproc: int = 1,
    use_cv2: bool = True,
    quality: int = 0,
) -> np.ndarray:
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)

    NOTE: `use_cv2` is around x10 faster
    NOTE: quality doesn't really change speed
    NOTE: nproc > 1 is slower

    FIXME: try to reduce the serialization/deserialization size for mp

    There might be inconsistent instance segmentation between the json file and
    png file:
    - https://github.com/mcordts/cityscapesScripts/issues/136
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

    _seg2bdry = partial(
        seg2bdry,
        ignore_mask=ignore_mask,
        radius=radius,
        use_cv2=use_cv2,
        quality=quality,
    )

    _single_run = partial(
        instance_sensitive_run,
        mask=mask,
        inst_mask=inst_mask,
        ignore_classes=ignore_classes,
        cls_inst=cls_inst,
        func_seg2bdry=_seg2bdry,
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


def instance_insensitive_seg2edges(
    mask: np.ndarray,
    radius: int = 2,
    num_classes: int = 34,
    ignore_classes: List[int] = [2, 3],
    nproc: int = 1,
    use_cv2: bool = True,
    quality: int = 0,
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

    _seg2bdry = partial(
        seg2bdry,
        ignore_mask=ignore_mask,
        radius=radius,
        use_cv2=use_cv2,
        quality=quality,
    )

    _single_run = partial(
        instance_insensitive_run,
        mask=mask,
        ignore_classes=ignore_classes,
        func_seg2bdry=_seg2bdry,
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
