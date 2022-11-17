#!/usr/bin/env python3

from warnings import warn

import numpy as np

from pyEdgeEval.preprocess import binary_thin
from pyEdgeEval.utils import mask2bdry

__all__ = [
    "loop_mask2edge",
    "loop_instance_mask2edge",
]


def loop_mask2edge(
    mask,
    ignore_indices,
    radius,
    thin=False,
    use_cv2=True,
    quality=0,
):
    """mask2edge with looping"""
    assert mask.ndim == 3
    num_labels, h, w = mask.shape

    # make ignore mask
    ignore_mask = np.zeros((h, w), dtype=np.uint8)
    for i in ignore_indices:
        ignore_mask += mask[i]

    edges = np.zeros_like(mask)
    for label in range(num_labels):
        m = mask[label]

        if label in ignore_indices:
            continue

        # if there are no class labels in the mask
        if not np.count_nonzero(m):
            continue

        edge = mask2bdry(
            mask=m,
            ignore_mask=ignore_mask,
            radius=radius,
            use_cv2=use_cv2,
            quality=quality,
        )

        # thin the boundaries
        if thin:
            edge = binary_thin(edge).astype(np.uint8)

        edges[label] = edge

    return edges


def loop_instance_mask2edge(
    mask,
    inst_mask,
    inst_labelIds,
    ignore_indices,
    radius,
    thin=False,
    use_cv2=True,
    quality=0,
    _inst_len=5,
    _inst_id_dig=2,
):
    """mask2edge with looping (instance sensitive)"""
    assert mask.ndim == 3
    num_labels, h, w = mask.shape

    # make ignore mask
    ignore_mask = np.zeros((h, w), dtype=np.uint8)
    for i in ignore_indices:
        ignore_mask += mask[i]

    # make sure that instance labels are sorted
    inst_labels = sorted(inst_labelIds)
    if inst_labels[0] == 0:
        # when the first label is 0, it's probably not right
        warn(
            "inst labels has labelId of 0, but this should be the background Id"
        )
    min_inst_id = inst_labels[0] * (10 ** (_inst_len - _inst_id_dig))

    # create a lookup dictionary {label: [instances]}
    label_inst = {}
    _cand_insts = np.unique(inst_mask)
    for inst_label in _cand_insts:
        if inst_label < min_inst_id:
            continue

        # convert to string and fill
        inst_label = str(inst_label).zfill(_inst_len)

        _label = int(inst_label[:_inst_id_dig])
        _inst = int(inst_label[_inst_id_dig:])
        if _label not in label_inst.keys():
            label_inst[_label] = [_inst]
        else:
            label_inst[_label].append(_inst)

    args = dict(
        radius=radius,
        use_cv2=use_cv2,
        quality=quality,
    )

    edges = np.zeros_like(mask)
    for label in range(num_labels):
        m = mask[label]

        if label in ignore_indices:
            continue

        # if there are no class labels in the mask
        if not np.count_nonzero(m):
            continue

        dist = mask2bdry(mask=m, ignore_mask=ignore_mask, **args)

        # per instance boundaries
        if label in label_inst.keys():
            instances = label_inst[label]  # list
            for instance in instances:
                iid = int(str(label) + str(instance).zfill(3))
                _mask = inst_mask == iid
                dist = dist | mask2bdry(
                    mask=_mask,
                    ignore_mask=ignore_mask,
                    **args,
                )

        # thin the boundaries
        if thin:
            dist = binary_thin(dist).astype(np.uint8)

        edges[label] = dist

    return edges
