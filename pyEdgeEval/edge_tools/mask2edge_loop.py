#!/usr/bin/env python3

import numpy as np

from pyEdgeEval.utils import mask2bdry

__all__ = [
    "loop_mask2edge",
    "loop_instance_mask2edge",
]


def loop_mask2edge(
    mask,
    ignore_indices,
    radius,
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

        edges[label] = mask2bdry(
            mask=m,
            ignore_mask=ignore_mask,
            radius=radius,
            use_cv2=use_cv2,
            quality=quality,
        )

    return edges


def loop_instance_mask2edge(
    mask,
    inst_mask,
    inst_labelIds,
    ignore_indices,
    radius,
    use_cv2=True,
    quality=0,
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

    # create a lookup dictionary {label: [instances]}
    label_inst = {}
    _cand_insts = np.unique(inst_mask)
    for inst_label in _cand_insts:
        if inst_label < inst_labels[0] * 1000:  # 24000
            continue
        _label = int(str(inst_label)[:2])
        _inst = int(str(inst_label)[2:])
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

        edges[label] = dist

    return edges
