#!/usr/bin/env python3

import numpy as np
from PIL import Image

from pyEdgeEval.common.multi_label.evaluate_boundaries import (
    evaluate_boundaries_threshold,
)
from pyEdgeEval.datasets.cityscapes_attributes import (
    CITYSCAPES_labelIds,
    CITYSCAPES_label2trainId,
    CITYSCAPES_inst_labelIds,
)
from pyEdgeEval.common.utils import check_thresholds
from pyEdgeEval.utils import (
    mask2bdry,
    mask2onehot,
    mask_label2trainId,
    # edge_label2trainId,
)

# flip key-value
CITYSCAPES_trainId2label = {v: k for k, v in CITYSCAPES_label2trainId.items()}


def one_label_mask2edge(
    label,
    mask,
    ignore_indices,
    radius,
    use_cv2=True,
    quality=0,
):
    num_labels, h, w = mask.shape
    assert label < num_labels
    ignore_mask = np.zeros((h, w), dtype=np.uint8)
    for i in ignore_indices:
        ignore_mask += mask[i]

    m = mask[label]

    if label in ignore_indices:
        return np.zeros_like(m)
    if not np.count_nonzero(m):
        return np.zeros_like(m)

    return mask2bdry(
        mask=m,
        ignore_mask=ignore_mask,
        radius=radius,
        use_cv2=use_cv2,
        quality=quality,
    )


def one_label_instance_mask2edge(
    label,
    mask,
    inst_mask,
    inst_labelIds,
    ignore_indices,
    radius,
    use_cv2=True,
    quality=0,
):
    num_labels, h, w = mask.shape
    assert label < num_labels
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

    m = mask[label]

    args = dict(
        radius=radius,
        use_cv2=use_cv2,
        quality=quality,
    )

    if label in ignore_indices:
        return np.zeros_like(m)
    if not np.count_nonzero(m):
        return np.zeros_like(m)

    edge = mask2bdry(
        mask=m,
        ignore_mask=ignore_mask,
        **args,
    )

    # per instance boundaries
    if label in label_inst.keys():
        instances = label_inst[label]  # list
        for instance in instances:
            iid = int(str(label) + str(instance).zfill(3))
            _mask = inst_mask == iid
            edge = edge | mask2bdry(
                mask=_mask,
                ignore_mask=ignore_mask,
                **args,
            )

    return edge


def _evaluate_single(
    seg_path,
    inst_path,
    pred_path,
    category,
    scale,
    max_dist,
    thresholds,
    apply_thinning,
    apply_nms,
    kill_internal,
    skip_if_nonexistent,
    radius,
    labels=CITYSCAPES_labelIds,
    ignore_indices=[2, 3],
    **kwargs,
):
    """Evaluate a single sample (sub-routine)

    NOTE: don't set defaults for easier debugging
    """
    # checks and converts thresholds
    thresholds = check_thresholds(thresholds)

    cat_idx = category - 1  # category is indexed from 1

    # get corresponding label
    label = CITYSCAPES_trainId2label[cat_idx]

    # load everything (not trainIds)
    seg_label = Image.open(seg_path)
    # _seg = np.array(seg_label)
    # h, w = _seg.shape
    w, h = seg_label.size
    height, width = int(h * scale + 0.5), int(w * scale + 0.5)
    pred = Image.open(pred_path)

    # rescale
    seg_label = seg_label.resize((width, height), Image.Resampling.NEAREST)
    pred = pred.resize((width, height), Image.Resampling.NEAREST)

    seg_label = np.array(seg_label)
    pred = np.array(pred)
    pred = (pred / 255).astype(float)

    # convert to onehot
    onehot_mask = mask2onehot(seg_label, labels=labels)

    # generate edge
    if inst_path:
        inst_mask = Image.open(inst_path)
        inst_mask = inst_mask.resize((width, height), Image.Resampling.NEAREST)
        inst_mask = np.array(inst_mask)
        edge = one_label_instance_mask2edge(
            label=label,
            mask=onehot_mask,
            inst_mask=inst_mask,
            inst_labelIds=CITYSCAPES_inst_labelIds,
            ignore_indices=ignore_indices,
            radius=radius,
        )
    else:
        edge = one_label_mask2edge(
            label=label,
            mask=onehot_mask,
            ignore_indices=ignore_indices,
            radius=radius,
        )

    # convert to trainIds (edge and segmentation)
    # edge = edge_label2trainId(edge=edge_label, label2trainId=label2trainId)
    seg = mask_label2trainId(
        mask=seg_label, label2trainId=CITYSCAPES_label2trainId
    )

    # cat_edge = edge[cat_idx, :, :]
    if kill_internal:
        # load segmentation map
        assert edge.shape == seg.shape
        # obtain binary map
        # need to be careful where the category starts
        # some datasets will skip 0 and start from 1 (like sbd)
        cat_seg = seg == cat_idx
    else:
        cat_seg = None

    # evaluate multi-label boundaries
    count_r, sum_r, count_p, sum_p = evaluate_boundaries_threshold(
        thresholds=thresholds,
        pred=pred,
        gt=edge,
        gt_seg=cat_seg,
        max_dist=max_dist,
        apply_thinning=apply_thinning,
        kill_internal=kill_internal,
        skip_if_nonexistent=skip_if_nonexistent,
        apply_nms=apply_nms,
        nms_kwargs=dict(
            r=1,
            s=5,
            m=1.01,
            half_prec=False,
        ),
    )

    return count_r, sum_r, count_p, sum_p


def otf_cityscapes_eval_single(kwargs):
    """Wrapper function to unpack all the kwargs"""
    return _evaluate_single(**kwargs)
