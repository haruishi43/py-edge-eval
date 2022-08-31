#!/usr/bin/env python3

import numpy as np
from PIL import Image

from pyEdgeEval.common.multi_label import (
    decode_png,
    load_scaled_edge,
    evaluate_boundaries_threshold,
)
from pyEdgeEval.common.utils import check_thresholds


def _evaluate_single(
    edge_path,
    seg_path,
    pred_path,
    category,
    scale,
    max_dist,
    thresholds,
    apply_thinning,
    apply_nms,
    kill_internal,
    skip_if_nonexistent,
    num_classes,
    **kwargs,
):
    """Evaluate a single sample (sub-routine)

    NOTE: don't set defaults for easier debugging
    """
    # checks and converts thresholds
    thresholds = check_thresholds(thresholds)

    # load gt edge
    edge, (height, width) = load_scaled_edge(edge_path, scale)
    edge = decode_png(edge, num_classes)
    cat_idx = category - 1
    cat_edge = edge[cat_idx, :, :]

    # load pred
    pred = Image.open(pred_path)
    pred = pred.resize((width, height), Image.Resampling.NEAREST)
    pred = np.array(pred)
    pred = (pred / 255).astype(float)

    if kill_internal:
        # load segmentation map
        seg = Image.open(seg_path)
        seg = seg.resize((width, height), Image.Resampling.NEAREST)
        seg = np.array(seg)
        assert edge.shape[1:] == seg.shape
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
        gt=cat_edge,
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


def cityscapes_eval_single(kwargs):
    """Wrapper function to unpack all the kwargs"""
    return _evaluate_single(**kwargs)
