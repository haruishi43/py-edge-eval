#!/usr/bin/env python3

import numpy as np
from skimage.util import img_as_float
from skimage.io import imread

from pyEdgeEval.common.binary_label.evaluate_boundaries import (
    evaluate_boundaries_threshold,
)
from pyEdgeEval.common.utils import check_thresholds
from pyEdgeEval.utils import loadmat


def load_bsds_gt_boundaries(path: str, new_loader: bool = False):
    """BSDS GT Boundaries

    - there are multiple boundaries because there are multiple annotators
    - uint8
    """
    if new_loader:
        from pymatreader import read_mat

        gt = read_mat(path)["groundTruth"]  # list
        num_gts = len(gt)
        return [gt[i]["Boundaries"] for i in range(num_gts)]
    else:
        # FIXME: confusing data organization with scipy
        gt = loadmat(path, False)["groundTruth"]  # np.ndarray
        num_gts = gt.shape[1]
        return [gt[0, i]["Boundaries"][0, 0] for i in range(num_gts)]


def load_predictions(path: str):
    img = imread(path)
    assert (
        img.dtype == np.uint8
    ), f"ERR: img needs to be uint8, but got{img.dtype}"
    return img_as_float(img)


def _evaluate_single(
    gt_path,
    pred_path,
    scale,
    max_dist,
    thresholds,
    apply_thinning,
    apply_nms,
    **kwargs,
):
    """Evaluate a single sample (sub-routine)

    NOTE: don't set defaults for easier debugging
    """
    # checks and converts thresholds
    thresholds = check_thresholds(thresholds)

    pred = load_predictions(pred_path)
    gt = load_bsds_gt_boundaries(gt_path)

    # TODO: scale inputs

    # evaluate multi-label boundaries
    count_r, sum_r, count_p, sum_p = evaluate_boundaries_threshold(
        thresholds=thresholds,
        pred=pred,
        gt=gt,
        max_dist=max_dist,
        apply_thinning=apply_thinning,
        apply_nms=apply_nms,
        nms_kwargs=dict(
            r=1,
            s=5,
            m=1.01,
            half_prec=False,
        ),
    )

    return count_r, sum_r, count_p, sum_p


def bsds_eval_single(kwargs):
    """Wrapper function to unpack all the kwargs"""
    return _evaluate_single(**kwargs)
