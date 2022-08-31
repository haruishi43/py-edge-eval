#!/usr/bin/env python3

import numpy as np

from pyEdgeEval._lib import correspond_pixels
from pyEdgeEval.preprocess import binary_thin, fast_nms


def evaluate_boundaries_threshold(
    thresholds: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    max_dist: float = 0.0075,
    apply_thinning: bool = True,
    apply_nms: bool = False,
    nms_kwargs=dict(
        r=1,
        s=5,
        m=1.01,
        half_prec=False,
    ),
):
    """
    Evaluate the accuracy of a predicted boundary and a range of thresholds

    - Single GT

    Args:
        thresholds: a 1D array specifying the thresholds
        pred: the predicted boundaries as a (H,W) floating point array where
            each pixel represents the strength of the predicted boundary
        gts: list of ground truth boundary, as returned
            by the `load_boundary` or `boundary` methods
        max_dist: (default=0.02) maximum distance parameter
            used for determining pixel matches. This value is multiplied by the
            length of the diagonal of the image to get the threshold used
            for matching pixels.
        apply_thinning: (default=True) if True, apply morphologial
            thinning to the predicted boundaries before evaluation
        apply_nms: (default=False) apply a fast nms preprocess
        nms_kwargs: arguments for nms process

    Returns:
        tuple `(count_r, sum_r, count_p, sum_p, thresholds)` where each
            of the first four entries are arrays that can be used to compute
            recall and precision at each threshold with:
            ```
            recall = count_r / (sum_r + (sum_r == 0))
            precision = count_p / (sum_p + (sum_p == 0))
            ```
    """

    sum_p = np.zeros(thresholds.shape)
    count_p = np.zeros(thresholds.shape)
    sum_r = np.zeros(thresholds.shape)
    count_r = np.zeros(thresholds.shape)

    if apply_nms:
        pred = fast_nms(
            img=pred,
            **nms_kwargs,
        )

    for i_t, thresh in enumerate(list(thresholds)):

        _pred = pred >= thresh

        if apply_thinning:
            _pred = binary_thin(_pred)

        if gt.any():
            match1, match2, cost, oc = correspond_pixels(
                _pred, gt, max_dist=max_dist
            )
            match1 = match1 > 0
            match2 = match2 > 0

            # Recall
            sum_r[i_t] = gt.sum()
            count_r[i_t] = match2.sum()

            # Precision
            sum_p[i_t] = _pred.sum()
            count_p[i_t] = match1.sum()
        else:
            sum_r[i_t] = 0
            count_r[i_t] = 0
            sum_p[i_t] = _pred.sum()  # keep track of false positives
            count_p[i_t] = 0

    return count_r, sum_r, count_p, sum_p


def evaluate_boundaries_threshold_multiple_gts(
    thresholds: np.ndarray,
    pred: np.ndarray,
    gts: np.ndarray,
    max_dist: float = 0.0075,
    apply_thinning: bool = True,
    apply_nms: bool = False,
    nms_kwargs=dict(
        r=1,
        s=5,
        m=1.01,
        half_prec=False,
    ),
):
    """
    Evaluate the accuracy of a predicted boundary and a range of thresholds

    - Assumes that there are multiple GTs

    Args:
        thresholds: a 1D array specifying the thresholds
        pred: the predicted boundaries as a (H,W) floating point array where
            each pixel represents the strength of the predicted boundary
        gts: list of ground truth boundary, as returned
            by the `load_boundary` or `boundary` methods
        max_dist: (default=0.02) maximum distance parameter
            used for determining pixel matches. This value is multiplied by the
            length of the diagonal of the image to get the threshold used
            for matching pixels.
        apply_thinning: (default=True) if True, apply morphologial
            thinning to the predicted boundaries before evaluation
        apply_nms: (default=False) apply a fast nms preprocess
        nms_kwargs: arguments for nms process

    Returns:
        tuple `(count_r, sum_r, count_p, sum_p, thresholds)` where each
            of the first four entries are arrays that can be used to compute
            recall and precision at each threshold with:
            ```
            recall = count_r / (sum_r + (sum_r == 0))
            precision = count_p / (sum_p + (sum_p == 0))
            ```
    """

    sum_p = np.zeros(thresholds.shape)
    count_p = np.zeros(thresholds.shape)
    sum_r = np.zeros(thresholds.shape)
    count_r = np.zeros(thresholds.shape)

    if apply_nms:
        pred = fast_nms(
            img=pred,
            **nms_kwargs,
        )

    for i_t, thresh in enumerate(list(thresholds)):

        _pred = pred >= thresh

        acc_prec = np.zeros(_pred.shape, dtype=bool)

        if apply_thinning:
            _pred = binary_thin(_pred)

        for gt in gts:

            match1, match2, cost, oc = correspond_pixels(
                _pred, gt, max_dist=max_dist
            )
            match1 = match1 > 0
            match2 = match2 > 0

            # Precision accumulator
            acc_prec = acc_prec | match1

            # Recall
            sum_r[i_t] += gt.sum()
            count_r[i_t] += match2.sum()

        # Precision
        sum_p[i_t] = _pred.sum()
        count_p[i_t] = acc_prec.sum()

    return count_r, sum_r, count_p, sum_p
