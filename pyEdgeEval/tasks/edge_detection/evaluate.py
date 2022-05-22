#!/usr/bin/env python3

from typing import Tuple

import numpy as np

from pyEdgeEval._lib import correspond_pixels
from pyEdgeEval.preprocess.thin import binary_thin


def evaluate_boundaries_bin(
    pred: np.ndarray,
    gts: np.ndarray,
    max_dist: float = 0.0075,
    apply_thinning: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the accuracy of a predicted boundary.

    :param pred: the predicted boundaries as a (H,W)
        binary array
    :param gts: a list of ground truth boundaries, as returned
        by the `load_boundaries` or `boundaries` methods
    :param max_dist: (default=0.0075) maximum distance parameter
        used for determining pixel matches. This value is multiplied by the
        length of the diagonal of the image to get the threshold used
        for matching pixels.
    :param apply_thinning: (default=True) if True, apply morphologial
        thinning to the predicted boundaries before evaluation
    :return: tuple `(count_r, sum_r, count_p, sum_p)` where each of
        the four entries are float values that can be used to compute
        recall and precision with:
        ```
        recall = count_r / (sum_r + (sum_r == 0))
        precision = count_p / (sum_p + (sum_p == 0))
        ```
    """
    acc_prec = np.zeros(pred.shape, dtype=bool)
    pred = pred != 0

    if apply_thinning:
        pred = binary_thin(pred)

    sum_r = 0
    count_r = 0
    for gt in gts:
        match1, match2, cost, oc = correspond_pixels(
            pred, gt, max_dist=max_dist
        )
        match1 = match1 > 0
        match2 = match2 > 0
        # Precision accumulator
        acc_prec = acc_prec | match1
        # Recall
        sum_r += gt.sum()
        count_r += match2.sum()

    # Precision
    sum_p = pred.sum()
    count_p = acc_prec.sum()

    return count_r, sum_r, count_p, sum_p


def evaluate_boundaries_threshold(
    thresholds: np.ndarray,
    pred: np.ndarray,
    gts: np.ndarray,
    max_dist: float = 0.0075,
    apply_thinning: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the accuracy of a predicted boundary and a range of thresholds

    :param pred: the predicted boundaries as a (H,W)
        floating point array where each pixel represents the strength of the
        predicted boundary
    :param gts: a list of ground truth boundaries, as returned
        by the `load_boundaries` or `boundaries` methods
    :param thresholds: a 1D array specifying the thresholds
    :param max_dist: (default=0.0075) maximum distance parameter
        used for determining pixel matches. This value is multiplied by the
        length of the diagonal of the image to get the threshold used
        for matching pixels.
    :param apply_thinning: (default=True) if True, apply morphologial
        thinning to the predicted boundaries before evaluation
    :return: tuple `(count_r, sum_r, count_p, sum_p, thresholds)` where each
        of the first four entries are arrays that can be used to compute
        recall and precision at each threshold with:
        ```
        recall = count_r / (sum_r + (sum_r == 0))
        precision = count_p / (sum_p + (sum_p == 0))
        ```
        The thresholds are also returned.
    """

    sum_p = np.zeros(thresholds.shape)
    count_p = np.zeros(thresholds.shape)
    sum_r = np.zeros(thresholds.shape)
    count_r = np.zeros(thresholds.shape)

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


def evaluate_single_sample_bin(
    sample,
    func_load_pred,
    func_load_gt,
    max_dist: float = 0.0075,
    apply_thinning: bool = True,
):
    """subroutine wrapper for parallel

    `func_load_pred` and `func_load_gt` take `sample` as argument to load a pair of
    prediction and GT data
    """
    pred = func_load_pred(sample)
    gt_b = func_load_gt(sample)
    count_r, sum_r, count_p, sum_p = evaluate_boundaries_bin(
        pred=pred,
        gts=gt_b,
        max_dist=max_dist,
        apply_thinning=apply_thinning,
    )
    return count_r, sum_r, count_p, sum_p


def evaluate_single_sample_threshold(
    sample,
    thresholds,
    func_load_pred,
    func_load_gt,
    max_dist: float = 0.0075,
    apply_thinning: bool = True,
):
    """subroutine wrapper for parallel

    `func_load_pred` and `func_load_gt` take `sample` as argument to load a pair of
    prediction and GT data
    """
    pred = func_load_pred(sample)
    gt_b = func_load_gt(sample)
    count_r, sum_r, count_p, sum_p = evaluate_boundaries_threshold(
        thresholds=thresholds,
        pred=pred,
        gts=gt_b,
        max_dist=max_dist,
        apply_thinning=apply_thinning,
    )
    return count_r, sum_r, count_p, sum_p
