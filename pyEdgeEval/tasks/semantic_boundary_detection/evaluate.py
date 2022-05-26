#!/usr/bin/env python3

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

from pyEdgeEval._lib import correspond_pixels
from pyEdgeEval.preprocess.thin import binary_thin


def kill_internal_prediction(
    pred: np.ndarray,
    gt: np.ndarray,
    seg: np.ndarray,
    max_dist: float = 0.02,
) -> np.ndarray:
    """Remove predicted pixels inside boundaries

    NOTE: the distance transform may differ from MATLAB implementation
    NOTE: might not work correctly when using instance sensitive boundaries
    """
    diag = np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)
    buffer = diag * max_dist

    # buggy output when input is only 0s or 1s
    distmap = distance_transform_edt(1 - gt)
    killmask = np.invert((distmap > buffer) * seg)
    assert killmask.shape == pred.shape
    return pred * killmask


def evaluate_boundaries_bin(
    pred: np.ndarray,
    gt: np.ndarray,
    gt_seg: Optional[np.ndarray] = None,
    max_dist: float = 0.02,  # 0.0075
    apply_thinning: bool = True,
    kill_internal: bool = True,
):
    pred = pred != 0

    if apply_thinning:
        pred = binary_thin(pred)

    if gt.any():
        if kill_internal:
            assert isinstance(
                gt_seg, np.ndarray
            ), "ERR: `seg` is not np.ndarray"
            pred = kill_internal_prediction(
                pred=pred,
                gt=gt,
                seg=gt_seg,
                max_dist=max_dist,
            )

        match1, match2, cost, oc = correspond_pixels(
            pred, gt, max_dist=max_dist
        )
        match1 = match1 > 0
        match2 = match2 > 0
        # Recall
        sum_r = gt.sum()
        count_r = match2.sum()

        # Precision
        # TODO: check if using pred after kill_internal is correct
        sum_p = pred.sum()
        count_p = match1.sum()
    else:
        sum_r = 0
        count_r = 0
        sum_p = pred.sum()
        count_p = 0

    return count_r, sum_r, count_p, sum_p


def evaluate_boundaries_threshold(
    thresholds: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    gt_seg: Optional[np.ndarray] = None,
    max_dist: float = 0.02,  # 0.0075
    apply_thinning: bool = True,
    kill_internal: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the accuracy of a predicted boundary and a range of thresholds

    :param pred: the predicted boundaries as a (H,W)
        floating point array where each pixel represents the strength of the
        predicted boundary
    :param gt: ground truth boundary, as returned
        by the `load_boundary` or `boundary` methods
    :param thresholds: a 1D array specifying the thresholds
    :param max_dist: (default=0.02) maximum distance parameter
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

    NOTE: compared to BSDS500, we don't have multiple GTs
    """
    # Handle thresholds
    if isinstance(thresholds, int):
        thresholds = np.linspace(
            1.0 / (thresholds + 1), 1.0 - 1.0 / (thresholds + 1), thresholds
        )
    elif isinstance(thresholds, np.ndarray):
        if thresholds.ndim != 1:
            raise ValueError(
                "thresholds array should have 1 dimension, "
                "not {}".format(thresholds.ndim)
            )
        pass
    else:
        raise ValueError(
            "thresholds should be an int or a NumPy array, not "
            "a {}".format(type(thresholds))
        )

    sum_p = np.zeros(thresholds.shape)
    count_p = np.zeros(thresholds.shape)
    sum_r = np.zeros(thresholds.shape)
    count_r = np.zeros(thresholds.shape)

    for i_t, thresh in enumerate(list(thresholds)):

        _pred = pred >= thresh

        if apply_thinning:
            _pred = binary_thin(_pred)

        # skip correspond pixels when gt is empty
        if gt.any():
            if kill_internal:
                assert isinstance(
                    gt_seg, np.ndarray
                ), "ERR: `seg` is not np.ndarray"
                _pred = kill_internal_prediction(
                    pred=_pred,
                    gt=gt,
                    seg=gt_seg,
                    max_dist=max_dist,
                )

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
            sum_p[i_t] = _pred.sum()
            count_p[i_t] = 0

    return count_r, sum_r, count_p, sum_p


def evaluate_single_sample(
    sample,
    category,
    func_load_pred,
    func_load_gt,
    thresholds: Optional[np.ndarray] = None,
    max_dist: float = 0.0075,
    apply_thinning: bool = True,
    kill_internal: bool = False,
):
    assert isinstance(category, int)
    cat_pred = func_load_pred(sample, category=category)
    gt, seg, present_categories = func_load_gt(sample)

    assert len(gt.shape) == 3
    cat_gt = gt[category - 1, :, :]  # 0 indexed

    if kill_internal:
        seg = seg == category  # 0 is background
    else:
        seg = None

    # reduce scale

    if thresholds is None:
        count_r, sum_r, count_p, sum_p = evaluate_boundaries_bin(
            pred=cat_pred,
            gt=cat_gt,
            gt_seg=seg,
            max_dist=max_dist,
            apply_thinning=apply_thinning,
            kill_internal=kill_internal,
        )
    else:
        count_r, sum_r, count_p, sum_p = evaluate_boundaries_threshold(
            thresholds=thresholds,
            pred=cat_pred,
            gt=cat_gt,
            gt_seg=seg,
            max_dist=max_dist,
            apply_thinning=apply_thinning,
            kill_internal=kill_internal,
        )

    return count_r, sum_r, count_p, sum_p
