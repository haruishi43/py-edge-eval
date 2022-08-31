#!/usr/bin/env python3

from typing import Optional

import numpy as np

from pyEdgeEval._lib import correspond_pixels
from pyEdgeEval.preprocess import binary_thin, fast_nms

from .options import kill_internal_prediction


def evaluate_boundaries_threshold(
    thresholds: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    gt_seg: Optional[np.ndarray] = None,
    max_dist: float = 0.02,  # 0.0075
    apply_thinning: bool = True,
    kill_internal: bool = False,
    skip_if_nonexistent: bool = False,
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

    Args:
        thresholds: a 1D array specifying the thresholds
        pred: the predicted boundaries as a (H,W) floating point array where
            each pixel represents the strength of the predicted boundary
        gt: ground truth boundary, as returned by the `load_boundary` or
            `boundary` methods
        gt_seg: ground truth segmentation data needed for `kill_internal`
        max_dist: (default=0.02) maximum distance parameter used for
            determining pixel matches. This value is multiplied by the
            length of the diagonal of the image to get the threshold used
            for matching pixels.
        apply_thinning: (default=True) if True, apply morphologial
            thinning to the predicted boundaries before evaluation
        kill_internal: (default=True) remove countors inside the
            segmentation mask
        skip_if_nonexistent: (default=True) this will skip the evaluation and
            disregards all false positives if there are no boundaries in the GT
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

    NOTE: compared to BSDS500, we don't have multiple GTs
    """

    sum_p = np.zeros(thresholds.shape)
    count_p = np.zeros(thresholds.shape)
    sum_r = np.zeros(thresholds.shape)
    count_r = np.zeros(thresholds.shape)

    if skip_if_nonexistent:
        # skip when category is not present in GT (for pre-seal)
        if np.all(gt == 0):
            return count_r, sum_r, count_p, sum_p

    if apply_nms:
        pred = fast_nms(
            img=pred,
            **nms_kwargs,
        )

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
            sum_p[i_t] = _pred.sum()  # keep track of false positives
            count_p[i_t] = 0

    return count_r, sum_r, count_p, sum_p
