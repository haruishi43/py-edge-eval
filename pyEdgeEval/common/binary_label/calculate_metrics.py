#!/usr/bin/env python3

import numpy as np
from skimage.util import img_as_float
from skimage.io import imread

from pyEdgeEval.common.metrics import (
    compute_rec_prec_f1,
    interpolated_max_scores,
)
from pyEdgeEval.common.binary_label.evaluate_boundaries import (
    evaluate_boundaries_threshold,
)
from pyEdgeEval.common.utils import check_thresholds
from pyEdgeEval.utils import (
    track_parallel_progress,
    track_progress,
    loadmat,
)

__all__ = ["calculate_metrics"]


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
    gts = load_bsds_gt_boundaries(gt_path)

    # TODO: scale inputs

    # evaluate multi-label boundaries
    count_r, sum_r, count_p, sum_p = evaluate_boundaries_threshold(
        thresholds=thresholds,
        pred=pred,
        gts=gts,
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


def wrapper_eval_single(kwargs):
    """Wrapper function to unpack all the kwargs"""
    return _evaluate_single(**kwargs)


def calculate_metrics(
    thresholds,
    samples,
    nproc=8,
):
    """Main function to calculate boundary metrics

    :param thresholds: thresholds used for evaluation which can be in the form of
        (int, float, list, np.ndarray)
    :param samples: list of dicts containing file paths and evaluation parameters
    :param nproc: integer that specifies the number of processes to spawn

    :return: tuple of results
    """

    # initial run (process heavy)
    if nproc > 1:
        sample_metrics = track_parallel_progress(
            wrapper_eval_single,
            samples,
            nproc=nproc,
            keep_order=True,
        )
    else:
        sample_metrics = track_progress(
            wrapper_eval_single,
            samples,
        )

    # check and convert
    thresholds = check_thresholds(thresholds)

    # initialize array
    n_thresh = thresholds.shape[0]
    count_r_overall = np.zeros((n_thresh,))
    sum_r_overall = np.zeros((n_thresh,))
    count_p_overall = np.zeros((n_thresh,))
    sum_p_overall = np.zeros((n_thresh,))

    count_r_best = 0
    sum_r_best = 0
    count_p_best = 0
    sum_p_best = 0

    # calculate metrics
    sample_results = []
    for sample_index, sample_data in enumerate(samples):
        count_r, sum_r, count_p, sum_p = sample_metrics[sample_index]

        count_r_overall += count_r
        sum_r_overall += sum_r
        count_p_overall += count_p
        sum_p_overall += sum_p

        # Compute precision, recall and F1
        rec, prec, f1 = compute_rec_prec_f1(count_r, sum_r, count_p, sum_p)

        # Gather OIS metrics

        # Find best F1 score
        best_ndx = np.argmax(f1)

        count_r_best += count_r[best_ndx]
        sum_r_best += sum_r[best_ndx]
        count_p_best += count_p[best_ndx]
        sum_p_best += sum_p[best_ndx]

        sample_results.append(
            dict(
                name=sample_data["name"],
                threshold=thresholds[best_ndx],
                recall=rec[best_ndx],
                precision=prec[best_ndx],
                f1=f1[best_ndx],
            )
        )

    # Computer overall precision, recall and F1
    rec_overall, prec_overall, f1_overall = compute_rec_prec_f1(
        count_r_overall, sum_r_overall, count_p_overall, sum_p_overall
    )

    # Interpolated way to find ODS scores
    best_threshold, best_rec, best_prec, best_f1 = interpolated_max_scores(
        thresholds, rec_overall, prec_overall
    )

    # Find best F1 score
    # best_i_ovr = np.argmax(f1_overall)

    threshold_results = []
    for thresh_i in range(n_thresh):
        threshold_results.append(
            dict(
                threshold=thresholds[thresh_i],
                recall=rec_overall[thresh_i],
                precision=prec_overall[thresh_i],
                f1=f1_overall[thresh_i],
            )
        )

    # Calculate AUC
    prec_inc = 0.01  # hard-coded
    rec_unique, rec_unique_ndx = np.unique(rec_overall, return_index=True)
    prec_unique = prec_overall[rec_unique_ndx]
    if rec_unique.shape[0] > 1:
        prec_interp = np.interp(
            np.arange(0, 1, prec_inc),
            rec_unique,
            prec_unique,
            left=0.0,
            right=0.0,
        )
        area_pr = prec_interp.sum() * prec_inc
    else:
        area_pr = 0.0

    # Calculate AP
    ap = 0
    for t in np.arange(0, 1, 0.01):
        _r = rec_overall >= t
        p = np.max(prec_overall[_r], initial=0)
        ap = ap + p / 101

    # Calculate OIS metrics
    rec_best, prec_best, f1_best = compute_rec_prec_f1(
        float(count_r_best),
        float(sum_r_best),
        float(count_p_best),
        float(sum_p_best),
    )

    overall_result = dict(
        ODS_threshold=best_threshold,
        ODS_recall=best_rec,
        ODS_precision=best_prec,
        ODS_f1=best_f1,
        OIS_recall=rec_best,
        OIS_precision=prec_best,
        OIS_f1=f1_best,
        AUC=area_pr,
        AP=ap,
    )

    return sample_results, threshold_results, overall_result
