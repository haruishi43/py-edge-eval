#!/usr/bin/env python3

import numpy as np

from pyEdgeEval.common.metrics import (
    compute_rec_prec_f1,
    interpolated_max_scores,
)
from pyEdgeEval.common.utils import check_thresholds
from pyEdgeEval.utils import (
    track_parallel_progress,
    track_progress,
)

__all__ = ["calculate_metrics"]


def calculate_metrics(
    eval_single,
    thresholds,
    samples,
    nproc=8,
):
    """Main function to calculate boundary metrics

    Args:
        eval_single (Callable): function that takes samples (dict) as input
        threhsolds (int, float, list, np.ndarray): thresholds used for evaluation
        samples (dict): list of dicts containing sample info
        nproc (int): integer that specifies the number of processes to spawn

    Returns:
        dict of metrics
    """

    # initial run (process heavy)
    if nproc > 1:
        sample_metrics = track_parallel_progress(
            eval_single,
            samples,
            nproc=nproc,
            keep_order=True,
        )
    else:
        sample_metrics = track_progress(
            eval_single,
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

    # OIS scores
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

        # best_thresh, best_rec, best_prec, best_f1 = interpolated_max_scores(thresholds, rec, prec)

        # Find best F1 score
        best_ndx = np.argmax(f1)

        # Gather OIS metrics

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
