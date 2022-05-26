#!/usr/bin/env python3

from typing import Any, Callable, List, Tuple, Union

import numpy as np

from .metrics import (
    OverallResult,
    OverallResultSBD,
    SampleResult,
    SingleResult,
    ThresholdResult,
    compute_rec_prec_f1,
)
from pyEdgeEval.utils import (
    track_parallel_progress,
    track_progress,
)


def base_evaluation_wo_threshold(
    samples: List[Any],
    wrapper: Callable[[Any], Any],
    nproc: int = 8,
) -> SingleResult:
    """
    Perform an evaluation of predictions against ground truths for an image

    This evaluation assumes that the predictions are already binary
    """
    # initial run (process heavy)
    if nproc > 1:
        sample_data = track_parallel_progress(
            wrapper,
            samples,
            nproc=nproc,
            keep_order=True,
        )
    else:
        sample_data = track_progress(
            wrapper,
            samples,
        )

    # initialize variables
    overall_count_r = 0
    overall_sum_r = 0
    overall_count_p = 0
    overall_sum_p = 0

    # FIXME: map reduce?
    for count_r, sum_r, count_p, sum_p in sample_data:
        overall_count_r += count_r
        overall_sum_r += sum_r
        overall_count_p += count_p
        overall_sum_p += sum_p

    # Computer overall precision, recall and F1
    rec, prec, f1 = compute_rec_prec_f1(
        overall_count_r, overall_sum_r, overall_count_p, overall_sum_p
    )

    # NOTE: no precision-recall curve and AP (area-under-curve)

    return SingleResult(rec, prec, f1)


def base_pr_evaluation(
    thresholds: np.ndarray,
    samples: List[Any],
    wrapper: Callable[[Any], Any],
    nproc: int = 8,
    is_sbd: bool = False,
) -> Tuple[
    List[SampleResult],
    List[ThresholdResult],
    Union[OverallResult, OverallResultSBD],
]:
    """
    Perform an evaluation of predictions against ground truths for an image
    set over a given set of thresholds.

    :param samples: the names of the samples that are to be evaluated
    :param thresholds: a 1D array specifying the thresholds
    :param wrapper: a callable that returns evaluation data given sample
    :return: `(sample_results, threshold_results, overall_result)`
        where `sample_results` is a list of `SampleResult` named tuples with one
        for each sample, `threshold_results` is a list of `ThresholdResult`
        named tuples, with one for each threshold and `overall_result`
        is an `OverallResult` named tuple giving the over all results. The
        attributes in these structures will now be described:

    `SampleResult`:
    - `sample_name`: the name identifying the sample to which this result
        applies
    - `threshold`: the threshold at which the best F1-score was obtained for
        the given sample
    - `recall`: the recall score obtained at the best threshold
    - `precision`: the precision score obtained at the best threshold
    - `f1`: the F1-score obtained at the best threshold

    `ThresholdResult`:
    - `threshold`: the threshold value to which this result applies
    - `recall`: the average recall score for all samples
    - `precision`: the average precision score for all samples
    - `f1`: the average F1-score for all samples

    `OverallResult`:
    - `threshold`: the threshold at which the best average F1-score over
        all samples is obtained
    - `recall`: the average recall score for all samples at `threshold`
    - `precision`: the average precision score for all samples at `threshold`
    - `f1`: the average F1-score for all samples at `threshold`
    - `best_recall`: the average recall score for all samples at the best
        threshold *for each individual sample*
    - `best_precision`: the average precision score for all samples at the
        best threshold *for each individual sample*
    - `best_f1`: the average F1-score for all samples at the best threshold
        *for each individual sample*
    - `area_pr`: the area under the precision-recall curve at `threshold`
    `
    """
    # initial run (process heavy)
    if nproc > 1:
        sample_data = track_parallel_progress(
            wrapper,
            samples,
            nproc=nproc,
            keep_order=True,
        )
    else:
        sample_data = track_progress(
            wrapper,
            samples,
        )

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
    for sample_index, sample_id in enumerate(samples):
        count_r, sum_r, count_p, sum_p = sample_data[sample_index]

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
            SampleResult(
                sample_id,
                thresholds[best_ndx],
                rec[best_ndx],
                prec[best_ndx],
                f1[best_ndx],
            )
        )

    # Computer overall precision, recall and F1
    rec_overall, prec_overall, f1_overall = compute_rec_prec_f1(
        count_r_overall, sum_r_overall, count_p_overall, sum_p_overall
    )

    # Find best F1 score
    best_i_ovr = np.argmax(f1_overall)

    threshold_results = []
    for thresh_i in range(n_thresh):
        threshold_results.append(
            ThresholdResult(
                thresholds[thresh_i],
                rec_overall[thresh_i],
                prec_overall[thresh_i],
                f1_overall[thresh_i],
            )
        )

    # Calculate AUC
    # FIXME: why do we use increments of 0.01?
    rec_unique, rec_unique_ndx = np.unique(rec_overall, return_index=True)
    prec_unique = prec_overall[rec_unique_ndx]
    if rec_unique.shape[0] > 1:
        prec_interp = np.interp(
            np.arange(0, 1, 0.01), rec_unique, prec_unique, left=0.0, right=0.0
        )
        area_pr = prec_interp.sum() * 0.01
    else:
        area_pr = 0.0

    # Calculate AP
    ap = 0
    for t in np.arange(0, 1, 0.01):
        _r = rec_overall >= t
        p = np.max(prec_overall[_r], initial=0)
        ap = ap + p / 101

    # Calculate ODS metrics

    rec_best, prec_best, f1_best = compute_rec_prec_f1(
        float(count_r_best),
        float(sum_r_best),
        float(count_p_best),
        float(sum_p_best),
    )

    if is_sbd:
        overall_result = OverallResultSBD(
            thresholds[best_i_ovr],  # ODS threshold
            rec_overall[best_i_ovr],  # ODS Recall
            prec_overall[best_i_ovr],  # ODS Precision
            f1_overall[best_i_ovr],  # ODS F
            rec_best,  # OIS Recall
            prec_best,  # OIS Precision
            f1_best,  # OIS F
            area_pr,  # AUC
            ap,
        )
    else:
        overall_result = OverallResult(
            thresholds[best_i_ovr],  # ODS threshold
            rec_overall[best_i_ovr],  # ODS Recall
            prec_overall[best_i_ovr],  # ODS Precision
            f1_overall[best_i_ovr],  # ODS F
            rec_best,  # OIS Recall
            prec_best,  # OIS Precision
            f1_best,  # OIS F
            area_pr,  # AUC
        )

    return sample_results, threshold_results, overall_result
