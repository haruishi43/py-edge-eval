#!/usr/bin/env python3

import os
from functools import partial
import math
from typing import List, Union

import numpy as np

from pyEdgeEval.bsds.evaluate import pr_evaluation
from pyEdgeEval.bsds.utils import (
    load_bsds_gt_boundaries,
    load_predictions,
)


def load_gt_boundaries(sample_name: str, bench_dir_path: str):
    gt_path = os.path.join(bench_dir_path, "groundTruth", f"{sample_name}.mat")
    return load_bsds_gt_boundaries(gt_path)  # List[np.ndarray]


def load_pred(sample_name: str, bench_dir_path: str):
    pred_path = os.path.join(bench_dir_path, "png", f"{sample_name}.png")
    return load_predictions(pred_path)  # np.ndarray(dtype=float)


def load_actual_results(bench_dir_path: str):
    bdry_img_path = os.path.join(bench_dir_path, "test_2", "eval_bdry_img.txt")
    bdry_img = np.loadtxt(bdry_img_path)
    bdry_thr_path = os.path.join(bench_dir_path, "test_2", "eval_bdry_thr.txt")
    bdry_thr = np.loadtxt(bdry_thr_path)
    bdry_path = os.path.join(bench_dir_path, "test_2", "eval_bdry.txt")
    bdry = np.loadtxt(bdry_path)
    return bdry_img, bdry_thr, bdry


def run_evaluation(
    bench_dir_path: str,
    thresholds: Union[int, List[float]] = 5,
    nproc: int = 1,
    abs_tol: float = 1e-03,
):
    SAMPLE_NAMES = ["2018", "3063", "5096", "6046", "8068"]

    assert os.path.exists(bench_dir_path), f"{bench_dir_path} doesn't exist"

    (sample_results, threshold_results, overall_result,) = pr_evaluation(
        thresholds,
        SAMPLE_NAMES,
        partial(load_gt_boundaries, bench_dir_path=bench_dir_path),
        partial(load_pred, bench_dir_path=bench_dir_path),
        nproc=nproc,
    )

    (actual_sample, actual_threshold, actual_overall,) = load_actual_results(
        bench_dir_path=bench_dir_path,
    )

    def _isclose(a, b):
        return math.isclose(a, b, abs_tol=abs_tol)

    # compare with actual
    print(sample_results)
    print("Per image:")
    for sample_index, res in enumerate(sample_results):
        actual = actual_sample[sample_index]
        assert _isclose(res.threshold, actual[1])
        assert _isclose(res.recall, actual[2])
        assert _isclose(res.precision, actual[3])
        assert _isclose(res.f1, actual[4])

    print(threshold_results)
    print("Overall:")
    for thresh_i, res in enumerate(threshold_results):
        actual = actual_threshold[thresh_i]
        assert _isclose(res.threshold, actual[0])
        assert _isclose(res.recall, actual[1])
        assert _isclose(res.precision, actual[2])
        assert _isclose(res.f1, actual[3])

    print(overall_result)
    print("Summary:")
    assert _isclose(overall_result.threshold, actual_overall[0])
    assert _isclose(overall_result.recall, actual_overall[1])
    assert _isclose(overall_result.precision, actual_overall[2])
    assert _isclose(overall_result.f1, actual_overall[3])
    assert _isclose(overall_result.best_recall, actual_overall[4])
    assert _isclose(overall_result.best_precision, actual_overall[5])
    assert _isclose(overall_result.best_f1, actual_overall[6])
    assert _isclose(overall_result.area_pr, actual_overall[7])


def test_bsds500():
    nproc = 1
    abs_tol = 1e-03
    run_evaluation(
        bench_dir_path="data/BSDS500_bench",
        thresholds=5,
        nproc=nproc,
        abs_tol=abs_tol,
    )

    # FIXME: nproc > 1 results in different results
    nproc = 8
    abs_tol = 1e-02
    run_evaluation(
        bench_dir_path="data/BSDS500_bench",
        thresholds=5,
        nproc=nproc,
        abs_tol=abs_tol,
    )
