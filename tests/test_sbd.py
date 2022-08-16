#!/usr/bin/env python3

"""Test SBD

- SBD_bench consists of single category evaluation (class 14 (human))
"""

import os.path as osp
import math
from typing import List, Union

import numpy as np

from pyEdgeEval.evaluators import SBDEvaluator


def load_actual_results(bench_dir_path: str):
    # TODO: might also want to compare "_ev1.txt" files
    bdry_img_path = osp.join(bench_dir_path, "outdir", "eval_bdry_img.txt")
    bdry_img = np.loadtxt(bdry_img_path)
    bdry_thr_path = osp.join(bench_dir_path, "outdir", "eval_bdry_thr.txt")
    bdry_thr = np.loadtxt(bdry_thr_path)
    bdry_path = osp.join(bench_dir_path, "outdir", "eval_bdry.txt")
    bdry = np.loadtxt(bdry_path)
    return bdry_img, bdry_thr, bdry


def run_evaluation(
    bench_dir_path: str,
    thresholds: Union[int, List[float]] = 99,
    nproc: int = 1,
    abs_tol: float = 1e-03,
):
    SAMPLE_NAMES = ["2008_000051", "2008_000195"]
    CATEGORY = 15  # human

    assert osp.exists(bench_dir_path), f"{bench_dir_path} doesn't exist"

    evaluator = SBDEvaluator(
        dataset_root=osp.join(bench_dir_path, "datadir"),
        pred_root=osp.join(bench_dir_path, "indir"),
    )

    # force
    evaluator.set_sample_names(sample_names=SAMPLE_NAMES)

    evaluator.set_eval_params(
        eval_mode="pre-seal",
        apply_thinning=True,
        instance_sensitive=False,
    )

    overall_metric = evaluator.evaluate_category(
        category=CATEGORY,
        thresholds=thresholds,
        nproc=nproc,
        save_dir=None,
    )

    (actual_sample, actual_threshold, actual_overall,) = load_actual_results(
        bench_dir_path=bench_dir_path,
    )

    def _isclose(a, b):
        return math.isclose(a, b, abs_tol=abs_tol)

    # compare with actual
    # print(sample_results)
    # print("Per image:")
    # for sample_index, res in enumerate(sample_results):
    #     actual = actual_sample[sample_index]
    #     assert _isclose(res.threshold, actual[1])
    #     assert _isclose(res.recall, actual[2])
    #     assert _isclose(res.precision, actual[3])
    #     assert _isclose(res.f1, actual[4])

    # print(threshold_results)
    # print("Overall:")
    # for thresh_i, res in enumerate(threshold_results):
    #     actual = actual_threshold[thresh_i]
    #     assert _isclose(res.threshold, actual[0])
    #     assert _isclose(res.recall, actual[1])
    #     assert _isclose(res.precision, actual[2])
    #     assert _isclose(res.f1, actual[3])

    print(overall_metric)
    print("Summary:")
    assert _isclose(overall_metric["ODS_threshold"], actual_overall[0])
    assert _isclose(overall_metric["ODS_recall"], actual_overall[1])
    assert _isclose(overall_metric["ODS_precision"], actual_overall[2])
    assert _isclose(overall_metric["ODS_f1"], actual_overall[3])
    assert _isclose(overall_metric["OIS_recall"], actual_overall[4])
    assert _isclose(overall_metric["OIS_precision"], actual_overall[5])
    assert _isclose(overall_metric["OIS_f1"], actual_overall[6])
    assert _isclose(overall_metric["AUC"], actual_overall[7])


def test_sbd():
    nproc = 1
    abs_tol = 1e-02
    run_evaluation(
        bench_dir_path="data/SBD_bench",
        thresholds=99,
        nproc=nproc,
        abs_tol=abs_tol,
    )

    # FIXME: nproc > 1 results in different results
    nproc = 8
    abs_tol = 1e-02
    run_evaluation(
        bench_dir_path="data/SBD_bench",
        thresholds=99,
        nproc=nproc,
        abs_tol=abs_tol,
    )
