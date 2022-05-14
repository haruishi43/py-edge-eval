#!/usr/bin/env python3

"""
Verify the functionality of the evaluation suite.

Executes the evaluation procedure against five sampels and outputs the
results. Compare them with the results from the BSDS dataset to verify
that this Python port works properly.

The outputs of this script should be the same with the outputs in `test_2`
(`BSR/bench/data`). The slight difference in the matches might be due to
how MATLAB and Python handles data types (unsure).
"""

import argparse
import os
from functools import partial

from pyEdgeEval.bsds.evaluate import pr_evaluation
from pyEdgeEval.bsds.utils import (
    load_bsds_gt_boundaries,
    load_predictions,
    save_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify the BSDS-500 boundary " "evaluation suite"
    )
    parser.add_argument(
        "bench_path",
        type=str,
        help="the root path of the BSDS-500 benchmark",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="directory to output the results",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="number of parallel processes",
    )
    return parser.parse_args()


def load_gt_boundaries(sample_name: str, bench_dir_path: str):
    gt_path = os.path.join(bench_dir_path, "groundTruth", f"{sample_name}.mat")
    return load_bsds_gt_boundaries(gt_path)  # List[np.ndarray]


def load_pred(sample_name: str, bench_dir_path: str):
    pred_path = os.path.join(bench_dir_path, "png", f"{sample_name}.png")
    return load_predictions(pred_path)  # np.ndarray(dtype=float)


def test(bench_dir_path: str, output_dir_path: str, nproc: int):
    SAMPLE_NAMES = ["2018", "3063", "5096", "6046", "8068"]
    N_THRESHOLDS = 5

    assert os.path.exists(bench_dir_path), f"{bench_dir_path} doesn't exist"
    assert os.path.exists(output_dir_path), f"{output_dir_path} doesn't exist"

    (sample_results, threshold_results, overall_result,) = pr_evaluation(
        N_THRESHOLDS,
        SAMPLE_NAMES,
        partial(load_gt_boundaries, bench_dir_path=bench_dir_path),
        partial(load_pred, bench_dir_path=bench_dir_path),
        nproc=nproc,
    )

    print("Per image:")
    for sample_index, res in enumerate(sample_results):
        print(
            "{:<10d} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}".format(
                sample_index + 1,
                res.threshold,
                res.recall,
                res.precision,
                res.f1,
            )
        )

    print("")
    print("Overall:")
    for thresh_i, res in enumerate(threshold_results):
        print(
            "{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}".format(
                res.threshold, res.recall, res.precision, res.f1
            )
        )

    print("")
    print(
        "Summary: (threshold, recall, precision, f1, best recall, best precision, best f1, Area under PR"
    )
    print(
        "{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}"
        "{:<10.6f}".format(
            overall_result.threshold,
            overall_result.recall,
            overall_result.precision,
            overall_result.f1,
            overall_result.best_recall,
            overall_result.best_precision,
            overall_result.best_f1,
            overall_result.area_pr,
        )
    )

    # save the results
    save_results(
        path=output_dir_path,
        sample_results=sample_results,
        threshold_results=threshold_results,
        overall_result=overall_result,
    )


def main():
    args = parse_args()
    test(
        bench_dir_path=args.bench_path,
        output_dir_path=args.output_dir,
        nproc=args.nproc,
    )


if __name__ == "__main__":
    main()
