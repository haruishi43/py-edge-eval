#!/usr/bin/env python3

import os
import argparse
from functools import partial

import numpy as np

from pyEdgeEval.datasets.bsds.evaluate import pr_evaluation
from pyEdgeEval.datasets.bsds.utils import (
    load_bsds_gt_boundaries,
    load_predictions,
    save_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BSDS output")
    parser.add_argument(
        "bsds_path", type=str, help="the root path of the BSDS-500 dataset",
    )
    parser.add_argument(
        "pred_path", type=str, help="the root path of the predictions",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="the root path of where the results are populated",
    )
    parser.add_argument(
        "--use-val",
        action="store_true",
        help="val or test",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="99",
        help="the number of thresholds (could be a list of floats); use 99 for eval",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="the number of parallel threads",
    )
    parser.add_argument(
        "--pred-suffix",
        type=str,
        default=".png",
        help="suffix and extension",
    )

    return parser.parse_args()


def parse_bsds500_sample_names(data_root: str, split: str):
    dir = os.path.join(data_root, "images")
    assert os.path.exists(dir), f"ERR: {dir} does not exist"
    names = []
    files = os.listdir(os.path.join(dir, split))
    for fn in files:
        dir, filename = os.path.split(fn)
        name, ext = os.path.splitext(filename)
        if ext.lower() == ".jpg":
            names.append(os.path.join(split, name))
    return names


def load_gt_boundaries(sample_name: str, bsds_dir_path: str):
    gt_path = os.path.join(
        bsds_dir_path, "groundTruth", f"{sample_name}.mat"
    )
    return load_bsds_gt_boundaries(gt_path)  # List[np.ndarray]


def load_pred(
    sample_name: str,
    pred_dir_path: str,
    suffix: str = ".png",
    ensure_same_size_func=None,
):
    pred_path = os.path.join(
        pred_dir_path, f"{sample_name}{suffix}"
    )
    pred = load_predictions(pred_path)  # np.ndarray(dtype=float)

    # FIXME: the shapes are different?
    if ensure_same_size_func is not None:
        gt = ensure_same_size_func(sample_name)
        gt_shape = gt[0].shape
        pred = pred[:gt_shape[0], :gt_shape[1]]
        pred = np.pad(
            pred,
            [(0, gt_shape[0] - pred.shape[0]), (0, gt_shape[1] - pred.shape[1])],
            mode="constant",
        )

    return pred


def evaluate_bsds500(
    bsds_path: str,
    pred_path: str,
    output_path: str,
    pred_suffix: str,
    use_val: bool,
    thresholds: str,
    nproc: int,
):
    """Evaluate BSDS500"""
    assert os.path.exists(bsds_path), f"ERR: {bsds_path} doesn't exist"
    assert os.path.exists(pred_path), f"ERR: {pred_path} doesn't exist"
    assert os.path.exists(output_path), f"ERR: {output_path} doesn't exist"

    thresholds = thresholds.strip()
    try:
        n_thresholds = int(thresholds)
        thresholds = n_thresholds
    except ValueError:
        try:
            if thresholds.startswith("[") and thresholds.endswith("]"):
                thresholds = thresholds[1:-1]
                thresholds = np.array(
                    [float(t.strip()) for t in thresholds.split(",")]
                )
            else:
                print(
                    "Bad threshold format; should be a python list of floats (`[a, b, c]`)"
                )
                return
        except ValueError:
            print(
                "Bad threshold format; should be a python list of ints (`[a, b, c]`)"
            )
            return

    # sample name ('split/sample')
    if use_val:
        sample_names = parse_bsds500_sample_names(data_root=bsds_path, split="val")
    else:
        sample_names = parse_bsds500_sample_names(data_root=bsds_path, split="test")

    _load_gt_boundaries = partial(load_gt_boundaries, bsds_dir_path=bsds_path)
    _load_pred = partial(load_pred, pred_dir_path=pred_path, suffix=pred_suffix)

    (
        sample_results,
        threshold_results,
        overall_result,
    ) = pr_evaluation(
        thresholds=thresholds,
        sample_names=sample_names,
        load_gts=_load_gt_boundaries,
        load_pred=_load_pred,
        nproc=nproc,
    )

    # print("Per image:")
    # for sample_index, res in enumerate(sample_results):
    #     print(
    #         "{:<10d} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}".format(
    #             sample_index + 1, res.threshold, res.recall, res.precision, res.f1
    #         )
    #     )

    # print("")
    # print("Per threshold:")
    # for thresh_i, res in enumerate(threshold_results):
    #     print(
    #         "{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}".format(
    #             res.threshold, res.recall, res.precision, res.f1
    #         )
    #     )

    print("")
    print("Summary:")
    print(
        "{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}".format(
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

    # save results
    save_results(
        path=output_path,
        sample_results=sample_results,
        threshold_results=threshold_results,
        overall_result=overall_result,
    )


def main():
    args = parse_args()

    evaluate_bsds500(
        bsds_path=args.bsds_path,
        pred_path=args.pred_path,
        output_path=args.output_path,
        pred_suffix=args.pred_suffix,
        use_val=args.use_val,
        thresholds=args.thresholds,
        nproc=args.nproc,
    )


if __name__ == "__main__":
    main()
