#!/usr/bin/env python3

import argparse
import os
from functools import partial

import numpy as np

from pyEdgeEval.datasets.sbd.evaluate import per_category_pr_evaluation
from pyEdgeEval.datasets.sbd.utils import (
    load_instance_insensitive_gt,
    load_instance_sensitive_gt,
    save_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SBD output")
    parser.add_argument(
        "sbd_path", type=str, help="the root path of the SBD dataset",
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
        "--category",
        type=int,
        help="the category number to evaluate",
    )
    parser.add_argument(
        "--inst-sensitive",
        action="store_true",
        help="instance sensitive",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="option to remove the thinning process (i.e. uses raw predition)",
    )
    parser.add_argument(
        "--kill-internal",
        action="store_true",
        help="kill internal contour",
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

    return parser.parse_args()


def parse_sbd_sample_names(data_root: str, split: str):

    file_path = os.path.join(data_root, f'{split}.txt')
    with open(file_path, 'r') as f:
        sample_names = f.read().splitlines()

    return sample_names


def load_gt_boundaries(sample_name: str, data_root: str, instance_sensitive: bool = False):
    if instance_sensitive:
        cls_path = os.path.join(
            data_root, "datadir", "cls", f"{sample_name}.mat"
        )
        inst_path = os.path.join(
            data_root, "datadir", "inst", f"{sample_name}.mat"
        )
        return load_instance_sensitive_gt(cls_path=cls_path, inst_path=inst_path)
    else:
        gt_path = os.path.join(
            data_root, "datadir", "cls", f"{sample_name}.mat"
        )
        return load_instance_insensitive_gt(gt_path)


def load_pred(sample_name: str, category: int, data_root: str):
    """Load prediction

    Since loading predictions using image extensions is complicated, we need a custom loader

    """
    ...


def evaluate_sbd(
    sbd_path: str,
    pred_path: str,
    output_path: str,
    category: int,
    instance_sensitive: bool,
    apply_thinning: bool,
    kill_internal: bool,
    thresholds: str,
    nproc: int,
):
    """Evaluate SBD"""
    assert os.path.exists(sbd_path), f"{sbd_path} doesn't exist"
    assert os.path.exists(pred_path), f"{pred_path} doesn't exist"
    assert os.path.exists(output_path), f"{output_path} doesn't exist"

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

    sample_names = parse_sbd_sample_names(data_root=sbd_path, split='val')

    _load_gt_boundaries = partial(load_gt_boundaries, data_root=sbd_path, instance_sensitive=instance_sensitive)
    _load_pred = partial(load_pred, category=category, data_root=pred_path),

    (sample_results, threshold_results, overall_result,) = per_category_pr_evaluation(
        thresholds=thresholds,
        category=category,
        sample_names=sample_names,
        load_gt=_load_gt_boundaries,
        load_pred=_load_pred,
        apply_thinning=apply_thinning,
        kill_internal=kill_internal,
        nproc=nproc,
    )

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

    # save the results
    save_results(
        path=output_path,
        category=category,
        sample_results=sample_results,
        threshold_results=threshold_results,
        overall_result=overall_result,
    )


def main():
    args = parse_args()

    evaluate_sbd(
        sbd_path=args.bsds_path,
        pred_path=args.pred_path,
        output_path=args.output_path,
        category=args.category,
        instance_sensitive=args.inst_sensitive,
        apply_thinning=not args.raw,
        kill_internal=args.kill_internal,
        thresholds=args.thresholds,
        nproc=args.nproc,
    )


if __name__ == "__main__":
    main()
