#!/usr/bin/env python3

import argparse
import os
from functools import partial

import numpy as np
from PIL import Image

from pyEdgeEval.datasets.cityscapes.evaluate import per_category_pr_evaluation
from pyEdgeEval.datasets.cityscapes.utils import (
    load_gt,
    save_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Cityscapes output")
    parser.add_argument(
        "cityscapes_path", type=str, help="the root path of the cityscapes dataset",
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
        "--pre-seal",
        action="store_true",
        help="prior to SEAL, the evaluations were not as strict",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="option to remove the thinning process (i.e. uses raw predition)",
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


def parse_cityscapes_sample_names(data_root: str, split: str):

    file_path = os.path.join(data_root, 'splits', f'{split}.txt')
    with open(file_path, 'r') as f:
        sample_names = f.read().splitlines()

    return sample_names


def load_gt_boundaries(
    sample_name: str,
    data_root: str,
    split: str = 'val',
    instance_sensitive: bool = True,
    gt_dir: str = 'gtEval',
):
    if instance_sensitive:
        edge_path = os.path.join(data_root, gt_dir, split, f"{sample_name}_gtProc_isedge.png")
        seg_path = os.path.join(data_root, gt_dir, split, f"{sample_name}_gtFine_labelTrainIds.png")
        assert os.path.exists(edge_path), f"ERR: {edge_path} is not valid"
        assert os.path.exists(seg_path), f"ERR: {seg_path} is not valid"
        return load_gt(edge_path=edge_path, seg_path=seg_path, num_trainIds=19)
    else:
        edge_path = os.path.join(data_root, gt_dir, split, f"{sample_name}_gtProc_edge.png")
        seg_path = os.path.join(data_root, gt_dir, split, f"{sample_name}_gtFine_labelTrainIds.png")
        assert os.path.exists(edge_path), f"ERR: {edge_path} is not valid"
        assert os.path.exists(seg_path), f"ERR: {seg_path} is not valid"
        return load_gt(edge_path=edge_path, seg_path=seg_path, num_trainIds=19)


def load_pred(
    sample_name: str,
    data_root: str,
    category: int,
    suffix: str = "_leftImg8bit.png",
):
    """Load prediction

    Since loading predictions using image extensions is complicated, we need a custom loader
    We assume that the predictions are located inside the `data_root` directory and
    the predictions are separated by categories.
    `{data_root}/{category}/<img>{suffix}`

    - The output is a single class numpy array.
    - instead of .bmp, it is better to save predictions as .png
    """
    sample_name = sample_name.split("/")[1]  # assert city/img
    pred_path = os.path.join(
        data_root,
        f"class_{str(category).zfill(3)}",
        f"{sample_name}{suffix}",
    )
    pred = np.array(Image.open(pred_path))
    pred = (pred / 255).astype(float)
    return pred


def evaluate_cityscapes(
    cityscapes_path: str,
    pred_path: str,
    output_path: str,
    category: int,
    suffix: str,
    pre_seal: bool,
    apply_thinning: bool,
    thresholds: str,
    nproc: int,
):
    """Evaluate Cityscapes"""
    assert os.path.exists(cityscapes_path), f"{cityscapes_path} doesn't exist"
    assert os.path.exists(pred_path), f"{pred_path} doesn't exist"
    assert os.path.exists(output_path), f"{output_path} doesn't exist"

    assert 0 < category < 21, f"category needs to be between 1 ~ 20, but got {category}"

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

    if pre_seal:
        max_dist = 0.02
        kill_internal = True
        instance_sensitive = False
    else:
        max_dist = 0.0035
        kill_internal = False
        instance_sensitive = True

    sample_names = parse_cityscapes_sample_names(data_root=cityscapes_path, split='val')

    _load_gt_boundaries = partial(
        load_gt_boundaries,
        data_root=cityscapes_path,
        split='val',
        instance_sensitive=instance_sensitive,
    )
    _load_pred = partial(
        load_pred,
        category=category,
        data_root=pred_path,
        suffix=suffix,
    )

    (sample_results, threshold_results, overall_result,) = per_category_pr_evaluation(
        thresholds=thresholds,
        category=category,
        sample_names=sample_names,
        load_gt=_load_gt_boundaries,
        load_pred=_load_pred,
        max_dist=max_dist,
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

    # might need to specify suffix
    suffix = "_leftImg8bit.png"

    apply_thinning = not args.raw

    evaluate_cityscapes(
        cityscapes_path=args.cityscapes_path,
        pred_path=args.pred_path,
        output_path=args.output_path,
        category=args.category,
        suffix=suffix,
        pre_seal=args.pre_seal,
        apply_thinning=apply_thinning,
        thresholds=args.thresholds,
        nproc=args.nproc,
    )


if __name__ == "__main__":
    main()
