#!/usr/bin/env python3

import os

import numpy as np

from skimage.util import img_as_float
from skimage.io import imread

from pyEdgeEval.utils import loadmat


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
    assert os.path.exists(path), f"ERR: cannot load {path}"
    img = imread(path)
    assert (
        img.dtype == np.uint8
    ), f"ERR: img needs to be uint8, but got{img.dtype}"
    return img_as_float(img)


def save_results(
    path: str,
    sample_results,
    threshold_results,
    overall_result,
):
    """Save results as BSDS500 format"""
    assert os.path.exists(path), f"ERR: {path} doesn't exist"

    # save per sample results
    tmp_line = "{i:<10d} {thrs:<10.6f} {rec:<10.6f} {prec:<10.6f} {f1:<10.6f}\n"
    with open(os.path.join(path, "eval_bdry_img.txt"), "w") as f:
        for i, res in enumerate(sample_results):
            f.write(
                tmp_line.format(
                    i=i,
                    thrs=res.threshold,
                    rec=res.recall,
                    prec=res.precision,
                    f1=res.f1,
                )
            )

    # save per threshold results
    tmp_line = "{thrs:<10.6f} {rec:<10.6f} {prec:<10.6f} {f1:<10.6f}\n"
    with open(os.path.join(path, "eval_bdry_thr.txt"), "w") as f:
        for res in threshold_results:
            f.write(
                tmp_line.format(
                    thrs=res.threshold,
                    rec=res.recall,
                    prec=res.precision,
                    f1=res.f1,
                )
            )

    # save summary results
    with open(os.path.join(path, "eval_bdry.txt"), "w") as f:
        f.write(
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
