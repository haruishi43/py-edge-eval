#!/usr/bin/env python3

import os

import numpy as np
from PIL import Image

from pyEdgeEval.utils import mkdir_or_exist

from .edge_encoding import rgb_decoding


def load_gt(
    edge_path: str,
    seg_path: str,
    num_trainIds: int = 19,
):
    # get edge map
    edge = rgb_decoding(
        edge_path=edge_path,
        num_trainIds=num_trainIds,
        is_png=True,
    )

    # get segmentation mask (trainIds)
    seg = np.array(Image.open(seg_path))

    # NOTE: won't use
    present_categories = None

    return edge, seg, present_categories


def save_results(
    path: str,
    category: int,
    sample_results,
    threshold_results,
    overall_result,
):
    """Save results as BSDS500 format"""
    assert os.path.exists(path), f"ERR: {path} doesn't exist"

    # FIXME: change the name of the output file so that we know the category

    cat_name = "class_" + str(category).zfill(3)

    cat_dir = os.path.join(path, cat_name)
    mkdir_or_exist(cat_dir)

    # save per sample results
    tmp_line = "{i:<10d} {thrs:<10.6f} {rec:<10.6f} {prec:<10.6f} {f1:<10.6f}\n"
    with open(os.path.join(cat_dir, "eval_bdry_img.txt"), "w") as f:
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
    with open(os.path.join(cat_dir, "eval_bdry_thr.txt"), "w") as f:
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
    with open(os.path.join(cat_dir, "eval_bdry.txt"), "w") as f:
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
