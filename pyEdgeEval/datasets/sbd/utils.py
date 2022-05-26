#!/usr/bin/env python3

import copy
import os

import numpy as np
from scipy.sparse import csc_matrix

from pyEdgeEval.utils import loadmat, mkdir_or_exist


def sparse2numpy(data: csc_matrix):
    """helper function to convert compressed sparse column matrix to numpy array"""
    assert isinstance(
        data, csc_matrix
    ), f"ERR: input is not csc_matrix, but got {type(data)}"
    return data.toarray()


def load_sbd_gt_cls_mat(path: str, new_loader: bool = False):
    """Load Per Class Ground Truth Annoation"""
    if new_loader:
        gt = loadmat(path, True)["GTcls"]
        boundaries = gt["Boundaries"]  # list[csc_matrix]
        segmentation = gt["Segmentation"]  # np.ndarray(h, w)
        present_categories = gt["CategoriesPresent"]  # np.ndarray()

        assert len(segmentation.shape) == 2
        h, w = segmentation.shape
        num_categories = len(boundaries)
        np_boundaries = np.zeros([num_categories, h, w], dtype=np.uint8)
        for cat in range(num_categories):
            np_boundaries[cat] = sparse2numpy(boundaries[cat])

    else:
        gt = loadmat(path, False)["GTcls"][0, 0]
        boundaries = gt[0]  # np.ndarray(np.ndarray(csc_matrix))
        segmentation = gt[1]  # np.ndarray(h, w)
        present_categories = gt[2].squeeze()  # np.ndarray()

        assert len(segmentation.shape) == 2
        h, w = segmentation.shape
        num_categories = len(boundaries)
        np_boundaries = np.zeros([num_categories, h, w], dtype=np.uint8)
        for cat in range(num_categories):
            np_boundaries[cat] = sparse2numpy(boundaries[cat][0])

    # TODO: do checks for present categories?
    return np_boundaries, segmentation, present_categories


def load_sbd_gt_inst_mat(path: str, new_loader: bool = False):
    """Load Per Instance Ground Truth Annotation

    NOTE: seg and bdry is indexed by instances (not category)
    """
    if new_loader:
        gt = loadmat(path, True)["GTinst"]
        segmentation = gt["Segmentation"]
        boundaries = gt["Boundaries"]
        categories = gt["Categories"]

        assert len(segmentation.shape) == 2
        h, w = segmentation.shape
        num_categories = len(boundaries)
        np_boundaries = np.zeros([num_categories, h, w], dtype=np.uint8)
        for cat in range(num_categories):
            np_boundaries[cat] = sparse2numpy(boundaries[cat])

    else:
        gt = loadmat(path, False)["GTinst"][0, 0]
        segmentation = gt[0]
        boundaries = gt[1]
        categories = gt[2].squeeze()

        assert len(segmentation.shape) == 2
        h, w = segmentation.shape
        num_categories = len(boundaries)
        np_boundaries = np.zeros([num_categories, h, w], dtype=np.uint8)
        for cat in range(num_categories):
            np_boundaries[cat] = sparse2numpy(boundaries[cat][0])

    return np_boundaries, segmentation, categories


def load_instance_insensitive_gt(path: str, new_loader: bool = False):
    return load_sbd_gt_cls_mat(path, new_loader)


def load_instance_sensitive_gt(
    cls_path: str, inst_path: str, new_loader: bool = False
):
    cls_bdry, cls_seg, present_cats = load_sbd_gt_cls_mat(cls_path, new_loader)
    inst_bdry, inst_seg, inst_cats = load_sbd_gt_inst_mat(inst_path, new_loader)

    for inst_cat in inst_cats:
        assert (
            inst_cat in present_cats
        ), f"ERR: instance category of {inst_cat} not available in {present_cats}"

    # create a new bdry map and add instance boundary pixels
    new_bdry = copy.deepcopy(cls_bdry)
    for i, inst_cat in enumerate(inst_cats):
        _inst_bdry = inst_bdry[i]
        # NOTE: inst_cat is indexed from 1
        new_bdry[inst_cat - 1] = new_bdry[inst_cat - 1] | _inst_bdry

    return new_bdry, cls_seg, present_cats


def save_results(
    path: str,
    category: int,
    sample_results,
    threshold_results,
    overall_result,
):
    """Save results as SBD format"""
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
            "{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}".format(
                overall_result.threshold,
                overall_result.recall,
                overall_result.precision,
                overall_result.f1,
                overall_result.best_recall,
                overall_result.best_precision,
                overall_result.best_f1,
                overall_result.area_pr,
                overall_result.ap,
            )
        )
