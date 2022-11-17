#!/usr/bin/env python3

from copy import deepcopy

import cv2
import numpy as np
from PIL import Image

# from skimage.io import imread

from pyEdgeEval.common.multi_label.evaluate_boundaries import (
    evaluate_boundaries_threshold,
)
from pyEdgeEval.common.utils import check_thresholds
from pyEdgeEval.utils import loadmat, sparse2numpy


def load_sbd_gt_cls_mat(path: str, new_loader: bool = False):
    """Load Per Class Ground Truth Annoation"""
    if new_loader:
        gt = loadmat(path, True)["GTcls"]
        boundaries = gt["Boundaries"]  # list[csc_matrix]
        segmentation = gt["Segmentation"]  # np.ndarray(h, w)
        present_categories = gt["CategoriesPresent"]  # np.ndarray()
        if isinstance(present_categories, int):
            present_categories = np.array([present_categories])
        if present_categories.ndim == 0:
            # single value, or no instances
            present_categories = present_categories[None]
        assert present_categories.ndim == 1, f"{present_categories.ndim}"

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
        present_categories = gt[2]  # np.ndarray()
        if present_categories.ndim > 1:
            present_categories = present_categories.squeeze()
        if present_categories.ndim == 0:
            # single value, or no instances
            present_categories = present_categories[None]

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
        if isinstance(categories, int):
            categories = np.array([categories])
        if categories.ndim == 0:
            # single value, or no instances
            categories = categories[None]
        assert categories.ndim == 1, f"{categories.ndim}"

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
        categories = gt[2]
        if categories.ndim > 1:
            categories = categories.squeeze()
        if categories.ndim == 0:
            # single value, or no instances
            categories = categories[None]

        assert len(segmentation.shape) == 2
        h, w = segmentation.shape
        num_categories = len(boundaries)
        np_boundaries = np.zeros([num_categories, h, w], dtype=np.uint8)
        for cat in range(num_categories):
            np_boundaries[cat] = sparse2numpy(boundaries[cat][0])

    return np_boundaries, segmentation, categories


def load_instance_insensitive_gt(cls_path: str, new_loader: bool = False):
    return load_sbd_gt_cls_mat(cls_path, new_loader)


def load_instance_sensitive_gt(
    cls_path: str, inst_path: str, new_loader: bool = False
):
    cls_bdry, cls_seg, present_cats = load_sbd_gt_cls_mat(cls_path, new_loader)
    inst_bdry, _, inst_cats = load_sbd_gt_inst_mat(inst_path, new_loader)

    for inst_cat in inst_cats:
        assert (
            inst_cat in present_cats
        ), f"ERR: instance category of {inst_cat} not available in {present_cats}"

    # create a new bdry map and add instance boundary pixels
    new_bdry = deepcopy(cls_bdry)
    for i, inst_cat in enumerate(inst_cats):
        _inst_bdry = inst_bdry[i]
        # NOTE: inst_cat is indexed from 1
        new_bdry[inst_cat - 1] = new_bdry[inst_cat - 1] | _inst_bdry

    return new_bdry, cls_seg, present_cats


def load_reanno_instance_insensitive_gt(cls_path: str):
    """Just a wrapper"""
    return load_sbd_gt_cls_mat(cls_path, True)


def load_reanno_instance_sensitive_gt(cls_path: str, inst_path: str):
    """Re-annoated GTs have minor changes to how the instances are saved

    inst_bdry contains already preprocessed instance boundaries here, wheras
    before, it contained arrays for each of the instances and were indexed
    for each instances.

    we also need to use the new_loader, because old loader cannot read it correctly
    """
    cls_bdry, cls_seg, present_cats = load_sbd_gt_cls_mat(cls_path, True)
    inst_bdry, _, inst_cats = load_sbd_gt_inst_mat(inst_path, True)

    for inst_cat in inst_cats:
        assert (
            inst_cat in present_cats
        ), f"ERR: instance category of {inst_cat} not available in {present_cats}"

    assert (
        cls_bdry.shape == inst_bdry.shape
    ), "the two bdries should have equal shape"

    # create a new bdry map and add instance boundary pixels
    new_bdry = deepcopy(cls_bdry)
    for cat in present_cats:
        # NOTE: bdry idx doesn't include background (0)
        _inst_bdry = inst_bdry[cat - 1]
        new_bdry[cat - 1] = new_bdry[cat - 1] | _inst_bdry

    return new_bdry, cls_seg, present_cats


def _evaluate_single(
    cls_path,
    inst_path,
    pred_path,
    category,
    scale,
    max_dist,
    thresholds,
    apply_thinning,
    apply_nms,
    kill_internal,
    skip_if_nonexistent,
    **kwargs,
):
    """Evaluate a single sample (sub-routine)

    NOTE: don't set defaults for easier debugging
    """
    # checks and converts thresholds
    thresholds = check_thresholds(thresholds)

    if inst_path:
        # instance sensitive
        edge, seg, present_categories = load_instance_sensitive_gt(
            cls_path=cls_path,
            inst_path=inst_path,
        )
    else:
        # instance insensitive
        edge, seg, present_categories = load_instance_insensitive_gt(
            cls_path=cls_path,
        )

    # NOTE: background ID is 0
    # TODO: would be smart to return here using `present_categories`

    cat_idx = category - 1
    cat_edge = edge[cat_idx, :, :]

    # rescale
    (h, w) = cat_edge.shape
    height, width = int(h * scale + 0.5), int(w * scale + 0.5)
    cat_edge = cv2.resize(cat_edge, (width, height), cv2.INTER_NEAREST)
    if kill_internal:
        seg = cv2.resize(seg, (width, height), cv2.INTER_NEAREST)

        # SBD explicitly starts from 1 (because of matlab)
        # 0 is the background
        cat_seg = seg == category
    else:
        cat_seg = None

    # load pred
    pred = Image.open(pred_path)
    pred = pred.resize((width, height), Image.Resampling.NEAREST)
    # pred = imread(pred_path)
    # pred = cv2.resize(pred, (width, height), cv2.INTER_NEAREST)
    pred = np.array(pred)
    pred = (pred / 255).astype(float)

    # evaluate multi-label boundaries
    count_r, sum_r, count_p, sum_p = evaluate_boundaries_threshold(
        thresholds=thresholds,
        pred=pred,
        gt=cat_edge,
        gt_seg=cat_seg,
        max_dist=max_dist,
        apply_thinning=apply_thinning,
        kill_internal=kill_internal,
        skip_if_nonexistent=skip_if_nonexistent,
        apply_nms=apply_nms,
        nms_kwargs=dict(
            r=1,
            s=5,
            m=1.01,
            half_prec=False,
        ),
    )

    return count_r, sum_r, count_p, sum_p


def sbd_eval_single(kwargs):
    """Wrapper function to unpack all the kwargs"""
    return _evaluate_single(**kwargs)
