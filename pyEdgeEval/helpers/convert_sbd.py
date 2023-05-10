#!/usr/bin/env python3

"""Helper functions for SBD dataset"""

import os.path as osp
from copy import deepcopy
from warnings import warn
from typing import Optional

import numpy as np
from PIL import Image

from pyEdgeEval.common.multi_label import (
    rgb_multilabel_encoding,
    add_ignore_pixel,
    convert_inst_seg,
)
from pyEdgeEval.edge_tools import loop_instance_mask2edge, loop_mask2edge
from pyEdgeEval.datasets.sbd_attributes import (
    SBD_labelIds,
    SBD_inst_labelIds,
    SBD_label2trainId,
)
from pyEdgeEval.utils import (
    loadmat,
    mask2onehot,
    edge_label2trainId,
    sparse2numpy,
)


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
    warn("This function has been deprecated.", DeprecationWarning)
    return load_sbd_gt_cls_mat(cls_path, new_loader)


def load_instance_sensitive_gt(
    cls_path: str, inst_path: str, new_loader: bool = False
):
    """Loads instance sensitive ground truth annotation from .mat file.

    This function has been deprecated.
    """
    warn("This function has been deprecated.", DeprecationWarning)

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
    warn("This function does not work correctly for 'thin' evaluation")
    # TODO: the radius and thinning is not applied here
    # we should do this in this function, but this will change
    # the results for pre-processing the edges.
    return load_sbd_gt_cls_mat(cls_path, True)


def load_reanno_instance_sensitive_gt(cls_path: str, inst_path: str):
    """Re-annoated GTs have minor changes to how the instances are saved

    inst_bdry contains already preprocessed instance boundaries here, wheras
    before, it contained arrays for each of the instances and were indexed
    for each instances.

    we also need to use the new_loader, because old loader cannot read it correctly
    """
    warn("This function does not work correctly for 'thin' evaluation")
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

    # TODO: the radius and thinning is not applied here
    # we should do this in this function, but this will change
    # the results for pre-processing the edges.

    return new_bdry, cls_seg, present_cats


def check_path(path):
    """check and return path"""
    assert osp.exists(path), f"path {path}"
    return path


def get_samples(path):
    with open(path, "r") as f:
        sample_names = [line.rstrip("\n") for line in f]
    return sample_names


def convert_seg_label2train(seg, label2trainId):
    """Conver labelIds to trainIds

    NOTE: background, non-trainIds are all 255
    """
    out = np.ones_like(seg) * 255
    for labelId, trainId in label2trainId.items():
        out[seg == labelId] = trainId
    return out


def convert_mat2np(
    cls_path: str,
    inst_path: Optional[str] = None,
    radius: int = 2,
    thin: bool = False,
    outer_pixel: int = 21,
    new_loader: bool = False,
):
    """Segmentation maps to boundaries

    Since the preprocessed boundaries are fixed to a certain radius,
    we need to create custom boundaries from MATLAB files.

    Args:
        cls_path (str)
        inst_path (str)
        radius (int)
        thin (bool)
        outer_pixel (int)
        new_loader (bool)
    Returns:
        boundary and segmentation maps

    NOTE: added outer pixel introduced in SEAL
    """

    if thin and (radius != 1):
        warn("Thin, but radius was set to >1")

    # get segmentation from MATLAB file
    _, cls_seg, present_cats = load_sbd_gt_cls_mat(cls_path, new_loader)

    # add ignore pixels around the image
    cls_seg = add_ignore_pixel(cls_seg, border_px=5, ignore_id=outer_pixel)

    # convert mask to onehot vector
    mask = mask2onehot(cls_seg, labels=SBD_labelIds)

    if inst_path:
        # get instance segmentation from MATLAB file
        _, inst_seg, inst_cats = load_sbd_gt_inst_mat(inst_path, new_loader)

        # convert instance segmentation
        new_inst_seg = convert_inst_seg(inst_seg, inst_cats, present_cats)

        # generate edges
        bdry = loop_instance_mask2edge(
            mask=mask,
            inst_mask=new_inst_seg,
            inst_labelIds=SBD_inst_labelIds,
            ignore_indices=[outer_pixel],
            radius=radius,
            thin=thin,
        )
        bdry = edge_label2trainId(bdry, SBD_label2trainId)

        return bdry, cls_seg, new_inst_seg
    else:
        # generate edges
        bdry = loop_mask2edge(
            mask=mask,
            ignore_indices=[outer_pixel],
            radius=radius,
            thin=thin,
        )
        bdry = edge_label2trainId(bdry, SBD_label2trainId)

        return bdry, cls_seg


def routine(
    sample_name: str,
    data_dir: str,
    save_dir: str,
    radius: int = 2,
    thin: bool = False,
    edge_suffix: str = "_edge.png",
    isedge_suffix: str = "_isedge.png",
    label_suffix: str = "_labelIds.png",
    train_suffix: str = "_trainIds.png",
    inst_suffix: str = "_instanceIds.png",
    save_segs: bool = False,
):
    """Routine

    1. save instance sensitive
    2. save non instance sensitive
    """
    if thin:
        # force radius=1 for thinning
        radius = 1

    save_dir = check_path(save_dir)

    # we don't touch `cls_crf`
    cls_dir = check_path(osp.join(data_dir, "cls_orig"))
    inst_dir = check_path(osp.join(data_dir, "inst_orig"))

    cls_fp = check_path(osp.join(cls_dir, f"{sample_name}.mat"))
    inst_fp = check_path(osp.join(inst_dir, f"{sample_name}.mat"))

    # convert MATLAB files to numpy
    ISbdry, seg, inst = convert_mat2np(
        cls_path=cls_fp,
        inst_path=inst_fp,
        radius=radius,
        thin=thin,
        outer_pixel=21,
        new_loader=False,
    )
    nonISbdry, _ = convert_mat2np(
        cls_path=cls_fp,
        inst_path=None,
        radius=radius,
        thin=thin,
        outer_pixel=21,
        new_loader=False,
    )

    # save bdry
    nonISbdry_img = Image.fromarray(rgb_multilabel_encoding(nonISbdry))
    nonISbdry_img.save(osp.join(save_dir, f"{sample_name}{edge_suffix}"))
    ISbdry_img = Image.fromarray(rgb_multilabel_encoding(ISbdry))
    ISbdry_img.save(osp.join(save_dir, f"{sample_name}{isedge_suffix}"))

    if save_segs:
        # save refined segmentation map
        label_img = Image.fromarray(seg)
        label_img.save(osp.join(save_dir, f"{sample_name}{label_suffix}"))
        train_img = Image.fromarray(
            convert_seg_label2train(seg, SBD_label2trainId)
        )
        train_img.save(osp.join(save_dir, f"{sample_name}{train_suffix}"))

        # save instance
        inst_img = Image.fromarray(inst)
        inst_img.save(osp.join(save_dir, f"{sample_name}{inst_suffix}"))


# Reannotated Test Data


def load_reanno_samples(path: str, new_loader: bool = False):
    """Load sample names for the re-annotated validation dataset (introduced in SEAL)

    Args:
        path (str): path to `test.mat`
        new_loader (bool)

    Returns:
        list of sample names (xxxx_xxxxxx)
    """
    if new_loader:
        print("not implemented yet")
        test = loadmat(path, True)["listTest"]
        assert isinstance(
            test, (list, tuple)
        ), f".mat should contain lists but got {type(test)}"
    else:
        test = loadmat(path, False)["listTest"]
        # HACK: nested structured array of string
        test = [t.tolist()[0] for t in test.squeeze()]

    return test


def reanno_routine(
    sample_name: str,
    data_dir: str,
    save_dir: str,
    edge_suffix: str = "_reanno_edge.png",
    isedge_suffix: str = "_reanno_isedge.png",
):
    save_dir = check_path(save_dir)
    cls_dir = check_path(osp.join(data_dir, "cls"))
    inst_dir = check_path(osp.join(data_dir, "inst"))

    cls_fp = check_path(osp.join(cls_dir, f"{sample_name}.mat"))
    inst_fp = check_path(osp.join(inst_dir, f"{sample_name}.mat"))

    # IS
    ISbdry, _, _ = load_reanno_instance_sensitive_gt(
        cls_path=cls_fp,
        inst_path=inst_fp,
    )
    # nonIS
    nonISbdry, _, _ = load_reanno_instance_insensitive_gt(cls_path=cls_fp)

    # save bdry
    nonISbdry_img = Image.fromarray(rgb_multilabel_encoding(nonISbdry))
    nonISbdry_img.save(osp.join(save_dir, f"{sample_name}{edge_suffix}"))
    ISbdry_img = Image.fromarray(rgb_multilabel_encoding(ISbdry))
    ISbdry_img.save(osp.join(save_dir, f"{sample_name}{isedge_suffix}"))
