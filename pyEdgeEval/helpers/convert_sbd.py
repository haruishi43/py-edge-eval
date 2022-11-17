#!/usr/bin/env python3

"""Helper functions for SBD dataset"""

import os.path as osp
from warnings import warn
from typing import Optional

import numpy as np
from PIL import Image

from pyEdgeEval.common.multi_label import rgb_multilabel_encoding
from pyEdgeEval.edge_tools import loop_instance_mask2edge, loop_mask2edge
from pyEdgeEval.datasets.sbd import (
    load_sbd_gt_cls_mat,
    load_sbd_gt_inst_mat,
    load_reanno_instance_insensitive_gt,
    load_reanno_instance_sensitive_gt,
)
from pyEdgeEval.datasets.sbd_attributes import (
    SBD_labelIds,
    SBD_inst_labelIds,
    SBD_label2trainId,
)
from pyEdgeEval.utils import loadmat, mask2onehot, edge_label2trainId


def check_path(path):
    """check and return path"""
    assert osp.exists(path), f"path {path}"
    return path


def get_samples(path):
    with open(path, "r") as f:
        sample_names = [line.rstrip("\n") for line in f]
    return sample_names


def add_ignore_pixel(
    cls_seg: np.ndarray,
    border_px: int = 5,
    ignore_id: int = 21,
):
    """Add ignore pixels around the segmentation boundaries

    In SEAL, this process was added since SBD has issues when segmentation map
    touches the image boundaries.
    """
    h, w = cls_seg.shape
    cls_seg[0:border_px, :] = ignore_id
    cls_seg[h - border_px : h, :] = ignore_id
    cls_seg[:, 0:border_px] = ignore_id
    cls_seg[:, w - border_px : w] = ignore_id
    return cls_seg


def convert_inst_seg(inst_seg, inst_cats, present_cats):
    """Convert the instance segmentation format so it's compatible with mask2edge

    Args:
        inst_seg (np.ndarray): original instance segmentation
        inst_cats (np.ndarray): instance labels
        present_cats (np.ndarray): present categories in the instance segmentation map
    Returns:
        output instance segmentation map (np.ndarray with np.int32)
    """
    # output requires int32
    out = np.zeros_like(inst_seg, dtype=np.int32)
    for present_cat in present_cats:
        tmp_id = str(present_cat).zfill(2)
        num_inst = 0
        for i, cat in enumerate(inst_cats):
            if present_cat == cat:
                out_id = int(tmp_id + str(num_inst).zfill(3))
                # indexed from 1
                out[inst_seg == (i + 1)] = out_id
                num_inst += 1
    return out


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
