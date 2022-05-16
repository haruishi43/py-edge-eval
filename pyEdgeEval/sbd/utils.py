#!/usr/bin/env python3

import numpy as np
from scipy.sparse import csc_matrix

from pyEdgeEval.utils import loadmat


def sparse2numpy(data: csc_matrix):
    """helper function to convert compressed sparse column matrix to numpy array"""
    assert isinstance(data, csc_matrix)
    return data.toarray()


def load_sbd_gt_cls_mat(path: str, new_loader: bool = False):
    """Load Per Class Ground Truth Annoation"""
    if new_loader:
        gt = loadmat(path, True)["GTcls"]
        boundaries = gt["Boundaries"]  # list[csc_matrix]
        segmentation = gt["Segmentation"]  # np.ndarray(h, w)
        present_categories = gt["CategoriesPresent"] - 1  # np.ndarray()

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
        present_categories = gt[2].squeeze() - 1  # np.ndarray()

        assert len(segmentation.shape) == 2
        h, w = segmentation.shape
        num_categories = len(boundaries)
        np_boundaries = np.zeros([num_categories, h, w], dtype=np.uint8)
        for cat in range(num_categories):
            np_boundaries[cat] = sparse2numpy(boundaries[cat][0])

    # TODO: do checks for present categories?
    return np_boundaries, segmentation, present_categories


def load_sbd_gt_inst_mat(path: str, new_loader: bool = False):
    """Load Per Instance Ground Truth Annotation"""
    if new_loader:
        gt = loadmat(path, True)["GTinst"]
        segmentation = gt["Segmentation"]
        boundaries = gt["Boundaries"]
        categories = gt["Categories"] - 1

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
        categories = gt[2].squeeze() - 1

        assert len(segmentation.shape) == 2
        h, w = segmentation.shape
        num_categories = len(boundaries)
        np_boundaries = np.zeros([num_categories, h, w], dtype=np.uint8)
        for cat in range(num_categories):
            np_boundaries[cat] = sparse2numpy(boundaries[cat][0])

    return np_boundaries, segmentation, categories


def load_sbd_gts(path: str):
    """SBD GT Boundaries"""
    ...
