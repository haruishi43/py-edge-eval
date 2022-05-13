#!/usr/bin/env python3

import os

import numpy as np

from skimage.util import img_as_float
from skimage.io import imread

# libraries used for .mat io
from scipy.io import loadmat as scipy_loadmat
try:
    # prefer `pymatreader`
    from pymatreader import read_mat as new_loadmat
except ImportError:
    try:
        from mat73 import loadmat as new_loadmat
    except ImportError:
        print("WARN: cannot load newer versions of .mat files")
        new_loadmat = None


def loadmat(
    path: str,
    use_mat73: bool = False,
):
    assert os.path.exists(path), f"{path} doesn't exist"
    if use_mat73:
        assert new_loadmat is not None, \
            "ERR: need modules that can load newer .mat"
        mat = new_loadmat(path)
    else:
        mat = scipy_loadmat(path)

    return mat


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
    assert img.dtype == np.uint8, f"ERR: img needs to be uint8, but got{img.dtype}"
    return img_as_float(img)
