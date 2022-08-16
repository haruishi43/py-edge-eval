#!/usr/bin/env python3

import os

# libraries used for .mat io
from scipy.io import loadmat as scipy_loadmat
from scipy.sparse import csc_matrix

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
        assert (
            new_loadmat is not None
        ), "ERR: need modules that can load newer .mat"
        mat = new_loadmat(path)
    else:
        mat = scipy_loadmat(path)

    return mat


def sparse2numpy(data: csc_matrix):
    """helper function to convert compressed sparse column matrix to numpy array"""
    assert isinstance(
        data, csc_matrix
    ), f"ERR: input is not csc_matrix, but got {type(data)}"
    return data.toarray()
