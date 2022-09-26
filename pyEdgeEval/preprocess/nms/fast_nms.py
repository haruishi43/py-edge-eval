#!/usr/bin/env python3

import os
import cv2
import numpy as np
from scipy.io import loadmat

from pyEdgeEval._lib import nms

from .toolbox import conv_tri, grad2


# NOTE:
#    In NMS, `if edge < interp: out = 0`, I found that sometimes edge is very close to interp.
#    `edge = 10e-8` and `interp = 11e-8` in C, while `edge = 10e-8` and `interp = 9e-8` in python.
#    ** Such slight differences (11e-8 - 9e-8 = 2e-8) in precision **
#    ** would lead to very different results (`out = 0` in C and `out = edge` in python). **
#    Sadly, C implementation is not expected but needed :(


def fast_nms(
    img: np.ndarray,
    r: int = 1,
    s: int = 5,
    m: float = 1.01,
    half_prec: bool = False,
    return_as_uint8: bool = False,
) -> np.ndarray:
    """NMS for binary edges

    Args:
        img (np.ndarray): edge (np.uint8 or float)
        r (int): radius for nms supr
        s (int): radius for supr boundaries
        m (float): multiplier for conservative supr

    Returns:
        supressed edge

    References:
    - https://github.com/pdollar/edges/blob/master/private/edgesNmsMex.cpp

    Current runtime is around 20ms
    """

    if img.dtype == np.uint8:
        # NOTE: input image must be normalized between 0~1
        img = (img / 255.0).astype(np.float64)

    assert (img.dtype == np.float64) or (
        img.dtype == np.float32
    ), f"ERR: input dtype should be float64 or float32 but got {img.dtype}"

    edge = conv_tri(img, 1)

    if half_prec:
        # although there aren't much speed ups from using half precision,
        # using this may save memory
        edge = np.float32(edge)
    ox, oy = grad2(conv_tri(edge, 4))
    oxx, _ = grad2(ox)
    oxy, oyy = grad2(oy)

    # sometimes oxx + 1e-5 = 0, and causes true divide warnings
    oxx[np.where(oxx == 0)] = 1e-5

    val = oyy * np.sign(-oxy) / oxx
    ori = np.mod(np.arctan(val), np.pi)
    # r, s, m = 1, 5, float(1.01)
    out = nms(edge, ori, r=r, s=s, m=m)

    if return_as_uint8:
        out = np.clip(out, 0, 1)
        # NOTE: in MATLAB, uint8(x) means round(x).astype(uint8) in numpy
        out = np.round(out * 255).astype(np.uint8)

    return out


"""
Helper Functions:

Probably won't use...
"""


def nms_process_one_image(img, save_path=None, save=True):
    """NMS single image

    Args:
        img (np.ndarray): edge, model output
        save_path (str): save path
        save (bool): if True, save .png

    Returns:
        edge
    """

    if save and save_path is not None:
        assert os.path.splitext(save_path)[-1] == ".png"

    out = fast_nms(img, r=1, s=5, m=1.01, return_as_uint8=True)

    if save:
        # remove cv2 dependencies
        cv2.imwrite(save_path, out)
    return out


def nms_results(result_dir, save_dir, loader):
    assert os.path.exists(result_dir), f"ERR: {result_dir} does not exist"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for file in os.listdir(result_dir):
        save_name = os.path.join(save_dir, f"{os.path.splitext(file[0])}.png")
        if os.path.isfile(save_name):
            print(f"file: {save_name} exists... skipping")
        img_path = os.path.join(result_dir, file)
        # load image
        img = loader(img_path)
        # nms preprocess
        out = fast_nms(img, r=1, s=5, m=1.01, return_as_uint8=True)
        # save output
        cv2.imwrite(save_name, out)


def nms_all_results(
    model_name_list, result_dir, save_dir, key=None, file_format=".mat"
):
    if not isinstance(model_name_list, list):
        model_name_list = [model_name_list]
    assert file_format in {".mat", ".npy"}
    assert os.path.isdir(result_dir)

    for model_name in model_name_list:
        model_save_dir = os.path.join(save_dir, model_name)
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)

        for file in os.listdir(result_dir):
            save_name = os.path.join(
                model_save_dir, "{}.png".format(os.path.splitext(file)[0])
            )
            if os.path.isfile(save_name):
                continue

            if os.path.splitext(file)[-1] != file_format:
                continue
            abs_path = os.path.join(result_dir, file)
            if file_format == ".mat":
                assert key is not None
                image = loadmat(abs_path)[key]
            elif file_format == ".npy":
                image = np.load(abs_path)
            else:
                raise NotImplementedError
            out = fast_nms(image, r=1, s=5, m=1.01, return_as_uint8=True)
            # save output
            cv2.imwrite(save_name, out)
