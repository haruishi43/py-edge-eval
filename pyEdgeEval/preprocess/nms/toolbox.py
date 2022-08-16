#!/usr/bin/env python3

from typing import Tuple

import numpy as np
from scipy import signal


def conv_tri(
    image: np.ndarray,
    r: int,
    s: int = 1,
) -> np.ndarray:
    """2D image convolution with a triangle filter (no fast)
    See https://github.com/pdollar/toolbox/blob/master/channels/convTri.m
    Note: signal.convolve2d does not support float16('single' in MATLAB)
    """
    if image.size == 0 or (r == 0 and s == 1):
        return image
    if r <= 1:
        p = 12 / r / (r + 2) - 2
        f = np.array([[1, p, 1]]) / (2 + p)
        r = 1
    else:
        f = (
            np.array([list(range(1, r + 1)) + [r + 1] + list(range(r, 0, -1))])
            / (r + 1) ** 2
        )
    f = f.astype(image.dtype)
    image = np.pad(image, ((r, r), (r, r)), mode="symmetric")

    # FIXME: remove scipy dependencies
    image = signal.convolve2d(
        signal.convolve2d(image, f, "valid"), f.T, "valid"
    )
    if s > 1:
        t = int(np.floor(s / 2) + 1)
        image = image[
            t - 1 : image.shape[0] - (s - t) + 1 : s,
            t - 1 : image.shape[1] - (s - t) + 1 : s,
        ]
    return image


def grad2(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """numerical gradients along x and y directions (no fast)
    See https://github.com/pdollar/toolbox/blob/master/channels/gradient2.m
    Note: np.gradient return [oy, ox], MATLAB version return [ox, oy]
    """
    assert image.ndim == 2
    oy, ox = np.gradient(image)
    return ox, oy
