#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import distance_transform_edt


def kill_internal_prediction(
    pred: np.ndarray,
    gt: np.ndarray,
    seg: np.ndarray,
    max_dist: float = 0.02,
) -> np.ndarray:
    """Remove predicted pixels inside boundaries

    NOTE: the distance transform may differ from MATLAB implementation
    NOTE: might not work correctly when using instance sensitive boundaries
    """
    diag = np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)
    buffer = diag * max_dist

    # buggy output when input is only 0s or 1s
    distmap = distance_transform_edt(1 - gt)
    killmask = np.invert((distmap > buffer) * seg)
    assert killmask.shape == pred.shape
    return pred * killmask
