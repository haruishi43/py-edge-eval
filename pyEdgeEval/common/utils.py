#!/usr/bin/env python3

import numpy as np


def check_thresholds(thresholds):
    if isinstance(thresholds, int):
        # integers == number of thresholds between 0 and 1
        thresholds = np.linspace(
            1.0 / (thresholds + 1), 1.0 - 1.0 / (thresholds + 1), thresholds
        )
    elif isinstance(thresholds, float):
        # single specific threshold
        thresholds = np.array([thresholds])
    elif isinstance(thresholds, list):
        # multiple specific thresholds
        thresholds = np.array(thresholds)
    else:
        raise ValueError(
            "thresholds should be an int or a NumPy array, not "
            "a {}".format(type(thresholds))
        )

    assert isinstance(
        thresholds, np.ndarray
    ), f"ERR: thresholds must be np.ndarray, but got {type(thresholds)}"
    if thresholds.ndim != 1:
        raise ValueError(
            "thresholds array should have 1 dimension, "
            "not {}".format(thresholds.ndim)
        )

    return thresholds
