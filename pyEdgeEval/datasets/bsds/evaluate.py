#!/usr/bin/env python3

from functools import partial
from typing import Callable, List, Union

import numpy as np

from pyEdgeEval.tasks.common.evaluations import (
    base_evaluation_wo_threshold,
    base_pr_evaluation,
)
from pyEdgeEval.tasks.edge_detection import (
    evaluate_single_sample_bin,
    evaluate_single_sample_threshold,
)


def evaluation_wo_threshold(
    sample_names: List[str],
    load_gts: Callable[[str], np.ndarray],
    load_pred: Callable[[str], np.ndarray],
    max_dist: float = 0.0075,
    apply_thinning: bool = True,
    nproc: int = 8,
):
    # intialize the partial function for evaluating boundaries
    _wrapper = partial(
        evaluate_single_sample_bin,
        func_load_pred=load_pred,
        func_load_gt=load_gts,
        max_dist=max_dist,
        apply_thinning=apply_thinning,
    )

    return base_evaluation_wo_threshold(
        samples=sample_names,
        wrapper=_wrapper,
        nproc=nproc,
    )


def pr_evaluation(
    sample_names: List[str],
    thresholds: Union[int, np.ndarray],
    load_gts: Callable[[str], np.ndarray],
    load_pred: Callable[[str], np.ndarray],
    max_dist: float = 0.0075,
    apply_thinning: bool = True,
    nproc: int = 8,
):
    # check threshold values
    if isinstance(thresholds, int):
        thresholds = np.linspace(
            1.0 / (thresholds + 1), 1.0 - 1.0 / (thresholds + 1), thresholds
        )
    elif isinstance(thresholds, np.ndarray):
        if thresholds.ndim != 1:
            raise ValueError(
                "thresholds array should have 1 dimension, "
                "not {}".format(thresholds.ndim)
            )
        pass
    else:
        raise ValueError(
            "thresholds should be an int or a NumPy array, not "
            "a {}".format(type(thresholds))
        )

    # intialize the partial function for evaluating boundaries
    _wrapper = partial(
        evaluate_single_sample_threshold,
        thresholds=thresholds,
        func_load_pred=load_pred,
        func_load_gt=load_gts,
        max_dist=max_dist,
        apply_thinning=apply_thinning,
    )

    return base_pr_evaluation(
        thresholds=thresholds,
        samples=sample_names,
        wrapper=_wrapper,
        nproc=nproc,
    )
