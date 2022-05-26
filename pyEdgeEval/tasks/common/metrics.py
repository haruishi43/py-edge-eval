#!/usr/bin/env python3

from collections import namedtuple
from typing import Tuple, Union

import numpy as np


SampleResult = namedtuple(
    "SampleResult", ["sample_name", "threshold", "recall", "precision", "f1"]
)
ThresholdResult = namedtuple(
    "ThresholdResult", ["threshold", "recall", "precision", "f1"]
)
OverallResult = namedtuple(
    "OverallResult",
    [
        "threshold",
        "recall",
        "precision",
        "f1",
        "best_recall",
        "best_precision",
        "best_f1",
        "area_pr",
    ],
)
OverallResultSBD = namedtuple(
    "OverallResult",
    [
        "threshold",
        "recall",
        "precision",
        "f1",
        "best_recall",
        "best_precision",
        "best_f1",
        "area_pr",
        "ap",
    ],
)
SingleResult = namedtuple(
    "SingleResult",
    [
        "recall",
        "precision",
        "f1",
    ],
)


# Metrics


def recall(
    count_r: Union[float, np.ndarray],
    sum_r: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    return count_r / (sum_r + (sum_r == 0))


def precision(
    count_p: Union[float, np.ndarray],
    sum_p: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    return count_p / (sum_p + (sum_p == 0))


def f1(
    prec: Union[float, np.ndarray],
    rec: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    f1_denom = prec + rec + ((prec + rec) == 0)
    return 2.0 * prec * rec / f1_denom


def compute_rec_prec_f1(
    count_r: Union[float, np.ndarray],
    sum_r: Union[float, np.ndarray],
    count_p: Union[float, np.ndarray],
    sum_p: Union[float, np.ndarray],
) -> Tuple[
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
]:
    rec = recall(count_r, sum_r)
    prec = precision(count_p, sum_p)
    return rec, prec, f1(prec, rec)
