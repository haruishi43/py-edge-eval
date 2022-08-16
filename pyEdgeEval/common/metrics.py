#!/usr/bin/env python3

from typing import Tuple, Union

import numpy as np

__all__ = [
    "recall",
    "precision",
    "f1",
    "compute_rec_prec_f1",
]


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


def interpolated_max_scores(
    thresholds,
    rec,
    prec,
):
    best_threshold = thresholds[0]
    best_rec = rec[0]
    best_prec = prec[0]
    best_f1 = f1(best_prec, best_rec)

    for i in range(1, len(thresholds)):
        for d in np.linspace(0, 1, num=101):
            t = thresholds[i] * d + thresholds[i - 1] * (1 - d)
            r = rec[i] * d + rec[i - 1] * (1 - d)
            p = prec[i] * d + prec[i - 1] * (1 - d)
            f = f1(p, r)
            if f > best_f1:
                best_threshold = t
                best_rec = r
                best_prec = p
                best_f1 = f

    return best_threshold, best_rec, best_prec, best_f1
