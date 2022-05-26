#!/usr/bin/env python3

"""Try to mimic the original MATLAB PR Curves
Reference: https://github.com/pdollar/edges/blob/master/edgesEvalPlot.m
"""

import os
from collections import namedtuple
from typing import Any, List, Optional

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from scipy.interpolate import interp1d


AlgorithmInfo = namedtuple(
    "AlgorithmInfo",
    [
        "name",  # name
        "threshold_results",  # path
        "overall_results",  # path
    ],
)


def _isometric_contour_line_template(
    ax=None,
):
    """Setup Basic Isometric Contour Line Plot"""
    if ax is None:
        ax = plt.gca()

    # plt.box(True)
    ax.set_frame_on(True)
    ax.grid(True)
    ax.axhline(0.5, 0, 1, linewidth=2, color=[0.7, 0.7, 0.7])
    for f in np.arange(0.1, 1, 0.1):
        r = np.arange(f, 1.01, 0.01)
        p = f * r / (2 * r - f)
        ax.plot(r, p, color=[0, 1, 0])
        ax.plot(p, r, color=[0, 1, 0])

    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlim=(0, 1), ylim=(0, 1))

    return ax


def _load_threshold_data(data_path):
    assert os.path.exists(data_path), f"ERR: {data_path} doesn't exist"
    pr = np.loadtxt(data_path)
    pr = pr[pr[:, 1] >= 1e-3]
    return pr


def _load_overall_results(data_path):
    assert os.path.exists(data_path), f"ERR: {data_path} doesn't exist"
    res = np.loadtxt(data_path)
    return res


def _calc_r50(pr):
    _, o = np.unique(pr[:, 2], return_index=True)
    r50 = interp1d(
        pr[o, 2],
        pr[o, 1],
        bounds_error=False,
        fill_value=np.nan,
    )(np.maximum(pr[o[0], 2], 0.5))

    return r50


def plot_pr_curve(
    algs: List[AlgorithmInfo],
    names: Optional[List[str]] = None,
    colors: Any = None,
    save_path: Optional[str] = None,
):
    assert isinstance(algs, list) and len(algs) > 0
    n = len(algs)
    if names is None:
        names = [a.name for a in algs]
    else:
        assert isinstance(names, list) and len(names) == n
    names = np.array(names)
    if colors is None:
        # https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib
        colors = cm.rainbow(np.linspace(0, 1, n))
    else:
        assert len(colors) == n
    colors = np.array(colors)

    # create basic template
    fig, ax = plt.subplots()
    ax = _isometric_contour_line_template(ax=ax)

    # load results for every algorithm (pr=[T, R, P, F])
    n = len(algs)
    hs, res, prs = [None] * n, np.zeros((n, 9), dtype=np.float32), []
    for i, alg in enumerate(algs):
        pr = _load_threshold_data(alg.threshold_results)
        res[i, :8] = _load_overall_results(alg.overall_results)
        res[i, 8] = _calc_r50(pr)
        prs.append(pr)
    prs = np.stack(prs, axis=0)

    # sort algorithms by ODS score
    o = np.argsort(res[:, 3])[::-1]
    res, prs, names, colors = res[o, :], prs[o], names[o], colors[o]

    # plot results for every algorithm (plot best last)
    for i in range(n - 1, -1, -1):
        hs[i] = ax.plot(
            prs[i, :, 1],
            prs[i, :, 2],
            linestyle="-",
            linewidth=3,
            color=colors[i],
        )[0]
        prefix = "ODS={:.3f}, OIS={:.3f}, AP={:.3f}, R50={:.3f}".format(
            *res[i, [3, 6, 7, 8]]
        )
        prefix += " - {}".format(names[i])
        print(prefix)  # should remove

    # add legends
    legend_texts = [
        "[F={:.2f}] {}".format(res[i, 3], names[i]) for i in range(n)
    ]
    ax.legend(hs, legend_texts, loc="lower left")

    if save_path:
        plt.savefig(save_path)

    # don't really need this, but calling this will close the figure
    plt.show()
