#!/usr/bin/env python3

"""Try to mimic the original MATLAB PR Curves
Reference: https://github.com/xwjabc/hed/blob/master/eval/edges/edgesEvalPlot.m
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

from pyEdgeEval.visualization.pr_curve import (
    _calc_r50,
    _isometric_contour_line_template,
    _load_overall_results,
    _load_threshold_data,
)


def _plot_bsds500_human(ax=None):
    """Human Performance for BSDS500"""
    if ax is None:
        ax = plt.gca()
    h = ax.plot(
        0.7235,
        0.9014,
        marker="o",
        markersize=8,
        color=[0, 0.5, 0],
        markerfacecolor=[0, 0.5, 0],
        markeredgecolor=[0, 0.5, 0],
    )
    return ax, h


def plot_pr_curve(
    algs, names=None, colors=None, plot_human: bool = True, save_path=None
):
    assert isinstance(algs, list) and len(algs) > 0
    n = len(algs)
    if names is None:
        names = [a.name for a in algs]
    else:
        assert isinstance(names, list) and len(names) == n
    names = np.array(names)

    _n = n + 1 if plot_human else n
    if colors is None:
        # https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib
        colors = cm.rainbow(np.linspace(0, 1, _n))
    else:
        assert len(colors) == _n
    colors = np.array(colors)

    # create basic template
    fig, ax = plt.subplots()
    ax = _isometric_contour_line_template(ax=ax)

    # plot human (first)
    if plot_human:
        ax, h = _plot_bsds500_human(ax=ax)

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
        print(prefix)

    # add legends
    legend_texts = [
        "[F={:.2f}] {}".format(res[i, 3], names[i]) for i in range(n)
    ]

    # prepend human
    if plot_human:
        hs = h + hs
        legend_texts = ["[F=.80] Human"] + legend_texts

    ax.legend(hs, legend_texts, loc="lower left")

    if save_path:
        plt.savefig(save_path)

    # don't really need this, but calling this will close the figure
    plt.show()
