#!/usr/bin/env python3

"""Try to mimic the original codes from InvDet CASENet
Reference: https://github.com/Lavender105/DFF/blob/master/lib/matlab/eval/function/plot_pr.m

Since we don't threshold the results for deep learning approach, we cannot
define precision-recall curve for newer methods.
The recall scores of the curves are not monotonically increasing (due to the fact that
post-processing istaken after thresholding in measuring the precision and recall rates).

Methods like InvDet and DSN removed their evaluation results so we cannot test PR curves.
"""
