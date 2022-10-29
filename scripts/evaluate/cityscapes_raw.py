#!/usr/bin/env python3

from pyEdgeEval.helpers.evaluate_cityscapes import evaluate_cityscapes_raw


if __name__ == "__main__":
    evaluate_cityscapes_raw(gt_dir='gtEval')
