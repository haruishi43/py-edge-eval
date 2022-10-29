#!/usr/bin/env python3

from pyEdgeEval.helpers.evaluate_cityscapes import evaluate_cityscapes_thin


if __name__ == "__main__":
    evaluate_cityscapes_thin(gt_dir='gtEval')
