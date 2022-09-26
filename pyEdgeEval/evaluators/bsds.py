#!/usr/bin/env python3

import os
import os.path as osp

from pyEdgeEval.common.binary_label import (
    calculate_metrics,
    save_results,
)
from pyEdgeEval.datasets import bsds_eval_single
from pyEdgeEval.utils import print_log

from .base import BaseBinaryEvaluator


class BSDS500Evaluator(BaseBinaryEvaluator):

    # Dataset specific attributes
    GT_DIR = "groundTruth"
    GT_SUFFIX = ".mat"
    PRED_SUFFIX = ".png"

    def __init__(
        self,
        dataset_root: str,
        pred_root: str,
        split: str = "test",
    ):
        self.dataset_root = dataset_root
        self.pred_root = pred_root

        assert split in ("val", "test")
        self.split = split

        self.GT_root = osp.join(
            self.dataset_root,
            self.GT_DIR,
        )

        try:
            self.set_sample_names()
        except Exception:
            print_log(
                "Tried to set sample_names, but couldn't",
                logger=self._logger,
            )

    def set_sample_names(
        self,
        sample_names=None,
    ):
        if sample_names is None:
            # load sample_names by going into the split file
            sample_names = []

            # NOTE: for benchmark, there are no split
            bsds_dir = osp.join(self.GT_root, self.split)

            print_log(f"Loading samples from {bsds_dir}", logger=self._logger)
            files = os.listdir(osp.join(self.GT_root, self.split))

            for fn in files:
                dir, filename = osp.split(fn)
                name, ext = osp.splitext(filename)
                if ext.lower() == ".mat":
                    # split/12345.mat
                    sample_names.append(osp.join(self.split, name))

        assert isinstance(
            sample_names, list
        ), f"ERR: sample_names should be a list but got {type(sample_names)}"
        assert len(sample_names) > 0, "ERR: sample_names is empty"
        self._sample_names = sample_names

    def set_eval_params(
        self,
        scale: float = 1.0,
        apply_thinning: bool = True,
        apply_nms: bool = False,
        max_dist: float = 0.0075,
        **kwargs,
    ) -> None:

        assert 0 < scale <= 1, f"ERR: scale ({scale}) is not valid"
        self.scale = scale
        self.apply_thinning = apply_thinning
        self.apply_nms = apply_nms
        self.max_dist = max_dist

    @property
    def eval_params(self):
        return dict(
            scale=self.scale,
            apply_thinning=self.apply_thinning,
            apply_nms=self.apply_nms,
            max_dist=self.max_dist,
        )

    def _before_evaluation(self):
        assert osp.exists(
            self.dataset_root
        ), f"ERR: {self.dataset_root} does not exist"
        assert osp.exists(self.GT_root), f"ERR: {self.GT_root} does not exist"
        assert osp.exists(
            self.pred_root
        ), f"ERR: {self.pred_root} does not exist"

    def evaluate(
        self,
        thresholds,
        nproc,
        save_dir,
        no_split_dir=False,
    ):
        self._before_evaluation()

        # populate data (samples)
        data = []
        for sample_name in self.sample_names:

            gt_path = osp.join(self.GT_root, f"{sample_name}{self.GT_SUFFIX}")
            if no_split_dir:
                _sample_name = sample_name.replace(f"{self.split}/", "")
                pred_path = osp.join(
                    self.pred_root, f"{_sample_name}{self.PRED_SUFFIX}"
                )
            else:
                pred_path = osp.join(
                    self.pred_root, f"{sample_name}{self.PRED_SUFFIX}"
                )

            assert osp.exists(gt_path), f"ERR: {gt_path} is not valid"
            assert osp.exists(pred_path), f"ERR: {pred_path} is not valid"

            data.append(
                dict(
                    name=sample_name,
                    thresholds=thresholds,
                    gt_path=gt_path,
                    pred_path=pred_path,
                    **self.eval_params,
                )
            )

        assert len(data) > 0, "ERR: no evaluation data"

        # evaluate
        (sample_metrics, threshold_metrics, overall_metric) = calculate_metrics(
            eval_single=bsds_eval_single,
            thresholds=thresholds,
            samples=data,
            nproc=nproc,
        )

        # save metrics
        if save_dir:
            print_log("Saving results", logger=self._logger)
            save_results(
                root=save_dir,
                sample_metrics=sample_metrics,
                threshold_metrics=threshold_metrics,
                overall_metric=overall_metric,
            )

        # TODO: better printing
        metrics = [
            "ODS_recall",
            "ODS_precision",
            "ODS_f1",
            "OIS_recall",
            "OIS_precision",
            "OIS_f1",
            "AP",
        ]
        print_log("Printing out Results:\n", logger=self._logger)
        for m in metrics:
            print_log(f"{m}:\t{overall_metric[m]}", logger=self._logger)

        return overall_metric
