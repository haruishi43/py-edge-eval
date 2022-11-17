#!/usr/bin/env python3

import os.path as osp

from pyEdgeEval.common.multi_label import (
    calculate_metrics,
    save_category_results,
)
from pyEdgeEval.datasets import sbd_eval_single
from pyEdgeEval.utils import print_log

from .base import BaseMultilabelEvaluator


class SBDEvaluator(BaseMultilabelEvaluator):
    """SBD evaluator"""

    # Dataset specific attributes
    CLASSES = (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining table",
        "dog",
        "horse",
        "motor bike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    )
    CLS_DIR = "cls"
    INST_DIR = "inst"
    GT_SUFFIX = ".mat"
    PRED_SUFFIX = ".bmp"

    def __init__(
        self,
        dataset_root: str,
        pred_root: str,
        split: str = "val",
        **kwargs,
    ):
        self.dataset_root = dataset_root
        self.pred_root = pred_root

        assert split in ("val", "test")
        self.split = split

        self.CLS_root = osp.join(
            self.dataset_root,
            self.CLS_DIR,
        )
        self.INST_root = osp.join(
            self.dataset_root,
            self.INST_DIR,
        )

        # we can try to load some sample names
        try:
            self.set_sample_names()
        except Exception:
            print_log(
                "Tried to set sample_names, but couldn't",
                logger=self._logger,
            )

    def set_sample_names(self, sample_names=None, split_file=None):
        """priortizes `sample_names` more than `split_file`"""
        if sample_names is None:
            # load sample_names from split file
            if split_file is None:
                split_file = osp.join(self.dataset_root, f"{self.split}.txt")

            assert osp.exists(split_file), f"ERR: {split_file} does not exist!"

            print_log(f"Loading samples from {split_file}", logger=self._logger)
            with open(split_file, "r") as f:
                sample_names = f.read().splitlines()

        assert isinstance(
            sample_names, list
        ), f"ERR: sample_names should be a list but got {type(sample_names)}"
        assert len(sample_names) > 0, "ERR: sample_names is empty"
        self._sample_names = sample_names

    def set_eval_params(
        self,
        eval_mode=None,
        scale: float = 1.0,
        apply_thinning: bool = False,
        apply_nms: bool = False,
        instance_sensitive: bool = True,
        max_dist: float = 0.02,
        skip_if_nonexistent: bool = False,
        kill_internal: bool = False,
        **kwargs,
    ) -> None:

        assert 0 < scale <= 1, f"ERR: scale ({scale}) is not valid"
        self.scale = scale
        self.apply_thinning = apply_thinning
        self.apply_nms = apply_nms

        if eval_mode == "pre-seal":
            print_log("Using Pre-SEAL params", logger=self._logger)
            self.max_dist = 0.02
            self.kill_internal = True
            self.skip_if_nonexistent = True
            self.instance_sensitive = False
        elif eval_mode == "post-seal":
            print_log("Using Post-SEAL params", logger=self._logger)
            self.max_dist = 0.02
            self.kill_internal = False
            self.skip_if_nonexistent = False
            self.instance_sensitive = True
        elif eval_mode == "high-quality":
            print_log(
                "Using params for high-quality annotations", logger=self._logger
            )
            self.max_dist = 0.0075
            self.kill_internal = False
            self.skip_if_nonexistent = False
            self.instance_sensitive = True
        else:
            print_log("Using custom params", logger=self._logger)
            self.max_dist = max_dist
            self.kill_internal = kill_internal
            self.skip_if_nonexistent = skip_if_nonexistent
            self.instance_sensitive = instance_sensitive

        if self.kill_internal and self.instance_sensitive:
            print_log(
                "kill_internal and instance_sensitive are both True which will conflict with each either",
                logger=self._logger,
            )

    @property
    def eval_params(self):
        return dict(
            scale=self.scale,
            apply_thinning=self.apply_thinning,
            apply_nms=self.apply_nms,
            max_dist=self.max_dist,
            kill_internal=self.kill_internal,
            skip_if_nonexistent=self.skip_if_nonexistent,
        )

    def _before_evaluation(self):
        assert osp.exists(
            self.dataset_root
        ), f"ERR: {self.dataset_root} does not exist"
        assert osp.exists(self.CLS_root), f"ERR: {self.CLS_root} does not exist"
        assert osp.exists(
            self.INST_root
        ), f"ERR: {self.INST_root} does not exist"
        assert osp.exists(
            self.pred_root
        ), f"ERR: {self.pred_root} does not exist"

    def evaluate_category(
        self,
        category,
        thresholds,
        nproc,
        save_dir,
    ):
        self._before_evaluation()
        assert (
            0 < category < len(self.CLASSES) + 1
        ), f"ERR: category={category} is not in range ({len(self.CLASSES) + 1})"

        # populate data (samples)
        data = []
        for sample_name in self.sample_names:
            # FIXME: naming scheme differs sometimes
            category_dir = f"class_{str(category).zfill(3)}"

            # GT file paths
            cls_path = osp.join(self.CLS_root, f"{sample_name}{self.GT_SUFFIX}")
            if self.instance_sensitive:
                inst_path = osp.join(
                    self.INST_root, f"{sample_name}{self.GT_SUFFIX}"
                )
                assert osp.exists(inst_path), f"ERR: {inst_path} is not valid"
            else:
                inst_path = None

            assert osp.exists(cls_path), f"ERR: {cls_path} is not valid"

            # prediction file path
            # sample_name = sample_name.split("/")[1]  # assert city/img
            pred_path = osp.join(
                self.pred_root,
                category_dir,
                f"{sample_name}{self.PRED_SUFFIX}",
            )

            data.append(
                dict(
                    name=sample_name,
                    category=category,
                    thresholds=thresholds,
                    # file paths
                    cls_path=cls_path,
                    inst_path=inst_path,
                    pred_path=pred_path,
                    **self.eval_params,
                )
            )

        assert len(data) > 0, "ERR: no evaluation data"

        # evaluate
        (sample_metrics, threshold_metrics, overall_metric) = calculate_metrics(
            eval_single=sbd_eval_single,
            thresholds=thresholds,
            samples=data,
            nproc=nproc,
        )

        # save metrics
        if save_dir:
            print_log("Saving category results", logger=self._logger)
            save_category_results(
                root=save_dir,
                category=category,
                sample_metrics=sample_metrics,
                threshold_metrics=threshold_metrics,
                overall_metric=overall_metric,
            )

        return overall_metric
