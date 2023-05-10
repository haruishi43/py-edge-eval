#!/usr/bin/env python3

import os.path as osp
from warnings import warn

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

    GT_DIR = "gtEval"  # pre-processed GTs

    EDGE_SUFFIX = None
    ISEDGE_SUFFIX = None
    RAW_EDGE_SUFFIX = "_raw_edge.png"
    THIN_EDGE_SUFFIX = "_thin_edge.png"
    RAW_ISEDGE_SUFFIX = "_raw_isedge.png"
    THIN_ISEDGE_SUFFIX = "_thin_isedge.png"

    SEG_SUFFIX = "_labelIds.png"
    PRED_SUFFIX = ".png"

    def __init__(
        self,
        dataset_root: str,
        pred_root: str,
        split: str = "val",
        thin: bool = False,
        gt_dir=None,
        pred_suffix=None,
        **kwargs,
    ):
        self.dataset_root = dataset_root
        self.pred_root = pred_root

        assert split in ("val", "test")
        self.split = split
        self.thin = thin
        if self.thin:
            print_log(
                "Using `thin` mode; setting the suffix respectively",
                logger=self._logger,
            )
            self.EDGE_SUFFIX = self.THIN_EDGE_SUFFIX
            self.ISEDGE_SUFFIX = self.THIN_ISEDGE_SUFFIX
        else:
            print_log(
                "Using `raw` mode; setting the suffix respectively",
                logger=self._logger,
            )
            self.EDGE_SUFFIX = self.RAW_EDGE_SUFFIX
            self.ISEDGE_SUFFIX = self.RAW_ISEDGE_SUFFIX

        # change dataset directory and suffix
        if gt_dir:
            print_log(f"changing GT_DIR to {gt_dir}", logger=self._logger)
            self.GT_DIR = gt_dir
        if pred_suffix:
            print_log(
                f"changing PRED_SUFFIX to {pred_suffix}", logger=self._logger
            )
            self.PRED_SUFFIX = pred_suffix

        # set parameters
        self.gtEval_root = osp.join(self.dataset_root, self.GT_DIR)

        if pred_suffix is not None:
            print_log(f"changed suffix to {pred_suffix}")
            self.PRED_SUFFIX = pred_suffix

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

            if instance_sensitive:
                warn("Pre-SEAL mode doesn't support instance sensitive")
            self.instance_sensitive = False
        elif eval_mode == "post-seal":
            print_log("Using Post-SEAL params", logger=self._logger)
            self.max_dist = 0.02
            if instance_sensitive:
                self.instance_sensitive = True
                self.kill_internal = False
                self.skip_if_nonexistent = False
            else:
                self.instance_sensitive = False
                self.kill_internal = True
                self.skip_if_nonexistent = True
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
            num_classes=len(self.CLASSES),
        )

    def _before_evaluation(self):
        assert (
            self._sample_names is not None
        ), "ERR: no samples yet. load them before evaluation"
        assert osp.exists(
            self.dataset_root
        ), f"ERR: {self.dataset_root} does not exist"
        assert osp.exists(
            self.gtEval_root
        ), f"ERR: {self.gtEval_root} does not exist"
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
            if self.instance_sensitive:
                edge_path = osp.join(
                    self.gtEval_root,
                    f"{sample_name}{self.ISEDGE_SUFFIX}",
                )
            else:
                edge_path = osp.join(
                    self.gtEval_root,
                    f"{sample_name}{self.EDGE_SUFFIX}",
                )
            seg_path = osp.join(
                self.gtEval_root,
                f"{sample_name}{self.SEG_SUFFIX}",
            )
            assert osp.exists(edge_path), f"ERR: {edge_path} is not valid"
            assert osp.exists(seg_path), f"ERR: {seg_path} is not valid"

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
                    edge_path=edge_path,
                    seg_path=seg_path,
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


class ReannoSBDEvaluator(SBDEvaluator):
    """Reannotated SBD validation split evaluator."""

    RAW_EDGE_SUFFIX = "_reanno_raw_edge.png"
    THIN_EDGE_SUFFIX = "_reanno_thin_edge.png"
    RAW_ISEDGE_SUFFIX = "_reanno_raw_isedge.png"
    THIN_ISEDGE_SUFFIX = "_reanno_thin_isedge.png"

    def __init__(
        self,
        dataset_root: str,
        pred_root: str,
        split: str = "val",
        thin: bool = False,
        gt_dir=None,
        pred_suffix=None,
        **kwargs,
    ):
        self.dataset_root = dataset_root
        self.pred_root = pred_root

        assert split in ("val", "test")
        self.split = split
        self.thin = thin
        if self.thin:
            print_log(
                "Using `thin` mode; setting the suffix respectively",
                logger=self._logger,
            )
            self.EDGE_SUFFIX = self.THIN_EDGE_SUFFIX
            self.ISEDGE_SUFFIX = self.THIN_ISEDGE_SUFFIX
        else:
            print_log(
                "Using `raw` mode; setting the suffix respectively",
                logger=self._logger,
            )
            self.EDGE_SUFFIX = self.RAW_EDGE_SUFFIX
            self.ISEDGE_SUFFIX = self.RAW_ISEDGE_SUFFIX

        # change dataset directory and suffix
        if gt_dir:
            print_log(f"changing GT_DIR to {gt_dir}", logger=self._logger)
            self.GT_DIR = gt_dir
        if pred_suffix:
            print_log(
                f"changing PRED_SUFFIX to {pred_suffix}", logger=self._logger
            )
            self.PRED_SUFFIX = pred_suffix

        # set parameters
        self.gtEval_root = osp.join(self.dataset_root, self.GT_DIR)

        if pred_suffix is not None:
            print_log(f"changed suffix to {pred_suffix}")
            self.PRED_SUFFIX = pred_suffix

        # we can try to load some sample names
        try:
            self.set_sample_names()
        except Exception:
            print_log(
                "Tried to set sample_names, but couldn't",
                logger=self._logger,
            )

    def set_eval_params(
        self,
        eval_mode=None,
        scale: float = 1.0,
        apply_thinning: bool = False,
        apply_nms: bool = False,
        instance_sensitive: bool = True,
        max_dist: float = 0.0075,
        **kwargs,
    ) -> None:
        print_log(
            "Using params for re-annotated test split", logger=self._logger
        )
        assert 0 < scale <= 1, f"ERR: scale ({scale}) is not valid"
        self.scale = scale
        self.apply_thinning = apply_thinning
        self.apply_nms = apply_nms

        self.max_dist = max_dist
        self.kill_internal = False
        self.skip_if_nonexistent = False
        self.instance_sensitive = instance_sensitive

        if self.kill_internal and self.instance_sensitive:
            print_log(
                "kill_internal and instance_sensitive are both True which will conflict with each either",
                logger=self._logger,
            )
