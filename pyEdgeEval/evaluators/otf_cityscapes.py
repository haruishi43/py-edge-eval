#!/usr/bin/env python3

import os.path as osp

from pyEdgeEval.common.multi_label import (
    calculate_metrics,
    save_category_results,
)
from pyEdgeEval.utils import print_log

from .cityscapes import CityscapesEvaluator


class OTFCityscapesEvaluator(CityscapesEvaluator):
    """On-The-Fly cityscapes dataset evaluator

    - On-The-Fly (OTF) creation of GT edges
        - needs GT segmentation and instance maps
    - Scales the masks first before generating the edges
        - non-OTF could create edges that are too thin if scaled down
    """

    # Dataset specific attributes
    SEG_SUFFIX = "_gtFine_labelIds.png"
    INST_SUFFIX = "_gtFine_instanceIds.png"

    radius = 2

    def __init__(
        self,
        radius: int = 1,
        **kwargs,
    ):
        self.radius = radius
        super().__init__(**kwargs)

    @property
    def eval_params(self):
        return dict(
            scale=self.scale,
            apply_thinning=self.apply_thinning,
            apply_nms=self.apply_nms,
            max_dist=self.max_dist,
            kill_internal=self.kill_internal,
            skip_if_nonexistent=self.skip_if_nonexistent,
            radius=self.radius,
            num_labels=34,
            ignore_labels=[2, 3],
            num_classes=len(self.CLASSES),  # not used
        )

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

        # create data
        data = []
        for sample_name in self.sample_names:

            # FIXME: naming scheme differs sometimes
            category_dir = f"class_{str(category).zfill(3)}"

            # GT file paths
            seg_path = osp.join(
                self.gtFine_root,
                f"{sample_name}{self.SEG_SUFFIX}",
            )
            assert osp.exists(seg_path), f"ERR: {seg_path} is not valid"
            if self.instance_sensitive:
                inst_path = osp.join(
                    self.gtFine_root,
                    f"{sample_name}{self.INST_SUFFIX}",
                )
                assert osp.exists(inst_path), f"ERR: {inst_path} is not valid"
            else:
                inst_path = None

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
                    seg_path=seg_path,
                    inst_path=inst_path,
                    pred_path=pred_path,
                    **self.eval_params,
                )
            )

        assert len(data) > 0, "ERR: no evaluation data"

        # evaluate
        (sample_metrics, threshold_metrics, overall_metric) = calculate_metrics(
            thresholds=thresholds,
            samples=data,
            nproc=nproc,
            dataset_type="otf_cityscapes",
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
