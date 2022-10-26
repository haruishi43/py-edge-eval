#!/usr/bin/env python3

"""Cityscapes Evaluator with forced half scale

Need to create half scale edge GTs with the prefix `_half_edge.png`

This is different from CASENet, SEAL, and DFF way of evaluating.

The evaluation outcomes are generally lower because we have lower recall.
"""

from pyEdgeEval.utils import print_log

from .cityscapes import CityscapesEvaluator


class HalfCityscapesEvaluator(CityscapesEvaluator):
    """Half-scale Cityscapes dataset evaluator

    - used GTs that are preprocessed to half scale
    - half scale evaluations are common for this dataset to speed up the process
    """

    # Dataset specific attributes
    RAW_EDGE_SUFFIX = "_gtProc_half_raw_edge.png"
    THIN_EDGE_SUFFIX = "_gtProc_half_thin_edge.png"
    RAW_ISEDGE_SUFFIX = "_gtProc_half_raw_isedge.png"
    THIN_ISEDGE_SUFFIX = "_gtProc_half_thin_isedge.png"

    def set_eval_params(
        self,
        eval_mode=None,
        apply_thinning: bool = False,
        apply_nms: bool = False,
        instance_sensitive: bool = True,
        max_dist: float = 0.0035,
        skip_if_nonexistent: bool = False,
        kill_internal: bool = False,
        **kwargs,
    ) -> None:

        self.apply_thinning = apply_thinning
        self.apply_nms = apply_nms

        if eval_mode == "pre-seal":
            print_log("Using Pre-SEAL params", logger=self._logger)
            assert (
                not instance_sensitive
            ), "Pre-SEAL configuration doesn't support instance sensitive edges"
            self.max_dist = 0.02
            self.kill_internal = True
            self.skip_if_nonexistent = True
            self.instance_sensitive = False
        elif eval_mode == "post-seal":
            print_log("Using Post-SEAL params", logger=self._logger)
            print_log(f"Using max_dist: {max_dist}", logger=self._logger)
            print_log(
                f"Using instance sensitive={instance_sensitive}",
                logger=self._logger,
            )
            self.max_dist = max_dist
            if instance_sensitive:
                # instance-sensitive
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
            scale=1,  # don't tamper with scale since we're using half scale already
            apply_thinning=self.apply_thinning,
            apply_nms=self.apply_nms,
            max_dist=self.max_dist,
            kill_internal=self.kill_internal,
            skip_if_nonexistent=self.skip_if_nonexistent,
            num_classes=len(self.CLASSES),
        )
