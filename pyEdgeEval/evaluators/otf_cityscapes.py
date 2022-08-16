#!/usr/bin/env python3

from .cityscapes import CityscapesEvaluator


class OTFCityscapesEvaluator(CityscapesEvaluator):
    """On-The-Fly cityscapes dataset evaluator

    - On-The-Fly (OTF) creation of GT edges
        - needs GT segmentation and instance maps
    """

    # Dataset specific attributes
    INST_SUFFIX = "_gtFine_instanceIds.png"

    radius = 2

    def __init__(
        self,
        radius: int = 2,
        **kwargs,
    ):
        self.radius = radius
        super().__init__(**kwargs)
