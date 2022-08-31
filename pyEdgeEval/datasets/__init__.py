#!/usr/bin/env python3

# binary edge
from .bsds import bsds_eval_single

# multilabel edge
from .cityscapes import cityscapes_eval_single
from .sbd import sbd_eval_single
from .otf_cityscapes import otf_cityscapes_eval_single

__all__ = [
    "bsds_eval_single",
    "cityscapes_eval_single",
    "sbd_eval_single",
    "otf_cityscapes_eval_single",
]
