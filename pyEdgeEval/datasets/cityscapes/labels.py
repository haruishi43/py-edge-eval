#!/usr/bin/env python3

"""
References:
- https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
"""

import numpy as np

from cityscapesscripts.helpers.labels import labels

__all__ = [
    "convert_label2trainId",
    "inst_labelIds",
    "label_mapping",
]

label2trainId = {label.id: label.trainId for label in labels}
trainId2name = {label.trainId: label.name for label in labels}
trainId2color = {label.trainId: label.color for label in labels}

# NOTE: same as trainId?
label_mapping = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}

inst_labelIds = [
    24,  # "person"
    25,  # "rider"
    26,  # "car"
    27,  # "truck"
    28,  # "bus"
    31,  # "train"
    32,  # "motorcycle"
    33,  # "bicycle"
]


def convert_label2trainId(label_img: np.ndarray) -> np.ndarray:
    """python version of `labelid2trainid` function"""

    if len(label_img.shape) == 2:
        h, w = label_img.shape
    elif len(label_img.shape) == 3:
        h, w, c = label_img.shape
        assert c == 1, f"ERR: input label has {c} channels which should be 1"
    else:
        raise ValueError()

    # 1. create an array populated with 255
    trainId_img = 255 * np.ones((h, w), dtype=np.uint8)  # 8-bit array

    # 2. map all pixels in the `label_mapping` dict
    for labelId, trainId in label_mapping.items():
        idx = label_img == labelId
        trainId_img[idx] = trainId

    return trainId_img
