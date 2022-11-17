#!/usr/bin/env python3

"""SBD Dataset attributes

- default labelIds adds background ID as 0
- add ignore pixel for the image boundary (21)
"""

# all labels (+ background + ignore)
SBD_labelIds = list(range(20 + 1 + 1))

# removed unused labels (trainId that is 255 and -1)
SBD_label2trainId = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    # 21: 20,
}

SBD_trainId2name = {  # same as instance labels
    0: "aeroplane",
    1: "bicycle",
    2: "bird",
    3: "boat",
    4: "bottle",
    5: "bus",
    6: "car",
    7: "cat",
    8: "chair",
    9: "cow",
    10: "dining table",
    11: "dog",
    12: "horse",
    13: "motor bike",
    14: "person",
    15: "potted plant",
    16: "sheep",
    17: "sofa",
    18: "train",
    19: "tv/monitor",
}

# doesn't include 0 and ignore pixels
SBD_inst_labelIds = list(range(1, 20 + 1))
