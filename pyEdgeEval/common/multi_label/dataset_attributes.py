#!/usr/bin/env python3

# all labels except for -1
CITYSCAPES_labelIds = list(range(34))

# removed unused labels (trainId that is 255 and -1)
CITYSCAPES_label2trainId = {
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

CITYSCAPES_trainId2name = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

CITYSCAPES_inst_labelIds = [
    24,  # "person"
    25,  # "rider"
    26,  # "car"
    27,  # "truck"
    28,  # "bus"
    31,  # "train"
    32,  # "motorcycle"
    33,  # "bicycle"
]


if __name__ == "__main__":
    from cityscapesscripts.helpers.labels import labels

    cs_label2trainId = {label.id: label.trainId for label in labels}
    # _trainId2name = {label.trainId: label.name for label in labels}
    # _trainId2color = {label.trainId: label.color for label in labels}

    print(labels)
    print(cs_label2trainId)
    print(len(cs_label2trainId))

    # checks
    for label, trainId in CITYSCAPES_label2trainId.items():
        assert cs_label2trainId[label] == trainId
