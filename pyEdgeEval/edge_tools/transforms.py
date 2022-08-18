#!/usr/bin/env python3

"""Generic functions for converting masks to edges

NOTE: no dataset specific functions (move then to subdirectories)
"""

from pyEdgeEval.utils import (
    mask_to_onehot,
    mask_label2trainId,
    edge_label2trainId,
)

from .mask2edge_loop import (
    loop_mask2edge,
    loop_instance_mask2edge,
)
from .mask2edge_mp import (
    mp_mask2edge,
    mp_instance_mask2edge,
)


def mask2edge(
    run_type: str,
    instance_sensitive: bool,
    **kwargs,
):
    """Wrapper for all mask2edge

    FIXME: might be unnecessary
    """

    if run_type == "loop":
        if instance_sensitive:
            return loop_instance_mask2edge(**kwargs)
        else:
            return loop_mask2edge(**kwargs)
    elif run_type == "mp":
        if instance_sensitive:
            return mp_instance_mask2edge(**kwargs)
        else:
            return mp_mask2edge(**kwargs)
    else:
        raise ValueError(f"{run_type} is not a valid run type")


class Mask2Edge(object):

    """Transform function"""

    LABEL_IDS = None
    label2trainId = None

    _mask2bdry_kwargs = dict()

    def __init__(
        self,
        labelIds,
        ignore_labelIds=[],
        label2trainId=None,
        radius: int = 2,
        use_cv2: bool = True,
        quality: int = 0,
    ):
        assert len(labelIds) > 0, "ERR: there should be more than 1 labels"
        self.LABEL_IDS = labelIds
        assert isinstance(
            ignore_labelIds, list
        ), "ERR: ignore labelIds should be a list"
        self.ignore_labelIds = ignore_labelIds
        assert radius >= 1, "ERR: radius should be equal or greater than 1"
        self.radius = radius

        # parameters for mask2bdry function
        self._mask2bdry_kwargs = dict(
            radius=self.radius,
            use_cv2=use_cv2,
            quality=quality,
        )

        # if we need to map output edge labels to trainIds
        self.label2trainId = label2trainId

    def __call__(self, mask):
        assert mask.ndim == 2
        onehot_mask = mask_to_onehot(mask, num_classes=len(self.LABEL_IDS))
        edge = mask2edge(
            run_type="loop",
            instance_sensitive=False,
            mask=onehot_mask,
            ignore_labelIds=self.ignore_labelIds,
            **self._mask2bdry_kwargs,
        )

        if self.label2trainId:
            # for convinence, we will convert both mask and edge to trainId
            edge = edge_label2trainId(
                edge=edge, label2trainId=self.label2trainId
            )
            mask = mask_label2trainId(
                mask=mask, label2trainId=self.label2trainId
            )

        return dict(
            edge=edge,
            mask=mask,
        )


class InstanceMask2Edge(Mask2Edge):

    """Transform function (instance sensitive)"""

    def __init__(
        self,
        inst_labelIds,
        **kwargs,
    ):
        assert isinstance(inst_labelIds, (list, tuple))
        self.inst_labelIds = inst_labelIds

        super().__init__(**kwargs)

    def __call__(self, mask, inst_mask):
        assert mask.ndim == 2
        assert mask.shape == inst_mask.shape
        # mask is uint8, inst_mask is int32
        onehot_mask = mask_to_onehot(mask, num_classes=len(self.LABEL_IDS))
        edge = mask2edge(
            run_type="loop",
            instance_sensitive=True,
            mask=onehot_mask,
            inst_mask=inst_mask,
            inst_labelIds=self.inst_labelIds,
            ignore_labelIds=self.ignore_labelIds,
            **self._mask2bdry_kwargs,
        )

        if self.label2trainId:
            # for convinence, we will convert both mask and edge to trainId
            edge = edge_label2trainId(
                edge=edge, label2trainId=self.label2trainId
            )
            mask = mask_label2trainId(
                mask=mask, label2trainId=self.label2trainId
            )

        return dict(
            edge=edge,
            mask=mask,
        )
