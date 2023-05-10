#!/usr/bin/env python3

import numpy as np


def add_ignore_pixel(
    cls_seg: np.ndarray,
    border_px: int = 5,
    ignore_id: int = 0,
):
    """Add ignore pixels around the segmentation boundaries

    In SEAL, this process was added since SBD has issues when segmentation map
    touches the image boundaries.
    """
    h, w = cls_seg.shape
    cls_seg[0:border_px, :] = ignore_id
    cls_seg[h - border_px : h, :] = ignore_id
    cls_seg[:, 0:border_px] = ignore_id
    cls_seg[:, w - border_px : w] = ignore_id
    return cls_seg


def convert_inst_seg(inst_seg, inst_cats, present_cats):
    """Convert the instance segmentation format so it's compatible with mask2edge

    Args:
        inst_seg (np.ndarray): original instance segmentation
        inst_cats (np.ndarray): instance labels
        present_cats (np.ndarray): present categories in the instance segmentation map
    Returns:
        output instance segmentation map (np.ndarray with np.int32)
    """
    # output requires int32
    out = np.zeros_like(inst_seg, dtype=np.int32)
    for present_cat in present_cats:
        tmp_id = str(present_cat).zfill(2)
        num_inst = 0
        for i, cat in enumerate(inst_cats):
            if present_cat == cat:
                out_id = int(tmp_id + str(num_inst).zfill(3))
                # indexed from 1
                out[inst_seg == (i + 1)] = out_id
                num_inst += 1
    return out
