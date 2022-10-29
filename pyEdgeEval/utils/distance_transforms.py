#!/usr/bin/env python3

import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

__all__ = ["mask2bdry"]


def cv2_mask2bdry(m, ignore_mask, radius, quality):
    inner = cv2.distanceTransform(
        ((m + ignore_mask) > 0).astype(np.uint8), cv2.DIST_L2, quality
    )
    outer = cv2.distanceTransform(
        ((1.0 - m) > 0).astype(np.uint8), cv2.DIST_L2, quality
    )
    dist = outer + inner

    dist[dist > radius] = 0
    dist = (dist > 0).astype(np.uint8)
    return dist


def scipy_mask2bdry(m, ignore_mask, radius):
    inner = distance_transform_edt((m + ignore_mask) > 0)
    outer = distance_transform_edt(1.0 - m)
    dist = outer + inner

    dist[dist > radius] = 0
    dist = (dist > 0).astype(np.uint8)
    return dist


def mask2bdry(
    mask: np.ndarray,
    ignore_mask: np.ndarray,
    radius: int,
    use_cv2: bool = True,
    quality: int = 0,
) -> np.ndarray:
    if use_cv2:
        return cv2_mask2bdry(
            m=mask,
            ignore_mask=ignore_mask,
            radius=radius,
            quality=quality,
        )
    else:
        return scipy_mask2bdry(
            m=mask,
            ignore_mask=ignore_mask,
            radius=radius,
        )
