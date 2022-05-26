#!/usr/bin/env python3

import sys

import numpy as np
from PIL import Image


def default_encoding(trainIds_edges: np.ndarray):
    num_trainIds, h, w = trainIds_edges.shape

    cat_edge_map = np.zeros((h, w), dtype=np.uint32)
    for trainId in range(num_trainIds):
        edge_map = trainIds_edges[trainId]
        cat_edge_map = cat_edge_map + (2**trainId) * edge_map

    return cat_edge_map


def rgb_encoding(trainIds_edges: np.ndarray):
    num_trainIds, h, w = trainIds_edges.shape

    cat_edge_b = np.zeros((h, w), dtype=np.uint8)
    cat_edge_g = np.zeros((h, w), dtype=np.uint8)
    cat_edge_r = np.zeros((h, w), dtype=np.uint8)
    cat_edge_png = np.zeros((h, w, 3), dtype=np.uint8)

    for trainId in range(num_trainIds):
        edge_map = trainIds_edges[trainId]
        if trainId >= 0 and trainId < 8:
            cat_edge_b = cat_edge_b + (2**trainId) * edge_map
        elif trainId >= 8 and trainId < 16:
            cat_edge_g = cat_edge_g + (2 ** (trainId - 8)) * edge_map
        elif trainId >= 16 and trainId < 24:
            cat_edge_r = cat_edge_r + (2 ** (trainId - 16)) * edge_map
        else:
            raise ValueError()

    cat_edge_png[:, :, 0] = cat_edge_r
    cat_edge_png[:, :, 1] = cat_edge_g
    cat_edge_png[:, :, 2] = cat_edge_b

    return cat_edge_png


def binary_decoding(edge_path: str, h: int, w: int, num_trainIds: int):
    """Load binary file to edge map
    - Assumes that the input file is a 32-bit array
    - The saved file does not have shape information
    - output type is `uint32`
    """
    # load from binary file
    b = np.fromfile(edge_path, dtype=np.uint32)
    if b.dtype.byteorder == ">" or (
        b.dtype.byteorder == "=" and sys.byteorder == "big"
    ):
        b = b[:, ::-1]

    b = b.reshape(h, w)[:, :, None]  # reshape and make it 3 channels
    b = b.reshape(h, w)[:, :, None]  # reshape and make it 3 channels
    ub = np.unpackbits(
        b.view(np.uint8),
        axis=2,
        count=num_trainIds,
        bitorder="little",
    )

    edge = np.transpose(ub, (2, 0, 1))
    return edge


def rgb_decoding(
    edge_path: str,
    num_trainIds: int,
    scale: float,
    is_png: bool = True,
):
    """Load RGB file to edge map
    - tested with '.png' file
    - output type is `uint8`
    """
    edge = Image.open(edge_path)
    _edge = np.array(edge)
    (h, w, _) = _edge.shape
    oh, ow = int(h * scale + 0.5), int(w * scale + 0.5)
    edge = edge.resize((ow, oh), Image.NEAREST)

    if is_png:
        edge = np.array(edge, dtype=np.uint8)
        edge = np.unpackbits(
            edge,
            axis=2,
        )[:, :, -1 : -(num_trainIds + 1) : -1]
    else:
        # tif
        edge = np.array(edge).astype(np.uint32)  # int32 -> uint32
        edge = edge[:, :, None]
        edge = np.unpackbits(
            edge.view(np.uint8),
            axis=2,
            count=num_trainIds,
            bitorder="little",
        )

    edge = np.ascontiguousarray(edge.transpose(2, 0, 1))
    return edge
