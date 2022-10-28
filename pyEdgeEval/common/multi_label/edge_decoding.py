#!/usr/bin/env python3

"""Decoding functions for multi-label edges"""

import sys

import numpy as np
from PIL import Image


def binary_multilabel_decoding(
    edge_path: str,
    h: int,
    w: int,
    num_classes: int,
):
    """Load binary file to edge map

    - Assumes that the input file is a 32-bit array
    - The saved file does not have shape information
    - output type is `uint32`

    TODO: currently not used (deprecated)
    """
    # load from binary file
    b = np.fromfile(edge_path, dtype=np.uint32)
    if b.dtype.byteorder == ">" or (
        b.dtype.byteorder == "=" and sys.byteorder == "big"
    ):
        b = b[:, ::-1]

    b = b.reshape(h, w)[:, :, None]  # reshape and make it 3 channels

    # TODO: scale?

    ub = np.unpackbits(
        b.view(np.uint8),
        axis=2,
        count=num_classes,
        bitorder="little",
    )

    edge = np.transpose(ub, (2, 0, 1))
    return edge


def load_scaled_edge(edge_path: str, scale: float):
    """Load edge from file and scale it

    Returns:
        edge: PIL.Image.Image
        (height, width): tuple of int
    """
    edge = Image.open(edge_path)
    (w, h) = edge.size
    # _edge = np.array(edge)  # allocation costs
    # (h, w, _) = _edge.shape
    height, width = int(h * scale + 0.5), int(w * scale + 0.5)
    edge = edge.resize((width, height), Image.Resampling.NEAREST)
    return edge, (height, width)


def decode_png(edge: Image.Image, num_classes: int):
    """Decode png format Image into multi-label edge"""
    edge = np.array(edge, dtype=np.uint8)
    edge = np.unpackbits(
        edge,
        axis=2,
    )[:, :, -1 : -(num_classes + 1) : -1]
    edge = np.ascontiguousarray(edge.transpose(2, 0, 1))
    return edge


def decode_tif(edge: Image.Image, num_classes: int):
    """Decode tif format Image into multi-label edge"""
    edge = np.array(edge).astype(np.uint32)  # int32 -> uint32
    edge = edge[:, :, None]
    edge = np.unpackbits(
        edge.view(np.uint8),
        axis=2,
        count=num_classes,
        bitorder="little",
    )
    edge = np.ascontiguousarray(edge.transpose(2, 0, 1))
    return edge
