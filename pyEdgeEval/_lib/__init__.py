#!/usr/bin/env python3

try:
    from pyEdgeEval._lib.correspond_pixels import correspond_pixels
except ImportError:
    raise ImportError("`correspond_pixels` hasn't been compiled yet")

__all__ = ["correspond_pixels"]
