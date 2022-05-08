#!/bin/bash

pip uninstall pyEdgeEval
rm -rf build
rm -rf dist
rm -rf pyEdgeEval.egg-info
rm pyEdgeEval/_lib/correspond_pixels.cpp
rm pyEdgeEval/_lib/correspond_pixels.cpython*
