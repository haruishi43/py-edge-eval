#!/bin/bash

if [[ $(pip freeze | grep "pyEdgeEval") ]]; then
  echo "pip uninstall pyEdgeEval"
  pip uninstall -y pyEdgeEval
else
  echo "no pip module"
fi
rm -rf build
rm -rf dist
rm -rf pyEdgeEval.egg-info
rm pyEdgeEval/_lib/correspond_pixels.cpp
rm pyEdgeEval/_lib/correspond_pixels.cpython*
rm pyEdgeEval/_lib/nms.cpp
rm pyEdgeEval/_lib/nms.cpython*
