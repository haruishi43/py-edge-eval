# Licenses

- C/C++ files used for `correspond_pixels` module are borrowed from BSDS500 benchmark which are under __GNU General Public License v2.0__. The files retain their license (marked in the header). Note that I have added [changelog](./pyEdgeEval/_lib/README.md) for any changes to their codes.
- `_lib/src/benms.cc` is borrowed from [Structured Edge Detection Toolbox V3.0](https://github.com/pdollar/edges) which is under the Microsoft Research License.
- I have borrowed and modified codebases from [`mmcv`](https://github.com/open-mmlab/mmcv) (specifically `logger.py`, `path.py`, `progressbar.py` and `timer.py`) which are licensed under __Apache 2.0 license__.
- Please repect their licenses and agree to their terms before using this software.

The SBD benchmark (Cityscapes benchmark is a derivative of this work):
```
@InProceedings{BharathICCV2011,
  author       = "Bharath Hariharan and Pablo Arbelaez and Lubomir Bourdev and Subhransu Maji and Jitendra Malik",
  title        = "Semantic Contours from Inverse Detectors",
  booktitle    = "International Conference on Computer Vision (ICCV)",
  year         = "2011",
}
```

The BSDS500 benchmark (SBD benchmark is a derivative of this work):
```
@InProceedings{BSDS500TPAMI2011,
  author       = "P. Arbelaez, M. Maire, C. Fowlkes and J. Malik.",
  title        = "Contour Detection and Hierarchical Image Segmentation",
  booktitle    = "IEEE TPAMI",
  year         = "2011",
}
```

MMCV:
```
@misc{mmcv,
  title={{MMCV: OpenMMLab} Computer Vision Foundation},
  author={MMCV Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmcv}},
  year={2018}
}
```
