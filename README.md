# Python Edge Evaluation Tools

Edge detection tasks heavily relies on the original codes used in BSDS300/500 written in [MATLAB and C++](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).
In the field of computer vision, various edge detection algorithms are now resorting to Python and the various machine learning libraries.
However, due to the fact that not everyone has access to MATLAB and that the original benchmark codes are outdated, evaluating these algorithms, especially on remote servers (i.e., linux environments, docker containers), has been difficult.
This library aims to remove these limitations and make it easy for models to be evaluated and benchmarked.
The original C++ codes used in the MATLAB benchmarks are ported with Cython and the evaluation scripts are rewritten in Python3.
The codebase is created to be extensible and supports various tasks and datasets as well as different evaluation protocols.
To test the validity of the evaluation code, `pyEdgeEval`'s results are compared with the results of the original MATLAB codes.

`pyEdgeEval` is:
- an alternative to the original MATLAB benchmark
- light with minimal dependencies
- modular and easily customizable
- fast (uses multiprocessing and Cython)
- implements common preprocessing algorithms
- supports various tasks and datasets
- supports various evaluation protocols
- edge generation tools
- etc...

Supported tasks:
- Edge Detection
- Semantic Boundary Detection

Supported datasets:
- [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
- [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
- [Cityscapes](https://www.cityscapes-dataset.com) (semantic boundary detection)

Disclaimers:
- The evaluation code does not output results that exactly match the original MATLAB benchmark. This could be for various reasons. The results are, for the most part, close enough. I recommend NOT to compare the results from this evaluation directly with results obtained through the MATLAB code for this reason.
- The codes and algorithms are not perfect. I will not take responsibility for how the code is used (check the license). If there are some bugs or improvements, please submit issues or pull requests.

# Installation

## Dependencies

- `python >= 3.8` (tested on 3.8.x)
- `cv2`

## Installation guide

```Bash
# Install dependencies
pip install -r requirements.txt

# install cv2 (e.g. use pip)
pip install opencv-python

# Option 1. install without cloning the project (only tested on ubuntu with python 3.8)
pip install pyEdgeEval

# Option 2. install as a pip package (install as a package)
git clone https://github.com/haruishi43/py-edge-eval.git
pip install -e .
```

# Converting Cityscapes Dataset for SBD

Script:

```Bash
python scripts/convert_datasets/cityscapes.py --nproc=8
```

NOTE:
- Beaware that using multi-processing will consume at most 10GB per process (I'm working on debugging memory allocation issues).
- `--nonIS` will generate non-IS boundaries.
- The script will generate full resolution training dataset, full resolution validation dataset, and half resolution validation dataset (both raw/thin for validation).

# Evaluation for each datasets

## BSDS500

Script:

```Bash
python scripts/evaluate_bsds500.py <path/to/bsds500> <path/to/pred> <path/to/output> --thresholds=5 --nproc=8
```

Tested with [@xwjabc's HED implementation](https://github.com/xwjabc/hed).
Setting `--nproc` will drastically improve the evaluation.
However, due to the randomness in the original MATLAB (C++) codebase, the results will be different (at most +-0.001 difference).


## SBD

Script:

```Bash
python scripts/evaluate_sbd.py <path/to/sbd> <path/to/pred> <path/to/output> --categories=15 --thresholds=5 --nproc=8
```


## CityScapes

First, create GT data using this script:
```Bash
# if you plan on evaluating with instance-sensitive edges (IS edges)
python scripts/convert_dataset/cityscapes.py --nproc 8
# if you plan on evaluating with non-instance-sensitive edges
python scripts/convert_dataset/cityscapes.py --nonIS --nproc 8
```
The scripts will create two types of edges (raw and thin) for two different scales (half and full).

Evaluation script:
```Bash
python scripts/evaluate_cityscapes.py <path/to/cityscapes> <path/to/predictions> <path/to/output> --categories='[1, 14]' --thresholds 99 --nproc 8
```

`--thin` will enable thinning on predictions and use thinned GTs.
For instance-insensitive edges, you would need to supply `--pre-seal` argument.
You can also preprocess the predictions by passing `--apply-thinning` and/or `--apply-nms` for thinning and NMS respectively.


# License

- The code is released under the MIT License (please refer to the LICENSE file for details).
- C/C++ files used for `correspond_pixels` module are borrowed from BSDS500 benchmark which are under __GNU General Public License v2.0__. The files retain their license (marked in the header). Note that I have added [changelog](./pyEdgeEval/_lib/README.md) for any changes to their codes.
- `_lib/src/benms.cc` is borrowed from [Structured Edge Detection Toolbox V3.0](https://github.com/pdollar/edges) which is under the Microsoft Research License.
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

# Acknowledgements

- [py-bsds500](https://github.com/Britefury/py-bsds500)
  - referenced Cython ports for `correspond_pixels` used in the original MATLAB code
  - referenced python version of the evaluation script for BSDS500
- [cityscapes-preprocess](https://github.com/Chrisding/cityscapes-preprocess)
  - MATLAB script for preprocessing and creating GT edges
- [seal](https://github.com/Chrisding/seal)
  - semantic boundary detection protocols
- [edge_eval_python](https://github.com/Walstruzz/edge_eval_python):
  - referenced implementation for `bwmorph` thinning algorithm and `nms` preprocessing
  - referenced PR-curve visualization codes
- [edges](https://github.com/pdollar/edges)
  - tools for processing edges (written in MATLAB and C++)
