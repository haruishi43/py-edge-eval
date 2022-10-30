# Python Edge Evaluation Tools

Edge detection tasks heavily relies on the original codes used in BSDS300/500 that runs on [MATLAB](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).
In the field of computer vision, various edge detection algorithms are now resorting to Python which supports various machine learning libraries.
However, not everyone has access to MATLAB and the original benchmark codes are outdated.
I created this open-source library, `pyEdgeEval`, to make it easier to evaluate and reproduce recent deep learning models for edge and boundary detection.
The original C++ codes used in the MATLAB benchmarks are ported with Cython and the evaluation scripts are rewritten in Python3.
This means that benchmarking could be easily done on almost any environment, especially on remote servers (i.e., linux environments, docker containers), which have been difficult before.
The codebase is designed to be extensible and supports various tasks and datasets as well as different evaluation protocols.
To test the validity of the evaluation code, `pyEdgeEval`'s results are compared with the results of the original MATLAB codes.
Besides benchmarking, `pyEdgeEval` adds various tools for edge detection such as `mask2edge` transformation.

`pyEdgeEval` is:
- a Python alternative to the original MATLAB benchmark
- light with minimal dependencies
- modular architecture and easily customizable
- relatively fast (uses multiprocessing and Cython)
- implements common preprocessing algorithms
- supports various tasks and datasets (extensible to other datasets)
- supports various evaluation protocols
- edge generation tools
- etc...

**Supported tasks**:
- Edge Detection
- Semantic Boundary Detection

**Supported datasets**:
- [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
- [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
- [Cityscapes](https://www.cityscapes-dataset.com) (semantic boundary detection)

*Disclaimers*:
- The evaluation code does not output results that exactly match the original MATLAB benchmark. This could be for various reasons such as random seeds for matching algorithm. The results are, for the most part, close enough (around 0.01% difference).
- The codes and algorithms are not perfect. I will not take responsibility for how the code is used (check the license(s)).
- If you find some bugs or want to improve this project, please submit issues or pull requests.

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
- I modified codes from other projects and their licenses applies to those files (please refer to [Licenses](./LICENSES.md)).

# Development

See [dev.md](./.readme/dev.md).
