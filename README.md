# Python Binding for Edge Evaluation Code

## Installation

```Bash
# Install dependencies
pip install -r requirements.txt
pip install opencv-python  # cv2

# Option 1. build the project (doesn't install as a package)
python setup.py build_ext --inplace

# Option 2. install as a pip package (install as a package)
git clone https://github.com/haruishi43/py-edge-eval.git
pip install -e .

# Option 3. install without cloning the project
pip install pyEdgeEval
```

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
python scripts/evaluate_sbd.py <path/to/sbd> <path/to/pred> <path/to/output> --category=15 --thresholds=5 --nproc=8
```

Per-category evaluation is currently supported.


## CityScapes

First, create GT data using this script:
```Bash
# if you plan on evaluating with instance-sensitive edges (IS edges)
python scripts/convert_dataset/cityscapes.py
# if you plan on evaluating with instance-insensitive edges
python scripts/convert_dataset/cityscapes.py --insensitive
```

Evaluation script:
```Bash
python scripts/evaluate_cityscapes.py <path/to/cityscapes> <path/to/predictions> <path/to/output> --category 14 --thresholds 99 --nproc 8
```

For instance-insensitive edges, you would need to supply `--pre-seal` argument.
