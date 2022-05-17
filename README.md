# Python Binding for Edge Evaluation Code

## Installation

```Bash
# Install dependencies
pip install -r requirements.txt

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
