# Python Binding for Edge Evaluation Code

## Installation

```Bash
# Install dependencies
pip install -r requirements.txt
# install mat loader of your choice
pip install pymatreader
pip install mat73

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
python scripts/evaluate_bsds500.py <path/to/bsds500> <path/to/pred> <path/to/output> --thresholds=5
```

Tested with [@xwjabc's HED implementation](https://github.com/xwjabc/hed).
For better testing, the number of thresholds should be 99 (but it will take a couple hours to finish testing).
