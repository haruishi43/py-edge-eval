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

## Developing

```Bash
# Install dependencies
pip install -r requirements-dev.txt

# [try to build]
# verbose build
pip install -e . -v

# [build distribution]
python -m build

# [convert wheel (for linux)]
auditwheel -v repair dist/<package>.whl
mv wheelhouse/<new package>.whl dist/
rm dist/<old package>.whl
```
