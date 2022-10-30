# Developing

## Installation, Building, and Packaging

```Bash
# Install dependencies
pip install -r requirements-dev.txt

# [try to build]
# Option 1. build as a package (`-v` for complete output)
pip install -e . -v
# Option 2. build, but don't install it as a pip package (doesn't conflict with pip installed versions)
python setup.py build_ext --inplace

# [build distribution]
python -m build

# [convert wheel (for linux)]
auditwheel -v repair dist/<package>.whl
mv wheelhouse/<new package>.whl dist/
rm dist/<old package>.whl
```

## Testing validity

BSDS500 provides a benchmark to test if the code is running as expected (on MATLAB).
I used the same benchmark and created a test script to check if the results are the same.
SBD also provides a similar benchmark which I converted to Python.

Theoratically the results should match exactly, but due to the results coming from two complete languages as well as the dependencies being diffferent, it is understandable to have some minor diferences.

For BSDS500 and SBD, I have provided a test script inside `tests`.
The results are very close and near identical, which means that the general functions are working.
Making more detailed test code is WIP.

I also test cityscapes' edge generation code (`convert_dataset/cityscapes.py`) and compared it with the outputs using MATLAB code.

### BSDS500

- Test code: `tests/test_bsds500.py`
- Benchmark data: `data/BSDS500_bench`

### SBD

- Test code: `tests/test_sbd.py`
- Benchmark data: `data/SBD_bench`

### Cityscapes

- Test code: WIP
- Benchmark data: `data/cityscapes_test`


## TODO

- [x] Cython `correspond_pixels` port
- [x] Cython `nms` port
- [x] Building script
- [x] Packaging script
- [x] BSDS evaluation script
- [x] BSDS evaluation test codes
- [x] Plot PR curves
- [x] NMS preprocess script
- [x] SBD evaluation script
- [ ] "thin" GTs for SBD
- [x] Cityscapes evaluation script
- [x] Multiprocessing for evaluation
- [ ] Set random seed for `correspond_pixels`
- [x] Move the scripts into the source code (currently moved to `scripts` for testing)
- [ ] unit test coverage for important functions (would like to make tests for all functions)
- [ ] Make a CLI interface (for evaluation/convert dataset)

## Bugs and Problems

- `pyEdgeEval` and MATLAB results differ slightly. This is due to various factors such as randomness in `correspond_pixels` and slight differences in preprocessing algorithms (`thin`, `kill_internal`, etc...). I would love to do a more comprehensive study on the differences MATLAB and Python, but I believe this benchmark code is currently robust enough to evaluate models. Note that MATLAB results and `pyEdgeEval` results should not be compared against each other because of this.
- Using multiprocessing causes a bug where every run produces different results. The seed for random number generator used in `correspond_pixels.pyx` causes the randomness which is the same as the original MATLAB code. To solve the issue, we would need to set the random seed every time we call `correspond_pixels` so that the results are reproducible (TODO).


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
