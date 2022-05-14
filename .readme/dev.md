# Developing

## Installation, Building, and Packaging

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

## TODO

- [x] Cython `correspond_pixels` port
- [x] Cython `nms` port
- [x] Building script
- [x] Packaging script
- [x] BSDS evaluation script
- [x] BSDS evaluation test codes
- [ ] Plot PR curves
- [ ] NMS preprocess script
- [ ] SBD evaluation script
- [x] Multiprocessing for evaluation
- [ ] Set random seed for `correspond_pixels`

## Bugs

- Using multiprocessing causes a bug where every run produces different results. The seed for random number generator used in `correspond_pixels.pyx` causes the randomness which is the same as the original MATLAB code. To solve the issue, we would need to set the random seed every time we call `correspond_pixels` so that the results are reproducible (TODO).
