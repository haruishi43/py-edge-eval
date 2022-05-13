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
- [ ] Multiprocessing for evaluation
