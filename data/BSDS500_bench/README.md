# BSDS500 test/sample data

The directories here, `groundTruth`, `png`, and `test_2` are from [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).
The included files are one of many tests included in `BSR/bench/data`, that provides a way to compare the matching algorithm in `pyEdgeEval` with the original MATLAB implementation.
Specifically, we implemented a part of the `test_bench.m`:
```
%% 2. boundary benchmark for results stored as contour images

imgDir = 'data/images';
gtDir = 'data/groundTruth';
pbDir = 'data/png';
outDir = 'data/test_2';
mkdir(outDir);
nthresh = 5;

tic;
boundaryBench(imgDir, gtDir, pbDir, outDir, nthresh);
toc;
```
since the other tests are mainly for segmentation tasks.
