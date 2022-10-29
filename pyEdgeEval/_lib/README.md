Note:
- The files in `src` and `include` are borrowed from the [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) without any changes. I respect their license and have added a changelog at the bottom. Their license applies to these files.
- `src/benms.cc` is borrowed from [pdollar's structured edge detection toolbox v3.0](https://github.com/pdollar/edges/blob/master/private/edgesNmsMex.cpp). This code is under MSR-LA.
- I created Cython APIs (`correspond_pixels.pyx` and `nms.pyx`). Thank you Britefury for your wonderful work in [py-bsds500](https://github.com/Britefury/py-bsds500).

TODO:
- [x] wrapper for pixel matching algorithm
- [x] wrapper for NMS
- [ ] clean up warnings (unused variables)
- [ ] add stub file for cython wrapped functions

Changelog for following files that follow GNU GPL License:
- 2022/5/12: no changes to the original files (the files are in `include` and `src`)
