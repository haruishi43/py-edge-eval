
When reading `.mat` files that are created with the newer vesions of MATLAB, `scipy` won't be able to open them.
You would need to implement your own HDF5 reader, or use one of the following packages:

```Bash
pip install pymatreader
pip install mat73
```

`pyEdgeEval.bsds.loadmat` already implements a way to load the newer `.mat` files by setting `use_mat73=True`.
