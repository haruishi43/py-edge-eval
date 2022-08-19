# Documents

## Update documentation to Github Pages

```Bash
# auto generate api rst
sphinx-apidoc -f -o docs/source -H pyEdgeEval ../pyEdgeEval/

# geneate html
make html

# check html
python -m http.server --directory docs/build/html 8889

# update gh-pages
make update-gh-pages
```

TODO:
- [x] basic workflow
- [ ] workflow for documenting APIs (automatic parsing of docstrings)
- [ ] docstrings for cython

NOTES:
- use [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) (`sphinx.ext.napoleon`)
  - [napoleon documentation](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
  - [Example Google Sytle Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
    - [output](https://11ohina017.github.io/google_style_code/index.html)
- sphinx guides:
  - [an example pypi project](https://pythonhosted.org/an_example_pypi_project/sphinx.html)
