# Documents

## Update documentation to Github Pages

```Bash
# auto generate api


# test document
make html
python -m http.server --directory docs/build/html 8889

# update gh-pages
make update-gh-pages
```

TODO:
- [x] basic workflow
- [ ] workflow for documenting APIs (automatic parsing of docstrings)

NOTES:
- use [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) (`sphinx.ext.napoleon`)
