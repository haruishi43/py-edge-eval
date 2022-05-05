#!/usr/bin/env python3

import os.path as osp

from setuptools import find_packages, setup
from setuptools.extension import Extension
from Cython.Build import cythonize


def readme():
    with open("README.md") as f:
        content = f.read()
    return content


def find_version():
    version_file = "pyEdgeEval/info.py"
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


def find_author():
    version_file = "pyEdgeEval/info.py"
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__author__"]


def find_email():
    version_file = "pyEdgeEval/info.py"
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__email__"]


def find_description():
    version_file = "pyEdgeEval/info.py"
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__description__"]


def find_url():
    version_file = "pyEdgeEval/info.py"
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__url__"]


def get_requirements(filename="requirements.txt"):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), "r") as f:
        requires = [line.replace("\n", "") for line in f.readlines()]
    return requires


def get_long_description():
    with open("README.md") as f:
        long_description = f.read()

    try:
        import github2pypi

        return github2pypi.replace_url(
            slug="haruishi43/equilib", content=long_description
        )
    except Exception:
        return long_description


def get_extentions():
    """Manage and generate Extensions

    - "correspond_pixels.pyx"
    """

    # FIXME: possibly glob it instead?
    source_files = [
        "correspond_pixels.pyx",
        "src/csa.cc",
        "src/Exception.cc",
        "src/kofn.cc",
        "src/match.cc",
        "src/Matrix.cc",
        "src/Random.cc",
        "src/String.cc",
        "src/Timer.cc",
    ]
    sources = []
    root = osp.join("pyEdgeEval", "_lib")
    for s in source_files:
        sources.append(osp.join(root, s))

    extensions = [
        Extension(
            "pyEdgeEval._lib.correspond_pixels",
            sources=sources,
            include_dirs=[osp.join(root, "include")],
            language="c++",
            extra_compile_args=["-DNOBLAS"],
        ),
    ]

    return extensions


setup(
    name="pyEdgeEval",
    version=find_version(),
    author=find_author(),
    author_email=find_email(),
    description=find_description(),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url=find_url(),
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.6",
    keywords=["Edge Detection", "Semantic Boundary Detection", "Computer Vision"],
    packages=find_packages(
        exclude=[
            "github2pypi",
            "tests",
            "scripts",
            "tools",
            "data",
            ".readme",
            "build",
        ]
    ),
    install_requires=["numpy"],
    include_package_data=False,  # used for MANIFEST.in
    zip_safe=False,
    ext_modules=cythonize(get_extentions()),
)
