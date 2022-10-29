#!/usr/bin/env python3

import os.path as osp

from setuptools import find_packages, setup
from setuptools.extension import Extension
from Cython.Build import cythonize

PKG_NAME = "pyEdgeEval"


def get_long_description():
    with open("README.md") as f:
        long_description = f.read()
    return long_description


def find_version():
    version_file = osp.join(PKG_NAME, "info.py")
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


def find_author():
    version_file = osp.join(PKG_NAME, "info.py")
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__author__"]


def find_email():
    version_file = osp.join(PKG_NAME, "info.py")
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__email__"]


def find_description():
    version_file = osp.join(PKG_NAME, "info.py")
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__description__"]


def find_url():
    version_file = osp.join(PKG_NAME, "info.py")
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__url__"]


def get_requirements(filename="requirements.txt"):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), "r") as f:
        requires = [line.replace("\n", "") for line in f.readlines()]
    return requires


def get_extentions():
    """Manage and generate Extensions

    - `correspond_pixels.pyx`
    - `nms.pyx`

    # FIXME: need to compile sourcefiles everytime... but I guess -fPIC is working some magic
    """

    ROOT = osp.join(PKG_NAME, "_lib")
    extensions = []

    def _add_sources(source_files, root=ROOT):
        _srcs = []
        for s in source_files:
            _srcs.append(osp.join(root, s))
        return _srcs

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
    sources = _add_sources(source_files)

    extensions += [
        Extension(
            f"{PKG_NAME}._lib.correspond_pixels",
            sources=sources,
            include_dirs=[osp.join(ROOT, "include")],
            language="c++",
            extra_compile_args=["-fPIC", "-DNOBLAS"],
        ),
    ]

    source_files = [
        "nms.pyx",
        "src/benms.cc",
        "src/Exception.cc",
        "src/Matrix.cc",
        "src/Random.cc",
        "src/String.cc",
        "src/Timer.cc",
    ]
    sources = _add_sources(source_files)

    extensions += [
        Extension(
            f"{PKG_NAME}._lib.nms",
            sources=sources,
            include_dirs=[osp.join(ROOT, "include")],
            language="c++",
            extra_compile_args=["-fPIC", "-DNOBLAS"],
        )
    ]

    return extensions


setup(
    name=PKG_NAME,
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    keywords=[
        "Edge Detection",
        "Semantic Boundary Detection",
        "Computer Vision",
    ],
    packages=find_packages(
        exclude=[
            "tests",
            "scripts",
            "tools",
            "data",
            ".readme",
            "build",
        ]
    ),
    setup_requires=["numpy", "Cython"],
    install_requires=["numpy", "Cython"],
    include_package_data=True,  # used for MANIFEST.in
    zip_safe=False,
    ext_modules=cythonize(
        get_extentions(),
        language_level="3",
    ),
)
