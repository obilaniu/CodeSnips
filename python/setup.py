#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Imports
#

import os
from   setuptools                       import setup, find_packages, Extension



#
# Setup
#

setup(
    # The basics
    name                 = "pysnips",
    version              = "0.0.0",
    author               = "Anonymous",
    author_email         = "anonymous@anonymous.com",
    license              = None,
    url                  = "https://github.com/obilaniu/CodeSnips",
    
    # Descriptions
    description          = "A personal collection of Python code snippets.",
    long_description     =
    """
    This is the Python subset of my personal collection of code snippets.
    """,
    classifiers          = [
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Logging",
        "Topic :: Utilities",
    ],
    
    # Sources
    packages             = find_packages(exclude=["scripts"]),
    ext_modules          = [
        Extension("pysnips.discrete.cksum.crc_native",
                  [os.path.join("pysnips", "discrete", "cksum", "crc_native.c")],
                  extra_compile_args=["-march=native"])
    ],
    install_requires     = [
        "numpy>=1.10",
        "Pillow>=4.0.0",
    ],
    extras_require       = {
        'PyTorch':  ["torch>=0.2.0",],
        'scipy':    ["scipy>=0.18.1",],
    },
    
    # Misc
    zip_safe             = False,
)

