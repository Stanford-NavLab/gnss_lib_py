#!/usr/bin/env python

"""Package setup file.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "09 Apr 2021"

import setuptools

setuptools.setup(
    name='gnss_lib_py',
    version='0.1.0',
    description='Python code to process, simulate and demonstrate using GNSS measurements',
    author='Ashwin Kanhere, Derek Knowles',
    author_email='akanhere@stanford.edu',
    url='https://github.com/Stanford-NavLab/gnss_lib_py',
    package_dir={"": "gnss_lib_py"},
    packages=setuptools.find_packages(where="gnss_lib_py"),
    python_requires=">=3.8" "<3.11",
     )