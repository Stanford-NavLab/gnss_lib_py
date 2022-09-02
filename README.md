[![build](https://github.com/Stanford-NavLab/gnss_lib_py/actions/workflows/python-app.yml/badge.svg)](https://github.com/Stanford-NavLab/gnss_lib_py/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/Stanford-NavLab/gnss_lib_py/branch/main/graph/badge.svg?token=1FBGEWRFM6)](https://codecov.io/gh/Stanford-NavLab/gnss_lib_py)
[![Documentation Status](https://readthedocs.org/projects/gnss_lib_py/badge/?version=latest)](https://gnss_lib_py.readthedocs.io/en/latest/?badge=latest)

gnss_lib_py
===========

`gnss_lib_py` is a modular Python tool for parsing, analyzing, and
visualizing Global Navigation Satellite Systems (GNSS) data and state
estimates.
It also provides an intuitive and modular framework allowing users to
quickly prototype, implement, and visualize GNSS algorithms.
`gnss_lib_py` is modular in the sense that multiple types of
algorithms can be easily exchanged for each other and extendable in
facilitating user-specific extensions of existing implementations.

<img src="https://raw.githubusercontent.com/Stanford-NavLab/gnss_lib_py/main/docs/source/img/skyplot.png" alt="satellite skyplot" width="600"/>

`gnss_lib_py` contains parsers for common file types used for
storing GNSS measurements, benchmark algorithms for processing
measurements into state estimates and visualization tools for measurements
and state estimates.
The modularity of `gnss_lib_py` is made possibly by the unifying
`NavData` class, which contains methods to add, remove and modify
numeric and string data consistently.
We provide standard row names for `NavData` elements on the
[reference page](https://gnss-lib-py.readthedocs.io/en/latest/reference/reference.html).
These names ensure cross compatability between different datasets and
algorithms.

Documentation
-------------
Full documentation is available on our [readthedocs website](https://gnss-lib-py.readthedocs.io/en/latest/index.html).


Code Organization
-----------------

`gnss_lib_py` is organized as:

```bash

   ├── data/                          # Location for data files
      └── unit_test/                  # Data files for unit testing
   ├── dev/                           # Code users do not wish to commit
   ├── docs/                          # Documentation files
   ├── gnss_lib_py/                   # gnss_lib_py source files
        ├── algorithms/               # Navigation algorithms
        ├── parsers/                  # Data parsers
        ├── utils/                    # GNSS and common utilities
        └── __init__.py
   ├── notebooks/                     # Interactive Jupyter notebooks
        ├── tutorials/                # Notebooks with tutorial code
   ├── results/                       # Location for result images/files
   ├── tests/                         # Tests for source files
      ├── algorithms/                 # Tests for files in algorithms
      ├── parsers/                    # Tests for files in parsers
      ├── utils/                      # Tests for files in utils
      └── test_gnss_lib_py.py         # High level checks for repository
   ├── CONTRIBUTORS.md                # List of contributors
   ├── build_docs.sh                  # Bash script to build docs
   ├── poetry.lock                    # Poetry specific Lock file
   ├── pyproject.toml                 # List of package dependencies
   └── requirements.txt               # List of packages for pip install
```
In the directory organization above:

  * The `algorithms` directory contains localization algorithms that
    work by passing in a `NavData` class. Currently, the following
    algorithms are implemented in the `algorithms`:

      * Weighted Least Squares
      * Calculating pseudorange residuals
  * The data parsers in the `parsers` directory allow for loading
    GNSS data into `gnss_lib_py`'s unifying `NavData` class.
    Currently, the following datasets and types are supported:

      * [2021 Google Android Derived Dataset](https://www.kaggle.com/c/google-smartphone-decimeter-challenge)
      * [2022 Google Android Derived Dataset](https://www.kaggle.com/competitions/smartphone-decimeter-2022)

  * The `utils` directory contains utilities used to handle
    GNSS measurements, time conversions, visualizations, satellite
    simulation, file operations, etc.

Installation
------------

`gnss_lib_py` is available through `pip` installation with:

```
pip install gnss-lib-py
```

For directions on how to install an editable or developer installation of `gnss_lib_py` on Linux, MacOS, and Windows, please
see the [install instructions](https://gnss-lib-py.readthedocs.io/en/latest/install.html).

Tutorials
---------
We have a range of tutorials on how to easily use this project. They can
all be found in the [tutorials section](https://gnss-lib-py.readthedocs.io/en/latest/tutorials/tutorials.html).

Reference
---------
References on the package contents, explanation of the benefits of our
custom NavData class, and function-level documentation can all be
found in the [reference section](https://gnss-lib-py.readthedocs.io/en/latest/reference/reference.html).

Contributing
------------
If you have a bug report or would like to contribute to our repository,
please follow the guide on the [contributing page](https://gnss-lib-py.readthedocs.io/en/latest/contributing/contributing.html).

Troubleshooting
---------------
Answers to common questions can be found in the [troubleshooting section](https://gnss-lib-py.readthedocs.io/en/latest/troubleshooting.html).

Attribution
-----------
This project is a product of the [Stanford NAV Lab](https://navlab.stanford.edu/)
and currently maintained by Ashwin Kanhere and Derek Knowles. If using
this project in your own work please cite the following:

```

   @inproceedings{knowlesmodular2022,
      title = {A Modular and Extendable GNSS Python Library},
      author={Knowles, Derek and Kanhere, Ashwin V and Bhamidipati, Sriramya and Gao, Grace},
      booktitle={Proceedings of the 35th International Technical Meeting of the Satellite Division of The Institute of Navigation (ION GNSS+ 2022)},
      institution = {Stanford University},
      year = {2022 [Online]},
      url = {https://github.com/Stanford-NavLab/gnss_lib_py},
   }
```

Additionally, we would like to thank [all contributors](https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/CONTRIBUTORS.md) to this project.
