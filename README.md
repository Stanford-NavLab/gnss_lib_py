[![build](https://github.com/Stanford-NavLab/gnss_lib_py/actions/workflows/python-app.yml/badge.svg)](https://github.com/Stanford-NavLab/gnss_lib_py/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/Stanford-NavLab/gnss_lib_py/branch/main/graph/badge.svg?token=1FBGEWRFM6)](https://codecov.io/gh/Stanford-NavLab/gnss_lib_py)


gnss_lib_py
===========

`gnss_lib_py` is a modular Python tool for parsing, analyzing, and
visualizing Global Navigation Satellite Systems (GNSS) data.
It also provides an intuitive and modular framework allowing users to
quickly prototype, implement, and visualize GNSS algorithms.
`gnss_lib_py` is modular in the sense that multiple types of
algorithms can be easily exchanged for each other and extendable in
facilitating user-specific extensions of existing implementations.

<img src="docs/source/img/skyplot.png" alt="satellite skyplot" width="600"/>

`gnss_lib_py` contains parsers for common file types used for
storing GNSS measurements, benchmark algorithms for processing
measurements into state estimates and visualization tools for measurements
and state estimates.
The modularity of `gnss_lib_py` is made possibly by the unifying
`NavData` class, which contains methods to add, remove and modify
numeric and string data consistently.
We provide standard row names for `NavData` elements on the
:ref:`reference page<reference>`.
These names ensure cross compatability between different datasets and
algorithms.

Documentation
-------------
Full documentation is available at https://gnss_lib_py.readthedocs.io


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
   ├── requirements.txt               # List of packages for pip install
   └── setup.py                       # Setup file
```
In the directory organization above:

  * The following algorithms are implemented in the `algorithms`
    directory and work by passing in a `NavData` class.

      * Weighted Least Squares
      * Calculating pseudorange residuals
  * The data parers in the `parsers` directory allow for loading
    GNSS data from the following sources into the
    `gnss_lib_py`'s unifying `NavData` class:

      * `2021 Google Android Derived Dataset <https://www.kaggle.com/c/google-smartphone-decimeter-challenge>`__
  * The following data visualization tools are available in the
    `utils` directory:

        * Skyplot: showing the movement of GNSS satellites during the
          elapsed time of the provided `NavData` class.
        * Metric plotting: allows you to plot a specific array of data
          from the `NavData` class
        * Metric plotting by constellation: allows you to plot a specific
          array of data but broken up by individual constellations and
          signal types.
        * Residual plotting: specifically optimized for plotting residuals.


Installation
------------
For directions on how to install the `gnss_lib_py` project, please
see the :ref:`install instructions<install>`.

Tutorials
---------
We have a range of tutorials on how to easily use this project. They can
all be found in the :ref:`tutorials section<tutorials>`.

Contributing
------------
If you have a bug report or would like to contribute to our repository,
please follow the guide in :ref:`contributing page<contributing>`.

Reference
---------
References on the package contents, explanation of the benefits of our
custom NavData class, and function-level documentation can all be
found on our :ref:`reference section<reference>`.

Troubleshooting
---------------
Answers to common questions can be found in :ref:`troubleshooting section<troubleshooting>`.

Attribution
-----------
This project is a product of the [Stanford NAV Lab](https://navlab.stanford.edu/)
and currently maintained by Ashwin Kanhere and Derek Knowles. If using
this project in your own work please cite the following:

```

   @inproceedings{knowlesmodular2022,
      title = {A Modular and Extendable GNSS Python Library},
      author={Knowles, Derek and Kanhere, Ashwin V and Bhamidipati, Sriramya and and Gao, Grace},
      booktitle={Proceedings of the 35th International Technical Meeting of the Satellite Division of The Institute of Navigation (ION GNSS+ 2022)},
      institution = {Stanford University},
      year = {2022 [Online]},
      url = {https://github.com/Stanford-NavLab/gnss_lib_py},
   }
```

Additionaly, we would like to thank [all contributors](https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/CONTRIBUTORS.md) to this project.
