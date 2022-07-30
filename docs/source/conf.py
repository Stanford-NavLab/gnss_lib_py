# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../gnss_lib_py/'))
sys.path.insert(0, os.path.abspath('../../gnss_lib_py/algorithms/'))
sys.path.insert(0, os.path.abspath('../../gnss_lib_py/parsers/'))
sys.path.insert(0, os.path.abspath('../../gnss_lib_py/utils/'))
sys.path.insert(0, os.path.abspath('../../tests/'))
sys.path.insert(0, os.path.abspath('../../tests/algorithms'))
sys.path.insert(0, os.path.abspath('../../tests/parsers'))
sys.path.insert(0, os.path.abspath('../../tests/utils'))


# -- Project information -----------------------------------------------------

project = 'gnss_lib_py'
copyright = '2022, Stanford NAV Lab'
author = 'Ashwin Kanhere, Derek Knowles'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
                'sphinx.ext.autodoc',
                'sphinx.ext.napoleon',
                'nbsphinx',
                'nbsphinx_link',
]

# Specify which files are source files for Sphinx
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# napoleon settings
napoleon_numpy_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# autodocs settings to include private members
autodoc_default_options = {
                            "members": True,
                            "undoc-members": True,
                            "private-members": True,
                            # "special-members": True,
                            "inherited-members": True,
                            "show-inheritance": True,
                           }


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
html_static_path = []

html_logo = "img/nav_lab_logo.png"

html_favicon = "img/nav_lab_fav.ico"

html_theme_options = {
    "style_nav_header_background" : "#8C1515",
    # "display_version" : True,
    "collapse_navigation" : False,
    # "sticky_navigation" : False,
    # "navigation_depth" : 4,
    "includehidden" : True,
    # "titles_only" : True,
    "logo_only" : False,
    "display_version" : True,
}

# document __init__ methods
autoclass_content = 'both'

# allow errors when building notebooks
nbsphinx_allow_errors = True
