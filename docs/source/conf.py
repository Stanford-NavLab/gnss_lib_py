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
import inspect
import subprocess
from os.path import relpath, dirname
from pygit2 import Repository

import gnss_lib_py

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
                'sphinx.ext.linkcode',
                'nbsphinx',
                'nbsphinx_link',
                'IPython.sphinxext.ipython_console_highlighting',
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
                            "inherited-members": False,
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

# Function to find URLs for the source code on GitHub for built docs

# The original code to find the head tag was taken from:
# https://gist.github.com/nlgranger/55ff2e7ff10c280731348a16d569cb73
# This code was modified to use the current commit when the code differs from
# main or a tag

#Default to the main branch
linkcode_revision = "main"


#Default to the main branch, default to main and tags not existing
linkcode_revision = "main"
in_main = False
tagged = False


# lock to commit number
cmd = "git log -n1 --pretty=%H"
head = subprocess.check_output(cmd.split()).strip().decode('utf-8')
# if we are on main's HEAD, use main as reference irrespective of
# what branch you are on
cmd = "git log --first-parent main -n1 --pretty=%H"
main = subprocess.check_output(cmd.split()).strip().decode('utf-8')
if head == main:
    in_main = True

# if we have a tag, use tag as reference, irrespective of what branch
# you are actually on
try:
    cmd = "git describe --exact-match --tags " + head
    tag = subprocess.check_output(cmd.split(" ")).strip().decode('utf-8')
    linkcode_revision = tag
    tagged = True
except subprocess.CalledProcessError:
    pass

# If the current branch is main, or a tag exists, use the branch name.
# If not, use the commit number
if not tagged and not in_main:
    linkcode_revision = head

linkcode_url = "https://github.com/Stanford-NavLab/gnss_lib_py/blob/" \
               + linkcode_revision + "/{filepath}#L{linestart}-L{linestop}"

print(f'linkcode_revision: {linkcode_revision}')
print(f'linkcode_revision: {main}')
print(f'linkcode_revision: {head}')


def linkcode_resolve(domain, info):
    """Return GitHub link to Python file for docs.

    This function does not return a link for non-Python objects.
    For Python objects, `domain == 'py'`, `info['module']` contains the
    name of the module containing the method being documented, and
    `info['fullname']` contains the name of the method.

    Notes
    -----
    Based off the numpy implementation of linkcode_resolve:
    https://github.com/numpy/numpy/blob/2f375c0f9f19085684c9712d602d22a2b4cb4c8e/doc/source/conf.py#L443
    Retrieved on 1 Jul, 2023.
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']
    submod = sys.modules.get(modname)
    if submod is None:
        return None
    # print('modname:', modname)
    # print('fullname:', fullname)
    # print(f"submod:{submod}")

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)
    filepath = None
    lineno = None

    if filepath is None:
        try:
            filepath = inspect.getsourcefile(obj)
        except Exception:
            filepath = None
        if not filepath:
            return None
        #NOTE: Re-export filtering turned off because
        # # Ignore re-exports as their source files are not within the gnss_lib_py repo
        # module = inspect.getmodule(obj)
        # if module is not None and not module.__name__.startswith("gnss_lib_py"):
        #     return "no_module_not_gnss_lib_py"

        try:
            source, lineno = inspect.getsourcelines(obj)
        except Exception:
            lineno = ""
        # The following line of code first finds the relative path from
        # the location of gnss_lib_py.__init__.py and then adds gnss_lib_py
        # to the beginning to give the path of that file from the root folder
        # of gnss_lib_py and the tests directory adjacent to it

        filepath = relpath(filepath, dirname(gnss_lib_py.__file__))
        filepath = os.path.join('gnss_lib_py', filepath)

    if lineno:
        linestart = lineno
        linestop = lineno + len(source) - 1
    else:
        linestart = ""
        linestop = ""

    codelink = linkcode_url.format(
            filepath=filepath, linestart=linestart, linestop=linestop)
    return codelink
