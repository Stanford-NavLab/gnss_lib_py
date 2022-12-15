"""Handle version and add submodules to gnss_lib_py namesppace."""

from importlib import metadata

# import submodules into gnss_lib_py namespace
from gnss_lib_py.algorithms import *
from gnss_lib_py.parsers import *
from gnss_lib_py.utils import *

# single location of version exists in pyproject.toml
__version__ = metadata.version("gnss-lib-py")
