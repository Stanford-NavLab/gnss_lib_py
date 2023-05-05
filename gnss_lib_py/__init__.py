"""Handle version and add submodules to gnss_lib_py namesppace."""

from importlib import metadata

# import submodules into gnss_lib_py namespace
from gnss_lib_py.algorithms.gnss_filters import *
from gnss_lib_py.algorithms.residuals import *
from gnss_lib_py.algorithms.snapshot import *

from gnss_lib_py.parsers.android import *
from gnss_lib_py.parsers.ephemeris import *
from gnss_lib_py.parsers.navdata import *
from gnss_lib_py.parsers.precise_ephemerides import *

from gnss_lib_py.utils.coordinates import *
from gnss_lib_py.utils.filters import *
from gnss_lib_py.utils.sim_gnss import *
from gnss_lib_py.utils.time_conversions import *
from gnss_lib_py.utils.visualizations import *

# single location of version exists in pyproject.toml
__version__ = metadata.version("gnss-lib-py")
