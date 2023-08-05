"""Handle version and add submodules to gnss_lib_py namesppace."""

from importlib import metadata

# import submodules into gnss_lib_py namespace
from gnss_lib_py.algorithms.gnss_filters import *
from gnss_lib_py.algorithms.residuals import *
from gnss_lib_py.algorithms.snapshot import *

from gnss_lib_py.parsers.android import *
from gnss_lib_py.parsers.clk import *
from gnss_lib_py.parsers.navdata import *
from gnss_lib_py.parsers.nmea import *
from gnss_lib_py.parsers.rinex_nav import *
from gnss_lib_py.parsers.rinex_obs import *
from gnss_lib_py.parsers.smartloc import *
from gnss_lib_py.parsers.sp3 import *

from gnss_lib_py.utils.coordinates import *
from gnss_lib_py.utils.ephemeris_downloader import *
from gnss_lib_py.utils.filters import *
from gnss_lib_py.utils.gnss_models import *
from gnss_lib_py.utils.sv_models import *
from gnss_lib_py.utils.time_conversions import *
from gnss_lib_py.utils.visualizations import *

# single location of version exists in pyproject.toml
__version__ = metadata.version("gnss-lib-py")
