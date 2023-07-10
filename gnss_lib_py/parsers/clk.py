"""Functions to process precise ephemerides .sp3 and .clk files.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "09 June 2022"

import os
from datetime import datetime, timezone

from scipy import interpolate

from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis

class Clk:
    """Class handling satellite clock bias data from precise ephemerides

    """
    def __init__(self):
        self.clk_bias = []
        self.utc_time = []
        self.tym = []

def parse_clockfile(input_path):
    """Clk specific loading and preprocessing for any GNSS constellation

    Parameters
    ----------
    input_path : string or path-like
        Path to clk file

    Returns
    -------
    clkdata : dict
        Populated gnss_lib_py.parsers.clk.Clk objects
        with key of `gnss_sv_id` for each satellite.

    Notes
    -----
    The format for .clk files can be viewed in [2]_.

    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2015

    References
    -----
    .. [2]  https://files.igs.org/pub/data/format/rinex_clock300.txt
            Accessed as of August 24, 2022
    """

    # Initial checks for loading sp3_path
    if not isinstance(input_path, (str, os.PathLike)):
        raise TypeError("input_path must be string or path-like")
    if not os.path.exists(input_path):
        raise FileNotFoundError("file not found")

    # Create a dictionary for complete clock data
    clkdata = {}

    # Read Clock file
    with open(input_path, 'r', encoding="utf-8") as infile:
        clk = infile.readlines()

    line = 0
    while True:
        if 'OF SOLN SATS' not in clk[line]:
            del clk[line]
        else:
            line +=1
            break

    line = 0
    while True:
        if 'END OF HEADER' not in clk[line]:
            line +=1
        else:
            del clk[0:line+1]
            break

    for  clk_val in clk:

        if len(clk_val) == 0 or clk_val[0:2]!='AS':
            continue

        timelist_val = clk_val.split()

        gnss_sv_id = timelist_val[1]
        if gnss_sv_id not in clkdata:
            clkdata[gnss_sv_id] = Clk()

        curr_time = datetime(year = int(timelist_val[2]), \
                             month = int(timelist_val[3]), \
                             day = int(timelist_val[4]), \
                             hour = int(timelist_val[5]), \
                             minute = int(timelist_val[6]), \
                             second = int(float(timelist_val[7])), \
                             tzinfo=timezone.utc)
        gps_millis = datetime_to_gps_millis(curr_time, add_leap_secs = False)
        clkdata[gnss_sv_id].utc_time.append(curr_time)
        clkdata[gnss_sv_id].tym.append(gps_millis)
        clkdata[gnss_sv_id].clk_bias.append(float(timelist_val[9]))

    return clkdata

def extract_clk(clkdata, sidx, ipos = 10, \
                     method='CubicSpline', verbose = False):
    """Computing interpolated function over clk data for any GNSS

    Parameters
    ----------
    clkdata : gnss_lib_py.parsers.clk.Clk
        Instance of GPS-only Clk class list with len == # sats
    sidx : int
        Nearest index within clk time series around which interpolated
        function needs to be centered
    ipos : int
        No. of data points from clk data on either side of sidx
        that will be used for computing interpolated function
    method : string
        Type of interpolation method used for clk data (the default is
        CubicSpline, which depicts third-order polynomial)

    Returns
    -------
    func_satbias : np.ndarray
        Instance with 1-D array of scipy.interpolate.interpolate.interp1d
        that is loaded with .clk data
    """

    if method=='CubicSpline':
        low_i = (sidx - ipos) if (sidx - ipos) >= 0 else 0
        high_i = (sidx + ipos) if (sidx + ipos) <= len(clkdata.tym) else -1

        if verbose:
            print('Nearest clk: ', sidx, clkdata.tym[sidx], clkdata.clk_bias[sidx])

        func_satbias = interpolate.CubicSpline(clkdata.tym[low_i:high_i], \
                                               clkdata.clk_bias[low_i:high_i])

    return func_satbias


def clk_snapshot(func_satbias, cxtime, hstep = 5e-1, method='CubicSpline'):
    """Compute satellite clock bias and drift from clk interpolated function

    Parameters
    ----------
    func_satbias : scipy.interpolate._cubic.CubicSpline
        Instance with interpolated function for satellite bias from .clk data
    cxtime : float
        Time at which satellite clock bias and drift is to be computed
    hstep : float
        Step size in milliseconds used to computing clock drift using
        central differencing (the default is 5e-1)
    method : string
        Type of interpolation method used for sp3 data (the default is
        CubicSpline, which depicts third-order polynomial)

    Returns
    -------
    satbias_clk : float
        Computed satellite clock bias (in seconds)
    satdrift_clk : float
        Computed satellite clock drift (in seconds/seconds)
    """

    if method=='CubicSpline':
        sat_t = func_satbias([cxtime-0.5*hstep, cxtime, cxtime+0.5*hstep])

    satbias_clk = sat_t[1]
    satdrift_clk = (sat_t[2]-sat_t[0]) / hstep

    return satbias_clk, (satdrift_clk * 1e3)
