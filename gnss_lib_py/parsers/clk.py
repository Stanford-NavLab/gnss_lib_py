"""Functions to process .clk and .CLK precise clock product files.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "09 June 2022"

import os
from datetime import datetime, timezone

import numpy as np
from scipy import interpolate

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils.constants import CONSTELLATION_CHARS, C
from gnss_lib_py.utils.time_conversions import gps_to_unix_millis
from gnss_lib_py.utils.time_conversions import gps_datetime_to_gps_millis

class Clk(NavData):
    """Clk specific loading and preprocessing for any GNSS constellation

    Parameters
    ----------
    input_paths : string or path-like or list of paths
        Path to measurement clk file(s).

    Returns
    -------
    clkdata : dict
        Populated gnss_lib_py.parsers.clk.Clk objects
        with key of `gnss_sv_id` for each satellite.

    Notes
    -----
    The format for .clk files can be viewed in [1]_.

    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2015

    References
    -----
    .. [1]  https://files.igs.org/pub/data/format/rinex_clock300.txt
            Accessed as of August 24, 2022

    """
    def __init__(self, input_path):
        """Clk loading and preprocessing.

        Parameters
        ----------
        input_path : string or path-like
            Path to clk file

        """
        super().__init__()

        gps_millis = []
        unix_millis = []
        gnss_sv_ids = []
        gnss_id = []
        sv_id = []
        b_sv_m = []

        # Initial checks for loading sp3_path
        if not isinstance(input_path, (str, os.PathLike)):
            raise TypeError("input_path must be string or path-like")
        if not os.path.exists(input_path):
            raise FileNotFoundError("file not found")

        # Read Clock file
        with open(input_path, 'r', encoding="utf-8") as infile:
            clk = infile.readlines()

        for  clk_val in clk:

            timelist_val = clk_val.split()

            if len(timelist_val) == 0 or timelist_val[0] != 'AS':
                continue

            gnss_sv_id = timelist_val[1]

            curr_time = datetime(year = int(timelist_val[2]), \
                                 month = int(timelist_val[3]), \
                                 day = int(timelist_val[4]), \
                                 hour = int(timelist_val[5]), \
                                 minute = int(timelist_val[6]), \
                                 second = int(float(timelist_val[7])), \
                                 tzinfo=timezone.utc)
            gps_millis_timestep = gps_datetime_to_gps_millis(curr_time)
            unix_millis_timestep = gps_to_unix_millis(gps_millis_timestep)
            gnss_sv_ids.append(gnss_sv_id)
            gnss_id.append(CONSTELLATION_CHARS[gnss_sv_id[0]])
            sv_id.append(int(gnss_sv_id[1:]))
            gps_millis.append(gps_millis_timestep)
            unix_millis.append(unix_millis_timestep)
            # clock bias is given in seconds, convert to meters
            b_sv_m.append(float(timelist_val[9]) * C)

        self["gps_millis"] = gps_millis
        self["unix_millis"] = unix_millis
        self["gnss_sv_id"] = np.array(gnss_sv_ids,dtype=object)
        self["gnss_id"] = np.array(gnss_id, dtype=object)
        self["sv_id"] = sv_id
        self["b_sv_m"] = b_sv_m

    def extract_clk(self, gnss_sv_id, sidx, ipos = 10, \
                         method='CubicSpline', verbose = False):
        """Computing interpolated function over clk data for any GNSS

        Parameters
        ----------
        gnss_sv_id : string
            PRN of satellite for which position should be determined.
        sidx : int
            Nearest index within clk time series around which interpolated
            function needs to be centered
        ipos : int
            No. of data points from clk data on either side of sidx
            that will be used for computing interpolated function
        method : string
            Type of interpolation method used for clk data (the default is
            CubicSpline, which depicts third-order polynomial)
        verbose : bool
            If true, prints extra debugging statements.

        Returns
        -------
        func_satbias : np.ndarray
            Instance with 1-D array of scipy.interpolate.interpolate.interp1d
            that is loaded with .clk data
        """

        clkdata = self.where("gnss_sv_id",gnss_sv_id)

        if method=='CubicSpline':
            low_i = (sidx - ipos) if (sidx - ipos) >= 0 else 0
            high_i = (sidx + ipos) if (sidx + ipos) <= len(clkdata["gps_millis"]) else -1

            if verbose:
                print('Nearest clk: ', sidx, clkdata["gps_millis"][sidx],
                                             clkdata["b_sv_m"][sidx])

            func_satbias = interpolate.CubicSpline(clkdata["gps_millis"][low_i:high_i], \
                                                   clkdata["b_sv_m"][low_i:high_i])

        return func_satbias


    def clk_snapshot(self, func_satbias, cxtime, hstep = 5e-1,
                     method='CubicSpline'):
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
