"""Functions to process precise ephemerides .sp3 and .clk files.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "09 June 2022"

import os
from datetime import datetime, timezone

import numpy as np
from scipy import interpolate

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils.constants import CONSTELLATION_CHARS
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis
from gnss_lib_py.utils.time_conversions import datetime_to_unix_millis

class Sp3(NavData):
    """sp3 specific loading and preprocessing for any GNSS constellation

    Parameters
    ----------
    input_path : string or path-like
        Path to sp3 file

    Notes
    -----
    The format for .sp3 files can be viewed in [1]_.

    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2015

    References
    ----------
    .. [1]  https://files.igs.org/pub/data/format/sp3d.pdf
            Accessed as of August 20, 2022
    """
    def __init__(self, input_path):

        super().__init__()

        gps_millis = []
        unix_millis = []
        gnss_sv_ids = []
        gnss_id = []
        sv_id = []
        x_sv_m = []
        y_sv_m = []
        z_sv_m = []

        # Initial checks for loading sp3_path
        if not isinstance(input_path, (str, os.PathLike)):
            raise TypeError("input_path must be string or path-like")
        if not os.path.exists(input_path):
            raise FileNotFoundError("file not found")

        # Load in the file
        with open(input_path, 'r', encoding="utf-8") as infile:
            data = [line.strip() for line in infile]

        # Loop through each line
        for dval in data:
            if len(dval) == 0:
                # No data
                continue

            if dval[0] == '*':
                # A new record
                # Get the date
                temp = dval.split()
                curr_time = datetime( int(temp[1]), int(temp[2]), \
                                      int(temp[3]), int(temp[4]), \
                                      int(temp[5]),int(float(temp[6])),\
                                      tzinfo=timezone.utc )
                gps_millis_timestep = datetime_to_gps_millis(curr_time,
                                                add_leap_secs = False)
                unix_millis_timestep = datetime_to_unix_millis(curr_time)

            if 'P' in dval[0]:
                # A satellite record.  Get the satellite number, and coordinate (X,Y,Z) info
                temp = dval.split()

                gnss_sv_id = temp[0][1:]

                gps_millis.append(gps_millis_timestep)
                unix_millis.append(unix_millis_timestep)
                gnss_sv_ids.append(gnss_sv_id)
                gnss_id.append(CONSTELLATION_CHARS[gnss_sv_id[0]])
                sv_id.append(int(gnss_sv_id[1:]))
                x_sv_m.append(float(temp[1])*1e3)
                y_sv_m.append(float(temp[2])*1e3)
                z_sv_m.append(float(temp[3])*1e3)

        self["gps_millis"] = gps_millis
        self["unix_millis"] = unix_millis
        self["gnss_sv_id"] = np.array(gnss_sv_ids,dtype=object)
        self["gnss_id"] = np.array(gnss_id, dtype=object)
        self["sv_id"] = sv_id
        self["x_sv_m"] = x_sv_m
        self["y_sv_m"] = y_sv_m
        self["z_sv_m"] = z_sv_m

    def extract_sp3(self, gnss_sv_id, sidx, ipos = 10, \
                         method = 'CubicSpline', verbose = False):
        """Computing interpolated function over sp3 data for any GNSS

        Parameters
        ----------
        gnss_sv_id : string
            PRN of satellite for which position should be determined.
        sidx : int
            Nearest index within sp3 time series around which interpolated
            function needs to be centered
        ipos : int
            No. of data points from sp3 data on either side of sidx
            that will be used for computing interpolated function
        method : string
            Type of interpolation method used for sp3 data (the default is
            CubicSpline, which depicts third-order polynomial)

        Returns
        -------
        func_satpos : np.ndarray
            Instance with 3-D array of scipy.interpolate.interpolate.interp1d
            that is loaded with .sp3 data
        """

        sp3data = self.where("gnss_sv_id",gnss_sv_id)

        func_satpos = np.empty((3,), dtype=object)
        func_satpos[:] = np.nan

        if method=='CubicSpline':
            low_i = (sidx - ipos) if (sidx - ipos) >= 0 else 0
            high_i = (sidx + ipos) if (sidx + ipos) <= len(sp3data["gps_millis"]) else -1

            if verbose:
                print('Nearest sp3: ', sidx, sp3data["gps_millis"][sidx], \
                                       sp3data["x_sv_m"][sidx],
                                       sp3data["y_sv_m"][sidx],
                                       sp3data["z_sv_m"][sidx])

            func_satpos[0] = interpolate.CubicSpline(sp3data["gps_millis"][low_i:high_i], \
                                                     sp3data["x_sv_m"][low_i:high_i])
            func_satpos[1] = interpolate.CubicSpline(sp3data["gps_millis"][low_i:high_i], \
                                                     sp3data["y_sv_m"][low_i:high_i])
            func_satpos[2] = interpolate.CubicSpline(sp3data["gps_millis"][low_i:high_i], \
                                                     sp3data["z_sv_m"][low_i:high_i])

        return func_satpos


    def sp3_snapshot(self, func_satpos, cxtime, hstep = 5e-1, method='CubicSpline'):
        """Compute satellite 3-D position and velocity from sp3 interpolated function

        Parameters
        ----------
        func_satpos : np.ndarray
            Instance with 3-D array of scipy.interpolate.interpolate.interp1d
            that is loaded with .sp3 data
        cxtime : float
            Time at which the satellite 3-D position and velocity needs to be
            computed, given 3-D array of interpolated functions
        hstep : float
            Step size in milliseconds used to computing 3-D velocity of any
            given satellite using central differencing the default is 5e-1)
        method : string
            Type of interpolation method used for sp3 data (the default is
            CubicSpline, which depicts third-order polynomial)

        Returns
        -------
        satpos_sp3 : 3-D array
            Computed satellite position in ECEF frame (Earth's rotation not included)
        satvel_sp3 : 3-D array
            Computed satellite velocity in ECEF frame (Earth's rotation not included)
        """
        if method=='CubicSpline':
            sat_x = func_satpos[0]([cxtime-0.5*hstep, cxtime, cxtime+0.5*hstep])
            sat_y = func_satpos[1]([cxtime-0.5*hstep, cxtime, cxtime+0.5*hstep])
            sat_z = func_satpos[2]([cxtime-0.5*hstep, cxtime, cxtime+0.5*hstep])

        satpos_sp3 = np.array([sat_x[1], sat_y[1], sat_z[1]])
        satvel_sp3 = np.array([ (sat_x[2]-sat_x[0]) / hstep, \
                                (sat_y[2]-sat_y[0]) / hstep, \
                                (sat_z[2]-sat_z[0]) / hstep ])

        return satpos_sp3, (satvel_sp3 * 1e3)
