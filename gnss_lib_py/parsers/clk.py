"""Functions to process .clk precise clock product files.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "09 June 2022"

import os
from datetime import datetime, timezone

import numpy as np

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.utils.constants import CONSTELLATION_CHARS, C
from gnss_lib_py.utils.time_conversions import gps_datetime_to_gps_millis

class Clk(NavData):
    """Clk specific loading and preprocessing for any GNSS constellation.

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
    def __init__(self, input_paths):
        """Clk loading and preprocessing.

        Parameters
        ----------
        input_paths : string or path-like or list of paths
            Path to measurement clk file(s).

        """
        super().__init__()

        if isinstance(input_paths, (str, os.PathLike)):
            input_paths = [input_paths]

        gps_millis = []
        gnss_sv_ids = []
        gnss_id = []
        sv_id = []
        b_sv_m = []

        for input_path in input_paths:
            # Initial checks for loading sp3_path
            if not isinstance(input_path, (str, os.PathLike)):
                raise TypeError("input_path must be string or path-like")
            if not os.path.exists(input_path):
                raise FileNotFoundError(input_path,"file not found")

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
                gnss_sv_ids.append(gnss_sv_id)
                gnss_id.append(CONSTELLATION_CHARS[gnss_sv_id[0]])
                sv_id.append(int(gnss_sv_id[1:]))
                gps_millis.append(gps_millis_timestep)
                # clock bias is given in seconds, convert to meters
                b_sv_m.append(float(timelist_val[9]) * C)

        self["gps_millis"] = gps_millis
        self["gnss_sv_id"] = np.array(gnss_sv_ids,dtype=object)
        self["gnss_id"] = np.array(gnss_id, dtype=object)
        self["sv_id"] = sv_id
        self["b_sv_m"] = b_sv_m

    def interpolate_clk(self, navdata, window=6, verbose=False):
        """Interpolate clock data from clk file, adding inplace to given
        NavData instance.

        Parameters
        ----------
        navdata : gnss_lib_py.navdata.navdata.NavData
            Instance of the NavData class that must include rows for
            ``gps_millis`` and ``gnss_sv_id``
        window : int
            Number of points to use in interpolation window.
        verbose : bool
            Flag (True/False) for whether to print intermediate steps useful
            for debugging/reviewing (the default is False)

        """
        # add satellite indexes if not present already.
        sv_idx_keys = ['b_sv_m', 'b_dot_sv_mps']

        for sv_idx_key in sv_idx_keys:
            if sv_idx_key not in navdata.rows:
                navdata[sv_idx_key] = np.nan

        available_svs_from_clk = np.unique(self["gnss_sv_id"])

        for gnss_sv_id in np.unique(navdata["gnss_sv_id"]):
            if verbose:
                print("interpolating clk for ",gnss_sv_id)
            navdata_id = navdata.where("gnss_sv_id",gnss_sv_id)
            navdata_id_gps_millis = np.atleast_1d(navdata_id["gps_millis"])

            if gnss_sv_id in available_svs_from_clk:
                clk_id = self.where("gnss_sv_id",gnss_sv_id)
                x_data = clk_id["gps_millis"]
                y_data = clk_id["b_sv_m"]

                if np.min(navdata_id_gps_millis) < x_data[0] \
                    or np.max(navdata_id_gps_millis) > x_data[-1]:
                    raise RuntimeError("clk data does not include all "\
                                     + "times in measurement file.")

                b_sv_m = np.zeros(len(navdata_id))
                b_dot_sv_mps = np.zeros(len(navdata_id))

                # iterate through needed polynomials so don't repeat fit
                insert_indexes = np.searchsorted(x_data,navdata_id_gps_millis)
                for insert_index in np.unique(insert_indexes):
                    max_index = min(len(clk_id)-1,insert_index+int(window/2))
                    min_index = max(0,insert_index-int(window/2))
                    x_window = x_data[min_index:max_index]
                    y_window = y_data[min_index:max_index]

                    insert_index_idxs = np.argwhere(insert_indexes==insert_index)
                    x_interpret = navdata_id_gps_millis[insert_index_idxs]

                    poly = np.polynomial.polynomial.Polynomial.fit(x_window, y_window, deg=3)
                    b_sv = poly(x_interpret)

                    polyder = poly.deriv()
                    b_dot_sv = polyder(x_interpret)

                    b_sv_m[insert_index_idxs] = b_sv
                    # multiply by 1000 since currently meters/milliseconds
                    b_dot_sv_mps[insert_index_idxs] = b_dot_sv*1000

                row_idx = navdata.argwhere("gnss_sv_id",gnss_sv_id)
                navdata["b_sv_m",row_idx] = b_sv_m
                navdata["b_dot_sv_mps",row_idx] = b_dot_sv_mps
