"""Functions to process .sp3 precise orbit files.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "09 June 2022"

import os
from datetime import datetime, timezone

import numpy as np

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.utils.constants import CONSTELLATION_CHARS
from gnss_lib_py.utils.time_conversions import gps_datetime_to_gps_millis


class Sp3(NavData):
    """sp3 specific loading and preprocessing for any GNSS constellation.

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
    def __init__(self, input_paths):
        """Sp3 loading and preprocessing.

        Parameters
        ----------
        input_paths : string or path-like or list of paths
            Path to measurement sp3 file(s).

        """
        super().__init__()

        if isinstance(input_paths, (str, os.PathLike)):
            input_paths = [input_paths]

        gps_millis = []
        gnss_sv_ids = []
        gnss_id = []
        sv_id = []
        x_sv_m = []
        y_sv_m = []
        z_sv_m = []

        for input_path in input_paths:
            # Initial checks for loading sp3_path
            if not isinstance(input_path, (str, os.PathLike)):
                raise TypeError("input_path must be string or path-like")
            if not os.path.exists(input_path):
                raise FileNotFoundError(input_path,"file not found")

            # Load in the file
            with open(input_path, 'r', encoding="utf-8") as infile:
                data = [line.strip() for line in infile]

            # Loop through each line
            gps_millis_timestep = 0
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
                    gps_millis_timestep = gps_datetime_to_gps_millis(curr_time)

                if 'P' in dval[0]:
                    # A satellite record.  Get the satellite number, and coordinate (X,Y,Z) info
                    temp = dval.split()

                    gnss_sv_id = temp[0][1:]

                    gps_millis.append(gps_millis_timestep)
                    gnss_sv_ids.append(gnss_sv_id)
                    gnss_id.append(CONSTELLATION_CHARS[gnss_sv_id[0]])
                    sv_id.append(int(gnss_sv_id[1:]))
                    x_sv_m.append(float(temp[1])*1e3)
                    y_sv_m.append(float(temp[2])*1e3)
                    z_sv_m.append(float(temp[3])*1e3)

        self["gps_millis"] = gps_millis
        self["gnss_sv_id"] = np.array(gnss_sv_ids,dtype=object)
        self["gnss_id"] = np.array(gnss_id, dtype=object)
        self["sv_id"] = sv_id
        self["x_sv_m"] = x_sv_m
        self["y_sv_m"] = y_sv_m
        self["z_sv_m"] = z_sv_m

    def interpolate_sp3(self, navdata, window=6, verbose=False):
        """Interpolate ECEF position data from sp3 file, inplace to
        given navdata instance.

        Parameters
        ----------
        navdata : gnss_lib_py.navdata.navdata.NavData
            Instance of the NavData class that must include rows for
            ``gps_millis`` and ``gnss_sv_id``.
        window : int
            Number of points to use in interpolation window.
        verbose : bool
            Flag (True/False) for whether to print intermediate steps useful
            for debugging/reviewing (the default is False)

        """
        # add satellite indexes if not present already.
        sv_idx_keys = ['x_sv_m', 'y_sv_m', 'z_sv_m', \
                       'vx_sv_mps','vy_sv_mps','vz_sv_mps']

        for sv_idx_key in sv_idx_keys:
            if sv_idx_key not in navdata.rows:
                navdata[sv_idx_key] = np.nan

        available_svs_from_sp3 = np.unique(self["gnss_sv_id"])

        for gnss_sv_id in np.unique(navdata["gnss_sv_id"]):
            if verbose:
                print("interpolating sp3 for ",gnss_sv_id)
            navdata_id = navdata.where("gnss_sv_id",gnss_sv_id)
            navdata_id_gps_millis = np.atleast_1d(navdata_id["gps_millis"])

            if gnss_sv_id in available_svs_from_sp3:
                sp3_id = self.where("gnss_sv_id",gnss_sv_id)
                x_data = sp3_id["gps_millis"]
                y_data = sp3_id[["x_sv_m","y_sv_m","z_sv_m"]]

                if np.min(navdata_id_gps_millis) < x_data[0] \
                    or np.max(navdata_id_gps_millis) > x_data[-1]:
                    raise RuntimeError("sp3 data does not include "\
                                     + "appropriate times in measurement"\
                                     + " file for SV ",gnss_sv_id)

                xyz_sv_m = np.zeros((6,len(navdata_id)))

                # iterate through needed polynomials so don't repeat fit
                insert_indexes = np.searchsorted(x_data,navdata_id_gps_millis)
                for insert_index in np.unique(insert_indexes):
                    max_index = min(len(sp3_id)-1,insert_index+int(window/2))
                    min_index = max(0,insert_index-int(window/2))
                    x_window = x_data[min_index:max_index]
                    y_window = y_data[:,min_index:max_index]

                    insert_index_idxs = np.argwhere(insert_indexes==insert_index)
                    x_interpret = navdata_id_gps_millis[insert_index_idxs]

                    # iterate through x,y, and z
                    for xyz_index in range(3):
                        poly = np.polynomial.polynomial.Polynomial.fit(x_window, y_window[xyz_index,:], deg=3)
                        xyz_sv = poly(x_interpret)

                        polyder = poly.deriv()
                        vxyz_sv = polyder(x_interpret)

                        xyz_sv_m[xyz_index,insert_index_idxs] = xyz_sv
                        # multiply by 1000 since currently meters/milliseconds
                        xyz_sv_m[xyz_index+3,insert_index_idxs] = vxyz_sv*1000

                row_idx = navdata.argwhere("gnss_sv_id",gnss_sv_id)
                navdata["x_sv_m",row_idx] = xyz_sv_m[0,:]
                navdata["y_sv_m",row_idx] = xyz_sv_m[1,:]
                navdata["z_sv_m",row_idx] = xyz_sv_m[2,:]
                navdata["vx_sv_mps",row_idx] = xyz_sv_m[3,:]
                navdata["vy_sv_mps",row_idx] = xyz_sv_m[4,:]
                navdata["vz_sv_mps",row_idx] = xyz_sv_m[5,:]
