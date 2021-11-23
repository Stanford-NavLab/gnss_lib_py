"""Functions to read data from NMEA files.

"""

__authors__ = "Shubh Gupta"
__date__ = "10 Apr 2021"

import os
import sys
from datetime import datetime
from collections import defaultdict
# append <path>/gnss_lib_py/gnss_lib_py/ to path
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))

import numpy as np
from scipy import interpolate

import core.constants as consts
from core.ephemeris import datetime2tow

class PreciseNav(object):
    """Class that contain satellite data.

    """
    def __init__(self, date, sat_position):
        """Initialize PreciseNav class.

        Parameters
        ----------
        date : datetime object
            ???.
        sat_position : tuple
            ??? contains (x, y, z, t).

        """
        self.date = date
        self.tow = datetime2tow(date,False)[1]
        self.xyzt = np.array(list(map(float, sat_position)))  # [km, km, km, mcs]

    def eph2pos(self):
        """Conversion from km to m ???

        Returns
        -------
        result : np.array
            x,y,z ECEF position of satellite in meters

        """
        result = self.xyzt[:3] * 1e3
        return result

    def time_offset(self):
        """Converts clock correction from microseconds to seconds

        Returns
        -------
        result : float
            clock correction in seconds

        """
        result = self.xyzt[3] / 1e6
        return result

def parse_sp3(path):
    """Convert SP3 ephemeris file into a python dictionary.

    Parameters
    ----------
    path : string
        File location of SP3 file to read.

    Returns
    -------
    nav_dict : dict
        Dicionary containing satellite objects mapped by PRN number.

    """
    with open(path) as fd:
        data = fd.readlines()
    nav_dict = defaultdict(list)
    for j, d in enumerate(data):
        if d[0] == '*':
            split = d.split()[1:]
            y, m, d, H, M = list(map(int, split[:-1]))
            s = int(float(split[-1]))
            date = datetime(y, m, d, H, M, s)
        elif d[0] == 'P' and date:  # GPS satellites
            prn, x, y, z, t = d[2:].split()[:5]
            nav_dict[d[1] + "%02d" % int(prn)] += [PreciseNav(date, (x, y, z, t))]
        else:
            continue
    return nav_dict

# Rotate to correct ECEF satellite positions
def flight_time_correct(X, Y, Z, flight_time):
    """Get the time-of-flight corrected ECEF satellite positions

    Uses the signal flight time to calculate the rotation to correct
    the satellite positions.

    Parameters
    ----------
    X : float
        ECEF X position of satellite.
    Y : float
        ECEF Y position of satellite.
    Z : float
        ECEF Z position of satellite.

    Returns
    -------
    X_corrected : float
        Corrected ECEF X position of satellite.
    Y_corrected : float
        Corrected ECEF Y position of satellite.
    Z_corrected : float
        Corrected ECEF Z position of satellite.

    """

    theta = consts.OMEGA_E_DOT * flight_time/1e6
    R = np.array([[np.cos(theta), np.sin(theta), 0.],
                  [-np.sin(theta), np.cos(theta), 0.],
                  [0., 0., 1.]])

    XYZ = np.array([X, Y, Z])
    rot_XYZ = R @  np.expand_dims(XYZ, axis=-1)
    X_corrected = rot_XYZ[0]
    Y_corrected = rot_XYZ[1]
    Z_corrected = rot_XYZ[2]

    return X_corrected, Y_corrected, Z_corrected

#
def interpol_sp3(sp3, prn, t, verbose=False):
    """Interpolate satellite position and correction for time t and prn.

    Parameters
    ----------
    sp3 : dict
        Dicionary containing satellite objects mapped by PRN number
        created by the parse_sp3() function.
    prn : int
        PRN number of satellite to interpolate.
    t : float ???
        Time to interpolate.

    Returns
    -------
    X_interp : float
        Interpolated ECEF X position of satellite.
    Y_interp : float
        Interpolated ECEF Y position of satellite.
    Z_interp : float
        Interpolated ECEF Z position of satellite.
    B_interp : float
        Interpolated time correction in distance (meters??).

    """
    inter_rad = 4
    # subar = sp3['G'+"%02d" % prn]
    subar = sp3[prn]
    low_i, high_i = 0, 0
    for i, ephem in enumerate(subar):
        if ephem.tow > t:
            low_i = max(0, i-inter_rad)
            high_i = min(i+inter_rad, len(subar))
            break

    if high_i-low_i<1:
        return 0., 0., 0., 0.

    _t = np.zeros(high_i-low_i)
    _X = np.zeros(high_i-low_i)
    _Y = np.zeros(high_i-low_i)
    _Z = np.zeros(high_i-low_i)
    _B = np.zeros(high_i-low_i)
    for i in range(low_i, high_i):
        _t[i-low_i] = subar[i].tow
        xyz = subar[i].eph2pos()
        _X[i-low_i] = xyz[0]
        _Y[i-low_i] = xyz[1]
        _Z[i-low_i] = xyz[2]
        _B[i-low_i] = subar[i].time_offset()

    X = interpolate.CubicSpline(_t, _X)
    Y = interpolate.CubicSpline(_t, _Y)
    Z = interpolate.CubicSpline(_t, _Z)
    B = interpolate.CubicSpline(_t, _B)

    # print( np.linalg.norm(np.array([X,Y,Z]) - gt_ecef) - c*B)
    X_interp = X(t)
    Y_interp = Y(t)
    Z_interp = Z(t)
    B_interp = consts.C*B(t)

    return X_interp, Y_interp, Z_interp, B_interp
