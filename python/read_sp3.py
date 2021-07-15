from datetime import datetime, timedelta
from io import BytesIO
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import interpolate
import constants

def datetime_to_tow(t):
    """
    Shubh got this from somewhere (need to determine where)
    """
    # DateTime to GPS week and TOW
    wk_ref = datetime(2014, 2, 16, 0, 0, 0, 0, None)
    refwk = 1780
    wk = (t - wk_ref).days // 7 + refwk
    tow = ((t - wk_ref) - timedelta((wk - refwk) * 7.0)).total_seconds()
    return tow

class PreciseNav(object):
    """
    Shubh wrote this
    """
    def __init__(self, date, sat_position):
        self.date = date
        self.tow = datetime_to_tow(date)
        self.xyzt = np.array(list(map(float, sat_position)))  # [km, km, km, mcs]

    def eph2pos(self):
        return self.xyzt[:3] * 1e3

    def time_offset(self):
        return self.xyzt[3] / 1e6

#Read SP3
def parse_sp3(path):
    """
    Shubh wrote this
    """
    print("\nParsing %s:" % path)
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
    """
    Shubh wrote this
    """
    theta = constants.WE * flight_time/1e6
    R = np.array([[np.cos(theta), np.sin(theta), 0.], [-np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])

    XYZ = np.array([X, Y, Z])
    rot_XYZ = R @  np.expand_dims(XYZ, axis=-1)
    return rot_XYZ[0], rot_XYZ[1], rot_XYZ[2]

# Interpolate satellite position and correction for time t and prn
def interpol_sp3(sp3, prn, t):
    """
    Shubh wrote this
    """
    inter_rad = 3
    subar = sp3['G'+"%02d" % prn]
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

    X = interpolate.interp1d(_t, _X)
    Y = interpolate.interp1d(_t, _Y)
    Z = interpolate.interp1d(_t, _Z)
    B = interpolate.interp1d(_t, _B)

    # print( np.linalg.norm(np.array([X,Y,Z]) - gt_ecef) - c*B)
    return X(t),Y(t),Z(t),constants.c*B(t)
