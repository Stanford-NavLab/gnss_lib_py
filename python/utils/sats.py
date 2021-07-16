########################################################################
# Author(s):    Ashwin Kanhere, Shubh Gupta
# Date:         16 July 2021
# Desc:         Utility functions to simulate satellite positions
########################################################################
import math
import numpy as np

#TODO: Consider moving to funcs.ephemeris

pi = math.pi
# Generate points in a circle
def PointsInCircum(r, n=100):
    """Shubh wrote this
    """
    return np.array([[math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r] for x in range(0, n+1)])


def sats_from_el_az(elaz_deg):
    """Ashwin wrote this
    """
    assert np.shape(elaz_deg)[1] == 2, "elaz_deg should be a Nx2 array"
    el = np.deg2rad(elaz_deg[:, 0])
    az = np.deg2rad(elaz_deg[:,1])
    unit_vect = np.zeros([3, np.shape(elaz_deg)[0]])
    unit_vect[0, :] = np.sin(az)*np.cos(el)
    unit_vect[1, :] = np.cos(az)*np.cos(el)
    unit_vect[2, :] = np.sin(el)
    sats_ned = 20200000*unit_vect
    return sats_ned.T