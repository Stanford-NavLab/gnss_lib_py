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
    """Generate uniformly spaced points in a circle

    Parameters
    ----------
    r : float
        Radius of circle
    
    n : int
        Number of points to be generated

    Returns
    -------
    pts : ndarray
        Generated points
    """
    pts = np.array([[math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r] for x in range(0, n+1)])
    return pts


def sats_from_el_az(elaz_deg):
    """Generate NED positions for satellites at given elevation and azimuth angles

    Parameters
    ----------
    elaz_deg : ndarray
        Nx2 array of elevation and azimuth angles [degrees]

    Returns
    -------
    sats_ned : ndarray
        Nx3 satellite NED positions
    """
    assert np.shape(elaz_deg)[1] == 2, "elaz_deg should be a Nx2 array"
    el = np.deg2rad(elaz_deg[:, 0])
    az = np.deg2rad(elaz_deg[:,1])
    unit_vect = np.zeros([3, np.shape(elaz_deg)[0]])
    unit_vect[0, :] = np.sin(az)*np.cos(el)
    unit_vect[1, :] = np.cos(az)*np.cos(el)
    unit_vect[2, :] = np.sin(el)
    sats_ned = np.transpose(20200000*unit_vect)
    return sats_ned