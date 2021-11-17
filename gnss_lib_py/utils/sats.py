"""Utility functions to simulate satellite positions.

"""

__authors__ = "Ashwin Kanhere, Shubh Gupta"
__date__ = "16 July 2021"

import math

import numpy as np

#TODO: Consider moving to core.ephemeris

pi = math.pi
# Generate points in a circle
def points_in_circum(r, n=100):
    """Generate uniformly spaced points in a circle

    Parameters
    ----------
    r : float
        Radius of circle

    n : int
        Number of points to be generated

    Returns
    -------
    pts : np.ndarray
        Generated points
    """
    pts = np.array([[math.cos(2*pi/n*x)*r,
                    math.sin(2*pi/n*x)*r] for x in range(0, n+1)])
    return pts


def sats_from_el_az(elaz_deg):
    """Generate NED satellite positions at given elevation and azimuth.

    Parameters
    ----------
    elaz_deg : np.ndarray
        Nx2 array of elevation and azimuth angles [degrees]

    Returns
    -------
    sats_ned : np.ndarray
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
