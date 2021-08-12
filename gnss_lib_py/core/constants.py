########################################################################
# Author(s):    Ashwin Kanhere
# Date:         13 July 2021
# Desc:         GPS constants
########################################################################

import numpy as np

class GPSConsts:
    """Class containing constants required for GPS navigation.

    Based on ECE 456 implementation [1]_.

    Attributes
    ----------
    A : float
        Semi-major axis of the earth [m]
    B : float
        Semi-minor axis of the earth [m]
    E : float
        Eccentricity of the earth = 0.08181919035596
    LAT_ACC_THRESH : float
        10 meter latitude accuracy
    MUEARTH : float
        :math:`G*M_E`, the "gravitational constant" for orbital
        motion about the Earth [m^3/s^2]
    OMEGAEDOT : float
        The sidereal rotation rate of the Earth
        (WGS-84) [rad/s]
    C : float
        speed of light [m/s]
    F : float
        Relativistic correction term [s/m^(1/2)]
    F1 : float
        GPS L1 frequency [Hz]
    F2 : float
        GPS L2 frequency [Hz]
    PI : float
        pi
    T_TRANS : float
        70 ms is the average time taken for signal transmission from GPS sats
    GRAV : float
        Acceleration due to gravity ENU frame of reference [m/s]
    WEEKSEC : float
        Number of seconds in a week [s]

    References
    ----------
    .. [1] Makela, Jonathan, ECE 456, Global Nav Satellite Systems, Fall 2017. 
      University of Illinois Urbana-Champaign. Coding Assignments.

    """
    def __init__(self):
        self.A = 6378137.
        self.B = 6356752.3145
        self.E = np.sqrt(1-(self.B**2)/(self.A**2))
        self.LAT_ACC_THRESH = 1.57e-6
        self.MUEARTH = 398600.5e9
        self.OMEGAEDOT = 7.2921151467e-5
        self.C = 299792458.
        self.F = -4.442807633e-10
        self.F1 = 1.57542e9
        self.F2 = 1.22760e9
        self.PI = 3.1415926535898
        self.T_TRANS = 70*0.001
        self.GRAV = -9.80665
        self.WEEKSEC = 604800


class CoordConsts:
    """Class containing constants required for coordinate conversion.

    Attributes
    ----------
    A : float
        Semi-major axis of the earth [m]
    B : float
        Semi-minor axis of the earth [m]
    ESQ : float
        Don't know what this is
    ESQ1: float
        Don't know what this is either
    """
    # TODO: Update docstring for ESQ and ESQ1

    def __init__(self):
        self.A = 6378137.
        self.B = 6356752.3145
        self.ESQ = 6.69437999014 * 0.001
        self.E1SQ = 6.73949674228 * 0.001
