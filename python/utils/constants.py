########################################################################
# Author(s):    Ashwin Kanhere
# Date:         13 July 2021
# Desc:         GPS constants
########################################################################

import numpy as np


class gpsconsts:
    def __init__(self):

        self.A = 6378137.                       # semi-major axis of the earth [m]
        self.B = 6356752.3145                   # semi-minor axis of the earth [m]
        self.E = np.sqrt(1-(self.b**2)/(self.a**2))            # eccentricity of the earth = 0.08181919035596
        self.LAT_ACC_THRESH = 1.57e-6      # 10 meter latitude accuracy
        self.MUEARTH = 398600.5e9               # G*Me, the "gravitational constant" for orbital
                                                # motion about the Earth [m^3/s^2]
        self.OMEGAEDOT = 7.2921151467e-5        # the sidereal rotation rate of the Earth
                                                # (WGS-84) [rad/s]
        self.C = 299792458.                     # speed of light [m/s]
        self.F = -4.442807633e-10               # Relativistic correction term [s/m^(1/2)]
        self.F1 = 1.57542e9                     # GPS L1 frequency [Hz]
        self.F2 = 1.22760e9                     # GPS L2 frequency [Hz]
        self.PI = 3.1415926535898               # pi
        self.T_TRANS = 70*0.001                 # 70 ms is the average time taken for signal transmission from GPS sats
        self.GRAV = -9.80665                    # Acceleration due to gravity ENU frame of reference [m/s]
        self.WEEKSEC = 604800                   # Number of seconds in a week [s]
