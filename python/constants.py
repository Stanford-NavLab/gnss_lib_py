import numpy as np

WE = 7.2921151467e-5
LIGHTSPEED = 2.99792458e8
WEEKSEC = 604800
GRAVITY = -9.80665

class gpsconsts:
    def __init__(self):

        self.a = 6378137.                       # semi-major axis of the earth [m]
        self.b = 6356752.3145                   # semi-minor axis of the earth [m]
        self.e = np.sqrt(1-(self.b**2)/(self.a**2))            # eccentricity of the earth = 0.08181919035596
        self.lat_accuracy_thresh = 1.57e-6      # 10 meter latitude accuracy
        self.muearth = 398600.5e9               # G*Me, the "gravitational constant" for orbital
                                                # motion about the Earth [m^3/s^2]
        self.OmegaEDot = 7.2921151467e-5        # the sidereal rotation rate of the Earth
                                                # (WGS-84) [rad/s]
        self.c = 299792458.                     # speed of light [m/s]
        self.F = -4.442807633e-10               # Relativistic correction term [s/m^(1/2)]
        self.f1 = 1.57542e9                     # GPS L1 frequency [Hz]
        self.f2 = 1.22760e9                     # GPS L2 frequency [Hz]
        self.pi = 3.1415926535898               # pi
        self.t_trans = 70*0.001                 ## 70 ms is the average time taken for signal transmission from GPS sats