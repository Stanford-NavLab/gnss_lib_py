"""Constants for GPS navigation and coordinate transforms.

Based on the UIUC ECE 456 implementation [1]_.

References
----------
.. [1] Makela, Jonathan, ECE 456, Global Nav Satellite Systems,
   Fall 2017. University of Illinois Urbana-Champaign. Coding Assignments.
.. [2] International GNSS Service (IGS), RINEX Working Group and Radio
   Technical Commission for Maritime Servies Special Committee. RINEX
   Version 3.04. http://acc.igs.org/misc/rinex304.pdf
.. [3] https://developer.android.com/reference/android/location/GnssStatus#constants_1
.. [4] https://developer.android.com/reference/android/location/GnssMeasurement#getCodeType()
.. [5] https://sys.qzss.go.jp/dod/en/constellation.html

"""

__authors__ = "Ashwin Kanhere"
__date__ = "13 July 2021"

from datetime import datetime, timezone
from numpy import sqrt


A = 6378137.
"""float : Semi-major axis (radius) of the Earth [m]."""

B = 6356752.3145
"""float : Semi-minor axis (radius) of the Earth [m]."""

E = sqrt(1.-(B**2)/(A**2)) # 0.08181919035596
"""float : Eccentricity of the shape (not orbit) of the Earth."""

E1SQ = 6.69437999014 * 0.001
"""float : First esscentricity squared of Earth (not orbit)."""

E2SQ = 6.73949674228 * 0.001
"""float : Second eccentricity squared of Earth (not orbit)."""

LAT_ACC_THRESH = 1.57e-6
""" float : 10 meter latitude accuracy."""

MU_EARTH = 398600.5e9
"""float : :math:`G*M_E`, the "gravitational constant" for orbital
motion about the Earth [m^3/s^2]."""

OMEGA_E_DOT = 7.2921151467e-5
"""float : The sidereal rotation rate of the Earth (WGS-84) [rad/s]."""

C = 299792458.
"""float : Speed of light [m/s]."""

F = -4.442807633e-10
"""float : Relativistic correction term [s/m^(1/2)]."""

F1 = 1.57542e9
"""float : GPS L1 frequency [Hz]."""

F2 = 1.22760e9
"""float : GPS L2 frequency [Hz]."""

T_TRANS = 70*0.001
"""float : Average time taken for signal transmission from GPS sats to
receivers."""

GRAV = -9.80665
"""float : Acceleration due to gravity ENU frame of reference [m/s]."""

WEEKSEC = 604800
""" int : Number of seconds in a week [s]."""

GPS_EPOCH_0 = datetime(1980, 1, 6, 0, 0, 0, 0, tzinfo=timezone.utc)
""" datetime.datetime: Starting time for GPS epoch"""

MILLIS_PER_DAY = 86400000

TROPO_DELAY_C1 = 2.47
"""float : First coefficient of simplified tropospheric delay model developed in [1]_."""

TROPO_DELAY_C2 = 0.0121
"""float : Second coefficient of simplified tropospheric delay model developed in [1]_."""

TROPO_DELAY_C3 = 1.33e-4
"""float : Third coefficient of simplified tropospheric delay model developed in [1]_."""

CONSTELLATION_CHARS = {'G' : 'gps',
                       'R' : 'glonass',
                       'S' : 'sbas',
                       'C' : 'beidou',
                       'E' : 'galileo',
                       'J' : 'qzss',
                       'I' : 'irnss',
                       }
"""dict : Satellite System identifier from Rinex specification p13 in [2]_."""

CONSTELLATION_ANDROID = {1 : 'gps',
                         3 : 'glonass',
                         2 : 'sbas',
                         5 : 'beidou',
                         6 : 'galileo',
                         4 : 'qzss',
                         7 : 'irnss',
                         0 : 'unknown',
                         }
"""dict : Satellite System identifier from GNSSStatus specification [3]_."""

CODE_TYPE_ANDROID = {
                     'gps' : {
                              'C' : "l1",
                              'Q' : "l5",
                              'X' : "l5",
                             },
                     'glonass' : {
                              'C' : "l1",
                             },
                     'qzss' : {
                              'C' : "l1",
                              'X' : "l5",
                             },
                     'galileo' : {
                              'C' : "e1",
                              'X' : "e5a",
                              'Q' : "e5a",
                             },
                     'beidou' : {
                              'I' : "b1",
                              'X' : "l5",
                             },
                    }
"""dict : GNSS code type identifier from GnssMeasurement specification [4]_."""

QZSS_PRN_SVN = {193 : 1,
                194 : 2,
                199 : 3,
                195 : 4,
                196 : 5,
               }
"""dict : Translation from PRN to SVN for the QZSS constellation [5]_."""
