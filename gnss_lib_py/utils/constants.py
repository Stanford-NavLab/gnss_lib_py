"""Constants for GPS navigation and coordinate transforms.

Based on the UIUC ECE 456 implementation [1]_.

References
----------
.. [1] Makela, Jonathan, ECE 456, Global Nav Satellite Systems,
   Fall 2017. University of Illinois Urbana-Champaign. Coding Assignments.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "13 July 2021"

from numpy import sqrt
from datetime import datetime, timezone

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
