"""Functions for coordinate conversions required by GPS.

Based on code from https://github.com/commaai/laika whose license is
copied below:

MIT License

Copyright (c) 2018 comma.ai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

__authors__ = "Shubh Gupta, Ashwin Kanhere"
__date__ = "20 July 2021"

import os
import sys
# append <path>/gnss_lib_py/gnss_lib_py/ to path
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))

import numpy as np

import core.constants as consts

def geodetic2ecef(geodetic, radians=False):
    """LLA to ECEF conversion.

    Parameters
    ----------
    geodetic : np.ndarray
        Float with WGS-84 LLA coordinates.
    radians : bool
        Flag of whether input [rad].

    Returns
    -------
    ecef : np.ndarray
        ECEF coordinates corresponding to input LLA.

    Notes
    -----
    Based on code from https://github.com/commaai/laika.

    """

    geodetic = np.array(geodetic)
    input_shape = geodetic.shape
    geodetic = np.atleast_2d(geodetic)

    ratio = 1.0 if radians else (np.pi / 180.0)
    lat = ratio*geodetic[:,0]
    lon = ratio*geodetic[:,1]
    alt = geodetic[:,2]

    xi = np.sqrt(1 - consts.E1SQ * np.sin(lat)**2)
    x = (consts.A / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (consts.A / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (consts.A / xi * (1 - consts.E1SQ) + alt) * np.sin(lat)
    ecef = np.array([x, y, z]).T
    ecef = np.reshape(ecef, input_shape)
    return ecef


def ecef2geodetic(ecef, radians=False):
    """ECEF to LLA conversion using Ferrari's method

    Parameters
    ----------
    ecef : np.ndarray
        Float with ECEF coordinates
    radians : bool
        Flag of whether output should be in radians

    Returns
    -------
    geodetic : np.ndarray
        Float with WGS-84 LLA coordinates corresponding to input ECEF

    Notes
    -----
    Based on code from https://github.com/commaai/laika.

    """

    ecef = np.atleast_1d(ecef)
    input_shape = ecef.shape
    ecef = np.atleast_2d(ecef)
    x, y, z = ecef[:, 0], ecef[:, 1], ecef[:, 2]

    ratio = 1.0 if radians else (180.0 / np.pi)

    # Convert from ECEF to geodetic using Ferrari's methods
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Ferrari.27s_solution
    r = np.sqrt(x * x + y * y)
    E1SQ = consts.A * consts.A - consts.B * consts.B
    F = 54 * consts.B * consts.B * z * z
    G = r * r + (1 - consts.E1SQ) * z * z - consts.E1SQ * E1SQ
    C = (consts.E1SQ * consts.E1SQ * F * r * r) / (pow(G, 3))
    S = np.cbrt(1 + C + np.sqrt(C * C + 2 * C))
    P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
    Q = np.sqrt(1 + 2 * consts.E1SQ * consts.E1SQ * P)
    r_0 =  -(P * consts.E1SQ * r) / (1 + Q) + np.sqrt(0.5 * consts.A * consts.A*(1 + 1.0 / Q) - \
          P * (1 - consts.E1SQ) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r)
    U = np.sqrt(pow((r - consts.E1SQ * r_0), 2) + z * z)
    V = np.sqrt(pow((r - consts.E1SQ * r_0), 2) + (1 - consts.E1SQ) * z * z)
    Z_0 = consts.B * consts.B * z / (consts.A * V)
    h = U * (1 - consts.B * consts.B / (consts.A * V))
    lat = ratio*np.arctan((z + consts.E2SQ * Z_0) / r)
    lon = ratio*np.arctan2(y, x)

    # stack the new columns and return to the original shape
    geodetic = np.column_stack((lat, lon, h))
    geodetic = np.reshape(geodetic, input_shape)
    return geodetic.reshape(input_shape)

class LocalCoord(object):
    """Class for conversions to NED (North-East-Down).

    Attributes
    ----------
    init_ecef : np.ndarray
        ECEF of origin of NED.
    ned2ecef_matrix : np.ndarray
        Rotation matrix to convert from NED to ECEF.
    ecef2ned_matrix : np.ndarray
        Rotation matrix to convert from ECEF to NED.

    Notes
    -----
    Based on code from https://github.com/commaai/laika.

    """

    def __init__(self, init_geodetic, init_ecef):
        #TODO: Add documentation for the __init__
        self.init_ecef = init_ecef
        lat, lon, _ = (np.pi/180)*np.array(init_geodetic)
        self.ned2ecef_matrix = np.array([[-np.sin(lat)*np.cos(lon), -np.sin(lon), -np.cos(lat)*np.cos(lon)],
                                         [-np.sin(lat)*np.sin(lon), np.cos(lon), -np.cos(lat)*np.sin(lon)],
                                         [np.cos(lat), 0, -np.sin(lat)]])
        self.ecef2ned_matrix = self.ned2ecef_matrix.T

    @classmethod
    def from_geodetic(cls, init_geodetic):
        """Instantiate class using NED origin in geodetic coordinates.

        Parameters
        ----------
        init_geodetic : np.ndarray
            Float with WGS-84 LLA coordinates of the NED origin

        Returns
        -------
        local_coord : LocalCoord
            Instance of LocalCoord object with corresponding NED origin.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        init_ecef = geodetic2ecef(init_geodetic)
        local_coord = LocalCoord(init_geodetic, init_ecef)
        return local_coord

    @classmethod
    def from_ecef(cls, init_ecef):
        """Instantiate class using the NED origin in ECEF coordinates.

        Parameters
        ----------
        init_ecef : np.ndarray
            Float with ECEF coordinates of the NED origin.

        Returns
        -------
        local_coord : LocalCoord
            Instance of LocalCoord object with corresponding NED origin.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        init_geodetic = ecef2geodetic(init_ecef)
        local_coord = LocalCoord(init_geodetic, init_ecef)
        return local_coord

    def ecef2ned(self, ecef):
        """Convert ECEF position vectors to NED position vectors.

        Parameters
        ----------
        ecef : np.ndarray
            Float with ECEF position vectors.

        Returns
        -------
        ned : np.ndarray
            Converted NED position vectors.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        ecef = np.array(ecef)
        # Convert to column vectors for calculation before returning in the same shape as the input
        input_shape = ecef.shape
        if input_shape[0] == 3:
            ned =  np.matmul(self.ecef2ned_matrix, (ecef - np.reshape(self.init_ecef, [3, -1])))
        elif input_shape[1]==3:
            ned = np.matmul(self.ecef2ned_matrix, (ecef.T - np.reshape(self.init_ecef, [3, -1])))
        ned = np.reshape(ned, input_shape)
        return ned

    def ecef2nedv(self, ecef):
        """Convert ECEF free vectors to NED free vectors.

        Parameters
        ----------
        ecef : np.ndarray
            Float with free vectors in the ECEF frame of reference.

        Returns
        -------
        ned : np.ndarray
            Converted free vectors in the NED frame of reference.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        ecef = np.array(ecef)
        # Convert to column vectors for calculation before returning in the same shape as the input
        input_shape = ecef.shape
        if input_shape[0] == 3:
            ned =  np.matmul(self.ecef2ned_matrix, ecef)
        elif input_shape[1]==3:
            ned = np.matmul(self.ecef2ned_matrix, ecef.T)
        ned = np.reshape(ned, input_shape)
        return ned

    def ned2ecef(self, ned):
        """Convert NED position vectors to ECEF position vectors.

        Parameters
        ----------
        ned : np.ndarray
            Float with position vectors in the NED frame of reference.

        Returns
        -------
        ecef : np.ndarray
            Converted position vectors in the ECEF frame of reference.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        ned = np.array(ned)
        # Convert to column vectors for calculation before returning in the same shape as the input
        input_shape = ned.shape
        if input_shape[0] == 3:
            ecef =  np.matmul(self.ned2ecef_matrix, ned) + np.reshape(self.init_ecef, [3, -1])
            return ecef
        elif input_shape[1]==3:
            ecef = np.matmul(self.ned2ecef_matrix, ned.T) + np.reshape(self.init_ecef, [3, -1])
            return ecef.T

    def ned2ecefv(self, ned):
        """Convert NED free vectors to ECEF free vectors.

        Parameters
        ----------
        ned : np.ndarray
            Float with free vectors in the NED frame of reference.

        Returns
        -------
        ecef : np.ndarray
            Converted free vectors in the ECEF frame of reference.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        ned = np.array(ned)
        # Convert to column vectors for calculation before returning in the same shape as the input
        input_shape = ned.shape
        if input_shape[0] == 3:
            ecef =  np.matmul(self.ned2ecef_matrix, ned)
            return ecef
        elif input_shape[1]==3:
            ecef = np.matmul(self.ned2ecef_matrix, ned.T)
            return ecef.T

    def geodetic2ned(self, geodetic):
        """Convert geodetic position vectors to NED position vectors.

        Parameters
        ----------
        geodetic : np.ndarray
            Float with geodetic position vectors.

        Returns
        -------
        ned : np.ndarray
            Converted NED position vectors.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        ecef = geodetic2ecef(geodetic)
        ned = self.ecef2ned(ecef)
        return ned

    def ned2geodetic(self, ned):
        """Convert geodetic position vectors to NED position vectors.

        Parameters
        ----------
        ned : np.ndarray
            Float with NED position vectors.

        Returns
        -------
        geodetic : np.ndarray
            Converted geodetic position vectors.

        Notes
        -----
        Based on code from https://github.com/commaai/laika.

        """

        ecef = self.ned2ecef(ned)
        geodetic = ecef2geodetic(ecef)
        return geodetic
