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

__authors__ = "Shubh Gupta, Ashwin Kanhere, Derek Knowles"
__date__ = "20 July 2021"

import numpy as np

import gnss_lib_py.utils.constants as consts

EPSILON = 1e-7

def geodetic_to_ecef(geodetic, radians=False):
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

    ratio = 1.0 if radians else (np.pi / 180.0)
    geodetic = np.array(geodetic)
    input_shape = geodetic.shape
    geodetic = np.atleast_2d(geodetic)
    if input_shape[0]==3:
        lat = ratio*geodetic[0,:]
        lon = ratio*geodetic[1,:]
        alt = geodetic[2,:]
    elif input_shape[1]==3:
        lat = ratio*geodetic[:,0]
        lon = ratio*geodetic[:,1]
        alt = geodetic[:,2]
    else:  # pragma: no cover
        raise ValueError('geodetic is incorrect shape ', geodetic.shape,
                        ' should be (N,3) or (3,N)')
    xi = np.sqrt(1 - consts.E1SQ * np.sin(lat)**2)
    x = (consts.A / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (consts.A / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (consts.A / xi * (1 - consts.E1SQ) + alt) * np.sin(lat)
    ecef = np.array([x, y, z]).T
    if input_shape[0]==3:
        ecef = ecef.T
    return ecef


def ecef_to_geodetic(ecef, radians=False):
    """ECEF to LLA conversion using Ferrari's method.

    Parameters
    ----------
    ecef : np.ndarray
        array where ECEF x, ECEF y, and ECEF z are either independent
        rows or independent columns, values should be floats
    radians : bool
        If False (default), output of lat/lon is returned in degrees.
        If True, output of lat/lon is returned in radians.

    Returns
    -------
    geodetic : np.ndarray
        Float with WGS-84 LLA coordinates corresponding to input ECEF.
        Order is returned as (lat, lon, h) and is returned in the same
        shape as the input. Height is in meters above the the WGS-84
        ellipsoid.

    Notes
    -----
    Based on code from https://github.com/commaai/laika.

    """

    ecef = np.atleast_2d(ecef)
    ecef = ecef.astype(np.float64)
    input_shape = ecef.shape
    if input_shape[0]==3:
        x_ecef, y_ecef, z_ecef = ecef[0, :], ecef[1, :], ecef[2, :]
    elif input_shape[1]==3:
        x_ecef, y_ecef, z_ecef = ecef[:, 0], ecef[:, 1], ecef[:, 2]
    else:  # pragma: no cover
        raise ValueError('Input ECEF vector has incorrect shape ', ecef.shape,
                        ' should be (N,3) or (3,N)')
    ratio = 1.0 if radians else (180.0 / np.pi)

    # Convert from ECEF to geodetic using Ferrari's methods
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Ferrari.27s_solution
    r = np.sqrt(x_ecef * x_ecef + y_ecef * y_ecef)
    E1SQ = consts.A * consts.A - consts.B * consts.B
    F = 54 * consts.B * consts.B * z_ecef * z_ecef
    G = r * r + (1 - consts.E1SQ) * z_ecef * z_ecef - consts.E1SQ * E1SQ
    C = (consts.E1SQ * consts.E1SQ * F * r * r) / (pow(G, 3))
    S = np.cbrt(1 + C + np.sqrt(C * C + 2 * C + EPSILON))
    P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
    Q = np.sqrt(1 + 2 * consts.E1SQ * consts.E1SQ * P)
    r_0 =  -(P * consts.E1SQ * r) / (1 + Q) + np.sqrt(0.5 * consts.A * consts.A*(1 + 1.0 / Q) - \
          P * (1 - consts.E1SQ) * z_ecef * z_ecef / (Q * (1 + Q)) - 0.5 * P * r * r)
    U = np.sqrt(pow((r - consts.E1SQ * r_0), 2) + z_ecef * z_ecef)
    V = np.sqrt(pow((r - consts.E1SQ * r_0), 2) + (1 - consts.E1SQ) * z_ecef * z_ecef)
    Z_0 = consts.B * consts.B * z_ecef / (consts.A * V)
    h = U * (1 - consts.B * consts.B / ((consts.A * V)))
    lat = ratio*np.arctan((z_ecef + consts.E2SQ * Z_0) / (r))
    lon = ratio*np.arctan2(y_ecef, x_ecef)
    # stack the new columns and return to the original shape
    geodetic = np.column_stack((lat, lon, h))
    if input_shape[0]==3:
        geodetic = np.row_stack((lat, lon, h))
    return geodetic

class LocalCoord(object):
    """Class for conversions to NED (North-East-Down).

    Attributes
    ----------
    init_ecef : np.ndarray
        ECEF of origin of NED.
    ned_to_ecef_matrix : np.ndarray
        Rotation matrix to convert from NED to ECEF.
    ecef_to_ned_matrix : np.ndarray
        Rotation matrix to convert from ECEF to NED.

    Notes
    -----
    Based on code from https://github.com/commaai/laika.

    """

    def __init__(self, init_geodetic, init_ecef):
        self.init_ecef = init_ecef
        if init_geodetic.shape[0]==3:
            lat = (np.pi/180.)*init_geodetic[0, 0]
            lon = (np.pi/180.)*init_geodetic[1, 0]
        elif init_geodetic.shape[1]==3:
            lat = (np.pi/180.)*init_geodetic[0, 0]
            lon = (np.pi/180.)*init_geodetic[0, 1]
        else:  # pragma: no cover
            raise ValueError('init_geodetic has incorrect size', len(init_geodetic),
                            ' must be of size 3')
        self.ned_to_ecef_matrix = np.array([[-np.sin(lat)*np.cos(lon), -np.sin(lon), -np.cos(lat)*np.cos(lon)],
                                            [-np.sin(lat)*np.sin(lon), np.cos(lon), -np.cos(lat)*np.sin(lon)],
                                            [np.cos(lat), 0, -np.sin(lat)]])
        self.ecef_to_ned_matrix = self.ned_to_ecef_matrix.T

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

        init_ecef = geodetic_to_ecef(init_geodetic)
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

        init_geodetic = ecef_to_geodetic(init_ecef)
        local_coord = LocalCoord(init_geodetic, init_ecef)
        return local_coord

    def ecef_to_ned(self, ecef):
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
            ned =  np.matmul(self.ecef_to_ned_matrix, (ecef - np.reshape(self.init_ecef, [3, -1])))
        elif input_shape[1]==3:
            ned = np.matmul(self.ecef_to_ned_matrix, (ecef.T - np.reshape(self.init_ecef, [3, -1])))
            ned = np.transpose(ned)
        return ned

    def ecef_to_nedv(self, ecef):
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
            ned =  np.matmul(self.ecef_to_ned_matrix, ecef)
        elif input_shape[1]==3:
            ned = np.matmul(self.ecef_to_ned_matrix, ecef.T)
            ned = ned.T
        return ned

    def ned_to_ecef(self, ned):
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
            ecef =  np.matmul(self.ned_to_ecef_matrix, ned) + np.reshape(self.init_ecef, [3, -1])
        elif input_shape[1]==3:
            ecef = np.matmul(self.ned_to_ecef_matrix, ned.T) + np.reshape(self.init_ecef, [3, -1])
            ecef = ecef.T
        return ecef

    def ned_to_ecefv(self, ned):
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
            ecef =  np.matmul(self.ned_to_ecef_matrix, ned)
        elif input_shape[1]==3:
            ecef = np.matmul(self.ned_to_ecef_matrix, ned.T)
            ecef = ecef.T
        return ecef

    def geodetic_to_ned(self, geodetic):
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

        ecef = geodetic_to_ecef(geodetic)
        ned = self.ecef_to_ned(ecef)
        return ned

    def ned_to_geodetic(self, ned):
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

        ecef = self.ned_to_ecef(ned)
        geodetic = ecef_to_geodetic(ecef)
        return geodetic

def ecef_to_el_az(rx_pos, sv_pos):
    """Calculate the elevation and azimuth from receiver to satellites.

    Vectorized to be able to be able to output the elevation and azimuth
    for multiple satellites at the same time.

    Parameters
    ----------
    rx_pos : np.ndarray
        1x3 vector containing ECEF [X, Y, Z] coordinate of receiver
    sv_pos : np.ndarray
        Nx3 array  containing ECEF [X, Y, Z] coordinates of satellites

    Returns
    -------
    el_az : np.ndarray
        Nx2 array containing the elevation and azimuth from the
        receiver to the requested satellites. Elevation and azimuth are
        given in decimal degrees.

    Notes
    -----
    Code based on method by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    """

    # conform receiver position to correct shape
    rx_pos = np.atleast_2d(rx_pos)
    if rx_pos.shape == (3,1):
        rx_pos = rx_pos.T
    if rx_pos.shape != (1,3):
        raise RuntimeError("Receiver ECEF position must be a " \
                          + "np.ndarray of shape 1x3.")

    # conform satellite position to correct shape
    sv_pos = np.atleast_2d(sv_pos)
    if sv_pos.shape[1] != 3:
        raise RuntimeError("Satellite ECEF position(s) must be a " \
                          + "np.ndarray of shape Nx3.")

    # Convert the receiver location to WGS84
    rx_lla = ecef_to_geodetic(rx_pos)

    # Create variables with the latitude and longitude in radians
    rx_lat, rx_lon = np.deg2rad(rx_lla[0,:2])

    # Create the 3 x 3 transform matrix from ECEF to VEN
    ecef_to_ven = np.array([[ np.cos(rx_lat)*np.cos(rx_lon),
                              np.cos(rx_lat)*np.sin(rx_lon),
                              np.sin(rx_lat)],
                            [-np.sin(rx_lon),
                              np.cos(rx_lon),
                              0.     ],
                            [-np.sin(rx_lat)*np.cos(rx_lon),
                             -np.sin(rx_lat)*np.sin(rx_lon),
                              np.cos(rx_lat)]])

    # Replicate the rx_pos array to be the same size as the satellite array
    rx_array = np.tile(rx_pos,(len(sv_pos),1))

    # Calculate the normalized pseudorange for each satellite
    pseudorange = (sv_pos - rx_array) / np.linalg.norm(sv_pos - rx_array,
                                             axis=1, keepdims=True)

    # Perform the transform of the normalized pseudorange from ECEF to VEN
    p_ven = np.dot(ecef_to_ven, pseudorange.T)

    # Calculate elevation and azimuth in degrees
    el_az = np.zeros([sv_pos.shape[0],2])
    el_az[:,0] = np.rad2deg((np.pi/2. - np.arccos(p_ven[0,:])))
    el_az[:,1] = np.rad2deg(np.arctan2(p_ven[1,:],p_ven[2,:]))

    # wrap from 0 to 360
    while np.any(el_az[:,1] < 0):
        el_az[:,1][el_az[:,1] < 0] += 360

    return el_az
