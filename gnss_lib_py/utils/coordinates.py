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
from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import loop_time, find_wildcard_indexes

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
        geodetic = np.vstack((lat, lon, h))
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
        3xN array  containing ECEF [X, Y, Z] coordinates of satellites

    Returns
    -------
    el_az : np.ndarray
        2XN array containing the elevation and azimuth from the
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
    if rx_pos.shape == (1,3):
        rx_pos = rx_pos.T
    if rx_pos.shape != (3, 1):
        raise RuntimeError("Receiver ECEF position must be a " \
                          + "np.ndarray of shape 3x1.")

    # conform satellite position to correct shape
    sv_pos = np.atleast_2d(sv_pos)
    if sv_pos.shape[0] != 3:
        raise RuntimeError("Satellite ECEF position(s) must be a " \
                          + "np.ndarray of shape 3xN.")

    # Convert the receiver location to WGS84
    rx_lla = ecef_to_geodetic(rx_pos)

    # Create variables with the latitude and longitude in radians
    rx_lat, rx_lon = np.deg2rad(rx_lla[:2,0])

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
    rx_array = np.tile(rx_pos,(1, sv_pos.shape[1]))

    # Calculate the normalized pseudorange for each satellite
    pseudorange = (sv_pos - rx_array) / np.linalg.norm(sv_pos - rx_array,
                                             axis=0, keepdims=True)

    # Perform the transform of the normalized pseudorange from ECEF to VEN
    p_ven = np.dot(ecef_to_ven, pseudorange)
    # Calculate elevation and azimuth in degrees
    el_az = np.zeros([2, sv_pos.shape[1]])
    el_az[0,:] = np.rad2deg((np.pi/2. - np.arccos(p_ven[0,:])))
    el_az[1,:] = np.rad2deg(np.arctan2(p_ven[1,:],p_ven[2,:]))

    # wrap from 0 to 360
    while np.any(el_az[1, :] < 0):
        el_az[1, :][el_az[1, :] < 0] += 360

    return el_az

def add_el_az(navdata, receiver_state, inplace=False):
    """Adds elevation and azimuth to NavData object.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        Instance of the NavData class. Must include ``gps_millis`` as
        well as satellite ECEF positions as ``x_sv_m``, ``y_sv_m``,
        ``z_sv_m``, ``gnss_id`` and ``sv_id``.
    receiver_state : gnss_lib_py.navdata.navdata.NavData
        Either estimated or ground truth receiver position in ECEF frame
        in meters as an instance of the NavData class with the
        following rows: ``x_rx*_m``, ``y_rx*_m``, ``z_rx*_m``,
        ``gps_millis``.
    inplace : bool
        If false (default) will add elevation and azimuth to a new
        NavData instance. If true, will add elevation and azimuth to the
        existing NavData instance.

    Returns
    -------
    data_el_az : gnss_lib_py.navdata.navdata.NavData
        If inplace is True, adds ``el_sv_deg`` and ``az_sv_deg`` to
        the input navdata and returns the same object.
        If inplace is False, returns ``el_sv_deg`` and ``az_sv_deg``
        in a new NavData instance along with ``gps_millis`` and the
        corresponding satellite and receiver rows.

    """

    # check for missing rows
    navdata.in_rows(["gps_millis","x_sv_m","y_sv_m","z_sv_m",
                     "gnss_id","sv_id"])
    receiver_state.in_rows(["gps_millis"])

    # check for receiver_state indexes
    rx_idxs = find_wildcard_indexes(receiver_state,["x_rx*_m",
                                                    "y_rx*_m",
                                                    "z_rx*_m"],max_allow=1)

    sv_el_az = None
    for timestamp, _, navdata_subset in loop_time(navdata,"gps_millis"):

        pos_sv_m = navdata_subset[["x_sv_m","y_sv_m","z_sv_m"]]
        # handle scenario with only a single SV returned as 1D array
        pos_sv_m = np.atleast_2d(pos_sv_m).reshape(3,-1)

        # find time index for receiver_state NavData instance
        rx_t_idx = np.argmin(np.abs(receiver_state["gps_millis"] - timestamp))

        pos_rx_m = receiver_state[[rx_idxs["x_rx*_m"][0],
                                   rx_idxs["y_rx*_m"][0],
                                   rx_idxs["z_rx*_m"][0]],
                                   rx_t_idx].reshape(-1,1)

        timestep_el_az = ecef_to_el_az(pos_rx_m, pos_sv_m)

        if sv_el_az is None:
            sv_el_az = timestep_el_az
        else:
            sv_el_az = np.hstack((sv_el_az,timestep_el_az))

    if inplace:
        navdata["el_sv_deg"] = sv_el_az[0,:]
        navdata["az_sv_deg"] = sv_el_az[1,:]
        return navdata

    data_el_az = NavData()
    data_el_az["gps_millis"] = navdata["gps_millis"]
    data_el_az["gnss_id"] = navdata["gnss_id"]
    data_el_az["sv_id"] = navdata["sv_id"]
    data_el_az["x_sv_m"] = navdata["x_sv_m"]
    data_el_az["y_sv_m"] = navdata["y_sv_m"]
    data_el_az["z_sv_m"] = navdata["z_sv_m"]
    data_el_az[rx_idxs["x_rx*_m"][0]] = receiver_state[rx_idxs["x_rx*_m"][0]]
    data_el_az[rx_idxs["y_rx*_m"][0]] = receiver_state[rx_idxs["y_rx*_m"][0]]
    data_el_az[rx_idxs["z_rx*_m"][0]] = receiver_state[rx_idxs["z_rx*_m"][0]]
    data_el_az["el_sv_deg"] = sv_el_az[0,:]
    data_el_az["az_sv_deg"] = sv_el_az[1,:]

    return data_el_az

def wrap_0_to_2pi(angles):
    """Wraps an arbitrary radian between [0, 2pi).

    Angles must be in radians.

    Parameters
    ----------
    angles : np.ndarray
        Array of angles in radians to wrap between 0 and 2pi.

    Returns
    -------
    angles : np.ndarray
        Angles wrapped between 0 and 2pi in radians.

    """
    angles = np.mod(angles, 2*np.pi)

    return angles


def el_az_to_enu_unit_vector(el_deg, az_deg):
    """
    Convert elevation and azimuth to ENU unit vectors.

    Parameters
    ----------
    el_deg : np.ndarray
        Elevation angle in degrees.

    az_deg : np.ndarray
        Azimuth angle in degrees.

    Returns
    -------
    unit_dir_mat : np.ndarray
        ENU unit vectors.
    """
    el_rad = np.deg2rad(el_deg)
    az_rad = np.deg2rad(az_deg)

    unit_dir_mat = np.vstack(
        (np.atleast_2d(np.cos(el_rad) * np.sin(az_rad)),
         np.atleast_2d(np.cos(el_rad) * np.cos(az_rad)),
         np.atleast_2d(np.sin(el_rad))
         )).T

    return unit_dir_mat
