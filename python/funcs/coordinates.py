########################################################################
# Author(s):    Shubh Gupta, Ashwin Kanhere
# Date:         20 July 2021
# Desc:         Functions for coordinate conversions required by GPS
########################################################################

import numpy as np
from utils.constants import CoordConsts

#Coordinate conversions (From https://github.com/commaai/laika)


def geodetic2ecef(geodetic, radians=False):
    """LLA to ECEF conversion 

    Parameters
    ----------
    geodetic : ndarray
        Float with WGS-84 LLA coordinates 
    radians : bool
        Flag of whether input is in radians

    Returns
    -------
    ecef : ndarray
        ECEF coordinates corresponding to input LLA

    Notes
    -----
    Based on code from https://github.com/commaai/laika

    """
    coordconsts = CoordConsts()
    geodetic = np.array(geodetic)
    input_shape = geodetic.shape
    geodetic = np.atleast_2d(geodetic)

    ratio = 1.0 if radians else (np.pi / 180.0)
    lat = ratio*geodetic[:,0]
    lon = ratio*geodetic[:,1]
    alt = geodetic[:,2]

    xi = np.sqrt(1 - coordconsts.ESQ * np.sin(lat)**2)
    x = (coordconsts.A / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (coordconsts.A / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (coordconsts.A / xi * (1 - coordconsts.ESQ) + alt) * np.sin(lat)
    ecef = np.array([x, y, z]).T
    ecef = np.reshape(ecef, input_shape)
    return ecef


def ecef2geodetic(ecef, radians=False):
    """ECEF to LLA conversion using Ferrari's method

    Parameters
    ----------
    ecef : ndarray
        Float with ECEF coordinates 
    radians : bool
        Flag of whether output should be in radians

    Returns
    -------
    geodetic : ndarray
        Float with WGS-84 LLA coordinates corresponding to input ECEF

    Notes
    -----
    Based on code from https://github.com/commaai/laika

    """
    coordconsts = CoordConsts()
    ecef = np.atleast_1d(ecef)
    input_shape = ecef.shape
    ecef = np.atleast_2d(ecef)
    x, y, z = ecef[:, 0], ecef[:, 1], ecef[:, 2]

    ratio = 1.0 if radians else (180.0 / np.pi)

    # Convert from ECEF to geodetic using Ferrari's methods
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Ferrari.27s_solution
    r = np.sqrt(x * x + y * y)
    Esq = coordconsts.A * coordconsts.A - coordconsts.B * coordconsts.B
    F = 54 * coordconsts.B * coordconsts.B * z * z
    G = r * r + (1 - coordconsts.ESQ) * z * z - coordconsts.ESQ * Esq
    C = (coordconsts.ESQ * coordconsts.ESQ * F * r * r) / (pow(G, 3))
    S = np.cbrt(1 + C + np.sqrt(C * C + 2 * C))
    P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
    Q = np.sqrt(1 + 2 * coordconsts.ESQ * coordconsts.ESQ * P)
    r_0 =  -(P * coordconsts.ESQ * r) / (1 + Q) + np.sqrt(0.5 * coordconsts.A * coordconsts.A*(1 + 1.0 / Q) - \
          P * (1 - coordconsts.ESQ) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r)
    U = np.sqrt(pow((r - coordconsts.ESQ * r_0), 2) + z * z)
    V = np.sqrt(pow((r - coordconsts.ESQ * r_0), 2) + (1 - coordconsts.ESQ) * z * z)
    Z_0 = coordconsts.B * coordconsts.B * z / (coordconsts.A * V)
    h = U * (1 - coordconsts.B * coordconsts.B / (coordconsts.A * V))
    lat = ratio*np.arctan((z + coordconsts.E1SQ * Z_0) / r)
    lon = ratio*np.arctan2(y, x)

    # stack the new columns and return to the original shape
    geodetic = np.column_stack((lat, lon, h))
    geodetic = np.reshape(geodetic, input_shape)
    return geodetic.reshape(input_shape)

class LocalCoord(object):
    """Class for conversions to NED (North-East-Down)

    Attributes
    ----------
    init_ecef : ndarray
        ECEF of origin of NED
    ned2ecef_matrix : ndarray
        Rotation matrix to convert from NED to ECEF
    ecef2ned_matrix : ndarray
        Rotation matrix to convert from ECEF to NED

    Methods
    -------
    from_geodetic(cls, init_geodetic)
        Instantiate a class from LLA of NED origin

    from_ecef(cls, init_ecef)
        Instantiate a class from ECEF of NED origin

    ecef2ned(ecef)
        Convert ECEF position vectors to NED position vectors

    ecef2nedv(ecef)
        Convert ECEF free vectors to NED free vectors
    
    ned2ecef(ned)
        Convert NED position vectors to ECEF position vectors

    ned2ecefv(ned)
        Convert NED free vectors to ECEF free vectors

    geodetic2ned(geodetic)
        Convert WGS-84 LLA position vectors to NED position vectors

    ned2geodetic(ned)
        Convert NED position vectors to WGS-84 LLA position vectors

    Notes
    -----
    Based on code from https://github.com/commaai/laika
    """

    def __init__(self, init_geodetic, init_ecef):
        self.init_ecef = init_ecef
        lat, lon, _ = (np.pi/180)*np.array(init_geodetic)
        self.ned2ecef_matrix = np.array([[-np.sin(lat)*np.cos(lon), -np.sin(lon), -np.cos(lat)*np.cos(lon)],
                                         [-np.sin(lat)*np.sin(lon), np.cos(lon), -np.cos(lat)*np.sin(lon)],
                                         [np.cos(lat), 0, -np.sin(lat)]])
        self.ecef2ned_matrix = self.ned2ecef_matrix.T

    @classmethod
    def from_geodetic(cls, init_geodetic):
        """Instantiate class using the NED origin in geodetic coordinates

        Parameters
        ----------
        init_geodetic : ndarray
            Float with WGS-84 LLA coordinates of the NED origin

        Returns
        -------
        local_coord : LocalCoord
            Instance of LocalCoord object with corresponding NED origin

        Notes
        -----
        Based on code from https://github.com/commaai/laika        
        """
        init_ecef = geodetic2ecef(init_geodetic)
        local_coord = LocalCoord(init_geodetic, init_ecef)
        return local_coord

    @classmethod
    def from_ecef(cls, init_ecef):        
        """Instantiate class using the NED origin in ECEF coordinates

        Parameters
        ----------
        init_ecef : ndarray
            Float with ECEF coordinates of the NED origin

        Returns
        -------
        local_coord : LocalCoord
            Instance of LocalCoord object with corresponding NED origin

        Notes
        -----
        Based on code from https://github.com/commaai/laika        
        """
        init_geodetic = ecef2geodetic(init_ecef)
        local_coord = LocalCoord(init_geodetic, init_ecef)
        return local_coord

    def ecef2ned(self, ecef):        
        """Convert ECEF position vectors to NED position vectors

        Parameters
        ----------
        ecef : ndarray
            Float with ECEF position vectors

        Returns
        -------
        ned : ndarray
            Converted NED position vectors

        Notes
        -----
        Based on code from https://github.com/commaai/laika        
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
        """Convert ECEF free vectors to NED free vectors

        Parameters
        ----------
        ecef : ndarray
            Float with free vectors in the ECEF frame of reference

        Returns
        -------
        ned : ndarray
            Converted free vectors in the NED frame of reference

        Notes
        -----
        Based on code from https://github.com/commaai/laika        
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
        """Convert NED position vectors to ECEF position vectors

        Parameters
        ----------
        ned : ndarray
            Float with position vectors in the NED frame of reference

        Returns
        -------
        ecef : ndarray
            Converted position vectors in the ECEF frame of reference

        Notes
        -----
        Based on code from https://github.com/commaai/laika        
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
        """Convert NED free vectors to ECEF free vectors

        Parameters
        ----------
        ned : ndarray
            Float with free vectors in the NED frame of reference

        Returns
        -------
        ecef : ndarray
            Converted free vectors in the ECEF frame of reference

        Notes
        -----
        Based on code from https://github.com/commaai/laika        
        """
        #Coordinate conversions (From https://github.com/commaai/laika)
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
        """Convert geodetic position vectors to NED position vectors

        Parameters
        ----------
        geodetic : ndarray
            Float with geodetic position vectors

        Returns
        -------
        ned : ndarray
            Converted NED position vectors

        Notes
        -----
        Based on code from https://github.com/commaai/laika        
        """
        #Coordinate conversions (From https://github.com/commaai/laika)
        ecef = geodetic2ecef(geodetic)
        ned = self.ecef2ned(ecef)
        return ned

    def ned2geodetic(self, ned):
        """Convert geodetic position vectors to NED position vectors

        Parameters
        ----------
        ned : ndarray
            Float with NED position vectors

        Returns
        -------
        geodetic : ndarray
            Converted geodetic position vectors

        Notes
        -----
        Based on code from https://github.com/commaai/laika        
        """
        #Coordinate conversions (From https://github.com/commaai/laika)
        ecef = self.ned2ecef(ned)
        geodetic = ecef2geodetic(ecef)
        return geodetic
