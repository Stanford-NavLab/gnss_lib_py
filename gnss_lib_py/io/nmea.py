"""Functions to read data from NMEA files.

"""

__authors__ = "Adam Dai, Shubh Gupta, Derek Knowles"
__date__ = "16 Jul 2021"

import os
import sys
import calendar
# append <path>/gnss_lib_py/gnss_lib_py/ to path
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))

import pynmea2
import numpy as np

from core import coordinates as coord

class NMEA():
    """Class used to parse through NMEA files

    """
    def __init__(self, filename):
        """Initialize NMEA class.

        Parameters
        ----------
        filename : str
            filepath to .NMEA file to read

        """
        self.gga_msgs = []
        self.rmc_msgs = []
        with open(filename, "r") as f:
            for line in f:
                try:
                    msg = pynmea2.parse(line, check = False)
                    if type(msg) == pynmea2.GGA:
                        self.gga_msgs.append(msg)
                    elif type(msg) == pynmea2.RMC:
                        self.rmc_msgs.append(msg)
                except pynmea2.ChecksumError as e:
                    pass

    def ecef_gt_w_time(self, date):
        """Get ECEF ground truth as well as measurement timesself.

        Measurment times are returned in UTC secondsself.

        Parameters
        ----------
        date : datetime object
            Calendar day of the start of recording.

        Returns
        -------
        ecef : np.array
            ECEF coordinates in array of shape [timesteps x 3] [m]
        times : np.array
            UTC time for ground truth measurements [s]
        geo_ls : np.array
            lat, lon, alt ground truth of shape [timesteps x 3]

        """
        geo_ls = []
        times = []

        # calculate date based on datestring
        date_ts = calendar.timegm(date.timetuple())

        for msg in self.gga_msgs:
            geo_ls.append([float(msg.latitude),
                           float(msg.longitude),
                           float(msg.altitude)])
            day_ts = (msg.timestamp.hour*3600 \
                   + msg.timestamp.minute*60 \
                   + msg.timestamp.second \
                   + msg.timestamp.microsecond*1e-6)
            times.append(date_ts + day_ts)

        ecef = coord.geodetic2ecef(geo_ls)
        times = np.array(times)
        geo_ls = np.array(geo_ls)

        return ecef, times, geo_ls


    def lla_gt(self):
        """Get latitude, longitude, and altitude ground truthself.

        Returns
        -------
        geo_ls : list
            A list of lists of latitude, longitude, and alitutde for
            each line of the NMEA file.

        """
        geo_ls = []
        for msg in self.gga_msgs:
            geo_ls.append([float(msg.latitude),
                           float(msg.longitude),
                           float(msg.altitude)])
        return np.array(geo_ls)

    def ecef_gt(self):
        """Get ECEF ground truth.

        Returns
        -------
        ecef??? : list????
            Returns ECEF coordinates of ground truth????

        """
        return coord.geodetic2ecef(self.lla_gt())
