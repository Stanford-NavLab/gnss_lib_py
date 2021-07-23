########################################################################
# Author(s):    Adam Dai, Shubh Gupta
# Date:         16 Jul 2021
# Desc:         Functions to read data from NMEA files
########################################################################

import os
import sys
import pynmea2
import datetime
import calendar
import numpy as np
sys.path.append("..")
from funcs import coordinates as coord

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
                    msg = pynmea2.parse(line)
                    if type(msg) == pynmea2.GGA:
                        self.gga_msgs.append(msg)
                    elif type(msg) == pynmea2.RMC:
                        self.rmc_msgs.append(msg)
                except:
                    print("WARNING: encountered checksum error while reading nmea")


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
            geo_ls.append([float(msg.latitude), float(msg.longitude), float(msg.altitude)])
        return geo_ls

    def ecef_gt(self):
        """Get ECEF ground truth.

        Returns
        -------
        ecef??? : list????
            Returns ECEF coordinates of ground truth????

        """
        return coord.geodetic2ecef(self.lla_gt())
