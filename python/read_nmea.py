import pynmea2
import gnss_lib.coordinates as coord
import numpy as np
import os, sys
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

class NMEA:
    def __init__(self, filename):
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
        geo_ls = []
        for msg in self.gga_msgs:
            geo_ls.append([float(msg.latitude), float(msg.longitude), float(msg.altitude)])
        return geo_ls

    def ecef_gt(self):
        return coord.geodetic2ecef(self.lla_gt())