"""Functions to read data from NMEA files.

"""

__authors__ = "Adam Dai, Shubh Gupta, Derek Knowles"
__date__ = "16 Jul 2021"

import pynmea2
import numpy as np
import pandas as pd
import calendar
import datetime

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils import coordinates as coord
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis
from pynmea2.nmea_utils import timestamp, datestamp

class NMEA(NavData):
    """Class used to parse through NMEA files

    """
    def __init__(self, filename, msg_types=['GGA', 'RMC'], float_coords=True):
        """Initialize NMEA class.

        Parameters
        ----------
        filename : str
            filepath to .NMEA file to read

        """
        pd_df = pd.DataFrame()
        field_dict = {}
        prev_timestamp = None
        with open(filename, "r") as f:
            for line in f:
                try:
                    msg = pynmea2.parse(line, check = False)
                    if 'timestamp' in list(msg.name_to_idx.keys()):
                        # find first timestamp
                        if prev_timestamp == None:
                            prev_timestamp = msg.timestamp

                        elif msg.timestamp != prev_timestamp:
                            # TODO: put in try/except block if there is no date
                            # convert timestamp and datestamp into gps_millis
                            time = field_dict.pop('timestamp')
                            date = field_dict.pop('datestamp')

                            dt = datetime.datetime.combine(datestamp(date), timestamp(time))
                            field_dict['gps_millis'] = datetime_to_gps_millis(dt)

                            new_row = pd.DataFrame([field_dict])
                            pd_df = pd.concat([pd_df, new_row])
                            field_dict = {}
                            prev_timestamp = msg.timestamp
                    if msg.sentence_type in msg_types:
                        if float_coords:
                            ignore = ['lat', 'lat_dir', 'lon', 'lon_dir']
                            try:
                                field_dict['lat'] = msg.latitude
                                field_dict['lon'] = msg.longitude
                            except NameError:
                                pass
                        else:
                            ignore = []
                        for field in msg.name_to_idx:
                            if field not in ignore:
                                field_dict[field] = msg.data[msg.name_to_idx[field]]
                except pynmea2.ChecksumError as e:
                    pass

            time = field_dict.pop('timestamp')
            date = field_dict.pop('datestamp')
            dt = datetime.datetime.combine(datestamp(date), timestamp(time))
            field_dict['gps_millis'] = datetime_to_gps_millis(dt)
            new_row = pd.DataFrame([field_dict])
            pd_df = pd.concat([pd_df, new_row])
            
        super().__init__(pandas_df=pd_df)
    

    def ecef_gt_w_time(self, date):
        """Get ECEF ground truth as well as measurement timesself.

        NavData times are returned in UTC secondsself.

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

        ecef = coord.geodetic_to_ecef(geo_ls)
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
        return coord.geodetic_to_ecef(self.lla_gt())
