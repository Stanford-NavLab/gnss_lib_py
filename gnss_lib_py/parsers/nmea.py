"""Functions to read data from NMEA files.

"""

__authors__ = "Ashwin Kanhere, Dalton Vega"
__date__ = "24 Jun, 2023"

import os
import datetime

import pynmea2
import numpy as np
import pandas as pd
from pynmea2.nmea_utils import timestamp, datestamp

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils.coordinates import geodetic_to_ecef
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis

class Nmea(NavData):
    """Class used to parse through NMEA files

    """
    def __init__(self, filename, msg_types=None,
                 check=False, keep_raw=False, include_ecef=False):
        """Read instance of NMEA file following NMEA 0183 standard.

        This class uses the NMEA parser from `pynmea2`, which supports
        the NMEA 0183 standard [1]_.
        With the introduction of the NMEA 2300 standard, an extra field
        is added to the RMC message type, as seen in Page 1-5 in [2]_.

        Parameters
        ----------
        filename : str or path-like
            filepath to NMEA file to read.
        msg_types : list
            List of strings describing messages that can be parsed.
            `None` is the default argument which defaults to `[GGA, RMC]`
            message types.
        check : bool
            `True` if the checksum at the end of the NMEA sentence should
            be ignored. `False` if the checksum should be checked and lines
            with incorrect checksums will be ignored.
        raw_coord : bool
            Flag for whether coordinates should be processed into commonly
            used latitude and longitude formats. If 'True', returned
            `NavData` has the same coordinates as the input NMEA file,
            including the cardinal directions.
            If `False`, the coordinates are processed into the decimal
            format between -180&deg; and 180&deg; for longitude and between
            -90&deg; and 90&deg; for latitude.

        References
        ----------
        .. [1] https://github.com/Knio/pynmea2/blob/5d3d2013bff9c5bce2e14132d21fff865b1e58fd/NMEA0183.pdf
               Accessed 24 June, 2023
        .. [2] https://www.sparkfun.com/datasheets/GPS/NMEA%20Reference%20Manual-Rev2.1-Dec07.pdf
               Accessed 24 June, 2023
        """
        if msg_types is None:
            # Use default message types
            msg_types = ['GGA', 'RMC']
        pd_df = pd.DataFrame()
        field_dict = {}
        prev_timestamp = None

        if not isinstance(filename, (str, os.PathLike)):
            raise TypeError("filename must be string or path-like")
        if not os.path.exists(filename):
            raise FileNotFoundError("file not found")

        with open(filename, "r", encoding='UTF-8') as open_file:
            for line in open_file:
                check_ind = line.find('*')
                if not check and '*' in line:
                    # This is the case where a checksum exists but
                    # the user wants to ignore it
                    check_ind = line.find('*')
                    line = line[:check_ind]
                try:
                    msg = pynmea2.parse(line, check = check)
                    if 'timestamp' in list(msg.name_to_idx.keys()):
                        # find first timestamp
                        if prev_timestamp is None:
                            prev_timestamp = msg.timestamp

                        elif msg.timestamp != prev_timestamp:
                            time = field_dict.pop('timestamp')
                            date = field_dict.pop('datestamp')

                            delta_t = datetime.datetime.combine(datestamp(date), timestamp(time))
                            field_dict['gps_millis'] = datetime_to_gps_millis(delta_t)

                            new_row = pd.DataFrame([field_dict])
                            pd_df = pd.concat([pd_df, new_row])
                            field_dict = {}
                            prev_timestamp = msg.timestamp
                    if msg.sentence_type in msg_types:
                        # Both GGA and RMC messages have the latitude and
                        # longitude in them and the following lines should
                        # extract the relevant coordinates in decimal form
                        # from the given degree and decimal minutes format
                        field_dict['lat_float'] = msg.latitude
                        field_dict['lon_float'] = msg.longitude
                        if keep_raw:
                            ignore = []
                        else:
                            ignore = ['lat', 'lat_dir', 'lon', 'lon_dir']
                        ignore += ['mag_variation',
                                  'mag_var_dir',
                                  'mode_indicator',
                                  'nav_status']
                        for field in msg.name_to_idx:
                            if field not in ignore:
                                field_dict[field] = msg.data[msg.name_to_idx[field]]
                except pynmea2.ChecksumError as check_err:
                    # If a checksum error is found, the transmitted message
                    # is wrong and that statement should be skipped
                    raise Exception('Cannot skip any messages. Need both' \
                                    + ' GGA and RMC messages for ' \
                                    + 'all of date, time, and all of LLH') \
                                        from check_err
            time = field_dict.pop('timestamp')
            date = field_dict.pop('datestamp')
            delta_t = datetime.datetime.combine(datestamp(date), timestamp(time))
            field_dict['gps_millis'] = datetime_to_gps_millis(delta_t)
            new_row = pd.DataFrame([field_dict])
            pd_df = pd.concat([pd_df, new_row])
        # As per `gnss_lib_py` standards, convert the heading from degrees
        # to radians
        pd_df['true_course_rad'] = (np.pi/180.)*pd_df['true_course'].astype(float)
        # Convert the given altitude value to float based on the given units

        # Assuming that altitude units are always meters
        pd_df['altitude'] = pd_df['altitude'].astype(float)
        super().__init__(pandas_df=pd_df)
        if include_ecef:
            self.include_ecef()


    @staticmethod
    def _row_map():
        """Map of row names from loaded Nmea to gnss_lib_py standard

        Returns
        -------
        row_map : Dict
            Dictionary of the form {old_name : new_name}
        """
        row_map = {'lat_float' : 'lat_rx_deg',
                   'lon_float' : 'lon_rx_deg',
                   'altitude' : 'alt_rx_m',
                   'spd_over_grnd': 'vx_rx_mps',
                   'true_course': 'heading_raw_rx_deg',
                   'true_course_rad' : 'heading_rx_rad'}
        return row_map


    def include_ecef(self):
        """Include ECEF coordinates for NMEA data.

        The ECEF coordinates are always added inplace to the same instance
        of Nmea that is input.
        """

        ecef = geodetic_to_ecef(self[['lat_rx_deg',
                                      'lon_rx_deg',
                                      'alt_rx_m']])
        self['x_rx_m'] = ecef[0,:]
        self['y_rx_m'] = ecef[1,:]
        self['z_rx_m'] = ecef[2,:]
