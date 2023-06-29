"""Parses Rinex files.

The Ephemeris Manager provides broadcast ephemeris for specific
satellites at a specific timestep. The EphemerisDownloader class should be
initialized and then the ``get_ephemeris`` function can be used to
retrieve ephemeris for specific satellites. ``get_ephemeris`` returns
the most recent broadcast ephemeris for the provided list of satellites
that was broadcast BEFORE the provided timestamp. For example GPS daily
ephemeris files contain data at a two hour frequency, so if the
timestamp provided is 5am, then ``get_ephemeris`` will return the 4am
data but not 6am. If provided a timestamp between midnight and 2am then
the ephemeris from around midnight (might be the day before) will be
provided. If no list of satellites is provided, then ``get_ephemeris``
will return data for all satellites.

When multiple observations are provided for the same satellite and same
timestep, the Ephemeris Manager will only return the first instance.
This is applicable when requesting ephemeris for multi-GNSS for the
current day. Same-day multi GNSS data is pulled from  same day. For
same-day multi-GNSS from https://igs.org/data/ which often has multiple
observations.

"""


__authors__ = "Shubh Gupta, Ashwin Kanhere"
__date__ = "13 July 2021"

import os
from datetime import datetime, timezone

import georinex
import numpy as np
import pandas as pd

import gnss_lib_py.utils.constants as consts
from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis
from gnss_lib_py.utils.ephemeris_downloader import EphemerisDownloader, DEFAULT_EPHEM_PATH

class Rinex(NavData):
    """Class to handle Rinex measurements.

    The Ephemeris Manager provides broadcast ephemeris for specific
    satellites at a specific timestep. The EphemerisDownloader class
    should be initialized and then the ``get_ephemeris`` function
    can be used to retrieve ephemeris for specific satellites.
    ``get_ephemeris`` returns the most recent broadcast ephemeris
    for the provided list of satellites that was broadcast BEFORE
    the provided timestamp. For example GPS daily ephemeris files
    contain data at a two hour frequency, so if the timestamp
    provided is 5am, then ``get_ephemeris`` will return the 4am data
    but not 6am. If provided a timestamp between midnight and 2am
    then the ephemeris from around midnight (might be the day
    before) will be provided. If no list of satellites is provided,
    then ``get_ephemeris`` will return data for all satellites.

    When multiple observations are provided for the same satellite
    and same timestep, the Ephemeris Manager will only return the
    first instance. This is applicable when requesting ephemeris for
    multi-GNSS for the current day. Same-day multi GNSS data is
    pulled from  same day. For same-day multi-GNSS from
    https://igs.org/data/ which often has multiple observations.

    Inherits from NavData().

    Attributes
    ----------
    iono_params : np.ndarray
        Array of ionosphere parameters ION ALPHA and ION BETA
    verbose : bool
        If true, prints debugging statements.

    """

    def __init__(self, input_paths, satellites=None, verbose=False):
        """Rinex specific loading and preprocessing

        Parameters
        ----------
        input_paths : string or path-like or list of paths
            Path to measurement Rinex file(s).
        satellites : List
            List of satellite IDs as a string, for example ['G01','E11',
            'R06']. Defaults to None which returns get_ephemeris for
            all satellites.

        """
        self.iono_params = None
        self.verbose = verbose
        pd_df = self.preprocess(input_paths, satellites)

        super().__init__(pandas_df=pd_df)


    def preprocess(self, rinex_paths, satellites):
        """Combine Rinex files and create pandas frame if necessary.

        Parameters
        ----------
        rinex_paths : string or path-like or list of paths
            Path to measurement Rinex file(s).
        satellites : List
            List of satellite IDs as a string, for example ['G01','E11',
            'R06']. Defaults to None which returns get_ephemeris for
            all satellites.

        Returns
        -------
        data : pd.DataFrame
            Combined rinex data from all files.

        """

        constellations = EphemerisDownloader.get_constellations(satellites)

        if isinstance(rinex_paths, (str, os.PathLike)):
            rinex_paths = [rinex_paths]

        data = pd.DataFrame()
        self.iono_params = []
        for rinex_path in rinex_paths:
            new_data = self._get_ephemeris_dataframe(rinex_path,
                                                     constellations)
            data = pd.concat((data,new_data), ignore_index=True)
            self.iono_params.append(self.get_iono_params(rinex_path))
        data.reset_index(inplace=True, drop=True)
        data.sort_values('time', inplace=True, ignore_index=True)

        if satellites is not None:
            data = data.loc[data['sv'].isin(satellites)]

        # Move sv to DataFrame columns, reset index
        data = data.reset_index(drop=True)
        # Replace datetime with gps_millis
        gps_millis = [np.float64(datetime_to_gps_millis(df_row['time'])) \
                        for _, df_row in data.iterrows()]
        data['gps_millis'] = gps_millis
        data = data.drop(columns=['time'])
        data = data.rename(columns={"sv":"sv_id"})
        if "GPSWeek" in data.columns:
            data = data.rename(columns={"GPSWeek":"gps_week"})
            if "GALWeek" in data.columns:
                data["gps_week"] = np.where(pd.isnull(data["gps_week"]),
                                                      data["GALWeek"],
                                                      data["gps_week"])
        elif "GALWeek" in data.columns:
            data = data.rename(columns={"GALWeek":"gps_week"})
        if len(data) == 0:
            raise RuntimeError("No ephemeris data available for the " \
                             + "given satellites")
        return data

    def postprocess(self):
        """Rinex specific post processing.

        """

        self['gnss_sv_id'] = self['sv_id']
        gnss_chars = [sv_id[0] for sv_id in np.atleast_1d(self['sv_id'])]
        gnss_nums = [sv_id[1:] for sv_id in np.atleast_1d(self['sv_id'])]
        gnss_id = [consts.CONSTELLATION_CHARS[gnss_char] for gnss_char in gnss_chars]
        self['gnss_id'] = np.asarray(gnss_id)
        self['sv_id'] = np.asarray(gnss_nums, dtype=int)

    def _get_ephemeris_dataframe(self, rinex_path, constellations=None):
        """Load/download ephemeris files and process into DataFrame

        Parameters
        ----------
        rinex_path : string or path-like
            Filepath to rinex file

        constellations : Set
            Set of satellites {"ConstIDSVID"}

        Returns
        -------
        data : pd.DataFrame
            Parsed ephemeris DataFrame
        """

        if constellations is not None:
            data = georinex.load(rinex_path,
                                 use=constellations,
                                 verbose=self.verbose).to_dataframe()
        else:
            data = georinex.load(rinex_path,
                                 verbose=self.verbose).to_dataframe()

        leap_seconds = self.load_leapseconds(rinex_path)
        if leap_seconds is None:
            data['leap_seconds'] = np.nan
        else:
            data['leap_seconds'] = leap_seconds
        data.dropna(how='all', inplace=True)
        data.reset_index(inplace=True)
        data['source'] = rinex_path
        data['t_oc'] = pd.to_numeric(data['time'] - datetime(1980, 1, 6, 0, 0, 0))
        #TODO: Use a constant for the time of GPS clock start
        data['t_oc']  = 1e-9 * data['t_oc'] - consts.WEEKSEC * np.floor(1e-9 * data['t_oc'] / consts.WEEKSEC)
        data['time'] = data['time'].dt.tz_localize('UTC')
        data.rename(columns={'M0': 'M_0', 'Eccentricity': 'e', 'Toe': 't_oe', 'DeltaN': 'deltaN', 'Cuc': 'C_uc', 'Cus': 'C_us',
                             'Cic': 'C_ic', 'Crc': 'C_rc', 'Cis': 'C_is', 'Crs': 'C_rs', 'Io': 'i_0', 'Omega0': 'Omega_0'}, inplace=True)

        return data

    def get_iono_params(self, rinex_path):
        """Gets ionosphere parameters from RINEX file header for calculation of
        ionosphere delay

        Parameters
        ----------
        rinex_path : string or path-like
            Filepath to rinex file

        Returns
        -------
        iono_params : np.ndarray
            Array of ionosphere parameters ION ALPHA and ION BETA
        """
        try:
            ion_alpha_str = georinex.rinexheader(rinex_path)['ION ALPHA'].replace('D', 'E')
            ion_alpha = np.array(list(map(float, ion_alpha_str.split())))
        except KeyError:
            ion_alpha = np.array([[np.nan]])
        try:
            ion_beta_str = georinex.rinexheader(rinex_path)['ION BETA'].replace('D', 'E')
            ion_beta = np.array(list(map(float, ion_beta_str.split())))
        except KeyError:
            ion_beta = np.array([[np.nan]])
        iono_params = np.vstack((ion_alpha, ion_beta))

        return iono_params

    def load_leapseconds(self, filename):
        """Read leapseconds from rinex file

        Parameters
        ----------
        filename : string
            Ephemeris filename

        Returns
        -------
        read_lp_sec : int or None
            Leap seconds read from file

        """
        with open(filename) as f:
            for line in f:
                if 'LEAP SECONDS' in line:
                    read_lp_sec = int(line.split()[0])
                    return read_lp_sec
                if 'END OF HEADER' in line:
                    return None

        return None

def get_time_cropped_rinex(timestamp, satellites=None,
                           ephemeris_directory=DEFAULT_EPHEM_PATH):
    """Add SV states using Rinex file.

    Provides broadcast ephemeris for specific
    satellites at a specific timestep
    ``add_sv_states_rinex`` returns the most recent broadcast ephemeris
    for the provided list of satellites that was broadcast BEFORE
    the provided timestamp. For example GPS daily ephemeris files
    contain data at a two hour frequency, so if the timestamp
    provided is 5am, then ``add_sv_states_rinex`` will return the 4am data
    but not 6am. If provided a timestamp between midnight and 2am
    then the ephemeris from around midnight (might be the day
    before) will be provided. If no list of satellites is provided,
    then ``add_sv_states_rinex`` will return data for all satellites.

    When multiple observations are provided for the same satellite
    and same timestep,  will only return the
    first instance. This is applicable when requesting ephemeris for
    multi-GNSS for the current day. Same-day multi GNSS data is
    pulled from  same day. For same-day multi-GNSS from
    https://igs.org/data/ which often has multiple observations.

    Parameters
    ----------
    timestamp : datetime.datetime
        Ephemeris data is returned for the timestamp day and
        includes all broadcast ephemeris whose broadcast timestamps
        happen before the given timestamp variable. Timezone should
        be added manually and is interpreted as UTC if not added.
    satellites : List
        List of satellite IDs as a string, for example ['G01','E11',
        'R06']. Defaults to None which returns get_ephemeris for
        all satellites.

    Returns
    -------
    data : gnss_lib_py.parsers.navdata.NavData
        ephemeris entries corresponding to timestamp

    Notes
    -----
    The Galileo week ``GALWeek`` is identical to the GPS Week
    ``GPSWeek``. See http://acc.igs.org/misc/rinex304.pdf page A26

    """

    ephemeris_downloader = EphemerisDownloader(ephemeris_directory)
    rinex_paths = ephemeris_downloader.get_ephemeris(timestamp,satellites)
    rinex_data = Rinex(rinex_paths, satellites=satellites)

    timestamp_millis = datetime_to_gps_millis(timestamp)
    time_cropped_data = rinex_data.where('gps_millis', timestamp_millis, "lesser")

    time_cropped_data = time_cropped_data.pandas_df().sort_values(
        'gps_millis').groupby('gnss_sv_id').last()
    if satellites is not None and len(time_cropped_data) < len(satellites):
        # if no data available for the given day, try looking at the
        # previous day, may occur when a time near to midnight
        # is provided. For example, 12:01am
        if len(time_cropped_data) != 0:
            satellites = list(set(satellites) - set(time_cropped_data.index))
        prev_day_timestamp = datetime(year=timestamp.year,
                                      month=timestamp.month,
                                      day=timestamp.day - 1,
                                      hour=23,
                                      minute=59,
                                      second=59,
                                      microsecond=999999,
                                      tzinfo=timezone.utc,
                                      )
        prev_rinex_paths = ephemeris_downloader.get_ephemeris(prev_day_timestamp,
                                                              satellites)
        # TODO: verify that the above statement doesn't need "False for timestamp"
        prev_rinex_data = Rinex(prev_rinex_paths, satellites=satellites)

        prev_data = prev_rinex_data.pandas_df().sort_values('gps_millis').groupby(
            'gnss_sv_id').last()
        rinex_data_df = pd.concat((time_cropped_data,prev_data))
        rinex_iono_params = prev_rinex_data.iono_params + rinex_data.iono_params
    else:
        rinex_data_df = time_cropped_data
        rinex_iono_params = rinex_data.iono_params

    rinex_data_df = rinex_data_df.reset_index()
    rinex_data = NavData(pandas_df=rinex_data_df)
    rinex_data.iono_params = rinex_iono_params

    return rinex_data
