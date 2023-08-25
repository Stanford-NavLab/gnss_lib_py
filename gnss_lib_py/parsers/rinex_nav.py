"""Parses Rinex .n files.

Loads rinex navigation files into a NavData object. Loading time can be
sped up significantly by passing in the "satellites" parameter which in
turn gets passed into the georinex library used to parse the rinex file.

Rinex files can be downloaded with the load_ephemeris function in the
utils/ephemeris_downloader.py file.

"""


__authors__ = "Ashwin Kanhere, Shubh Gupta, Dalton Vega"
__date__ = "13 July 2021"

import os
import warnings
from datetime import datetime, timezone
import sys

import numpy as np
import pandas as pd
import georinex as gr
from scipy.integrate import solve_ivp

import gnss_lib_py.utils.constants as consts
from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils.coordinates import pz90_to_inertial, inertial_to_pz90
from gnss_lib_py.utils.time_conversions import (datetime_to_gps_millis,
                                                gps_millis_to_tow,
                                                gps_millis_to_datetime,
                                                datetime_to_beidou_millis_since_gps_0,
                                                gps_datetime_to_gps_millis,
                                                tzinfo_to_utc)
from gnss_lib_py.utils.ephemeris_downloader import (load_ephemeris,
                                                    DEFAULT_EPHEM_PATH,
                                                    split_gnss_sv_ids)


def load_rinex_nav(timestamp, satellites=None,
                   rinex_nav_paths=None,
                   ephemeris_directory=DEFAULT_EPHEM_PATH,
                   verbose=False):
    """Load/download and trim rinex file for most appropriate data.

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

    For GLONASS and SBAS data, which is provided with a validity period,
    this function will also find the closest entry to the provided time.
    In many cases, this will be a time that corresponds to the satellite
    positions, velocities, and accelerations in the future.

    When multiple observations are provided for the same satellite
    and same timestep,  will only return the most applicable
    instance. This is applicable when requesting ephemeris for
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

    The Beidou week has a constant offset from the GPS week, which is noted
    in the constants by marking the GPS week when the Beidou clock epoch
    started.

    """
    if isinstance(timestamp, datetime):
        timestamp_millis = datetime_to_gps_millis(timestamp)
    elif isinstance(timestamp, (int, float)):
        timestamp_millis = timestamp
    elif isinstance(timestamp, np.ndarray) and \
        (np.issubdtype(timestamp.dtype, np.integer) or
         np.issubdtype(timestamp.dtype, np.floating)):
        timestamp_millis = timestamp
    else:
        raise TypeError("timestamp must be a datetime.datetime, a float/int " \
                        + "or np.ndarray containing ints/floats representing gps_millis")
    gnss_id, _ = split_gnss_sv_ids(gnss_sv_id=satellites)
    constellations = np.unique(gnss_id)
    rinex_nav_new_paths = load_ephemeris("rinex_nav", timestamp_millis,
                                     constellations=constellations,
                                     file_paths=rinex_nav_paths,
                                     download_directory=ephemeris_directory)
    rinex_data = RinexNav(rinex_nav_new_paths, satellites=satellites,
                          verbose=verbose)
    rinex_iono_params = rinex_data.iono_params

    time_cropped_rinex = trim_rinex_data(timestamp_millis, rinex_data)
    if satellites is not None and len(time_cropped_rinex) < len(satellites):
        # raise a RunTime warning because there are some satellites for which
        # ephimerides were not available
        non_rx_sats = list(set(satellites) - set(time_cropped_rinex['gnss_sv_id']))
        warnings.warn(f"Ephemeris data not available for {non_rx_sats}", RuntimeWarning)

    time_cropped_rinex.iono_params = rinex_iono_params

    return time_cropped_rinex


def trim_rinex_data(gps_millis, rinex_nav, constellations=None):
    """Find the appropriate closest rinex entry for the given gps_millis.

    For GPS, Galileo, Beidou, QZSS, and IRNSS, ephemeris parameters for
    the closest time in the past are used. For GLONASS and SBAS, the
    ephemeris parameters that are closest in time are used. In some cases,
    as per the RINEX v3 specification [1]_, this includes using parameters
    from the near future.

    Parameters
    ----------
    gps_millis : float
        Time at which SV states will be estimated and for which closest
        ephemeris parameters are needed.
    rinex_nav : gnss_lib_py.parsers.rinex_nav.RinexNav
        RinexNav object containing ephemeris parameters for all the
        constellations that are required.

    Returns
    -------
    trimmed_rinex : gnss_lib_py.parsers.navdata.NavData
        `NavData` instance containing `RinexNav` rows with the correspondingly
        trimmed epheemeris parameters for SV state estimation.

    References
    ----------
    ..  [1] http://acc.igs.org/misc/rinex304.pdf
    """
    # Split the rinex data into the GLONASS and SBAS part and find the
    # closest time to the query time
    #TODO: The tests for this should check that the lengths are correct
    # that the right times were extracted and that the corresponding
    # values are correct too
    # Those tests can also be used to test the get_time_cropped_rinex function
    # which is basically a wrapper for this function at this point
    if constellations is not None:
        if isinstance(constellations, str):
            constellations = [constellations]
        rinex_nav = rinex_nav.where('gnss_id', constellations, 'eq')
    rinex_gps_like, rinex_glonass_like = _split_rinex_nav(rinex_nav)
    if len(rinex_glonass_like) != 0:
        rinex_glonass_like['time_diff'] = np.abs(rinex_glonass_like['gps_millis']
                                                - gps_millis)
        rinex_glonass_like_df = rinex_glonass_like.pandas_df()
        closest_glonass_like_df = \
        rinex_glonass_like_df.loc[rinex_glonass_like_df.groupby('gnss_sv_id')['time_diff'].idxmin()]
        closest_glonass_like = NavData(pandas_df=closest_glonass_like_df.reset_index(drop=True))
        closest_glonass_like.remove(rows='time_diff', inplace=True)
    else:
        closest_glonass_like = NavData()
    # Split the rinex data into other constellations and find the closest
    # time, in the past, to the query time
    if len(rinex_gps_like) != 0:
        time_cropped_gps_like = rinex_gps_like.where('gps_millis', gps_millis, "lesser")
        #Convert back to to pd.DataFrame to utilize the strong sort_values
        # functionality
        time_cropped_gps_like = time_cropped_gps_like.pandas_df().sort_values(
                                'gps_millis').groupby('gnss_sv_id').last()
        time_cropped_gps_like = NavData(pandas_df=time_cropped_gps_like.reset_index())
    else:
        time_cropped_gps_like = NavData()
    trimmed_rinex = time_cropped_gps_like.concat(closest_glonass_like)

    return trimmed_rinex



class RinexNav(NavData):
    """Class to parse Rinex navigation files containing SV parameters.


    Loads rinex navigation files into a NavData object. Loading time can
    be sped up significantly by passing in the "satellites" parameter
    which in turn gets passed into the georinex library used to parse
    the rinex file.

    Inherits from NavData().

    Attributes
    ----------
    iono_params : dict
        Dictionary of the ionosphere correction terms of the form
        ``{gps_millis_at_day_start : {gnss_id : iono_array}}`` where
        ``gps_millis_at_day_start`` is the time (in gps_millis) for the
        beginning of the day corresponding to the ionosphere correction
        parameters in ``iono_array``.
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

        Also loads the ionospheric correction parameters from the Rinex
        file header and stores them in a dictionary while loading the
        pandas dataframe.

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

        if satellites is not None and len(satellites) != 0:
            constellations = set()
            for sat in satellites:
                constellations.add(sat[0])
        else:
            constellations = None

        if isinstance(rinex_paths, (str, os.PathLike)):
            rinex_paths = [rinex_paths]

        data = pd.DataFrame()
        self.iono_params = {}
        for rinex_path in rinex_paths:
            new_data, rinex_header = self._get_ephemeris_dataframe(rinex_path,
                                                                   constellations)
            data = pd.concat((data,new_data), ignore_index=True)
            # The pandas dataframe is indexed by a (time, sv) tuple and
            # the following line gets the date of the first entry and
            # converts it to an equivalent time in gps_millis
            first_time = new_data['time'][0]
            day_start_time = first_time.replace(hour=0,
                                                minute=0,
                                                second=0,
                                                tzinfo = timezone.utc)
            start_gps_millis = float(datetime_to_gps_millis(day_start_time))
            iono_params = self.get_iono_params(rinex_header,
                                               constellations)

            if start_gps_millis not in self.iono_params \
                or self.iono_params[start_gps_millis] is None:
                self.iono_params[start_gps_millis] = iono_params
            else:
                for constellation, value in self.iono_params.items():
                    if constellation not in \
                        self.iono_params[start_gps_millis].keys():
                            self.iono_params[start_gps_millis][constellation] \
                            = value
            #TODO: Find a more pythonic way to do this^
        data.reset_index(inplace=True, drop=True)
        data.sort_values('time', inplace=True, ignore_index=True)

        if satellites is not None:
            data = data.loc[data['sv'].isin(satellites)]

        # Move sv to DataFrame columns, reset index
        data = data.reset_index(drop=True)
        # Replace datetime with gps_millis
        # Use the vectorized version of datetime_to_gps_millis
        gps_millis = datetime_to_gps_millis(data['time'])
        # gps_millis = [np.float64(datetime_to_gps_millis(df_row['time'])) \
        #                 for _, df_row in data.iterrows()]
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
        if "BDTWeek" in data.columns:
            data = data.rename(columns={"BDTWeek":"beidou_week"})
        if len(data) == 0:
            raise RuntimeError("No ephemeris data available for the " \
                             + "given satellites")
        return data

    def postprocess(self):
        """Rinex specific post processing.

        This function breaks the input `sv` row containing the
        standard `gnss_sv_id` data into `gnss_id` and `sv_id` rows and
        also renames `sv` to `gnss_sv_id` to match the standard
        nomenclature.

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
        constellations : set
            Set of satellites {"ConstIDSVID"}

        Returns
        -------
        data : pd.DataFrame
            Parsed ephemeris DataFrame
        data_header : dict
            Header information from Rinex file.

        """

        if constellations is not None:
            data = gr.load(rinex_path,
                                 use=constellations,
                                 verbose=self.verbose).to_dataframe()
        else:
            data = gr.load(rinex_path,
                                 verbose=self.verbose).to_dataframe()
        data_header = gr.rinexheader(rinex_path)
        leap_seconds = self.load_leapseconds(data_header)
        data.dropna(how='all', inplace=True)
        data['leap_seconds'] = leap_seconds
        data.reset_index(inplace=True)
        data['source'] = rinex_path
        data['t_oc'] = pd.to_numeric(data['time'] - consts.GPS_EPOCH_0.replace(tzinfo=None))
        data['t_oc']  = 1e-9 * data['t_oc'] - consts.WEEKSEC * np.floor(1e-9 * data['t_oc'] / consts.WEEKSEC)
        data['time'] = data['time'].dt.tz_localize('UTC')
        # Rename Keplerian orbital parameters to match a GLP standard
        data.rename(columns={'M0': 'M_0', 'Eccentricity': 'e', 'Toe': 't_oe', 'DeltaN': 'deltaN', 'Cuc': 'C_uc', 'Cus': 'C_us',
                             'Cic': 'C_ic', 'Crc': 'C_rc', 'Cis': 'C_is', 'Crs': 'C_rs', 'Io': 'i_0', 'Omega0': 'Omega_0'}, inplace=True)
        data.rename(columns={'X': 'x_sv_m', 'dX': 'dx_sv_mps', 'dX2': 'dx2_sv_mps2',
                             'Y': 'y_sv_m', 'dY': 'dy_sv_mps', 'dY2': 'dy2_sv_mps2',
                             'Z': 'z_sv_m', 'dZ': 'dz_sv_mps', 'dZ2': 'dz2_sv_mps2'}, inplace=True)
        return data, data_header

    def get_iono_params(self, rinex_header, constellations=None):
        """Gets ionosphere parameters from RINEX file header for calculation of
        ionosphere delay.

        There are different possible ways of the header containing
        parameters for ionosphere delay parameters depending on whether
        the file version is v2 or v3. This function uses different keys
        for different Rinex versions and constellations and extracts the
        pertinent parameters into a common format.

        Parameters
        ----------
        rinex_header : dict
            Dictionary containing RINEX file header information
        constellations : list
            List of strings indicating which constellations ionosphere
            correction parameters are required for. Set to `None` by
            default, in which case the function gets all available
            parameters.

        Returns
        -------
        iono_params : dict
            Dictionary of the form ``{gnss_id : iono_array}``, where the
            shape of the array containing the ionospheric corrections.
        """
        iono_params = {}
        # If path ends in .n, then the file contains only GPS satellites
        # and is a Rinex v2 file
        if rinex_header['filetype']=='N' and rinex_header['systems']=='G':
            try:
                ion_alpha_str = rinex_header['ION ALPHA'].replace('D', 'E')
                ion_alpha = np.array(list(map(float, ion_alpha_str.split())))
                ion_beta_str = rinex_header['ION BETA'].replace('D', 'E')
                ion_beta = np.array(list(map(float, ion_beta_str.split())))
            except KeyError:
                ion_alpha = np.array([[np.nan]])
                ion_beta = np.array([[np.nan]])
            gps_iono_params = np.vstack((ion_alpha, ion_beta))
            iono_params['gps'] = gps_iono_params
        # If the path ends in .g, then the file contains GLONASS and no
        # ionospheric parameters. It is also a Rinex v2 file
        if rinex_header['filetype']=='G':
            iono_params = None
        # If the path ends in .rnx, then the file contains multiple
        # constellations, each with their own ionospheric parameters.
        # This type of file is a Rinex v3 file.
        if rinex_header['filetype']=='N' and rinex_header['systems']=='M':
            try:
                iono_corrs = rinex_header['IONOSPHERIC CORR']
                iono_corr_key = self._iono_corr_key()
                # If no constellations have been specified, then use all
                # possible constellations.
                if constellations is None:
                    constellations = list(iono_corr_key.keys())
                # Loop through each constellation and load the parameters
                for constellation in constellations:
                    try:
                        # Load the relevant keys for the corrections
                        const_keys = iono_corr_key[constellation]
                        if len(const_keys) == 2:
                            temp_iono_params = np.empty([len(const_keys), 4])
                            # Loop through the two constellation specific
                            #keys to load the parameters
                            for idx, const_key in enumerate(const_keys):
                                temp_iono_params[idx, :] = iono_corrs[const_key]
                        else:
                            temp_iono_params = np.asarray(iono_corrs[const_keys[0]])
                        iono_params[constellation] = temp_iono_params
                    except KeyError:
                        # if no iono parameters are found for a particular
                        # constellation, skip that constellation
                        continue
            except KeyError:
                iono_params = None
                warnings.warn("No ionospheric parameters found in RINEX file",
                              RuntimeWarning)
        return iono_params

    @staticmethod
    def _iono_corr_key():
        """Correlations between satellite name and iono param name.

        Returns
        -------
        iono_corr_key : list
            String names for ionospheric correction parameters within
            the rinex navigation file.

        """
        iono_corr_key = {}
        iono_corr_key['gps'] = ['GPSA', 'GPSB']
        iono_corr_key['galileo'] = ['GAL']
        iono_corr_key['beidou'] = ['BDSA', 'BDSB']
        iono_corr_key['qzss'] = ['QZSA', 'QZSB']
        iono_corr_key['irnss'] = ['IRNA', 'IRNB']
        return iono_corr_key

    @staticmethod
    def _clock_corr_key():
        clock_corr_key = {}
        clock_corr_key['gps'] = ['GPUT']
        clock_corr_key['glonass'] = ['GLUT']
        clock_corr_key['galileo'] = ['GAUT']
        clock_corr_key['beidou'] = ['BDUT']
        clock_corr_key['qzss'] = ['QZUT']
        clock_corr_key['irnss'] = ['IRUT']
        clock_corr_key['sbas'] = ['SBUT']
        return clock_corr_key

    def load_leapseconds(self, rinex_header):
        """Read leapseconds from Rinex file

        Parameters
        ----------
        rinex_header : dict
            Header information from Rinex file.

        Returns
        -------
        leap_seconds : int
            Leap seconds read from file, return ``np.nan``  if not found.

        """
        if rinex_header['systems']=='M':
            try:
                leap_seconds_line = rinex_header['LEAP SECONDS']
                leap_seconds = int(leap_seconds_line[4]+leap_seconds_line[5])
            except KeyError:
                leap_seconds = np.nan
        else:
            try:
                leap_seconds = int(rinex_header['LEAP SECONDS'].split()[0])
            except KeyError:
                leap_seconds = np.nan
        return leap_seconds


def _split_rinex_nav(rinex_nav):
    """Split the RinexNav instance into GPS like and GLONASS like parts.

    The GPS like part contains parameters for constellations that use
    orbital parameters for SV state estimation. The GLONASS like part
    contains parameters for constellations that use numerical integration
    for SV state estimation.

    Parameters
    ----------
    rinex_nav : gnss_lib_py.parsers.rinex_nav.RinexNav
        RinexNav instance containing ephemeris parameters, with one set
        of parameters per single satellite.

    Returns
    -------
    orbit_param_rinex : gnss_lib_py.parsers.rinex_nav.RinexNav
        RinexNav like instance of NavData containing orbital paramters
        for SV estimation of GPS like satellites.
    num_int_rinex : gnss_lib_py.parsers.rinex_nav.RinexNav
        RinexNav like instance of NavData containing numerical integration
        paramters for SV estimation of GLONASS like satellites.
    """
    orbit_param_rinex = rinex_nav.where('gnss_id', ['glonass', 'sbas'], 'neq')
    num_int_rinex = rinex_nav.where('gnss_id', ['glonass', 'sbas'], 'eq')
    return orbit_param_rinex, num_int_rinex

def rinex_to_sv_states(gps_millis, rinex_nav, rx_pos=None):
    """Estimate SV states for combined rinex data.

    This function splits the rinex data into GPS and GLONASS like, using
    separate SV state estimation methods for each. If a receiver position
    is given, this function will also estimate the time taken for the signal
    to travel from the satellite to the receiver and correct the SV
    estimate to match the state at the time of signal transmission.

    Parameters
    ----------
    gps_millis : float
        The query time at which SV states need to be calculated
    rinex_nav : gnss_lib_py.parsers.rinex_nav.RinexNav
        RinexNav or RinexNav like NavData instance containing ephemeris
        parameters.
    rx_pos : np.ndarray
        Estimated receiver position, used to estimate the time taken for
        the signal to reach the receiver and find the SV state estimate
        at the time of transmission. Default is None, in which case, no
        correction is made.

    Returns
    -------
    sv_states : gnss_lib_py.parsers.navdata.NavData
        NavData containing SV states and the rows `gps_millis`, `gnss_id`,
        `sv_id`, `gnss_sv_id`, `x_sv_m`, `y_sv_m`, `z_sv_m`, `vx_sv_mps`,
        `vy_sv_mps`, and `vz_sv_mps`.
    """
    sv_states = estimate_joint_sv_states(gps_millis, rinex_nav)
    sv_states.sort('gps_millis', inplace=True)
    if rx_pos is not None:
        # Correct for the time taken for the signal to travel to the
        # receiver and estimate the satellite positions at the time of
        # signal transmission
        rx_ecef = np.reshape(rx_ecef, [3, 1])
        _, true_range = _find_delxyz_range(sv_states, rx_ecef)
        t_corr = true_range/consts.C
        sv_states = estimate_joint_sv_states(gps_millis-1000.*t_corr, rinex_nav)
        sv_states.sort('gps_millis', inplace=True)

    return sv_states

def estimate_joint_sv_states(gps_millis, rinex_nav):
    """Given a time and rinex_nav, estimate SV states for all satellites.

    Parameters
    ----------
    gps_millis : float
        The query time at which SV states need to be calculated
    rinex_nav : gnss_lib_py.parsers.rinex_nav.RinexNav
        RinexNav or RinexNav like NavData instance containing ephemeris
        parameters.

    Returns
    -------
    sv_states : gnss_lib_py.parsers.navdata.NavData
        NavData containing SV states and the rows `gps_millis`, `gnss_id`,
        `sv_id`, `gnss_sv_id`, `x_sv_m`, `y_sv_m`, `z_sv_m`, `vx_sv_mps`,
        `vy_sv_mps`, and `vz_sv_mps`.
    """
    orbit_param_rinex, num_int_rinex = _split_rinex_nav(rinex_nav)
    sv_states = NavData()
    if len(orbit_param_rinex) != 0:
        # Estimate GPS and Galileo like states (with GPS time frame of reference)
        gps_orbit_param_rinex = orbit_param_rinex.where('gnss_id', 'beidou', 'neq')
        if len(gps_orbit_param_rinex)!=0:
            gps_like_sv_states = orbit_params_to_sv_states(gps_millis, gps_orbit_param_rinex)
            sv_states.concat(gps_like_sv_states, inplace=True)
        beidou_orbit_param_rinex = orbit_param_rinex.where('gnss_id', 'beidou', 'eq')
        if len(beidou_orbit_param_rinex)!=0:
            beidou_query_time = gps_millis - consts.GPS_BEIDOU_OFFSET_MILLIS
            beidou_like_sv_states = orbit_params_to_sv_states(beidou_query_time, beidou_orbit_param_rinex)
            beidou_like_sv_states['gps_millis'] = beidou_like_sv_states['gps_millis'] + consts.GPS_BEIDOU_OFFSET_MILLIS
            sv_states.concat(beidou_like_sv_states, inplace=True)
        # Estimate states for systems in the Beidou time frame of reference
    if len(num_int_rinex) != 0:
        num_int_sv_states = num_int_for_sv_states(gps_millis, num_int_rinex)
        sv_states.concat(num_int_sv_states, inplace=True)
    if len(sv_states)==0:
        warnings.warn('No SV states were estimated, check rinex_nav', RuntimeWarning)
    return sv_states


def orbit_params_to_sv_states(gps_millis, ephem_orbit_params):
    """Compute position and velocities for all satellites in ephemeris file
    given time of clock.

    `ephem` contains broadcast ephemeris parameters (similar in form to GPS
    broadcast parameters).
    Galileo week is the same as  GPS week but Beidou week is different,
    so this function accounts for the difference in the Beidou and GPS
    weeks.

    Must contain the following rows (description in [1]_):
    * :code:`gnss_id`
    * :code:`sv_id`
    * :code:`gps_week`/`beidou_week`
    * :code:`t_oe`
    * :code:`e`
    * :code:`omega`
    * :code:`Omega_0`
    * :code:`OmegaDot`
    * :code:`sqrtA`
    * :code:`deltaN`
    * :code:`IDOT`
    * :code:`i_0`
    * :code:`C_is`
    * :code:`C_ic`
    * :code:`C_rs`
    * :code:`C_rc`
    * :code:`C_uc`
    * :code:`C_us`

    Parameters
    ----------
    gps_millis : int
        Time at which measurements are needed, measured in milliseconds
        since start of GPS epoch [ms].
    ephem : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing ephemeris parameters of satellites
        for which states are required.

    Returns
    -------
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        NavData containing satellite positions, velocities, corresponding
        time with GNSS ID and SV number.

    Notes
    -----
    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    More details on the algorithm used to compute satellite positions
    from broadcast navigation message can be found in [3]_.

    Satellite velocity calculations based on algorithms introduced in [2]_.

    References
    ----------
    ..  [2] Misra, P. and Enge, P,
        "Global Positioning System: Signals, Measurements, and Performance."
        2nd Edition, Ganga-Jamuna Press, 2006.
    ..  [3] B. F. Thompson, S. W. Lewis, S. A. Brown, and T. M. Scott,
        “Computing GPS satellite velocity and acceleration from the broadcast
        navigation message,” NAVIGATION, vol. 66, no. 4, pp. 769–779, Dec. 2019,
        doi: 10.1002/navi.342.

    """

    # Convert time from GPS millis to TOW
    gps_week, gps_tow = gps_millis_to_tow(gps_millis)
    # Extract parameters

    c_is = ephem_orbit_params['C_is']
    c_ic = ephem_orbit_params['C_ic']
    c_rs = ephem_orbit_params['C_rs']
    c_rc = ephem_orbit_params['C_rc']
    c_uc = ephem_orbit_params['C_uc']
    c_us = ephem_orbit_params['C_us']
    delta_n   = ephem_orbit_params['deltaN']

    ecc        = ephem_orbit_params['e']     # eccentricity
    omega    = ephem_orbit_params['omega'] # argument of perigee
    omega_0  = ephem_orbit_params['Omega_0']
    sqrt_sma = ephem_orbit_params['sqrtA'] # sqrt of semi-major axis
    sma      = sqrt_sma**2      # semi-major axis

    sqrt_mu_a = np.sqrt(consts.MU_EARTH) * sqrt_sma**-3 # mean angular motion
    if 'gps_week' in ephem_orbit_params.rows:
        gpsweek_diff = (np.mod(gps_week,1024) - np.mod(ephem_orbit_params['gps_week'],1024))*consts.WEEKSEC
    elif 'beidou_week' in ephem_orbit_params.rows:
        beidou_week = gps_week - consts.GPS_BEIDOU_WEEK_OFFSET
        gpsweek_diff = (beidou_week - ephem_orbit_params['beidou_week'])*consts.WEEKSEC
    sv_posvel = NavData()
    sv_posvel['gnss_id'] = ephem_orbit_params['gnss_id']
    sv_posvel['sv_id'] = ephem_orbit_params['sv_id']
    sv_posvel['gps_millis'] = gps_millis

    delta_t = gps_tow - ephem_orbit_params['t_oe'] + gpsweek_diff

    # Calculate the mean anomaly with corrections
    ecc_anom = _compute_eccentric_anomaly(gps_week, gps_tow,
                                            ephem_orbit_params)

    cos_e   = np.cos(ecc_anom)
    sin_e   = np.sin(ecc_anom)
    e_cos_e = (1 - ecc*cos_e)

    # Calculate the true anomaly from the eccentric anomaly
    sin_nu = np.sqrt(1 - ecc**2) * (sin_e/e_cos_e)
    cos_nu = (cos_e-ecc) / e_cos_e
    nu_rad = np.arctan2(sin_nu, cos_nu)

    # Calcualte the argument of latitude iteratively
    phi_0 = nu_rad + omega
    phi   = phi_0
    for incl in range(5):
        cos_to_phi = np.cos(2.*phi)
        sin_to_phi = np.sin(2.*phi)
        phi_corr = c_uc * cos_to_phi + c_us * sin_to_phi
        phi = phi_0 + phi_corr

    # Calculate the longitude of ascending node with correction
    omega_corr = ephem_orbit_params['OmegaDot'] * delta_t

    # Also correct for the rotation since the beginning of the GPS week for which the Omega0 is
    # defined.  Correct for GPS week rollovers.

    # Also correct for the rotation since the beginning of the GPS week for
    # which the Omega0 is defined.  Correct for GPS week rollovers.
    omega = omega_0 - (consts.OMEGA_E_DOT*(gps_tow + gpsweek_diff)) + omega_corr

    # Calculate orbital radius with correction
    r_corr = c_rc * cos_to_phi + c_rs * sin_to_phi
    orb_radius      = sma*e_cos_e + r_corr

    ############################################
    ######  Lines added for velocity (1)  ######
    ############################################
    delta_e   = (sqrt_mu_a + delta_n) / e_cos_e
    dphi = np.sqrt(1 - ecc**2)*delta_e / e_cos_e
    # Changed from the paper
    delta_r   = (sma * ecc * delta_e * sin_e) + 2*(c_rs*cos_to_phi - c_rc*sin_to_phi)*dphi

    # Calculate the inclination with correction
    i_corr = c_ic*cos_to_phi + c_is*sin_to_phi + ephem_orbit_params['IDOT']*delta_t
    incl = ephem_orbit_params['i_0'] + i_corr

    ############################################
    ######  Lines added for velocity (2)  ######
    ############################################
    delta_i = 2*(c_is*cos_to_phi - c_ic*sin_to_phi)*dphi + ephem_orbit_params['IDOT']

    # Find the position in the orbital plane
    x_plane = orb_radius*np.cos(phi)
    y_plane = orb_radius*np.sin(phi)

    ############################################
    ######  Lines added for velocity (3)  ######
    ############################################
    delta_u = (1 + 2*(c_us * cos_to_phi - c_uc*sin_to_phi))*dphi
    dxp = delta_r*np.cos(phi) - orb_radius*np.sin(phi)*delta_u
    dyp = delta_r*np.sin(phi) + orb_radius*np.cos(phi)*delta_u
    # Find satellite position in ECEF coordinates
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    cos_i = np.cos(incl)
    sin_i = np.sin(incl)

    sv_posvel['x_sv_m'] = x_plane*cos_omega - y_plane*cos_i*sin_omega
    sv_posvel['y_sv_m'] = x_plane*sin_omega + y_plane*cos_i*cos_omega
    sv_posvel['z_sv_m'] = y_plane*sin_i

    ############################################
    ######  Lines added for velocity (4)  ######
    ############################################
    omega_dot = ephem_orbit_params['OmegaDot'] - consts.OMEGA_E_DOT
    sv_posvel['vx_sv_mps'] = (dxp * cos_omega
                        - dyp * cos_i*sin_omega
                        + y_plane  * sin_omega*sin_i*delta_i
                        - (x_plane * sin_omega + y_plane*cos_i*cos_omega)*omega_dot)

    sv_posvel['vy_sv_mps'] = (dxp * sin_omega
                        + dyp * cos_i * cos_omega
                        - y_plane  * sin_i * cos_omega * delta_i
                        + (x_plane * cos_omega - (y_plane*cos_i*sin_omega)) * omega_dot)

    sv_posvel['vz_sv_mps'] = dyp*sin_i + y_plane*cos_i*delta_i

    # Estimate SV clock corrections, including polynomial and relativistic
    # clock corrections
    clock_corr, _, _ = _estimate_sv_clock_corr(gps_millis,
                                                ephem_orbit_params)

    sv_posvel['b_sv_m'] = clock_corr

    return sv_posvel


def _find_delxyz_range(sv_posvel, rx_ecef):
    """Return difference of satellite and rx_pos positions and distance between them.

    Parameters
    ----------
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Satellite position and velocities.
    rx_ecef : np.ndarray
        3x1 Receiver 3D ECEF position [m].

    Returns
    -------
    del_pos : np.ndarray
        Difference between satellite positions and receiver position.
    true_range : np.ndarray
        Distance between satellite and receiver positions.
    """
    rx_ecef = np.reshape(rx_ecef, [3, 1])
    satellites = len(sv_posvel)
    sv_pos, _ = _extract_pos_vel_arr(sv_posvel)
    sv_pos = sv_pos.reshape(rx_ecef.shape[0], satellites)
    del_pos = sv_pos - np.tile(rx_ecef, (1, satellites))
    true_range = np.linalg.norm(del_pos, axis=0)
    return del_pos, true_range


def _extract_pos_vel_arr(sv_posvel):
    """Extract satellite positions and velocities into numpy arrays.

    Parameters
    ----------
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        NavData containing satellite position and velocity states.

    Returns
    -------
    sv_pos : np.ndarray
        ECEF satellite x, y and z positions 3xN [m].
    sv_vel : np.ndarray
        ECEF satellite x, y and z velocities 3xN [m].
    """
    sv_pos = sv_posvel[['x_sv_m', 'y_sv_m', 'z_sv_m']]
    sv_vel   = sv_posvel[['vx_sv_mps', 'vy_sv_mps', 'vz_sv_mps']]
    return sv_pos, sv_vel


def _compute_eccentric_anomaly(gps_week, gps_tow, ephem, tol=1e-5, max_iter=10):
    """Compute the eccentric anomaly from ephemeris parameters.

    This function extracts relevant parameters from the broadcast navigation
    ephemerides and then solves the equation `f(E) = M - E + e * sin(E) = 0`
    using the Newton-Raphson method.

    In the above equation `M` is the corrected mean anomaly, `e` is the
    orbit eccentricity and `E` is the eccentric anomaly, which is unknown.

    Parameters
    ----------
    gps_week : int
        Week of GPS calendar corresponding to time of clock.
    gps_tow : np.ndarray
        GPS time of the week at which positions are required [s].
    ephem : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing ephemeris parameters of satellites
        for which states are required.
    tol : float
        Tolerance for convergence of the Newton-Raphson.
    max_iter : int
        Maximum number of iterations for Newton-Raphson.

    Returns
    -------
    ecc_anom : np.ndarray
        Eccentric Anomaly of GNSS satellite orbits.

    """
    #Extract required parameters from ephemeris and GPS constants
    delta_n   = ephem['deltaN']
    mean_anom_0  = ephem['M_0']
    sqrt_sma = ephem['sqrtA'] # sqrt of semi-major axis
    sqrt_mu_a = np.sqrt(consts.MU_EARTH) * sqrt_sma**-3 # mean angular motion
    ecc        = ephem['e']     # eccentricity
    #Times for computing positions
    if 'gps_week' in ephem.rows:
        gpsweek_diff = (np.mod(gps_week,1024) - np.mod(ephem['gps_week'],1024))*consts.WEEKSEC
    elif 'beidou_week' in ephem.rows:
        beidou_week = gps_week - consts.GPS_BEIDOU_WEEK_OFFSET
        gpsweek_diff = (beidou_week - ephem['beidou_week'])*consts.WEEKSEC
    delta_t = gps_tow - ephem['t_oe'] + gpsweek_diff

    # Calculate the mean anomaly with corrections
    mean_anom_corr = delta_n * delta_t
    mean_anom = mean_anom_0 + (sqrt_mu_a * delta_t) + mean_anom_corr

    # Compute Eccentric Anomaly
    ecc_anom = mean_anom
    for _ in np.arange(0, max_iter):
        fun = mean_anom - ecc_anom + ecc * np.sin(ecc_anom)
        df_decc_anom = ecc*np.cos(ecc_anom) - 1.
        delta_ecc_anom   = -fun / df_decc_anom
        ecc_anom    = ecc_anom + delta_ecc_anom

    if np.any(delta_ecc_anom > tol): #pragma: no cover
        raise RuntimeWarning("Eccentric Anomaly may not have converged" \
                            + f"after {max_iter} steps. : dE = {delta_ecc_anom}")

    return ecc_anom


def _find_delxyz_range(sv_posvel, rx_ecef):
    """Return difference of satellite and rx_pos positions and distance between them.

    Parameters
    ----------
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Satellite position and velocities.
    rx_ecef : np.ndarray
        3x1 Receiver 3D ECEF position [m].

    Returns
    -------
    del_pos : np.ndarray
        Difference between satellite positions and receiver position.
    true_range : np.ndarray
        Distance between satellite and receiver positions.
    """
    rx_ecef = np.reshape(rx_ecef, [3, 1])
    satellites = len(sv_posvel)
    sv_pos, _ = _extract_pos_vel_arr(sv_posvel)
    sv_pos = sv_pos.reshape(rx_ecef.shape[0], satellites)
    del_pos = sv_pos - np.tile(rx_ecef, (1, satellites))
    true_range = np.linalg.norm(del_pos, axis=0)
    return del_pos, true_range


def _estimate_sv_clock_corr(gps_millis, ephem):
    """Calculate the modelled satellite clock delay

    Assumes that clock delays are calculated for E1 signals for Galileo
    satellites. The group delays are calculated according to the Galileo
    ICD[3]_.

    Parameters
    ---------
    gps_millis : int
        Time at which measurements are needed, measured in milliseconds
        since start of GPS epoch [ms].
    ephem : gnss_lib_py.parsers.navdata.NavData
        Satellite ephemeris parameters for measurement SVs.

    Returns
    -------
    clock_corr : np.ndarray
        Satellite clock corrections containing all terms [m].
    corr_polynomial : np.ndarray
        Polynomial clock perturbation terms [m].
    clock_relativistic : np.ndarray
        Relativistic clock correction terms [m].

    References
    ----------
    ..  [3] Galileo Open Service Signal In Space Interface Control Document,
        Issue 2.0, January 2021, Retreived on 24 August, 2023

    """
    # Extract required GPS constants
    ecc        = ephem['e']     # eccentricity
    sqrt_sma = ephem['sqrtA'] # sqrt of semi-major axis

    # if np.abs(delta_t).any() > 302400:
    #     delta_t = delta_t - np.sign(delta_t)*604800

    gps_week, gps_tow = gps_millis_to_tow(gps_millis)

    # Compute Eccentric Anomaly
    ecc_anom = _compute_eccentric_anomaly(gps_week, gps_tow, ephem)

    # Determine pseudorange corrections due to satellite clock corrections.
    # Calculate time offset from satellite reference time
    t_offset = gps_tow - ephem['t_oc']
    if np.abs(t_offset).any() > 302400:  # pragma: no cover
        t_offset = t_offset-np.sign(t_offset)*604800

    # Calculate clock corrections from the polynomial corrections in
    # broadcast message
    corr_polynomial = (ephem['SVclockBias']
                     + ephem['SVclockDrift']*t_offset
                     + ephem['SVclockDriftRate']*t_offset**2)

    # Calcualte the relativistic clock correction
    corr_relativistic = consts.F * ecc * sqrt_sma * np.sin(ecc_anom)

    # Calculate the total clock correction including the Tgd term
    if 'TGD' in ephem.rows:
        clk_corr = (corr_polynomial - ephem['TGD'] + corr_relativistic)
    elif 'BGDe5b' in ephem.rows:
        # Assuming single frequency measurements are used and that
        # E1 or E5b frequencies are used for state estimation
        clk_corr = (corr_polynomial - ephem['BGDe5b'] + corr_relativistic)
    elif 'TGD1' in ephem.rows:
        clk_corr = (corr_polynomial - ephem['TGD1'] + corr_relativistic)
    #Convert values to equivalent meters from seconds
    clk_corr = np.array(consts.C*clk_corr, ndmin=1)
    corr_polynomial = np.array(consts.C*corr_polynomial, ndmin=1)
    corr_relativistic = np.array(consts.C*corr_relativistic, ndmin=1)

    return clk_corr, corr_polynomial, corr_relativistic


def num_int_for_sv_states(gps_millis, num_int_rinex):
    rx_gnss_sv_id = np.unique(num_int_rinex['gnss_sv_id'])

    sv_posvel = NavData()
    for gnss_sv_id in rx_gnss_sv_id:
        num_int_rinex_sv = num_int_rinex.where('gnss_sv_id', gnss_sv_id, 'eq')
        times, est_pos, est_vel = num_int_single_sv(gps_millis, num_int_rinex_sv)
        sv_posvel_single = NavData()
        sv_posvel_single['gps_millis'] = gps_millis
        sv_posvel_single['x_sv_m'] = est_pos[0, :]
        sv_posvel_single['y_sv_m'] = est_pos[1, :]
        sv_posvel_single['z_sv_m'] = est_pos[2, :]
        sv_posvel_single['vx_sv_mps'] = est_vel[0, :]
        sv_posvel_single['vy_sv_mps'] = est_vel[1, :]
        sv_posvel_single['vz_sv_mps'] = est_vel[2, :]
        sv_posvel_single['b_sv_m'] = consts.C*num_int_rinex_sv['SVclockBias']
        #TODO: Figure out how to fill this for GLONASS
        sv_posvel_single['gnss_id'] = np.asarray([gnss_sv_id]*len(times), dtype=object)
        gnss_id, sv_id = split_gnss_sv_ids(gnss_sv_id=[gnss_sv_id])
        sv_posvel_single['gnss_id'] = np.asarray([gnss_id]*len(times), dtype=object)
        sv_posvel_single['sv_id'] = sv_id
        sv_posvel.concat(sv_posvel_single, inplace=True)
    sv_posvel.sort('gps_millis', inplace=True)
    return sv_posvel


def num_int_single_sv(gps_millis, num_int_rinex_sv):
    """
    Notes
    -----
    Algorithm used is from
    https://gssc.esa.int/navipedia/index.php/GLONASS_Satellite_Coordinates_Computation
    """
    #TODO: Figure out if this function should be used for a single SV
    #TODO: Vectorize this for multiple call times in gps_millis
    #TODO: See if this is correct for SBAS as well or if things need to
    # be changed
    # TODO: Strip the timezones from the times below and vectorize them
    # print('before conversion', gps_millis)
    # print(f'type before conversion', type(gps_millis))
    # print('Using GLONASS constants') # Tested with glonass constants, nothing changed
    if not isinstance(gps_millis, (np.ndarray, list, tuple)):
        gps_millis = np.atleast_1d(gps_millis)
    if np.size(gps_millis)==1:
        gps_millis = np.atleast_1d(gps_millis)
    # print('after conversion', gps_millis)
    # print(f'type gps_millis', type(gps_millis))
    # print(f'shape after conversion: {gps_millis.shape}')
    t_k = np.asarray([np.datetime64(gps_millis_to_datetime(gps_milli)) for gps_milli in gps_millis],
                     dtype=np.datetime64)
    t_e_timestamp =gps_millis_to_datetime(num_int_rinex_sv['gps_millis'])
    # remove the timezone information because numpy.datetime is not timezone aware
    t_e_timestamp = np.datetime64(t_e_timestamp.replace(tzinfo=None))
    TauN = -1*num_int_rinex_sv['SVclockBias']
    pos_pz90 = np.empty([3, 1])
    vel_pz90 = np.empty([3, 1])
    acc_pz90 = np.empty([3, 1])

    # Assign the positions, velocities, and accelerations from rinex file
    pos_pz90[0, 0] = num_int_rinex_sv['x_sv_m']
    vel_pz90[0, 0] = num_int_rinex_sv['dx_sv_mps']
    acc_pz90[0, 0] = num_int_rinex_sv['dx2_sv_mps2']
    pos_pz90[1, 0] = num_int_rinex_sv['y_sv_m']
    vel_pz90[1, 0] = num_int_rinex_sv['dy_sv_mps']
    acc_pz90[1, 0] = num_int_rinex_sv['dy2_sv_mps2']
    pos_pz90[2, 0] = num_int_rinex_sv['z_sv_m']
    vel_pz90[2, 0] = num_int_rinex_sv['dz_sv_mps']
    acc_pz90[2, 0] = num_int_rinex_sv['dz2_sv_mps2']
    delta = np.timedelta64(int(TauN*1e9), 'ns')

    #NOTE: The system level time corrections are not used because they
    # observed to be on the nanosecond level, If available, use the line
    # below to account for those corrections
    # delta = np.timedelta64(int((TauN + TauC)*1e9), 'ns')
    # # Offset described in RINEX documentation

    t_k_timestamps = t_k + delta
    theta_G0 = _find_gmst_at_midnight(t_e_timestamp)
    # print(f't_k_timestamp: {t_k_timestamps}')
    start_of_day = np.asarray([np.datetime64(t_k_case, 'D') for t_k_case in t_k_timestamps])
    t_k = (t_k_timestamps - start_of_day) / np.timedelta64(1, 's')
    t_e = (t_e_timestamp - start_of_day) / np.timedelta64(1, 's')

    theta_Ge = theta_G0 + consts.OMEGA_E_DOT * t_e

    pos_iner, vel_iner, acc_iner = pz90_to_inertial(pos_pz90, vel_pz90, acc_pz90, theta_Ge)
    s0 = np.reshape(np.vstack([pos_iner, vel_iner]), 6)
    if t_e <= np.min(t_k):
        # Span and evaluation times forward
        t_span_fwd = (t_e[0], np.max(t_k))
        t_eval_fwd = t_k
        # Span and evaluation times backward
        t_span_back = None
        t_eval_back = None

    elif t_e > np.max(t_k):
        # Span and evaluation times forward
        t_span_fwd = (t_e[0], np.min(t_k))
        t_eval_fwd = t_k
        # Span and evaluation times backward
        t_span_back = None
        t_eval_back = None
    else:
        # Span and evaluation times forward
        t_span_fwd = (t_e[0], np.max(t_k))
        t_eval_fwd = np.flatnonzero(t_k >= t_e[0])
        # Span and evaluation times backward
        t_span_back = (t_e[0], np.min(t_k))
        t_eval_back = np.flatnonzero(t_k < t_e[0])

    #TODO: See if the t_eval has to be removed
    # num_t = 10
    # t_eval = np.linspace(t_span_fwd[0], t_span_fwd[1], num_t)
    # print(f't_eval:{t_eval}')
    # print(f't_span:{t_span}')
    # print(f't_e:{t_e}')
    # print(f't_k:{t_k}')
    # print('initial condition', s0)

    sol_fwd = solve_ivp(_glonass_sv_dynamics,
                        t_span_fwd,
                        s0,
                        method='RK45',
                        t_eval=t_eval_fwd,
                        args=(acc_iner[0,0], acc_iner[1, 0], acc_iner[2,0]),
                        atol=1e-12)
    if t_span_back is not None:
        sol_back = solve_ivp(_glonass_sv_dynamics,
                             t_span_fwd,
                             s0,
                             method='RK45',
                             t_eval=t_eval_back,
                             args=(acc_iner[0,0], acc_iner[1, 0], acc_iner[2,0]),
                             atol=1e-12)
    t_out = sol_fwd.t
    y_out = sol_fwd.y
    if t_span_back is not None:
        t_out = np.concatenate((np.flip(sol_back.t), t_out))
        y_out = np.concatenate((np.flip(sol_back.y, axis=1), y_out))
    x_iner, y_iner, z_iner = y_out[0, :], y_out[1, :], y_out[2, :]
    pos_iner = np.vstack((x_iner, y_iner, z_iner))
    vx_iner, vy_iner, vz_iner = y_out[3, :], y_out[4, :], y_out[5, :]
    vel_iner = np.vstack((vx_iner, vy_iner, vz_iner))
    #TODO: Create a conversion back to PZ90 for the velocities and use
    # it to report the velocities
    theta_G_arr = theta_G0 + consts.OMEGA_E_DOT* t_out
    est_pos_pz90, est_vel_pz90 = inertial_to_pz90(pos_iner, vel_iner, theta_G_arr)
    # print('times')
    # for t in t_out:
    #     print(t)
    #     print(start_of_day + np.timedelta64(int(np.round(t, decimals=0)), 's'))
    # print('pos_out_x', est_pos_pz90[0, :])
    # print('pos_out_y', est_pos_pz90[1, :])
    # print('pos_out_z', est_pos_pz90[2, :])
    return t_out, est_pos_pz90, est_vel_pz90


def _find_gmst_at_midnight(timestamps):
    #TODO: Add a license statement here
    # https://github.com/jhaupt/Sidereal-Time-Calculator/blob/master/SiderealTimeCalculator.py
    # be a result of a constant (ish) offset in the GMST calculation
    # Testing and compared against the midnight
    #Code taken from:
    #Sidereal Time and Julian Date Calculator
    #Revision history: Justine Haupt, v1.0 (11/23/17)
    #calculate the Greenwhich mean sidereal time:
    #split TD into individual variables for month, day, etc. and convert to floats:
    if isinstance(timestamps, np.datetime64):
        timestamps = [timestamps]
    months = np.asarray([dt_item.astype('datetime64[M]').item().month for dt_item in timestamps])
    days = np.asarray([dt_item.astype('datetime64[D]').item().day for dt_item in timestamps])
    years = np.asarray([dt_item.astype('datetime64[Y]').item().year for dt_item in timestamps])
    #calculate the Julian date:
    julian_date = (367*years) - ((7*(years+((months+9)/12).astype(int)))/4).astype(int) \
                + ((275*months)/9).astype(int) \
                    + days + 1721013.5
    gmst = 18.697374558 + 24.06570982441908*(julian_date - 2451545)
    gmst = gmst % 24    #use modulo operator to convert to 24 hours
    gmst_rad = gmst / 24 * 2*np.pi # theta_G0 in radians
    return gmst_rad


def _glonass_sv_dynamics(t, s, acc_x_iner, acc_y_iner, acc_z_iner): # pragma: no cover
    """
    """
    #TODO: Test this function
    # Extract the values from the input initial conditions
    x_iner = s[0]
    y_iner = s[1]
    z_iner = s[2]
    vx_iner = s[3]
    vy_iner = s[4]
    vz_iner = s[5]

    rad = np.sqrt(x_iner**2 + y_iner**2 + z_iner**2)
    mu_bar = consts.MU_EARTH / (rad*rad)
    xa_bar = x_iner/rad
    ya_bar = y_iner/rad
    za_bar = z_iner/rad
    rho_bar = consts.A/rad

    xa_dot = vx_iner
    ya_dot = vy_iner
    za_dot = vz_iner
    vxa_dot = -mu_bar * xa_bar \
            + 3/2*consts.C20*mu_bar*xa_bar*rho_bar**2 * (1 - 5*za_bar**2) \
            + acc_x_iner
    vya_dot = -mu_bar * ya_bar \
            + 3/2*consts.C20*mu_bar*ya_bar*rho_bar**2 * (1 - 5*za_bar**2) \
            + acc_y_iner
    vza_dot = -mu_bar * za_bar \
            + 3/2*consts.C20*mu_bar*za_bar*rho_bar**2 * (3 - 5*za_bar**2) \
            + acc_z_iner
    s_dot = np.array([xa_dot, ya_dot, za_dot, vxa_dot, vya_dot, vza_dot])
    return s_dot

