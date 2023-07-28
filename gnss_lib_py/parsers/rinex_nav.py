"""Parses Rinex .n files.
#TODO: Remove this description and move to the tutorials and description
#in the documentation
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


__authors__ = "Ashwin Kanhere, Shubh Gupta"
__date__ = "13 July 2021"

import os
from datetime import datetime, timezone
import warnings

import numpy as np
import pandas as pd
import georinex as gr

import gnss_lib_py.utils.constants as consts
from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis, gps_millis_to_tow
from gnss_lib_py.utils.time_conversions import gps_millis_to_datetime
from gnss_lib_py.utils.ephemeris_downloader import load_ephemeris, DEFAULT_EPHEM_PATH


class RinexNav(NavData):
    """Class to handle Rinex files containing SV parameters.

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

        constellations = set()
        if satellites is not None and len(satellites) != 0:
            for sat in satellites:
                constellations.add(sat[0])

        if isinstance(rinex_paths, (str, os.PathLike)):
            rinex_paths = [rinex_paths]

        data = pd.DataFrame()
        self.iono_params = {}
        #TODO: Convert this to a dictionary
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
            if start_gps_millis not in self.iono_params:
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

        constellations : Set
            Set of satellites {"ConstIDSVID"}

        Returns
        -------
        data : pd.DataFrame
            Parsed ephemeris DataFrame
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
        if leap_seconds is None:
            data['leap_seconds'] = np.nan
        else:
            data['leap_seconds'] = leap_seconds
        data.dropna(how='all', inplace=True)
        data.reset_index(inplace=True)
        data['source'] = rinex_path
        data['t_oc'] = pd.to_numeric(data['time'] - consts.GPS_EPOCH_0.replace(tzinfo=None))
        data['t_oc']  = 1e-9 * data['t_oc'] - consts.WEEKSEC * np.floor(1e-9 * data['t_oc'] / consts.WEEKSEC)
        data['time'] = data['time'].dt.tz_localize('UTC')
        # Rename Keplerian orbital parameters to match a GLP standard
        # TODO: Change these to a standard nomenclature
        data.rename(columns={'M0': 'M_0', 'Eccentricity': 'e', 'Toe': 't_oe', 'DeltaN': 'deltaN', 'Cuc': 'C_uc', 'Cus': 'C_us',
                             'Cic': 'C_ic', 'Crc': 'C_rc', 'Cis': 'C_is', 'Crs': 'C_rs', 'Io': 'i_0', 'Omega0': 'Omega_0'}, inplace=True)
        data.rename(columns={'X': 'sv_x_m', 'dX': 'sv_dx_mps', 'dX2': 'sv_dx2_mps2',
                             'Y': 'sv_y_m', 'dY': 'sv_dy_mps', 'dY2': 'sv_dy2_mps2',
                             'Z': 'sv_z_m', 'dZ': 'sv_dz_mps', 'dZ2': 'sv_dz2_mps2'}, )
        return data, data_header

    def get_iono_params(self, rinex_header, constellations=None):
        """Gets ionosphere parameters from RINEX file header for calculation of
        ionosphere delay.

        There are different possible ways of the header containing
        parameters for ionosphere delay parameters. This function tries
        different keys to extract the pertinent parameters.

        Parameters
        ----------
        rinex_head : dict
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
        # If the path ends in .g, then the file constains GLONASS and no
        # ionospheric parameters
        if rinex_header['filetype']=='G':
            iono_params = None
        # If the path ends in .rnx, then the file contains multiple
        # constellations, each with their own ionospheric parameters
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
        iono_corr_key = {}
        iono_corr_key['gps'] = ['GPSA', 'GPSB']
        iono_corr_key['galileo'] = ['GAL']
        iono_corr_key['beidou'] = ['BDSA', 'BDSB']
        iono_corr_key['qzss'] = ['QZSA', 'QZSB']
        iono_corr_key['irnss'] = ['IRNA', 'IRNB']
        return iono_corr_key

    def load_leapseconds(self, rinex_header):
        """Read leapseconds from Rinex file

        Parameters
        ----------
        rinex_header : dict
            Header information from Rinex file

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
                leap_seconds = int(rinex_header['LEAP SECONDS'])
            except KeyError:
                leap_seconds = np.nan
        return leap_seconds

def rinex_to_sv_states(gps_millis, rinex_nav):
    # keplerian_rinex =
    # num_int_rinex =
    # keplerian_sv_states =
    # num_int_sv_states =
    # sv_states = keplerian_sv_states.concat(num_int_sv_states)
    # sv_states.sort('gps_millis', inplace=True)
    # rotate the states
    # return sv_states
    pass

def orbit_params_to_sv_states(gps_millis, ephem_orbit_params):
    """Compute position and velocities for all satellites in ephemeris file
    given time of clock.

    `ephem` contains broadcast ephemeris parameters (similar in form to GPS
    broadcast parameters).

    Must contain the following rows (description in [1]_):
    * :code:`gnss_id`
    * :code:`sv_id`
    * :code:`gps_week`
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
    from broadcast navigation message can be found in [1]_.

    Satellite velocity calculations based on algorithms introduced in [2]_.

    References
    ----------
    ..  [1] Misra, P. and Enge, P,
        "Global Positioning System: Signals, Measurements, and Performance."
        2nd Edition, Ganga-Jamuna Press, 2006.
    ..  [2] B. F. Thompson, S. W. Lewis, S. A. Brown, and T. M. Scott,
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
    gpsweek_diff = (np.mod(gps_week,1024) - np.mod(ephem_orbit_params['gps_week'],1024))*604800.
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
    nu_rad     = np.arctan2(sin_nu, cos_nu)

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
    gpsweek_diff = (np.mod(gps_week,1024) - np.mod(ephem['gps_week'],1024))*604800.
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


def _estimate_sv_clock_corr(gps_millis, ephem):
    """Calculate the modelled satellite clock delay

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
    clk_corr = (corr_polynomial - ephem['TGD'] + corr_relativistic)

    #Convert values to equivalent meters from seconds
    clk_corr = np.array(consts.C*clk_corr, ndmin=1)
    corr_polynomial = np.array(consts.C*corr_polynomial, ndmin=1)
    corr_relativistic = np.array(consts.C*corr_relativistic, ndmin=1)

    return clk_corr, corr_polynomial, corr_relativistic

def numerical_integration_for_sv_states(self, gps_millis):
    pass


def get_time_cropped_rinex(gps_millis, satellites=None,
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
    gps_millis : float
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

    constellations = set()
    for sat in satellites:
        constellations.add(consts.CONSTELLATION_CHARS[sat[0]])
    constellations = list(constellations)

    rinex_paths = load_ephemeris("rinex_nav",gps_millis,constellations,
                                 download_directory=ephemeris_directory,
                                 )
    rinex_data = RinexNav(rinex_paths, satellites=satellites)

    time_cropped_data = rinex_data.where('gps_millis', gps_millis, "lesser")

    time_cropped_data = time_cropped_data.pandas_df().sort_values(
        'gps_millis').groupby('gnss_sv_id').last()

    rinex_data_df = time_cropped_data
    rinex_iono_params = rinex_data.iono_params

    rinex_data_df = rinex_data_df.reset_index()
    rinex_data = NavData(pandas_df=rinex_data_df)
    rinex_data.iono_params = rinex_iono_params

    return rinex_data
