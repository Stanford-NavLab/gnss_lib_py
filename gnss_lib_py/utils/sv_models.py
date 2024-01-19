"""Model GNSS SV states (positions and velocities).

Functions to calculate GNSS SV positions and velocities for a given time.
"""

__authors__ = "Ashwin Kanhere, Bradley Collicott"
__date__ = "17 Jan, 2023"

import warnings

import numpy as np

from gnss_lib_py.parsers.sp3 import Sp3
from gnss_lib_py.parsers.clk import Clk
from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.parsers.rinex_nav import get_time_cropped_rinex, RinexNav
from gnss_lib_py.parsers.rinex_nav import _compute_eccentric_anomaly
from gnss_lib_py.parsers.rinex_nav import _estimate_sv_clock_corr
import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.coordinates import ecef_to_el_az
from gnss_lib_py.utils.time_conversions import gps_millis_to_tow
from gnss_lib_py.utils.ephemeris_downloader import DEFAULT_EPHEM_PATH, load_ephemeris
from gnss_lib_py.navdata.operations import loop_time, sort, concat, find_wildcard_indexes

def add_sv_states(navdata, source = 'precise', file_paths = None,
                  download_directory = DEFAULT_EPHEM_PATH,
                  verbose = False):
    """Add SV states to measurements using SP3 and CLK or Rinex files.

    If source is 'precise' then will use SP3 and CLK files.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        Instance of the NavData class that must include rows for
        ``gps_millis``, ``gnss_id``, ``sv_id``, and ``raw_pr_m``.
    source : string
        The method used to compute SV states. If 'precise', then will
        use SP3 and CLK precise files.
    file_paths : list, string or path-like
        Paths to existing ephemeris files if they exist.
    download_directory : string or path-like
        Directory where ephemeris files are downloaded if necessary.
    verbose : bool
        Prints extra debugging statements if true.

    Returns
    -------
    navdata_w_sv_states : gnss_lib_py.navdata.navdata.NavData
        Updated NavData class with satellite information computed.

    """
    if source == 'precise':
        navdata_w_sv_states = add_sv_states_precise(navdata,
                                file_paths = file_paths,
                                download_directory = download_directory,
                                verbose = verbose)
    else:
        raise RuntimeError('Only Precise SV state estimation supported')
    return navdata_w_sv_states


def add_sv_states_precise(navdata, file_paths = None,
                          download_directory = DEFAULT_EPHEM_PATH,
                          verbose=True):
    """Add SV states to measurements using SP3 and CLK files.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        Instance of the NavData class that must include rows for
        ``gps_millis``, ``gnss_id``, ``sv_id``, and ``raw_pr_m``.
    file_paths : list, string or path-like
        Paths to existing SP3 or CLK files if they exist.
    download_directory : string or path-like
        Directory where ephemeris files are downloaded if necessary.
    verbose : bool
        Prints extra debugging statements if true.

    Returns
    -------
    navdata_w_sv_states : gnss_lib_py.navdata.navdata.NavData
        Updated NavData class with satellite information computed.

    """

    # get unique gps_millis and constellations for ephemeris loader
    unique_gps_millis = np.unique(navdata["gps_millis"])
    constellations = np.unique(navdata["gnss_id"])

    # load sp3 files
    sp3_paths = load_ephemeris("sp3", gps_millis = unique_gps_millis,
                               constellations=constellations,
                               file_paths = file_paths,
                               download_directory=download_directory,
                               verbose=verbose
                              )
    sp3 = Sp3(sp3_paths)

    # load clk files
    clk_paths = load_ephemeris("clk", gps_millis = unique_gps_millis,
                               constellations=constellations,
                               file_paths = file_paths,
                               download_directory=download_directory,
                               verbose=verbose
                              )
    clk = Clk(clk_paths)

    # add SV states using sp3 and clk
    navdata_w_sv_states = single_gnss_from_precise_eph(navdata,
                                                       sp3,
                                                       clk,
                                                       verbose=verbose)

    return navdata_w_sv_states

def add_sv_states_rinex(measurements, ephemeris_path= DEFAULT_EPHEM_PATH,
                  constellations=['gps'], delta_t_dec = -2):
    """
    Add SV states (ECEF position and velocities) to measurements.

    Given received measurements, add SV states for measurements corresponding
    to received time and SV ID. If receiver position is given, that
    position is used to calculate the delay between signal transmission
    and reception, which is used to update the time at which the SV
    states are calculated.

    Columns for SV calculation: `gps_millis`, `gnss_id` and `sv_id`.
    Columns for Rx based correction: x_rx*_m, y_rx*_m and z_rx*_m

    Parameters
    ----------
    measurements : gnss_lib_py.navdata.navdata.NavData
        Recorded measurements with time of recpetion, GNSS ID and SV ID,
        corresponding to which SV states are calculated
    ephemeris_path : string or path-like
        Location where ephemeris files are stored. Files will be
        downloaded if they don't exist for the given date and constellation.
        If not given, default from
    constellations : list
        List of GNSS IDs for constellations are to be used. Others will
        be removed while processing the measurements
    delta_t_dec : int
        Decimal places after which times are considered as belonging to
        the same discrete time interval.

    Returns
    -------
    sv_states_all_time : gnss_lib_py.navdata.navdata.NavData
        Input measurements with rows containing SV states appended.
    """
    measurements_subset, ephem, _ = \
        _filter_ephemeris_measurements(measurements, constellations, ephemeris_path)
    sv_states_all_time = NavData()
    # Loop through the measurement file per time step
    for _, _, measure_frame in loop_time(measurements_subset,'gps_millis', \
                                                             delta_t_decimals=delta_t_dec):
        # Sort the satellites
        rx_ephem, _, inv_sort_order = _sort_ephem_measures(measure_frame, ephem)
        if rx_ephem.shape[1] != measure_frame.shape[1]: #pragma: no cover
            raise RuntimeError('Some ephemeris data is missing')
        try:
            # The following statement raises a KeyError if rows don't exist
            rx_rows_to_find = ['x_rx*_m', 'y_rx*_m', 'z_rx*_m']
            rx_idxs = find_wildcard_indexes(measure_frame,
                                                   rx_rows_to_find,
                                                   max_allow=1)
            rx_ecef = measure_frame[[rx_idxs["x_rx*_m"][0],
                                     rx_idxs["y_rx*_m"][0],
                                     rx_idxs["z_rx*_m"][0]]
                                     ,0]
            sv_states, _, _ = find_sv_location(measure_frame['gps_millis'], rx_ecef, rx_ephem)
        except KeyError:
            sv_states = find_sv_states(measure_frame['gps_millis'], rx_ephem)
        # Reverse the sorting
        sort(sv_states,ind=inv_sort_order,inplace=True)
        # Add them to new rows
        for row in sv_states.rows:
            if row not in ('gps_millis','gnss_id','sv_id'):
                measure_frame[row] = sv_states[row]
        if len(sv_states_all_time)==0:
            sv_states_all_time = measure_frame
        else:
            sv_states_all_time = concat(sv_states_all_time, measure_frame)
    return sv_states_all_time


def add_visible_svs_for_trajectory(rx_states,
                                   ephemeris_path=DEFAULT_EPHEM_PATH,
                                   constellations=['gps'], el_mask = 5.):
    """Wrapper to add visible satellite states for given times and positions.

    Given a sequence of times and rx points in ECEF, along with desired
    constellations, give SV states for satellites that are visible along
    trajectory at given times (assuming open sky conditions).

    rx_states must contain the following rows in order of increasing time:
    * :code:`gps_millis`
    * :code:`x_rx*_m`
    * :code:`y_rx*_m`
    * :code:`z_rx*_m`

    Parameters
    ----------
    rx_states : gnss_lib_py.navdata.navdata.NavData
        NavData containing position states of receiver at which SV states
        are needed.
    ephemeris_path : string
        Path at which ephemeris files are to be stored. Uses directory
        default if not given.
    constellations : list
        List of constellations for which states are to be estimated.
        Default is :code:`['gps']`. If :code:`None` is given, will estimate
        states for all available constellations.
    el_mask : float
        Elevation value above which satellites are considered visible.

    Returns
    -------
    sv_posvel_trajectory : gnss_lib_py.navdata.navdata.Navdata
        NavData instance containing


    """
    # Checks to ensure that the same number of times and states are given
    gps_millis = rx_states['gps_millis']
    assert len(gps_millis) == len(rx_states), \
        "Please give same number of times and ECEF points"
    assert isinstance(rx_states, NavData), \
        "rx_states must be a NavData instance"


    # Find starting time to download broadcast ephemeris file
    start_millis = gps_millis[0]

    # Initialize file with broadcast ephemeris parameters
    rinex_paths = load_ephemeris("rinex_nav",start_millis,constellations,
                                 download_directory=ephemeris_path,
                                 )
    ephem_all_sats = RinexNav(rinex_paths)

    # Find rows that correspond to receiver positions
    rx_rows_to_find = ['x_rx*_m', 'y_rx*_m', 'z_rx*_m']
    rx_idxs = find_wildcard_indexes(rx_states,rx_rows_to_find,
                                              max_allow=1)

    # Loop through all times and positions, estimated SV states and adding
    # them to a NavData instance that is returned
    sv_posvel_trajectory = NavData()
    for idx, milli in enumerate(gps_millis):
        rx_ecef = rx_states[[rx_idxs["x_rx*_m"][0],
                                rx_idxs["y_rx*_m"][0],
                                rx_idxs["z_rx*_m"][0]],
                                idx]
        ephem_viz = find_visible_ephem(milli, rx_ecef, ephem_all_sats, el_mask=el_mask)
        sv_posvel, _, _ = find_sv_location(milli, rx_ecef, ephem_viz)
        sv_posvel['gps_millis'] = milli
        if len(sv_posvel_trajectory) == 0:
            sv_posvel_trajectory = sv_posvel
        else:
            sv_posvel_trajectory = concat(sv_posvel_trajectory,sv_posvel)

    return sv_posvel_trajectory

def svs_from_el_az(elaz_deg):
    """Generate NED satellite positions for given elevation and azimuth.

    Given elevation and azimuth angles, with respect to the receiver,
    generate satellites in the NED frame of reference with the receiver
    position as the origin. Satellites are assumed to have a nominal
    distance of 20,200 km from the receiver (height of GNSS satellite orbit)

    Parameters
    ----------
    elaz_deg : np.ndarray
        Nx2 array of elevation and azimuth angles [degrees]

    Returns
    -------
    svs_ned : np.ndarray
        Nx3 satellite NED positions, simulated at a distance of 20,200 km
    """
    assert np.shape(elaz_deg)[0] == 2, "elaz_deg should be a 2xN array"
    el_deg = np.deg2rad(elaz_deg[0, :])
    az_deg = np.deg2rad(elaz_deg[1, :])
    unit_vect = np.zeros([3, np.shape(elaz_deg)[1]])
    unit_vect[0, :] = np.sin(az_deg)*np.cos(el_deg)
    unit_vect[1, :] = np.cos(az_deg)*np.cos(el_deg)
    unit_vect[2, :] = np.sin(el_deg)
    svs_ned = 20200000*unit_vect
    return svs_ned


def find_sv_states(gps_millis, ephem):
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
    ephem : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing ephemeris parameters of satellites
        for which states are required.

    Returns
    -------
    sv_posvel : gnss_lib_py.navdata.navdata.NavData
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

    c_is = ephem['C_is']
    c_ic = ephem['C_ic']
    c_rs = ephem['C_rs']
    c_rc = ephem['C_rc']
    c_uc = ephem['C_uc']
    c_us = ephem['C_us']
    delta_n   = ephem['deltaN']

    ecc        = ephem['e']     # eccentricity
    omega    = ephem['omega'] # argument of perigee
    omega_0  = ephem['Omega_0']
    sqrt_sma = ephem['sqrtA'] # sqrt of semi-major axis
    sma      = sqrt_sma**2      # semi-major axis

    sqrt_mu_a = np.sqrt(consts.MU_EARTH) * sqrt_sma**-3 # mean angular motion
    gpsweek_diff = (np.mod(gps_week,1024) - np.mod(ephem['gps_week'],1024))*604800.
    sv_posvel = NavData()
    sv_posvel['gnss_id'] = ephem['gnss_id']
    sv_posvel['sv_id'] = ephem['sv_id']
    # Deal with times being a single value or a vector with the same
    # length as the ephemeris
    sv_posvel['gps_millis'] = gps_millis

    delta_t = gps_tow - ephem['t_oe'] + gpsweek_diff

    # Calculate the mean anomaly with corrections
    ecc_anom = _compute_eccentric_anomaly(gps_week, gps_tow, ephem)

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
    omega_corr = ephem['OmegaDot'] * delta_t

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
    i_corr = c_ic*cos_to_phi + c_is*sin_to_phi + ephem['IDOT']*delta_t
    incl = ephem['i_0'] + i_corr

    ############################################
    ######  Lines added for velocity (2)  ######
    ############################################
    delta_i = 2*(c_is*cos_to_phi - c_ic*sin_to_phi)*dphi + ephem['IDOT']

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
    omega_dot = ephem['OmegaDot'] - consts.OMEGA_E_DOT
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
    clock_corr, _, _ = _estimate_sv_clock_corr(gps_millis, ephem)

    sv_posvel['b_sv_m'] = clock_corr

    return sv_posvel


def find_visible_ephem(gps_millis, rx_ecef, ephem, el_mask=5.):
    """Trim input ephemeris to keep only visible SVs.

    Parameters
    ----------
    gps_millis : int
        Time at which measurements are needed, measured in milliseconds
        since start of GPS epoch [ms].
    rx_ecef : np.ndarray
        3x1 row rx_pos ECEF position vector [m].
    ephem : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing satellite ephemeris parameters
        including gps_week and gps_tow for the ephemeris
    el_mask : float
        Minimum elevation of satellites considered visible.

    Returns
    -------
    eph : gnss_lib_py.navdata.navdata.NavData
        Ephemeris parameters of visible satellites

    """
    # Find positions and velocities of all satellites
    approx_posvel = find_sv_states(gps_millis - 1000.*consts.T_TRANS, ephem)
    # Find elevation and azimuth angles for all satellites
    approx_pos, _ = _extract_pos_vel_arr(approx_posvel)
    approx_el_az = ecef_to_el_az(np.reshape(rx_ecef, [3, 1]), approx_pos)
    # Keep attributes of only those satellites which are visible
    keep_ind = approx_el_az[0,:] > el_mask
    eph = ephem.copy(cols=np.nonzero(keep_ind))
    return eph


def find_visible_sv_posvel(rx_ecef, sv_posvel, el_mask=5.):
    """Trim input SV state NavData to keep only visible SVs.

    Parameters
    ----------
    rx_ecef : np.ndarray
        3x1 row rx_pos ECEF position vector [m].
    sv_posvel : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing satellite positions and velocities
        at the time at which visible satellites are needed.
    el_mask : float
        Minimum elevation of satellites considered visible.

    Returns
    -------
    vis_posvel : gnss_lib_py.navdata.navdata.NavData
        SV states of satellites that are visible

    """
    # Find positions and velocities of all satellites
    approx_posvel = sv_posvel.copy()
    # Find elevation and azimuth angles for all satellites
    approx_pos, _ = _extract_pos_vel_arr(approx_posvel)
    approx_el_az = ecef_to_el_az(np.reshape(rx_ecef, [3, 1]), approx_pos)
    # Keep attributes of only those satellites which are visible
    keep_ind = approx_el_az[0,:] > el_mask
    vis_posvel = sv_posvel.copy(cols=np.nonzero(keep_ind))
    return vis_posvel

def find_sv_location(gps_millis, rx_ecef, ephem=None, sv_posvel=None, get_iono=False):
    """Given time, return SV positions, difference from Rx, and ranges.

    Parameters
    ----------
    gps_millis : int
        Time at which measurements are needed, measured in milliseconds
        since start of GPS epoch [ms].
    rx_ecef : np.ndarray
        3x1 Receiver 3D ECEF position [m].
    ephem : gnss_lib_py.navdata.navdata.NavData
        DataFrame containing all satellite ephemeris parameters ephemeris,
        as indicated in :code:`find_sv_states`. Use None if using
        precomputed satellite positions and velocities instead.
    sv_posvel : gnss_lib_py.navdata.navdata.NavData
        Precomputed positions of satellites, use None if using broadcast
        ephemeris parameters instead.

    Returns
    -------
    sv_posvel : gnss_lib_py.navdata.navdata.NavData
        Satellite position and velocities (same if input).
    del_pos : np.ndarray
        Difference between satellite positions and receiver position.
    true_range : np.ndarray
        Distance between satellite and receiver positions.

    """
    rx_ecef = np.reshape(rx_ecef, [3, 1])
    if sv_posvel is None:
        assert ephem is not None, "Must provide ephemeris or positions" \
                                + " to find satellites states"
        sv_posvel = find_sv_states(gps_millis - 1000.*consts.T_TRANS, ephem)
        del_pos, true_range = _find_delxyz_range(sv_posvel, rx_ecef)
        t_corr = true_range/consts.C

        # Find satellite locations at (a more accurate) time of transmission
        sv_posvel = find_sv_states(gps_millis-1000.*t_corr, ephem)
    del_pos, true_range = _find_delxyz_range(sv_posvel, rx_ecef)
    t_corr = true_range/consts.C

    return sv_posvel, del_pos, true_range

def _filter_ephemeris_measurements(measurements, constellations,
                                   ephemeris_path = DEFAULT_EPHEM_PATH, get_iono=False):
    """Return subset of input measurements and ephmeris containing
    constellations and received SVs.

    Measurements are filtered to contain the intersection of received and
    desired constellations.
    Ephemeris is extracted from the given path and a subset containing
    SVs that are in measurements is returned.

    Parameters
    ----------
    measurements : gnss_lib_py.navdata.navdata.NavData
        Received measurements, that are filtered based on constellations.
    constellations : list
        List of strings indicating constellations required in output.
    ephemeris_path : string or path-like
        Path where the ephermis files are stored or downloaded to.

    Returns
    -------
    measurements_subset : gnss_lib_py.navdata.navdata.NavData
        Measurements containing desired constellations
    ephem : gnss_lib_py.navdata.navdata.NavData
        Ephemeris parameters for received SVs and constellations
    """
    measurements.in_rows(['gnss_id', 'sv_id', 'gps_millis'])
    # Check whether the rows are in the right format as needed.
    assert isinstance(measurements['gnss_id'].dtype, object)
    assert np.issubdtype(measurements['sv_id'].dtype, np.integer)
    assert np.issubdtype(measurements['gps_millis'].dtype, float)
    rx_const= np.unique(measurements['gnss_id'])
    # Check if required constellations are available, keep only required
    # constellations
    if constellations is None:
        constellations = list(consts.CONSTELLATION_CHARS.values())
    for const in constellations:
        if const not in rx_const:
            warnings.warn(const + " not available in received constellations", RuntimeWarning)
    rx_const_set = set(rx_const)
    req_const_set = set(constellations)
    keep_consts = req_const_set.intersection(rx_const_set)

    measurements_subset = measurements.where('gnss_id', keep_consts, condition="eq")

    # preprocessing of received quantities for downloading ephemeris file
    eph_sv = _combine_gnss_sv_ids(measurements)
    lookup_sats = list(np.unique(eph_sv))
    start_gps_millis = np.min(measurements['gps_millis'])
    # Download the ephemeris file for all the satellites in the measurement files
    ephem = get_time_cropped_rinex(start_gps_millis, lookup_sats,
                                   ephemeris_path)
    if get_iono:
        keys = list(ephem.iono_params.keys())
        key = keys[np.argmin([(start_gps_millis - key) for key in keys \
                            if (start_gps_millis - key) >= 0])]
        iono_params = ephem.iono_params[key]
    else:
        iono_params = None
    return measurements_subset, ephem, iono_params


def _combine_gnss_sv_ids(measurement_frame):
    """Combine string `gnss_id` and integer `sv_id` into single `gnss_sv_id`.

    `gnss_id` contains strings like 'gps' and 'glonass' and `sv_id` contains
    integers. The newly returned `gnss_sv_id` is formatted as `Axx` where
    `A` is a single letter denoting the `gnss_id` and `xx` denote the two
    digit `sv_id` of the satellite.

    Parameters
    ----------
    measurement_frame : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing measurements including `gnss_id` and
        `sv_id`.

    Returns
    -------
    gnss_sv_id : np.ndarray
	New row values that combine `gnss_id` and `sv_id` into a something
	similar to 'R01' or 'G12' for example.

    Notes
    -----
    For reference on strings and the contellation characters corresponding
    to them, refer to :code:`CONSTELLATION_CHARS` in
    `gnss_lib_py/utils/constants.py`.

    """
    constellation_char_inv = {const : gnss_char for gnss_char, const in consts.CONSTELLATION_CHARS.items()}
    gnss_chars = [constellation_char_inv[const] for const in np.array(measurement_frame['gnss_id'], ndmin=1)]
    gnss_sv_id = np.asarray([gnss_chars[col_num] + f'{sv:02}' for col_num, sv in enumerate(np.array(measurement_frame['sv_id'], ndmin=1))])
    return gnss_sv_id


def _sort_ephem_measures(measure_frame, ephem):
    """Sort measures and return indices for sorting and inverting sort.

    Parameters
    ----------
    measure_frame : gnss_lib_py.navdata.navdata.NavData
        Measurements received for a single time instance, to be sorted.
    ephem : gnss_lib_py.navdata.navdata.NavData
        Ephemeris parameters for all satellites for the closest time
        before the measurements were received.

    Returns
    -------
    rx_ephem : gnss_lib_py.navdata.navdata.NavData
        Ephemeris parameters for satellites from which measurements were
        received. Sorted by `gnss_sv_id`.
    sorted_sats_ind : np.ndarray
        Indices that sorts the original measurements by `gnss_sv_id`.
    inv_sort_order : np.ndarray
        Indices that invert the sort by `gnss_sv_id` to match the order
        in the input measurements.

    """
    gnss_sv_id = _combine_gnss_sv_ids(measure_frame)
    sorted_sats_ind = np.argsort(gnss_sv_id)
    inv_sort_order = np.argsort(sorted_sats_ind)
    sorted_sats = gnss_sv_id[sorted_sats_ind]
    rx_ephem = ephem.where('gnss_sv_id', sorted_sats, condition="eq")
    return rx_ephem, sorted_sats_ind, inv_sort_order


def _extract_pos_vel_arr(sv_posvel):
    """Extract satellite positions and velocities into numpy arrays.

    Parameters
    ----------
    sv_posvel : gnss_lib_py.navdata.navdata.NavData
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


def _find_delxyz_range(sv_posvel, rx_ecef):
    """Return difference of satellite and rx_pos positions and distance between them.

    Parameters
    ----------
    sv_posvel : gnss_lib_py.navdata.navdata.NavData
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

def single_gnss_from_precise_eph(navdata, sp3_parsed_file,
                                 clk_parsed_file, inplace=False,
                                 verbose = False):
    """Compute satellite states using .sp3 and .clk

    Either adds or replaces satellite ECEF position data and clock bias
    for any satellite that exists in the provided sp3 file and clk file.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        Instance of the NavData class that must include rows for
        ``gps_millis`` and ``gnss_sv_id``.
    sp3_parsed_file : gnss_lib_py.parsers.sp3.Sp3
        SP3 data
    clk_parsed_file : gnss_lib_py.parsers.clk.Clk
        Clk data
    inplace : bool
        If true, adds satellite positions and clock bias to the input
        navdata object, otherwise returns a new NavData object with the
        satellite rows added.
    verbose : bool
        Flag (True/False) for whether to print intermediate steps useful
        for debugging/reviewing (the default is False)

    Returns
    -------
    navdata : gnss_lib_py.navdata.navdata.NavData
        Updated NavData class with satellite information computed using
        precise ephemerides from .sp3 and .clk files
    """

    # combine gnss_id and sv_id into gnss_sv_ids
    if inplace:
        navdata["gnss_sv_id"] = _combine_gnss_sv_ids(navdata)
        sp3_parsed_file.interpolate_sp3(navdata, verbose=verbose)
        clk_parsed_file.interpolate_clk(navdata, verbose=verbose)
    else:
        new_navdata = navdata.copy()
        new_navdata["gnss_sv_id"] = _combine_gnss_sv_ids(new_navdata)
        sp3_parsed_file.interpolate_sp3(new_navdata, verbose=verbose)
        clk_parsed_file.interpolate_clk(new_navdata, verbose=verbose)

    if inplace:
        return None
    return new_navdata
