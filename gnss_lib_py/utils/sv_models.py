"""Model GNSS SV states (positions and velocities).

Functions to calculate GNSS SV positions and velocities for a given time.
"""

__authors__ = "Ashwin Kanhere, Bradley Collicott"
__date__ = "17 Jan, 2023"

import warnings

import numpy as np


from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.parsers.sp3 import Sp3
from gnss_lib_py.parsers.clk import Clk
from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.parsers.rinex_nav import (load_rinex_nav,
                                           RinexNav,

                                           _extract_pos_vel_arr,
                                           rinex_to_sv_states)
import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.coordinates import ecef_to_el_az
from gnss_lib_py.utils.ephemeris_downloader import DEFAULT_EPHEM_PATH
from gnss_lib_py.utils.time_conversions import gps_millis_to_tow, gps_millis_to_datetime
from gnss_lib_py.utils.ephemeris_downloader import (DEFAULT_EPHEM_PATH,
                                                    load_ephemeris,
                                                    combine_gnss_sv_ids)


def add_sv_states(navdata, source = 'precise', file_paths = None,
                  download_directory = DEFAULT_EPHEM_PATH,
                  verbose = False):
    """Add SV states to measurements using SP3 and CLK or Rinex files.

    If source is 'precise' then will use SP3 and CLK files.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
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
    navdata_w_sv_states : gnss_lib_py.parsers.navdata.NavData
        Updated NavData class with satellite information computed.

    """
    if source == 'precise':
        navdata_w_sv_states = add_sv_states_precise(navdata,
                                file_paths = file_paths,
                                download_directory = download_directory,
                                verbose = verbose)
    elif source == 'broadcast':
        navdata_w_sv_states = add_sv_states_broadcast(navdata,
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
    navdata : gnss_lib_py.parsers.navdata.NavData
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
    navdata_w_sv_states : gnss_lib_py.parsers.navdata.NavData
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

def add_sv_states_broadcast(measurements, file_paths = None,
                            download_directory= DEFAULT_EPHEM_PATH,
                            delta_t_dec = -2):
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
    measurements : gnss_lib_py.parsers.navdata.NavData
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
    sv_states_all_time : gnss_lib_py.parsers.navdata.NavData
        Input measurements with rows containing SV states appended.
    """
    measurements_subset, ephem, _ = \
        _filter_ephemeris_measurements(measurements, constellations, download_directory)
    sv_states_all_time = NavData()
    # Loop through the measurement file per time step
    for _, _, measure_frame in measurements_subset.loop_time('gps_millis', \
                                                             delta_t_decimals=delta_t_dec):
        # measure_frame = measure_frame.sort('sv_id', order="descending")
        # Sort the satellites
        rx_ephem, _, inv_sort_order = _sort_ephem_measures(measure_frame, ephem)
        if rx_ephem.shape[1] != measure_frame.shape[1]: #pragma: no cover
            raise RuntimeError('Some ephemeris data is missing')
        try:
            # The following statement raises a KeyError if rows don't exist
            rx_rows_to_find = ['x_rx*_m', 'y_rx*_m', 'z_rx*_m']
            rx_idxs = measure_frame.find_wildcard_indexes(
                                                   rx_rows_to_find,
                                                   max_allow=1)
            rx_ecef = measure_frame[[rx_idxs["x_rx*_m"][0],
                                     rx_idxs["y_rx*_m"][0],
                                     rx_idxs["z_rx*_m"][0]]
                                     ,0]
            sv_states = rinex_to_sv_states(measure_frame['gps_millis'], rx_ecef, rx_ephem)
        except KeyError:
            sv_states = rinex_to_sv_states(measure_frame['gps_millis'], rx_ephem)
        # Reverse the sorting
        sv_states = sv_states.sort(ind=inv_sort_order)
        # Add them to new rows
        for row in sv_states.rows:
            if row not in ('gps_millis','gnss_id','sv_id'):
                measure_frame[row] = sv_states[row]
        if len(sv_states_all_time)==0:
            sv_states_all_time = measure_frame
        else:
            sv_states_all_time.concat(measure_frame, inplace=True)
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
    rx_states : gnss_lib_py.parsers.navdata.NavData
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
    sv_posvel_trajectory : gnss_lib_py.parsers.navdata.Navdata
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
    # Load the rinex_nav file and trim to have only one entry per satellite
    ephem_all_sats = load_rinex_nav(rinex_paths, rinex_nav_paths=rinex_paths,
                                    ephemeris_directory=ephemeris_path)

    # Find rows that correspond to receiver positions
    rx_rows_to_find = ['x_rx*_m', 'y_rx*_m', 'z_rx*_m']
    rx_idxs = rx_states.find_wildcard_indexes(rx_rows_to_find,
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
        sv_posvel = rinex_to_sv_states(milli, rx_ecef, ephem_viz)
        sv_posvel['gps_millis'] = milli
        if len(sv_posvel_trajectory) == 0:
            sv_posvel_trajectory = sv_posvel
        else:
            sv_posvel_trajectory.concat(sv_posvel, inplace=True)

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


def find_visible_ephem(gps_millis, rx_ecef, ephem, el_mask=5.):
    """Trim input ephemeris to keep only visible SVs.

    Parameters
    ----------
    gps_millis : int
        Time at which measurements are needed, measured in milliseconds
        since start of GPS epoch [ms].
    rx_ecef : np.ndarray
        3x1 row rx_pos ECEF position vector [m].
    ephem : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing satellite ephemeris parameters
        including gps_week and gps_tow for the ephemeris
    el_mask : float
        Minimum elevation of satellites considered visible.

    Returns
    -------
    eph : gnss_lib_py.parsers.navdata.NavData
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
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing satellite positions and velocities
        at the time at which visible satellites are needed.
    el_mask : float
        Minimum elevation of satellites considered visible.

    Returns
    -------
    vis_posvel : gnss_lib_py.parsers.navdata.NavData
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

def _filter_ephemeris_measurements(measurements,
                                   ephemeris_path = DEFAULT_EPHEM_PATH,
                                   get_iono=False):
    """Return subset of input measurements and ephmeris containing
    constellations and received SVs.

    Measurements are filtered to contain the intersection of received and
    desired constellations.
    Ephemeris is extracted from the given path and a subset containing
    SVs that are in measurements is returned.

    Parameters
    ----------
    measurements : gnss_lib_py.parsers.navdata.NavData
        Received measurements, that are filtered based on constellations.
    constellations : list
        List of strings indicating constellations required in output.
    ephemeris_path : string or path-like
        Path where the ephermis files are stored or downloaded to.

    Returns
    -------
    measurements_subset : gnss_lib_py.parsers.navdata.NavData
        Measurements containing desired constellations
    ephem : gnss_lib_py.parsers.navdata.NavData
        Ephemeris parameters for received SVs and constellations
    """
    measurements.in_rows(['gnss_id', 'sv_id', 'gps_millis'])
    # Check whether the rows are in the right format as needed.
    isinstance(measurements['gnss_id'].dtype, object)
    isinstance(measurements['sv_id'].dtype, int)
    isinstance(measurements['gps_millis'].dtype, np.int64)
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
    eph_sv = combine_gnss_sv_ids(measurements)
    lookup_sats = list(np.unique(eph_sv))
    start_gps_millis = np.min(measurements['gps_millis'])
    # Download the ephemeris file for all the satellites in the measurement files
    ephem = load_rinex_nav(start_gps_millis, lookup_sats,
                                   ephemeris_path)
    if get_iono:
        keys = list(ephem.iono_params.keys())
        key = keys[np.argmin([(start_gps_millis - key) for key in keys \
                            if (start_gps_millis - key) >= 0])]
        iono_params = ephem.iono_params[key]
    else:
        iono_params = None
    return measurements_subset, ephem, iono_params


def _sort_ephem_measures(measure_frame, ephem):
    """Sort measures and return indices for sorting and inverting sort.

    Parameters
    ----------
    measure_frame : gnss_lib_py.parsers.navdata.NavData
        Measurements received for a single time instance, to be sorted.
    ephem : gnss_lib_py.parsers.navdata.NavData
        Ephemeris parameters for all satellites for the closest time
        before the measurements were received.

    Returns
    -------
    rx_ephem : gnss_lib_py.parsers.navdata.NavData
        Ephemeris parameters for satellites from which measurements were
        received. Sorted by `gnss_sv_id`.
    sorted_sats_ind : np.ndarray
        Indices that sorts the original measurements by `gnss_sv_id`.
    inv_sort_order : np.ndarray
        Indices that invert the sort by `gnss_sv_id` to match the order
        in the input measurements.

    """
    gnss_sv_id = combine_gnss_sv_ids(measure_frame)
    sorted_sats_ind = np.argsort(gnss_sv_id)
    inv_sort_order = np.argsort(sorted_sats_ind)
    sorted_sats = gnss_sv_id[sorted_sats_ind]
    rx_ephem = ephem.where('gnss_sv_id', sorted_sats, condition="eq")
    return rx_ephem, sorted_sats_ind, inv_sort_order


def single_gnss_from_precise_eph(navdata, sp3_parsed_file,
                                 clk_parsed_file, inplace=False,
                                 verbose = False):
    """Compute satellite states using .sp3 and .clk

    Either adds or replaces satellite ECEF position data and clock bias
    for any satellite that exists in the provided sp3 file and clk file.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
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
    navdata : gnss_lib_py.parsers.navdata.NavData
        Updated NavData class with satellite information computed using
        precise ephemerides from .sp3 and .clk files
    """

    # combine gnss_id and sv_id into gnss_sv_ids
    if inplace:
        navdata["gnss_sv_id"] = combine_gnss_sv_ids(navdata)
        sp3_parsed_file.interpolate_sp3(navdata, verbose=verbose)
        clk_parsed_file.interpolate_clk(navdata, verbose=verbose)
    else:
        new_navdata = navdata.copy()
        new_navdata["gnss_sv_id"] = combine_gnss_sv_ids(new_navdata)
        sp3_parsed_file.interpolate_sp3(new_navdata, verbose=verbose)
        clk_parsed_file.interpolate_clk(new_navdata, verbose=verbose)

    if inplace:
        return None
    return new_navdata
