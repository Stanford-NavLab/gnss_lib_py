"""Tests for the ephemeris class that returns satellite ephemerides.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "30 Aug 2022"

import os
from datetime import datetime, timezone

import pytest
import numpy as np

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.parsers.sp3 import Sp3
from gnss_lib_py.parsers.rinex_nav import (RinexNav,
                                           load_rinex_nav,
                                           get_closest_rinex_data,
                                           rinex_to_sv_states)
import gnss_lib_py.parsers.rinex_nav as rinex_nav
import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.ephemeris_downloader import (load_ephemeris,
                                                    combine_gnss_sv_ids)
from gnss_lib_py.utils.time_conversions import (gps_millis_to_tow,
                                                datetime_to_gps_millis,
                                                gps_millis_to_datetime)

@pytest.fixture(name="ephem_path", scope='session')
def fixture_ephem_path():
    """Location of ephemeris files for unit test

    Returns
    -------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded.
    """
    root_path = os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
    ephem_path = os.path.join(root_path, 'data/unit_test')
    return ephem_path


@pytest.fixture(name="all_gps_sats")
def fixture_all_gps_sats():
    """List of all GPS satellites that can be loaded by RinexNav.

    Returns
    -------
    all_gps_sats : list
        List of all GPS satellites.
    """
    all_gps_sats  = [f'G{sv:02d}' for sv in range(1,33)]
    return all_gps_sats


@pytest.fixture(name="rinex_nav_files_sats")
def fixture_rinex_nav_files_sats(all_gps_sats):
    """Names of files to be used to test RinexNav loading.

    The files do not contain all satellites and for speed considerations,
    the satellites available are listed here.

    Returns
    -------
    rinex_nav_files_sats : list
        List of tuple containing file name and list of satellites to load
        for that file.
    """
    rinex_nav_files_sats = [('BRDC00WRD_S_20230730000_01D_MN.rnx', None),
                            # ('brdc0730.17n', ['G01', 'G02']),
                            ('brdc1180.21n', all_gps_sats),
                            # ('brdc1190.21n', all_gps_sats),
                            # ('brdc1200.21n', all_gps_sats),
                            # ('brdc1360.20n', all_gps_sats),
                            ('brdc1370.20g', ['R01', 'R02']),
                            ('brdc1370.20n', ['G01', 'G02']),
                            ('brdc1370_no_leapseconds.20n', ['G01', 'G02']),
                            ('BRDM00DLR_R_20130010000_01D_MN.rnx', ['G01', 'G02', 'R01', 'R02', 'J01']),
                            ('BRDM00DLR_S_20230730000_01D_MN.rnx', ['G01']),
                            ('BRDM00DLR_S_20230730000_01D_MN_no_gps_iono.rnx',
                             ['G01', 'G02', 'R01', 'R02', 'E01', 'E02', 'C01', 'C02', 'J01', 'S22', 'S23']),
                            ('GOLD00USA_R_20201370000_01D_EN.rnx', ['E02', 'E03', 'E15']),
                            ('VOIM00MDG_R_20140010000_01D_MN.rnx', ['R03', 'R04']),
                            ('WTZS00DEU_R_20230800000_01D_MN.rnx', ['C02']),
                            ('ZIM200CHE_R_20201390000_01D_GN.rnx', ['G05', 'G06']),
                            # ('ZIM200CHE_R_20201390000_01D_RN.rnx', ['R12', 'R17']),
                            # ('zim21380.20g', ['R01', 'R02']),
                            # ('zim21380.20n', ['G02', 'G03'])
                            ]
    return rinex_nav_files_sats


@pytest.fixture(name="rinex_nav_files_no_iono")
def fixture_rinex_nav_files_no_iono():
    """List of files that do not have ionosphere correction parameters.

    Returns
    -------
    rinex_nav_files_no_iono : list
        List of files that do not have ionosphere correction parameters.
    """
    rinex_nav_files_no_iono = ["BRDM00DLR_R_20130010000_01D_MN.rnx",
                               "VOIM00MDG_R_20140010000_01D_MN.rnx",
                               "BRDC00WRD_S_20230730000_01D_MN.rnx",
                               "WTZS00DEU_R_20230800000_01D_MN.rnx",
                               ]
    return rinex_nav_files_no_iono


@pytest.fixture(name="rinex_nav_files_no_leapseconds")
def fixture_rinex_nav_files_no_leapseconds():
    """List of files that do not have leap seconds.

    Returns
    -------
    rinex_nav_files_no_leapseconds : list
        List of files that do not have leap seconds.
    """
    rinex_nav_files_no_leapseconds = ["brdc1370_no_leapseconds.20n"]
    return rinex_nav_files_no_leapseconds



def test_rinex_nav_init(ephem_path, rinex_nav_files_sats, rinex_nav_files_no_iono):
    """Test across multiple files.

    For speed, this code tests loading with particular satellites.

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded.
    rinex_nav_files_sats : list
        List of tuple containing file name and list of satellites to load
        for that file.
    rinex_nav_files_no_iono : list
        List of files that do not have ionosphere correction parameters.
    """

    rinex_nav_dir = os.path.join(ephem_path,"rinex","nav")
    rinex_paths = []
    for file, sats in rinex_nav_files_sats:
        rinex_path = os.path.join(rinex_nav_dir,file)
        rinex_paths.append(rinex_path)

        if file in rinex_nav_files_no_iono:
            # no iono
            with pytest.warns(RuntimeWarning) as warns:
                RinexNav(rinex_path, satellites = sats)
            assert len(warns) == 1
        else:
            RinexNav(rinex_path, satellites = sats)

def test_rinex_nav_init_multi(ephem_path):
    """Test across multiple files.

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded.

    """

    files = [
             "brdc1360.20n",
             "brdc1370.20n",
             "brdc1370.20g",
            ]

    rinex_nav_dir = os.path.join(ephem_path,"rinex","nav")
    rinex_paths = []
    for file in files:
        rinex_path = os.path.join(rinex_nav_dir,file)
        rinex_paths.append(rinex_path)
    RinexNav(rinex_paths)


@pytest.mark.parametrize('satellites',
                         [
                          ['G02'],
                          ['R01'],
                         ])
@pytest.mark.parametrize('ephem_time',
                         [
                          datetime(2020, 5, 16, 11, 47, 48, tzinfo=timezone.utc),
                          datetime(2020, 5, 16, 12, 10, 48, tzinfo=timezone.utc),
                         ])
def test_load_rinex_nav(ephem_path, ephem_time, satellites):
    """Create instance of Ephemeris manager and fetch ephemeris file.

    Check type and rows.

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded to.
    ephem_time : datetime.datetime
        Time at which state estimation is starting, the most recent ephemeris
        file before the start time will be fetched for GPS like constellations.
        For Glonass like constellations, the closest time will be fetched
        and this might be in the future as well.
    satellites : List
        List of satellites ['Const_IDSVID']

    """

    ephem_time_millis = datetime_to_gps_millis(ephem_time)

    ephem = load_rinex_nav(ephem_time_millis, satellites, ephem_path,
                                    verbose=True)

    # Test that ephem is of type gnss_lib_py.parsers.navdata.NavData
    assert isinstance(ephem, NavData)

    # check that there's one row per satellite
    assert len(ephem) == len(satellites)

    # time check for GPS and Galileo
    if "gps_week" in ephem.rows:
        for timestep in range(len(ephem)):
            if not np.isnan(ephem["gps_week",timestep]):
                gps_week, _ = gps_millis_to_tow(ephem["gps_millis",timestep])
                assert gps_week == ephem["gps_week",timestep]

    # Check that the correct time was loaded.
    if ephem_time == datetime(2020, 5, 16, 11, 47, 48, tzinfo=timezone.utc):
        if satellites == ['G02']:
            closest_time = datetime(2020, 5, 16, 10, 0, 0, tzinfo=timezone.utc)
            closest_time_millis = datetime_to_gps_millis(closest_time)
        elif satellites == ['R01']:
            closest_time = datetime(2020, 5, 16, 11, 45, 0, tzinfo=timezone.utc)
            closest_time_millis = datetime_to_gps_millis(closest_time)

    elif ephem_time == datetime(2020, 5, 16, 12, 10, 48, tzinfo=timezone.utc):
        if satellites == ['G02']:
            closest_time = datetime(2020, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
            closest_time_millis = datetime_to_gps_millis(closest_time)
        elif satellites == ['R01']:
            closest_time = datetime(2020, 5, 16, 12, 15, 0, tzinfo=timezone.utc)
            closest_time_millis = datetime_to_gps_millis(closest_time)

    np.testing.assert_almost_equal(ephem["gps_millis",0], closest_time_millis, decimal=-5)
    # The decimal -5 indicates that in this case, the closest time is off
    # by a few (16 seconds), which isn't incorrect but also isn't the
    # exact expected time.
    # Tests that NavData specific rows are present in ephem
    navdata_rows = ['gps_millis', 'sv_id', 'gnss_id']
    ephem.in_rows(navdata_rows)


@pytest.mark.parametrize('satellites',
                         [
                          ['G99'],
                         ])
@pytest.mark.parametrize('ephem_time',
                         [
                          datetime(2020, 5, 15, 3, 47, 48, tzinfo=timezone.utc),
                         ])
def test_get_ephem_fails(ephem_path, ephem_time, satellites):
    """Test when ephemeris manager should fail.

    Check type and rows.

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded to.
    ephem_time : datetime.datetime
        Time at which state estimation is starting, the most recent ephemeris
        file before the start time will be fetched.
    satellites : List
        List of satellites ['Const_IDSVID']

    """

    ephem_time = datetime_to_gps_millis(ephem_time)

    with pytest.raises(RuntimeError) as excinfo:
        load_rinex_nav(ephem_time, satellites, ephem_path)
    assert "ephemeris data" in str(excinfo.value)


def test_load_leapseconds(ephem_path):
    """Test load leapseconds.

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded to.

    """

    rinex_path = os.path.join(ephem_path,"rinex","nav",
                                   "brdc1370.20n")
    rinex_data = RinexNav(rinex_path)

    np.testing.assert_array_equal(rinex_data['leap_seconds'],
                                  np.array([18]*6))

    # check what happens when a file with an incomplete header is passed
    none_path = os.path.join(ephem_path,"rinex","nav",
                                   "brdc1370_no_leapseconds.20n")
    rinex_data = RinexNav(none_path)
    assert len(rinex_data.where('leap_seconds',np.nan)) == 1

def test_no_iono_params(ephem_path, rinex_nav_files_no_iono):
    """Test no iono params.

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded to.

    """
    with pytest.warns(RuntimeWarning) as warns:
        for rinex_file in rinex_nav_files_no_iono:
            rinex_path = os.path.join(ephem_path,"rinex","nav", rinex_file)
            RinexNav(rinex_path)


def test_gmst_calculation():
    """
    Notes
    -----
    Verified by comparing with the GMST calculated from
    https://www.omnicalculator.com/everyday-life/sidereal-time#how-to-calculate-sidereal-time-greenwich-sidereal-time-calculator-mean-and-apparent
    """
    query_datetime = np.datetime64('2020-05-15T02:00:00.000000')
    exp_hour = 15
    exp_min = 32
    exp_sec = 44
    exp_time = exp_hour + exp_min/60 + exp_sec/3600
    exp_time_rads = exp_time * 2 * np.pi / 24
    out_gmst_rad = rinex_nav._find_gmst_at_midnight(query_datetime)
    # np.testing.assert_almost_equal(exp_time_rads, out_gmst_rad, decimal=2)

    # Test that the function works for an array of datetimes
    query_datetime_2 = np.datetime64('2020-05-15T04:00:00.000000')
    exp_time_2 = exp_hour + exp_min/60 + exp_sec/3600
    exp_time_rads_2 = exp_time * 2 * np.pi / 24

    query_datetime_array = np.array([query_datetime, query_datetime_2])
    out_gmst_array = rinex_nav._find_gmst_at_midnight(query_datetime_array)
    np.testing.assert_almost_equal(np.asarray([exp_time_rads, exp_time_rads]),
                                    out_gmst_array, decimal=2)

    query_datetime_3 = np.datetime64('2020-05-16T00:00:00.000000')
    #NOTE: The sidereal time actually changes by more than 24 hours when
    # the day changes, hence the minutes are different here
    exp_hour_2 = 15
    exp_min_2 = 36
    exp_sec_2 = 44
    exp_time_2 = exp_hour_2 + exp_min_2/60 + exp_sec_2/3600
    exp_time_rads_2 = exp_time_2  * 2 * np.pi / 24
    query_datetime_array = np.array([query_datetime, query_datetime_2, query_datetime_3])
    out_gmst_array = rinex_nav._find_gmst_at_midnight(query_datetime_array)
    np.testing.assert_almost_equal(np.asarray([exp_time_rads, exp_time_rads, exp_time_rads_2]),
                                   out_gmst_array, decimal = 2)


@pytest.fixture(name="broadcast_error_bounds")
def fixture_broadcast_error_bounds():
    broadcast_error_bounds = {}
    broadcast_error_bounds['gps'] = 5
    broadcast_error_bounds['glonass'] = 11
    broadcast_error_bounds['galileo'] = 5
    broadcast_error_bounds['beidou'] = 50
    broadcast_error_bounds['qzss'] = 5
    return broadcast_error_bounds


@pytest.fixture(name="load_gps_millis")
def fixture_file_load_time():
    timestamp = datetime(2020, 5, 15, 5, 10, 0, tzinfo=timezone.utc)
    load_gps_millis = datetime_to_gps_millis(timestamp)
    return load_gps_millis


@pytest.fixture(name="sp3_file")
def fixture_sp3_file_path(ephem_path, load_gps_millis):
    sp3_file_paths = load_ephemeris("sp3", load_gps_millis, download_directory=ephem_path)
    sp3_file = Sp3(sp3_file_paths)
    return sp3_file


@pytest.fixture(name="rinex_nav_file")
def fixture_rinex_nav_file(ephem_path, load_gps_millis):
    # rinex_nav_paths = load_ephemeris("rinex_nav", load_gps_millis, download_directory=ephem_path)
    # rinex_nav_file = RinexNav(rinex_nav_paths)
    csv_path = os.path.join(ephem_path, 'rinex/nav', 'parsed_sv_state_test.csv')
    rinex_nav_file = NavData(csv_path=csv_path)
    return rinex_nav_file


@pytest.fixture(name="middle_gps_millis")
def fixture_middle_gps_millis():
    middle_of_the_day = datetime(2020, 5, 15, 15, 29, 42, tzinfo=timezone.utc)
    middle_gps_millis = datetime_to_gps_millis(middle_of_the_day)
    return middle_gps_millis


@pytest.fixture(name="sp3_constellations")
def fixture_sp3_constellations(middle_gps_millis, sp3_file):

    sp3_trimmed = sp3_file.where('gps_millis', [middle_gps_millis-3600*1e3, middle_gps_millis], 'between')
    sp3_constellations = {}
    sp3_constellations['gps'] = sp3_trimmed.where("gnss_id", "gps", "eq")
    sp3_constellations['glonass'] = sp3_trimmed.where('gnss_id', 'glonass', 'eq')
    sp3_constellations['galileo'] = sp3_trimmed.where('gnss_id', 'galileo', 'eq')
    sp3_constellations['beidou'] = sp3_trimmed.where('gnss_id', 'beidou', 'eq')
    sp3_constellations['qzss'] = sp3_trimmed.where('gnss_id', 'qzss', 'eq')
    return sp3_constellations


@pytest.mark.parametrize('constellation',
                         [
                          'gps',
                          'glonass',
                          'galileo',
                          'beidou',
                          'qzss',
                         ])
def test_sv_state_broadcast(constellation, sp3_constellations,
                            rinex_nav_file,
                            broadcast_error_bounds):
    query_times = np.unique(sp3_constellations[constellation]['gps_millis'])
    sp3_compare = sp3_constellations[constellation]
    sat_numbers = consts.NUMSATS[constellation]
    # Loading individual constellations separately to speed up the georinex process
    rinex_nav_constellation = rinex_nav_file.where('gnss_id', constellation, 'eq')
    est_positions = NavData()
    for time in query_times:
        trimmed_data = get_closest_rinex_data(time, rinex_nav_constellation)
        trimmed_data.sort('sv_id', inplace=True)
        est_positions_time = rinex_to_sv_states(time, trimmed_data)
        est_positions_time['gps_millis'] = time
        est_positions.concat(est_positions_time, inplace=True)
    all_times = len(np.unique(est_positions['gps_millis']))
    if len(np.unique(est_positions['sv_id'])) != len(np.unique(sp3_compare['sv_id'])):
        rx_num_sats = np.min([len(np.unique(est_positions['sv_id'])), len(np.unique(sp3_compare['sv_id']))])
    else:
        rx_num_sats = sat_numbers
    # Test that the errors aren't large
    x_err = np.ones([all_times, rx_num_sats])
    y_err = np.ones([all_times, rx_num_sats])
    z_err = np.ones([all_times, rx_num_sats])
    for row_idx, (gps_millis, _, sv_frame) in enumerate(est_positions.loop_time('gps_millis')):
        sp3_frame = sp3_compare.where('gps_millis', gps_millis, 'eq')
        sp3_sats = set(sp3_frame['sv_id'])
        est_sats = set(sv_frame['sv_id'])
        intersecting_sats = list(sp3_sats.intersection(est_sats))
        sp3_frame = sp3_frame.where('sv_id', intersecting_sats, 'eq')
        sv_frame = sv_frame.where('sv_id', intersecting_sats, 'eq')
        num_rx_sats_frame = len(intersecting_sats)
        sp3_frame.sort('sv_id', inplace=True)
        sv_frame.sort('sv_id', inplace=True)
        x_err[row_idx, :num_rx_sats_frame] = np.abs(sp3_frame['x_sv_m'] - sv_frame['x_sv_m'])
        y_err[row_idx, :num_rx_sats_frame] = np.abs(sp3_frame['y_sv_m'] - sv_frame['y_sv_m'])
        z_err[row_idx, :num_rx_sats_frame] = np.abs(sp3_frame['z_sv_m'] - sv_frame['z_sv_m'])

    assert np.max(x_err) < broadcast_error_bounds[constellation]
    assert np.max(y_err) < broadcast_error_bounds[constellation]
    assert np.max(z_err) < broadcast_error_bounds[constellation]

