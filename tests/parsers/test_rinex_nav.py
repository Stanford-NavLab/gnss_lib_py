"""Tests for the ephemeris class that returns satellite ephemerides.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "30 Aug 2022"

import os
from datetime import datetime, timezone

import pytest
import numpy as np

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.parsers.rinex_nav import RinexNav, load_rinex_nav
import gnss_lib_py.parsers.rinex_nav as rinex_nav
from gnss_lib_py.utils.time_conversions import gps_millis_to_tow
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis


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


def test_rinex_nav_init(ephem_path):
    """Test across multiple files.

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded.

    """

    no_iono = [
               "BRDM00DLR_R_20130010000_01D_MN.rnx",
               "VOIM00MDG_R_20140010000_01D_MN.rnx",
               "BRDC00WRD_S_20230730000_01D_MN.rnx",
               "WTZS00DEU_R_20230800000_01D_MN.rnx",
               ]

    rinex_nav_dir = os.path.join(ephem_path,"rinex","nav")
    rinex_paths = []
    for file in os.listdir(rinex_nav_dir):
        rinex_path = os.path.join(rinex_nav_dir,file)
        rinex_paths.append(rinex_path)

        if file in no_iono:
            # no iono
            with pytest.warns(RuntimeWarning) as warns:
                RinexNav(rinex_path)
            assert len(warns) == 1
        else:
            RinexNav(rinex_path)

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
        file before the start time will be fetched.
    satellites : List
        List of satellites ['Const_IDSVID']

    """

    ephem_time = datetime_to_gps_millis(ephem_time)

    ephem = load_rinex_nav(ephem_time, satellites, ephem_path,
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

def test_no_iono_params(ephem_path):
    """Test no iono params.

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded to.

    """

    rinex_path = os.path.join(ephem_path,"rinex","nav",
                                   "BRDM00DLR_S_20230730000_01D_MN_no_gps_iono.rnx")
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

