"""Tests for the ephemeris class that returns satellite ephemerides.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "30 Aug 2022"

import os
from datetime import datetime, timezone

import pytest
import numpy as np

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.parsers.rinex import Rinex, get_time_cropped_rinex
from gnss_lib_py.utils.time_conversions import gps_millis_to_tow


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
    ephem_path = os.path.join(root_path, 'data/unit_test/ephemeris')
    return ephem_path


@pytest.mark.parametrize('satellites',
                         [
                          ['G01'],
                          ['R01'],
                          ['E02'],
                          ['G01','R01'],
                          ['G01','R01','E02'],
                         ])
@pytest.mark.parametrize('ephem_time',
                         [
                          datetime(2020, 5, 15, 3, 47, 48),
                          datetime(2023, 3, 14, 23, 17, 13, tzinfo=timezone.utc),
                         ])
def test_get_time_cropped_rinex(ephem_path, ephem_time, satellites):
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

    if ephem_time.tzinfo is None:
        with pytest.warns(RuntimeWarning):
            ephem = get_time_cropped_rinex(ephem_time, satellites, ephem_path)
    else:
        ephem = get_time_cropped_rinex(ephem_time, satellites, ephem_path)


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
                          ['G01'],
                          ['E01'],
                          ['G01','R01'],
                         ])
@pytest.mark.parametrize('ephem_time',
                         [
                          datetime(2020, 5, 16, 0, 17, 1, tzinfo=timezone.utc),
                         ])
def test_prev_ephem(ephem_path, ephem_time, satellites):
    """Test scenario when timestamp is near after midnight.

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

    ephem = get_time_cropped_rinex(ephem_time, satellites, ephem_path)

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

    with pytest.raises(RuntimeError) as excinfo:
        get_time_cropped_rinex(ephem_time, satellites, ephem_path)
    assert "ephemeris data" in str(excinfo.value)


def test_load_leapseconds(ephem_path):
    """Test load leapseconds.

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded to.

    """

    rinex_path = os.path.join(ephem_path,"nasa",
                                   "brdc1370.20n")
    rinex_data = Rinex(rinex_path)
    assert rinex_data['leap_seconds'] == 18

    # check what happens when a file with an incomplete header is passed
    incomplete_path = os.path.join(ephem_path,"nasa",
                                   "brdc1370_no_leapseconds.20n")
    rinex_data = Rinex(incomplete_path)
    assert len(rinex_data.where('leap_seconds',np.nan)) == 1

    # check what happens when a file with an incomplete header is passed
    incomplete_path = os.path.join(ephem_path,"nasa",
                                   "brdc1370_incomplete.20n")
    assert rinex_data.load_leapseconds(incomplete_path) == None
