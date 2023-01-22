"""Tests for the ephemeris class that returns satellite ephemerides.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "30 Aug 2022"

import os
import pytest
from datetime import datetime, timezone

from gnss_lib_py.parsers.ephemeris import EphemerisManager
from gnss_lib_py.parsers.navdata import NavData


@pytest.fixture(name="ephem_path", scope='session')
def fixture_ephem_path():
    """Location of ephemeris files for unit test

    Returns
    -------
    root_path : string
        Folder location containing Android Derived 2021 measurements
    """
    root_path = os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
    ephem_path = os.path.join(root_path, 'data/unit_test/ephemeris')
    # Remove old files in the ephemeris directory, this tests the
    # download component of EphemerisManager()
    for dir_name in os.listdir(ephem_path):
        dir_loc = os.path.join(ephem_path, dir_name)
        for file_name in os.listdir(dir_loc):
            file_loc = os.path.join(dir_loc, file_name)
            if os.path.isfile(file_loc):
                os.remove(file_loc)
    return ephem_path


@pytest.fixture(name="ephem_time", scope="module")
def fixture_ephem_time():
    """Time for which corresponding ephemeris files will be used.

    Returns
    -------
    start_time : datetime.datetime
        Time at which state estimation starts
    """
    start_time = datetime(2020, 5, 15, 0, 47, 48, tzinfo=timezone.utc)
    return start_time


@pytest.fixture(name="ephem", scope="module")
def fixture_get_ephem(ephem_path, ephem_time):
    """Create instance of Ephemeris manager and fetch ephemeris file.

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded to.
    ephem_time : datetime.datetime
        Time at which state estimation is starting, the most recent ephemeris
        file before the start time will be fetched.

    Returns
    -------
    ephem : gnss_lib_py.parsers.navdata.NavData
    """
    ephem_man = EphemerisManager(ephem_path)
    #TODO: Find out why GLONASS and GALILEO ephimerides are taking too
    # long to download
    # svs = ['G02', 'G11', 'R01', 'R02', 'E01', 'E02']
    svs = ['G02', 'G11']
    ephem = ephem_man.get_ephemeris(ephem_time, svs)
    return ephem


def test_ephem_type(ephem):
    """Test that ephem is of type gnss_lib_py.parsers.navdata.NavData

    Parameters
    ----------
    ephem : gnss_lib_py.parsers.navdata.NavData
        Broadcast satellite ephemerides
    """
    assert isinstance(ephem, NavData)

def test_is_rows(ephem):
    """Tests that NavData specific rows are present in ephem

    Parameters
    ----------
    ephem : gnss_lib_py.parsers.navdata.NavData
        Broadcast satellite ephemerides
    """
    navdata_rows = ['gps_millis', 'sv_id', 'gps_week']
    ephem.in_rows(navdata_rows)
