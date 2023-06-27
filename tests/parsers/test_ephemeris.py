"""Tests for the ephemeris class that returns satellite ephemerides.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "30 Aug 2022"

import os
import ftplib
import requests
from datetime import datetime, timezone

import pytest
import numpy as np
import pandas as pd

from gnss_lib_py.parsers.ephemeris import EphemerisManager
from gnss_lib_py.parsers.navdata import NavData
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

@pytest.fixture(name="ephem_download_path", scope='session')
def fixture_ephem_download_path():
    """Location of ephemeris files for unit test

    Returns
    -------
    ephem_download_path : string
        Location where ephemeris files are stored/to be downloaded.
    """
    root_path = os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
    ephem_path = os.path.join(root_path, 'data/unit_test/ephemeris_download')
    return ephem_path

def remove_download_eph(ephem_download_path):
    """Remove previous directory

    Parameters
    ----------
    ephem_download_path : string
        Location where ephemeris files are stored/to be downloaded.

    """
    # Remove old files in the ephemeris directory, this tests the
    # download component of EphemerisManager()
    if os.path.isdir(ephem_download_path):
        for dir_name in os.listdir(ephem_download_path):
            dir_loc = os.path.join(ephem_download_path, dir_name)
            if os.path.isdir(dir_loc):
                for file_name in os.listdir(dir_loc):
                    file_loc = os.path.join(dir_loc, file_name)
                    if os.path.isfile(file_loc):
                        os.remove(file_loc)
                os.rmdir(dir_loc)
        os.rmdir(ephem_download_path)

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
def test_get_ephem(ephem_path, ephem_time, satellites):
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

    ephem_man = EphemerisManager(ephem_path, verbose=True)
    if ephem_time.tzinfo is None:
        with pytest.warns(RuntimeWarning):
            ephem = ephem_man.get_ephemeris(ephem_time, satellites)
    else:
        ephem = ephem_man.get_ephemeris(ephem_time, satellites)


    # Test that ephem is of type gnss_lib_py.parsers.navdata.NavData
    assert isinstance(ephem, NavData)
    assert isinstance(ephem_man.data, pd.DataFrame)

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

    ephem_man = EphemerisManager(ephem_path, verbose=True)
    ephem = ephem_man.get_ephemeris(ephem_time, satellites)

    # Test that ephem is of type gnss_lib_py.parsers.navdata.NavData
    assert isinstance(ephem, NavData)
    assert isinstance(ephem_man.data, pd.DataFrame)

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

@pytest.mark.parametrize('constellations',
                         [
                          set(['G']),
                          set(['G','R','E']),
                          None,
                         ])
@pytest.mark.parametrize('ephem_time',
                         [
                          datetime(2023, 3, 14, 23, 17, 13, tzinfo=timezone.utc),
                         ])
def test_load_ephem(ephem_path, ephem_time, constellations):
    """Create instance of Ephemeris manager and fetch ephemeris file.

    Check type and rows.

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded to.
    ephem_time : datetime.datetime
        Time at which state estimation is starting, the most recent ephemeris
        file before the start time will be fetched.
    constellations : List
        List of constellation types

    """

    ephem_man = EphemerisManager(ephem_path, verbose=True)
    ephem_man.load_data(ephem_time, constellations, same_day=True)

    # Test that ephem is of type gnss_lib_py.parsers.navdata.NavData
    assert isinstance(ephem_man.data, pd.DataFrame)

@pytest.mark.parametrize('fileinfo',
    [
     {'filepath': '/IGS/BRDC/2023/099/BRDC00WRD_S_20230990000_01D_MN.rnx.gz', 'url': 'igs-ftp.bkg.bund.de'},
     {'filepath': 'gnss/data/daily/2020/brdc/brdc1360.20n.Z', 'url': 'gdc.cddis.eosdis.nasa.gov'},
     {'filepath': 'gnss/data/daily/2020/brdc/brdc1360.20g.Z', 'url': 'gdc.cddis.eosdis.nasa.gov'},
     {'filepath': 'gnss/data/daily/2020/brdc/BRDC00IGS_R_20201360000_01D_MN.rnx.gz', 'url': 'gdc.cddis.eosdis.nasa.gov'},
     {'filepath': 'gnss/data/daily/2023/brdc/brdc0730.23n.gz', 'url': 'gdc.cddis.eosdis.nasa.gov'},
     {'filepath': 'gnss/data/daily/2023/brdc/brdc0730.23g.gz', 'url': 'gdc.cddis.eosdis.nasa.gov'},
     {'filepath': 'gnss/data/daily/2023/brdc/BRDC00IGS_R_20230730000_01D_MN.rnx.gz', 'url': 'gdc.cddis.eosdis.nasa.gov'},
     {'filepath': 'gnss/data/daily/2023/brdc/brdc0990.23n.gz', 'url': 'gdc.cddis.eosdis.nasa.gov'},
     ])
def test_download_ephem(ephem_download_path, fileinfo):
    """Test FTP download for ephemeris files.

    Parameters
    ----------
    ephem_download_path : string
        Location where ephemeris files are stored/to be downloaded to.
    fileinfo : dict
        Filenames for ephemeris with ftp server and constellation details

    """

    remove_download_eph(ephem_download_path)
    os.makedirs(ephem_download_path)

    ephem_man = EphemerisManager(ephem_download_path, verbose=True)

    filepath = fileinfo['filepath']
    url = fileinfo['url']
    directory = os.path.split(filepath)[0]
    filename = os.path.split(filepath)[1]
    if url == 'igs-ftp.bkg.bund.de':
        dest_filepath = os.path.join(ephem_man.data_directory, 'igs', filename)
    else:
        dest_filepath = os.path.join(ephem_man.data_directory, 'nasa', filename)

    # download the ephemeris file
    try:
        ephem_man.retrieve_file(url, directory, filename,
                           dest_filepath)
    except ftplib.error_perm as ftplib_exception:
        print(ftplib_exception)

    remove_download_eph(ephem_download_path)

@pytest.mark.parametrize('fileinfo',
    [
     {'filepath': 'IGS/BRDC/2023/099/BRDC00WRD_S_20230990000_01D_MN.rnx.gz',
       'url': 'http://igs.bkg.bund.de/root_ftp/'},
     ])
def test_request_igs(ephem_download_path, fileinfo):
    """Test requests download for igs files.

    The reason for this test is that Github workflow actions seem to
    block international IPs and so won't properly bind to the igs IP.

    Parameters
    ----------
    ephem_download_path : string
        Location where ephemeris files are stored/to be downloaded to.
    fileinfo : dict
        Filenames for ephemeris with ftp server and constellation details

    """

    remove_download_eph(ephem_download_path)
    os.makedirs(ephem_download_path)

    ephem_man = EphemerisManager(ephem_download_path, verbose=True)

    filepath = fileinfo['filepath']
    filename = os.path.split(filepath)[1]
    dest_filepath = os.path.join(ephem_man.data_directory, 'igs', filename)

    requests_url = fileinfo['url'] + fileinfo['filepath']


    fail_count = 0
    while fail_count < 3:
        try:
            response = requests.get(requests_url, timeout=5)
            break
        except ConnectionError:
            fail_count += 1

    with open(dest_filepath,'wb') as file:
        file.write(response.content)

    remove_download_eph(ephem_download_path)

def test_ftp_errors(ephem_download_path):
    """Test FTP download errors.

    Parameters
    ----------
    ephem_download_path : string
        Location where ephemeris files are stored/to be downloaded.

    """
    remove_download_eph(ephem_download_path)
    os.makedirs(ephem_download_path)

    fileinfo = {'filepath': 'gnss/data/daily/2020/brdc/notAfile.20n.Z', 'url': 'gdc.cddis.eosdis.nasa.gov'}

    ephem_man = EphemerisManager(ephem_download_path, verbose=True)

    filepath = fileinfo['filepath']
    url = fileinfo['url']
    directory = os.path.split(filepath)[0]
    filename = os.path.split(filepath)[1]
    dest_filepath = os.path.join(ephem_man.data_directory, 'nasa', filename)

    # download the ephemeris file
    with pytest.raises(ftplib.error_perm) as excinfo:
        ephem_man.retrieve_file(url, directory, filename,
                                dest_filepath)
    assert directory in str(excinfo.value)
    assert filename in str(excinfo.value)

    remove_download_eph(ephem_download_path)

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

    ephem_man = EphemerisManager(ephem_path, verbose=True)
    with pytest.raises(RuntimeError) as excinfo:
        ephem_man.get_ephemeris(ephem_time, satellites)
    assert "ephemeris data" in str(excinfo.value)

def test_get_constellation(ephem_path):
    """Test get constellation.

    Check type and rows.

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded to.

    """

    ephem_man = EphemerisManager(ephem_path, verbose=True)

    assert ephem_man.get_constellations(set('R')) is None

def test_load_leapseconds(ephem_path):
    """Test load leapseconds.

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded to.

    """

    ephem_man = EphemerisManager(ephem_path, verbose=True)

    # check what happens when a file with an incomplete header is passed
    incomplete_path = os.path.join(ephem_path,"nasa",
                                   "brdc1370_incomplete.20n")
    assert ephem_man.load_leapseconds(incomplete_path) is None
