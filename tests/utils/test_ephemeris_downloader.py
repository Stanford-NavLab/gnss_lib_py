"""Tests for the ephemeris class that returns satellite ephemerides.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "30 Aug 2022"

import os
import ftplib
from datetime import datetime, timezone, timedelta, time

import pytest
import requests
import numpy as np

import gnss_lib_py.utils.time_conversions as tc
import gnss_lib_py.utils.ephemeris_downloader as ed
# pylint: disable=protected-access

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
    ephem_path = os.path.join(root_path, 'data/unit_test/rinex_download')
    return ephem_path

def remove_download_eph(ephem_download_path):
    """Remove previous directory

    Parameters
    ----------
    ephem_download_path : string
        Location where ephemeris files are stored/to be downloaded.

    """
    # Remove old files in the ephemeris directory, this tests the
    # download component of EphemerisDownloader()
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

@pytest.mark.parametrize('ephem_datetime',
                         [
                          datetime(2023, 3, 14, 12, tzinfo=timezone.utc),
                          datetime.now().astimezone(timezone.utc),
                          [datetime(2018, 1, 14, 13, tzinfo=timezone.utc),
                           datetime(2021, 12, 20, 10, tzinfo=timezone.utc)],
                         ])
@pytest.mark.parametrize('constellations',
                         [
                          ["gps"],
                          {"glonass"},
                          np.array(["glonass","gps"]),
                         ])
def test_load_ephemeris_rinex_nav(ephem_datetime,constellations):
    """Test verifying ephemeris.

    Parameters
    ----------
    ephem_datetime : datetime.datetime
        Time for ephemeris check

    """
    gps_millis = tc.datetime_to_gps_millis(ephem_datetime)
    files = ed.load_ephemeris("rinex_nav",gps_millis,constellations,
                              verbose=True)


def test_extract_ephemeris_dates():
    """Test extracting the correct days from timestamps.

    Datetime format is: (Year, Month, Day, Hr, Min, Sec, Microsecond)

    """

    # check basic case
    noon_utc = datetime(2023, 7, 27, 12, 0, 0, 0, tzinfo=timezone.utc)
    dates = ed._extract_ephemeris_dates("rinex_nav", np.array([noon_utc]))
    assert dates == {datetime(2023, 7, 27).date()}

    # check that timezone conversion is happening
    eleven_pt = datetime(2023, 7, 26, 23, 0, 0, 0,
                    tzinfo=timezone(-timedelta(hours=8)))
    dates = ed._extract_ephemeris_dates("rinex_nav", np.array([eleven_pt]))
    assert dates == {datetime(2023, 7, 27).date()}

    # check add prev day if after before 2am
    for hour in [0,1,2]:
        two_am_utc = datetime(2023, 7, 27, hour, 0, 0, 0, tzinfo=timezone.utc)
        dates = ed._extract_ephemeris_dates("rinex_nav", np.array([two_am_utc]))
        assert dates == {datetime(2023, 7, 26).date(),
                         datetime(2023, 7, 27).date()}

    # check add next day if after 10pm
    for hour in [22,23]:
        ten_pm_utc = datetime(2023, 7, 26, hour, 0, 0, 0, tzinfo=timezone.utc)
        dates = ed._extract_ephemeris_dates("rinex_nav", np.array([ten_pm_utc]))
        assert dates == {datetime(2023, 7, 26).date(),
                         datetime(2023, 7, 27).date()}

    # check don't add next day if after 10pm on current day
    ten_pm_utc_today = datetime.combine(datetime.utcnow().date(),
                                        time(22,tzinfo=timezone.utc))
    dates = ed._extract_ephemeris_dates("rinex_nav", np.array([ten_pm_utc_today]))
    assert dates == {datetime.utcnow().date()}

    # check that across multiple days there aren't duplicates
    dates = ed._extract_ephemeris_dates("rinex_nav", np.array([noon_utc,
                                                          eleven_pt,
                                                          two_am_utc,
                                                          ten_pm_utc]))
    assert dates == {datetime(2023, 7, 26).date(),
                     datetime(2023, 7, 27).date()}

    # check for two separate days
    noon_utc_01 = datetime(2023, 7, 1, 12, 0, 0, 0, tzinfo=timezone.utc)
    dates = ed._extract_ephemeris_dates("rinex_nav", np.array([noon_utc,
                                                           noon_utc_01]))
    assert dates == {datetime(2023, 7, 27).date(),
                     datetime(2023, 7, 1).date()}

def test_ftp_errors(ephem_download_path):
    """Test FTP download errors.

    Parameters
    ----------
    ephem_download_path : string
        Location where ephemeris files are stored/to be downloaded.

    """
    remove_download_eph(ephem_download_path)
    os.makedirs(ephem_download_path)

    url = 'gdc.cddis.eosdis.nasa.gov'
    ftp_path = 'gnss/data/daily/2020/notafolder/notAfile.20n.Z'
    dest_filepath = os.path.join(ephem_download_path,'rinex',
                                 'nav',os.path.split(ftp_path)[1])
    # download the ephemeris file
    with pytest.raises(ftplib.error_perm) as excinfo:
        ed._ftp_download(url, ftp_path, dest_filepath)
    assert ftp_path in str(excinfo.value)

    remove_download_eph(ephem_download_path)

# @pytest.mark.parametrize('constellations',
#                          [
#                           set(['G']),
#                           set(['G','R','E']),
#                           None,
#                          ])
# @pytest.mark.parametrize('ephem_time',
#                          [
#                           datetime(2023, 3, 14, 23, 17, 13, tzinfo=timezone.utc),
#                          ])
# def test_load_ephem(ephem_path, ephem_time, constellations):
#     """Create instance of Ephemeris manager and fetch ephemeris file.
#
#     Check type and rows.
#
#     Parameters
#     ----------
#     ephem_path : string
#         Location where ephemeris files are stored/to be downloaded to.
#     ephem_time : datetime.datetime
#         Time at which state estimation is starting, the most recent ephemeris
#         file before the start time will be fetched.
#     constellations : List
#         List of constellation types
#
#     """
#
#     ephem_man = EphemerisDownloader(ephem_path, verbose=True)
#     rinex_paths = ephem_man.load_data(ephem_time, constellations, same_day=True)
#
#     # Test that ephem is of type gnss_lib_py.parsers.navdata.NavData
#     assert isinstance(rinex_paths, list)
#
#     for rinex_path in rinex_paths:
#         assert isinstance(rinex_path, (str, os.PathLike))
#
# @pytest.mark.parametrize('fileinfo',
#     [
#      {'filepath': '/IGS/BRDC/2023/099/BRDC00WRD_S_20230990000_01D_MN.rnx.gz', 'url': 'igs-ftp.bkg.bund.de'},
#      {'filepath': 'gnss/data/daily/2020/brdc/brdc1360.20n.Z', 'url': 'gdc.cddis.eosdis.nasa.gov'},
#      {'filepath': 'gnss/data/daily/2020/brdc/brdc1360.20g.Z', 'url': 'gdc.cddis.eosdis.nasa.gov'},
#      {'filepath': 'gnss/data/daily/2020/brdc/BRDC00IGS_R_20201360000_01D_MN.rnx.gz', 'url': 'gdc.cddis.eosdis.nasa.gov'},
#      {'filepath': 'gnss/data/daily/2023/brdc/brdc0730.23n.gz', 'url': 'gdc.cddis.eosdis.nasa.gov'},
#      {'filepath': 'gnss/data/daily/2023/brdc/brdc0730.23g.gz', 'url': 'gdc.cddis.eosdis.nasa.gov'},
#      {'filepath': 'gnss/data/daily/2023/brdc/BRDC00IGS_R_20230730000_01D_MN.rnx.gz', 'url': 'gdc.cddis.eosdis.nasa.gov'},
#      {'filepath': 'gnss/data/daily/2023/brdc/brdc0990.23n.gz', 'url': 'gdc.cddis.eosdis.nasa.gov'},
#      ])
# def test_download_ephem(ephem_download_path, fileinfo):
#     """Test FTP download for ephemeris files.
#
#     Parameters
#     ----------
#     ephem_download_path : string
#         Location where ephemeris files are stored/to be downloaded to.
#     fileinfo : dict
#         Filenames for ephemeris with ftp server and constellation details
#
#     """
#
#     remove_download_eph(ephem_download_path)
#     os.makedirs(ephem_download_path)
#
#     ephem_man = EphemerisDownloader(ephem_download_path, verbose=True)
#
#     filepath = fileinfo['filepath']
#     url = fileinfo['url']
#     directory = os.path.split(filepath)[0]
#     filename = os.path.split(filepath)[1]
#     dest_filepath = os.path.join(ephem_man.ephemeris_directory, 'rinex', filename)
#
#     # download the ephemeris file
#     try:
#         ephem_man.retrieve_file(url, directory, filename,
#                            dest_filepath)
#     except ftplib.error_perm as ftplib_exception:
#         print(ftplib_exception)
#
#     remove_download_eph(ephem_download_path)
#
# def download_igs(requests_url):
#     """Helper function to capture ConnectionError.
#
#     """
#     response = None
#     try:
#         response = requests.get(requests_url, timeout=5)
#     except requests.exceptions.ConnectionError:
#         print("ConnectionError.")
#
#     return response
#
# @pytest.mark.parametrize('fileinfo',
#     [
#      {'filepath': 'IGS/BRDC/2023/099/BRDC00WRD_S_20230990000_01D_MN.rnx.gz',
#        'url': 'http://igs.bkg.bund.de/root_ftp/'},
#      ])
# def test_request_igs(capsys, ephem_download_path, fileinfo):
#     """Test requests download for igs files.
#
#     The reason for this test is that Github workflow actions seem to
#     block international IPs and so won't properly bind to the igs IP.
#
#     Parameters
#     ----------
#     ephem_download_path : string
#         Location where ephemeris files are stored/to be downloaded to.
#     fileinfo : dict
#         Filenames for ephemeris with ftp server and constellation details
#
#     """
#
#     remove_download_eph(ephem_download_path)
#     os.makedirs(ephem_download_path)
#
#     ephem_man = EphemerisDownloader(ephem_download_path, verbose=True)
#
#     filepath = fileinfo['filepath']
#     filename = os.path.split(filepath)[1]
#     dest_filepath = os.path.join(ephem_man.ephemeris_directory, 'rinex', filename)
#
#     requests_url = fileinfo['url'] + fileinfo['filepath']
#
#     fail_count = 0
#     while fail_count < 3:# download the ephemeris file
#         response = download_igs(requests_url)
#         captured = capsys.readouterr()
#         if "ConnectionError." in captured.out:
#             fail_count += 1
#         else:
#             break
#
#     if response is None:
#         raise requests.exceptions.ConnectionError("IGS ConnectionError.")
#
#     with open(dest_filepath,'wb') as file:
#         file.write(response.content)
#
#     remove_download_eph(ephem_download_path)
#
