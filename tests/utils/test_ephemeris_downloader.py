"""Tests for the ephemeris class that returns satellite ephemerides.

"""

__authors__ = "Ashwin Kanhere, Derek Knowles"
__date__ = "30 Aug 2022"

import os
import ftplib
import gzip
from datetime import datetime, timezone, timedelta, time

import pytest
import requests
import numpy as np

from conftest import lazy_fixture
import gnss_lib_py.utils.time_conversions as tc
import gnss_lib_py.utils.ephemeris_downloader as ed
# pylint: disable=protected-access,too-few-public-methods

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

@pytest.fixture(name="all_ephem_paths", scope='session')
def fixture_all_ephem_paths(ephem_path):
    """Location of ephemeris files for unit test

    Parameters
    ----------
    ephem_path : string
        Location where ephemeris files are stored/to be downloaded.

    Returns
    -------
    all_ephem_paths : string
        Location of all unit test ephemeris files.

    """

    all_ephem_paths = []

    ephem_dirs = [
                       os.path.join(ephem_path,"rinex","nav"),
                       os.path.join(ephem_path,"sp3"),
                       os.path.join(ephem_path,"clk"),
                      ]
    for ephem_dir in ephem_dirs:
        all_ephem_paths += [os.path.join(ephem_dir,file) \
                            for file in os.listdir(ephem_dir)]

    return all_ephem_paths

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
    ephem_download_path = os.path.join(root_path, 'data/unit_test/ephemeris_downloader_tests')
    return ephem_download_path

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
        for file_dir_name in os.listdir(ephem_download_path):
            file_dir_loc = os.path.join(ephem_download_path, file_dir_name)
            if os.path.isdir(file_dir_loc):
                remove_download_eph(file_dir_loc)
            elif os.path.isfile(file_dir_loc):
                os.remove(file_dir_loc)
        os.rmdir(ephem_download_path)

@pytest.mark.parametrize('ephem_params',
                         [
                          (datetime(2020, 5, 16, 0, 17, 1, tzinfo=timezone.utc),
                           ["gps"]),
                          (datetime(2020, 5, 16, 4, 17, 1, tzinfo=timezone.utc),
                           ["glonass"]),
                          (datetime(2013, 1, 1, 15, 31, 59, tzinfo=timezone.utc),
                           ["qzss"]),
                          (datetime(2023, 3, 14, 8, 12, 34, tzinfo=timezone.utc),
                           ["galileo","sbas","irnss","beidou"]),
                         ])
@pytest.mark.parametrize('paths',
                         [
                          None,
                          "",
                          lazy_fixture("all_ephem_paths"),
                         ])
def test_load_ephemeris_rinex_nav(ephem_params, ephem_path, paths):
    """Test verifying ephemeris.

    Parameters
    ----------
    ephem_params : tuple
        Tuple of datetime.datetime for ephemeris clock time and
        constellation list for which to download ephemeris.
    ephem_path : string or path-like
        Directory where ephemeris files are downloaded if necessary.
    paths : string or path-like
        Paths to existing ephemeris files if they exist.

    """
    ephem_datetime,constellations = ephem_params
    gps_millis = tc.datetime_to_gps_millis(ephem_datetime)
    paths = ed.load_ephemeris("rinex_nav",gps_millis,constellations,
                              file_paths=paths,
                              download_directory=ephem_path,
                              verbose=True)

    # assert files in paths are not empty
    for path in paths:
        assert os.path.getsize(path) > 0.

@pytest.mark.parametrize('ephem_params',
                         [
                          (datetime(2020, 5, 17, 11, 17, 1, tzinfo=timezone.utc),
                           ["gps"]),
                          (datetime(2020, 5, 17, 19, 17, 1, tzinfo=timezone.utc),
                           ["glonass"]),
                          (datetime(2020, 5, 18, 11, 17, 1, tzinfo=timezone.utc),
                           ["gps"]),
                          (datetime(2020, 5, 18, 19, 17, 1, tzinfo=timezone.utc),
                           ["glonass"]),
                          (datetime(2014, 1, 1, 15, 31, 59, tzinfo=timezone.utc),
                           ["qzss"]),
                          (datetime(2023, 3, 21, 8, 12, 34, tzinfo=timezone.utc),
                           ["galileo","sbas","irnss","beidou"]),
                          (datetime(2023, 3, 21, 8, 12, 34, tzinfo=timezone.utc),
                           ["gps"]),
                         ])
@pytest.mark.parametrize('paths',
                         [
                          lazy_fixture("all_ephem_paths"),
                         ])
def test_rinex_different_monument(ephem_params, ephem_path, paths):
    """Test using a different monument.

    Parameters
    ----------
    ephem_params : tuple
        Tuple of datetime.datetime for ephemeris clock time and
        constellation list for which to download ephemeris.
    ephem_path : string or path-like
        Directory where ephemeris files are downloaded if necessary.
    paths : string or path-like
        Paths to existing ephemeris files if they exist.

    """
    ephem_datetime,constellations = ephem_params
    gps_millis = tc.datetime_to_gps_millis(ephem_datetime)
    paths = ed.load_ephemeris("rinex_nav",gps_millis,constellations,
                              file_paths=paths,
                              download_directory=ephem_path,
                              verbose=True)

    # assert files in paths are not empty
    for path in paths:
        assert os.path.getsize(path) > 0.

@pytest.mark.parametrize('ephem_datetime',
                         [
                          datetime(2021, 4, 28, 12, tzinfo=timezone.utc),
                         ])
@pytest.mark.parametrize('file_type',
                         [
                          "sp3",
                          "clk",
                         ])
@pytest.mark.parametrize('paths',
                         [
                          None,
                          lazy_fixture("all_ephem_paths"),
                         ])
def test_load_sp3_clk(ephem_datetime, file_type, ephem_path, paths):
    """Test verifying ephemeris.

    Parameters
    ----------
    ephem_datetime : datetime.datetime
        Days in UTC for which ephemeris needs to be retrieved.
    file_type : string
        File type to download either "rinex_nav", "sp3", or "clk".
    ephem_path : string or path-like
        Directory where ephemeris files are downloaded if necessary.
    paths : string or path-like
        Paths to existing ephemeris files if they exist.

    """
    gps_millis = tc.datetime_to_gps_millis(ephem_datetime)
    paths = ed.load_ephemeris(file_type,gps_millis,None,
                              file_paths=paths,
                              download_directory=ephem_path,
                              verbose=True)

    # assert files in paths are not empty
    for path in paths:
        assert os.path.getsize(path) > 0.

@pytest.mark.parametrize('ephem_datetime',
                         [
                          datetime(2017, 3, 14, 12, tzinfo=timezone.utc),
                          datetime(2020, 5, 17, 11, 17, 1, tzinfo=timezone.utc),
                          datetime(2021, 4, 28, 12, tzinfo=timezone.utc),
                          datetime(2023, 3, 14, 11, 17, 1, tzinfo=timezone.utc),
                         ])
@pytest.mark.parametrize('possible_types',
                         [
                          ["sp3_rapid_CODE"],
                          ["sp3_rapid_GFZ"],
                          ["sp3_final_CODE"],
                          ["sp3_short_CODE"],
                         ])
@pytest.mark.parametrize('paths',
                         [
                          lazy_fixture("all_ephem_paths"),
                         ])
def test_sp3_different_source(ephem_datetime, possible_types, paths):
    """Test using a different source for sp3.

    Parameters
    ----------
    ephem_datetime : datetime.datetime
        Ephemeris clock time
    possible_types : list
        What file types would fulfill the requirement in preference
        order.
    paths : string or path-like
        Paths to existing ephemeris files if they exist.

    """

    valid, file = ed._valid_ephemeris_in_paths(ephem_datetime.date(),
                                               possible_types,
                                               paths)
    assert valid
    assert os.path.getsize(file) > 0.

@pytest.mark.parametrize('ephem_datetime',
                         [
                          datetime(2017, 3, 14, 12, tzinfo=timezone.utc),
                          datetime(2018, 3, 14, 12, tzinfo=timezone.utc),
                          datetime(2018, 12, 30, 12, tzinfo=timezone.utc),
                          datetime(2019, 3, 14, 12, tzinfo=timezone.utc),
                          datetime(2020, 5, 17, 11, 17, 1, tzinfo=timezone.utc),
                          datetime(2021, 4, 28, 12, tzinfo=timezone.utc),
                          datetime(2023, 3, 14, 11, 17, 1, tzinfo=timezone.utc),
                         ])
@pytest.mark.parametrize('possible_types',
                         [
                          ["clk_rapid_CODE"],
                          ["clk_rapid_GFZ"],
                          ["clk_final_CODE"],
                          ["clk_final_WUM"],
                          ["clk_short_GFZ"],
                          ["clk_short_WUM"],
                          ["clk_short_CODE"],
                         ])
@pytest.mark.parametrize('paths',
                         [
                          lazy_fixture("all_ephem_paths"),
                         ])
def test_clk_different_source(ephem_datetime, possible_types, paths):
    """Test using a different source for clk.

    Parameters
    ----------
    ephem_datetime : datetime.datetime
        Ephemeris clock time
    possible_types : list
        What file types would fulfill the requirement in preference
        order.
    paths : string or path-like
        Paths to existing ephemeris files if they exist.

    """
    valid, file = ed._valid_ephemeris_in_paths(ephem_datetime.date(),
                                               possible_types,
                                               paths)
    assert valid
    assert os.path.getsize(file) > 0.

    # test version without any paths added
    valid, file = ed._valid_ephemeris_in_paths(ephem_datetime.date(),
                                               possible_types)
    assert not valid

def test_extract_ephemeris_dates():
    """Test extracting the correct days from timestamps.

    Datetime format is: (Year, Month, Day, Hr, Min, Sec, Microsecond)

    """

    # check basic case
    noon_utc = datetime(2023, 7, 27, 12, 0, 0, 0, tzinfo=timezone.utc)
    dates = ed._extract_ephemeris_dates("rinex_nav", np.array([noon_utc]))
    assert dates == [datetime(2023, 7, 27).date()]

    # check that timezone conversion is happening
    eleven_pt = datetime(2023, 7, 26, 23, 0, 0, 0,
                    tzinfo=timezone(-timedelta(hours=8)))
    dates = ed._extract_ephemeris_dates("rinex_nav", np.array([eleven_pt]))
    assert dates == [datetime(2023, 7, 27).date()]

    # check add prev day if after before 2am
    for hour in [0,1,2]:
        two_am_utc = datetime(2023, 7, 27, hour, 0, 0, 0, tzinfo=timezone.utc)
        dates = ed._extract_ephemeris_dates("rinex_nav", np.array([two_am_utc]))
        assert dates == [datetime(2023, 7, 26).date(),
                         datetime(2023, 7, 27).date()]

    # check add next day if after 10pm
    for hour in [22,23]:
        ten_pm_utc = datetime(2023, 7, 26, hour, 0, 0, 0, tzinfo=timezone.utc)
        dates = ed._extract_ephemeris_dates("rinex_nav", np.array([ten_pm_utc]))
        assert dates == [datetime(2023, 7, 26).date(),
                         datetime(2023, 7, 27).date()]

    # check don't add next day if after 10pm on current day
    ten_pm_utc_today = datetime.combine(datetime.now(timezone.utc).date(),
                                        time(22,tzinfo=timezone.utc))
    dates = ed._extract_ephemeris_dates("rinex_nav", np.array([ten_pm_utc_today]))
    assert dates == [datetime.now(timezone.utc).date()]

    # check that across multiple days there aren't duplicates
    dates = ed._extract_ephemeris_dates("rinex_nav", np.array([noon_utc,
                                                          eleven_pt,
                                                          two_am_utc,
                                                          ten_pm_utc]))
    assert dates == [datetime(2023, 7, 26).date(),
                     datetime(2023, 7, 27).date()]

    # test order is maintained in sorted order
    dates = ed._extract_ephemeris_dates("rinex_nav", np.array([noon_utc + timedelta(days=1),
                                                               noon_utc,
                                                               ]))
    assert dates == [datetime(2023, 7, 27).date(),
                     datetime(2023, 7, 28).date()]

    # check for two separate days
    noon_utc_01 = datetime(2023, 7, 1, 12, 0, 0, 0, tzinfo=timezone.utc)
    dates = ed._extract_ephemeris_dates("rinex_nav", np.array([noon_utc,
                                                           noon_utc_01]))
    assert dates == [datetime(2023, 7, 1).date(),
                     datetime(2023, 7, 27).date()]

def test_ftp_errors(tmp_path):
    """Test FTP download errors.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for test files.

    """
    url = 'gdc.cddis.eosdis.nasa.gov'
    ftp_path = 'gnss/data/daily/2020/notafolder/notAfile.20n.Z'
    dest_filepath = os.path.join(str(tmp_path),
                                 os.path.split(ftp_path)[1])
    # download the ephemeris file
    with pytest.raises(ftplib.error_perm) as excinfo:
        ed._ftp_download(url, ftp_path, dest_filepath)
    assert ftp_path in str(excinfo.value)
    assert not os.path.exists(dest_filepath)

def test_ftp_download_retry(monkeypatch, tmp_path):
    """Test _ftp_download retries on transient errors.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary directory for test files.

    """
    attempts = 0
    def mock_login(_url, _secure=False):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise EOFError("Connection closed unexpectedly")
        mock_ftp_obj = ftplib.FTP()
        mock_ftp_obj.retrbinary = lambda cmd, callback: callback(b"test")
        mock_ftp_obj.quit = lambda: None
        mock_ftp_obj.close = lambda: None
        return mock_ftp_obj

    monkeypatch.setattr(ed, "_ftp_login", mock_login)
    monkeypatch.setattr(ed, "_decompress_file", lambda path: None)

    dest = os.path.join(str(tmp_path), "test_file.gz")
    ed._ftp_download("example.com", "/path/to/file.gz", dest, verbose=True, retries=3)
    assert attempts == 2
    assert os.path.exists(dest)

def test_ftp_download_retry_failure(monkeypatch, tmp_path):
    """Test _ftp_download raises when all retries fail.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary directory for test files.

    """
    def mock_login(_url, _secure=False):
        raise TimeoutError("Connection timed out")

    monkeypatch.setattr(ed, "_ftp_login", mock_login)
    dest = os.path.join(str(tmp_path), "test_file.gz")
    with open(dest, "w", encoding="utf-8") as file_handle:
        file_handle.write("partial")
    with pytest.raises(TimeoutError):
        ed._ftp_download("example.com", "/path/to/file.gz", dest, verbose=True, retries=2)
    assert not os.path.exists(dest)

def test_http_errors(tmp_path):
    """Test HTTP download errors.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for test files.

    """
    url = 'https://igs.bkg.bund.de/root_ftp'
    file_path = '/IGS/BRDC/2020/notafolder/notAfile.rnx.gz'
    dest_filepath = os.path.join(str(tmp_path),
                                 os.path.split(file_path)[1])
    # create dummy file to ensure it gets cleaned up on failure
    with open(dest_filepath, 'w', encoding='utf-8') as handle:
        handle.write('dummy')

    with pytest.raises(requests.exceptions.RequestException) as excinfo:
        ed._http_download(url, file_path, dest_filepath, verbose=True)
    assert file_path in str(excinfo.value)
    assert not os.path.exists(dest_filepath)

def test_http_download_success(monkeypatch, tmp_path):
    """Test successful _http_download and file decompression.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary directory for test files.

    """
    compressed_content = gzip.compress(b"mocked rinex nav content")
    class MockResponse:
        """Mock HTTP response."""
        content = compressed_content
        status_code = 200
        def raise_for_status(self):
            """Mock raise_for_status."""

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())
    dest_path = os.path.join(str(tmp_path), "test_file.rnx.gz")
    ed._http_download("https://igs.bkg.bund.de/root_ftp", "/path/to/test_file.rnx.gz",
                      dest_path, verbose=True)
    decompressed_path = os.path.join(str(tmp_path), "test_file.rnx")
    assert os.path.exists(decompressed_path)
    with open(decompressed_path, "rb") as file_handle:
        assert file_handle.read() == b"mocked rinex nav content"

def test_download_ephemeris_http(monkeypatch, tmp_path):
    """Test _download_ephemeris routes HTTP URLs to _http_download.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary directory for test files.

    """
    downloaded = []
    def mock_http(_url, _path, dest, _verbose=False):
        downloaded.append(dest)
        decompressed = dest.rsplit('.', 1)[0]
        with open(decompressed, "w", encoding="utf-8") as file_handle:
            file_handle.write("content")

    monkeypatch.setattr(ed, "_http_download", mock_http)
    needed = [("https://example.com", "/dir/file.gz")]
    res = ed._download_ephemeris("rinex_nav", needed, str(tmp_path), verbose=True)
    assert len(res) == 1
    assert res[0].endswith("file")

def test_decompress_file_unix_compress(monkeypatch, tmp_path):
    """Test _decompress_file with .Z extension using unlzw3.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary directory for test files.

    """
    monkeypatch.setattr(ed.unlzw3, "unlzw", lambda _data: b"decompressed z content")
    z_file = os.path.join(str(tmp_path), "sample.Z")
    with open(z_file, "wb") as file_handle:
        file_handle.write(b"compressed z data")
    ed._decompress_file(z_file, remove_compressed=True)
    decompressed = os.path.join(str(tmp_path), "sample")
    assert os.path.exists(decompressed)
    assert not os.path.exists(z_file)
    with open(decompressed, "rb") as file_handle:
        assert file_handle.read() == b"decompressed z content"

def test_ftp_login_branches(monkeypatch):
    """Test _ftp_login both secure (TLS) and insecure branches.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.

    """
    tls_calls = []
    class MockFTPTLS:
        """Mock FTP_TLS object."""
        def __init__(self, host):
            self.host = host
        def login(self):
            """Mock login."""
            tls_calls.append("login")
        def prot_p(self):
            """Mock prot_p."""
            tls_calls.append("prot_p")

    insecure_calls = []
    class MockFTP:
        """Mock FTP object."""
        def __init__(self, host):
            self.host = host
        def login(self):
            """Mock login."""
            insecure_calls.append("login")

    monkeypatch.setattr(ed, "FTP_TLS", MockFTPTLS)
    monkeypatch.setattr(ed, "FTP", MockFTP)

    ftp_tls = ed._ftp_login("gdc.cddis.eosdis.nasa.gov", secure=True)
    assert tls_calls == ["login", "prot_p"]
    assert ftp_tls.host == "gdc.cddis.eosdis.nasa.gov"

    ftp_plain = ed._ftp_login("ftp.example.com", secure=False)
    assert insecure_calls == ["login"]
    assert ftp_plain.host == "ftp.example.com"

def test_ftp_download(ephem_download_path):
    """Test FTP download for ephemeris files.

    Parameters
    ----------
    ephem_download_path : string
        Location where ephemeris files are stored/to be downloaded to.

    """
    os.makedirs(ephem_download_path, exist_ok=True)

    past_dt = (datetime.now() - timedelta(days=2)).astimezone(timezone.utc)

    valid, igs_file = ed._valid_ephemeris_in_paths(past_dt.date(),
                                               ["rinex_nav_today"],
                                               file_paths=[])
    assert not valid
    valid, gps_file = ed._valid_ephemeris_in_paths(datetime(2017,3,14).date(),
                                               ["rinex_nav_gps"],
                                               file_paths=[])
    assert not valid
    needed_files = [igs_file, gps_file]

    try:
        paths = ed._download_ephemeris("rinex_nav",needed_files,
                                       ephem_download_path,
                                       verbose=True)

        # assert files in paths are not empty
        for path in paths:
            assert os.path.getsize(path) > 1E5
    except (ftplib.error_perm, requests.exceptions.RequestException) as net_exception:
        print(net_exception)

    # CODE rapid
    past_dt = (datetime.now() - timedelta(days=1,hours=12)).astimezone(timezone.utc)
    for file_type in ["sp3","clk"]:
        paths = ed.load_ephemeris(file_type,
                                  tc.datetime_to_gps_millis(past_dt),
                                  download_directory=ephem_download_path,
                                  verbose=True)
        assert os.path.getsize(paths[0]) > 1E5
        # add coverage of paths for the recent files
        ed.load_ephemeris(file_type,
                          tc.datetime_to_gps_millis(past_dt),
                          download_directory=ephem_download_path,
                          file_paths=paths,
                          verbose=True)

    # GFZ rapid
    past_dt = (datetime.now() - timedelta(days=5)).astimezone(timezone.utc)
    for file_type in ["sp3","clk"]:
        paths = ed.load_ephemeris(file_type,
                                  tc.datetime_to_gps_millis(past_dt),
                                  download_directory=ephem_download_path,
                                  verbose=True)
        assert os.path.getsize(paths[0]) > 1E5
        # add coverage of paths for the recent files
        ed.load_ephemeris(file_type,
                          tc.datetime_to_gps_millis(past_dt),
                          download_directory=ephem_download_path,
                          file_paths=paths,
                          verbose=True)

def test_load_ephemeris_fallbacks(monkeypatch, tmp_path):
    """Test fallback logic when preferred providers are missing.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary directory for test files.

    """
    def mock_ftp_download(_url, ftp_path, dest_filepath, _verbose=False, _retries=3):
        if any(prov in ftp_path for prov in ["WUM0MGXFIN", "wum", "BRDM00DLR"]):
            raise ftplib.error_perm(f"550 {ftp_path}: No such file")
        decompressed = dest_filepath.rsplit('.', 1)[0]
        with open(decompressed, "w", encoding="utf-8") as file_handle:
            file_handle.write("dummy")

    monkeypatch.setattr(ed, "_ftp_download", mock_ftp_download)

    # test when WUM0MGXFIN doesn't exist and is replaced
    paths = ed.load_ephemeris("clk",
                              tc.datetime_to_gps_millis(datetime(2020,6,7,12,
                                                        tzinfo=timezone.utc)),
                              download_directory=str(tmp_path),
                              verbose=True)
    assert "GFZ0MGXRAP" in paths[0]
    assert os.path.exists(paths[0])

    # test when wum doesn't exist and is replaced
    paths = ed.load_ephemeris("clk",
                              tc.datetime_to_gps_millis(datetime(2018,7,15,12,
                                                        tzinfo=timezone.utc)),
                              download_directory=str(tmp_path),
                              verbose=True)
    assert "gbm" in paths[0]
    assert os.path.exists(paths[0])

    # test when BRDM00DLR_R doesn't exist and is replaced
    paths = ed.load_ephemeris("rinex_nav",
                              tc.datetime_to_gps_millis(datetime(2013,12,3,12,
                                                        tzinfo=timezone.utc)),
                              download_directory=str(tmp_path),
                              verbose=True)
    assert "BRDC00IGS" in paths[0]
    assert os.path.exists(paths[0])

    # test when BRDM00DLR_S doesn't exist and is replaced
    paths = ed.load_ephemeris("rinex_nav",
                              tc.datetime_to_gps_millis(datetime(2019,11,26,12,
                                                        tzinfo=timezone.utc)),
                              download_directory=str(tmp_path),
                              verbose=True)
    assert "BRDC00IGS" in paths[0]
    assert os.path.exists(paths[0])

def test_load_ephemeris_fallback_third_attempt(monkeypatch, tmp_path):
    """Test 3rd fallback attempt to GPS/GLONASS when both BRDM and BRDC are missing.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary directory for test files.

    """
    def mock_ftp_download(_url, ftp_path, dest_filepath, _verbose=False, _retries=3):
        if "BRDM00DLR" in ftp_path or "BRDC00IGS" in ftp_path:
            raise ftplib.error_perm(f"550 {ftp_path}: No such file")
        decompressed = dest_filepath.rsplit('.', 1)[0]
        with open(decompressed, "w", encoding="utf-8") as file_handle:
            file_handle.write("dummy")

    monkeypatch.setattr(ed, "_ftp_download", mock_ftp_download)

    paths = ed.load_ephemeris("rinex_nav",
                              tc.datetime_to_gps_millis(datetime(2013, 12, 3, 12,
                                                        tzinfo=timezone.utc)),
                              download_directory=str(tmp_path),
                              verbose=True)
    assert len(paths) == 2
    assert any("13n" in p for p in paths)
    assert any("13g" in p for p in paths)

def download_igs(requests_url):
    """Helper function to capture ConnectionError.

    Parameters
    ----------
    requests_url : string
        URL for requests input.

    """
    response = None
    try:
        response = requests.get(requests_url, timeout=5)
    except requests.exceptions.ConnectionError:
        print("ConnectionError.")

    return response

def test_request_igs(capsys, tmp_path):
    """Test requests download for igs files.

    The reason for this test is that Github workflow actions seem to
    block international IPs and so won't properly bind to the igs IP.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Location where ephemeris files are stored/downloaded.

    """
    today_dt = datetime.now(timezone.utc)
    gps_millis_today = tc.datetime_to_gps_millis(today_dt)

    _, needed_files = ed._verify_ephemeris("rinex_nav",
                                            gps_millis_today)

    url, ftp_path = needed_files[0]
    requests_url = url.rstrip('/') + '/' + ftp_path.lstrip('/')

    filename = os.path.split(ftp_path)[1]
    dest_filepath = os.path.join(str(tmp_path), filename)

    fail_count = 0
    while fail_count < 3:# download the ephemeris file
        response = download_igs(requests_url)
        captured = capsys.readouterr()
        if "ConnectionError." in captured.out:
            fail_count += 1
        else:
            break

    if response is None:
        pytest.skip("IGS ConnectionError: Server unreachable from current network.")

    with open(dest_filepath,'wb') as file:
        file.write(response.content)

def test_ephemeris_fails():
    """Test ways the ephemeris downloader should fail.

    """

    with pytest.raises(RuntimeError) as excinfo:
        ed._extract_ephemeris_dates("rinex_obs",
                                    np.array([datetime.now().astimezone(timezone.utc)]))
    assert "file_type" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        ed._valid_ephemeris_in_paths(datetime.now().date(),"rinex_nav")
    assert "possible_type" in str(excinfo.value)

def test_verify_ephemeris():
    """Test verify ephemeris for today's.

    """

    gps_millis = tc.datetime_to_gps_millis(datetime.now().astimezone(timezone.utc))

    existing_paths, needed_files = ed._verify_ephemeris("rinex_nav",
                                                        gps_millis,
                                                        verbose=True)
    assert len(existing_paths) == 0
    assert needed_files[0][0] == 'https://igs.bkg.bund.de/root_ftp'

    clk_dates = [
                 tc.datetime_to_gps_millis(datetime(2017, 3, 14, 12,
                                           tzinfo=timezone.utc)),
                 tc.datetime_to_gps_millis(datetime(2018, 3, 14, 12,
                                           tzinfo=timezone.utc)),
                 tc.datetime_to_gps_millis(datetime(2018, 12, 30, 12,
                                           tzinfo=timezone.utc)),
                 tc.datetime_to_gps_millis(datetime(2019, 3, 14, 12,
                                           tzinfo=timezone.utc)),
                ]
    for gps_millis in clk_dates:
        existing_paths, needed_files = ed._verify_ephemeris("clk",
                                                            gps_millis,
                                                            verbose=True)
        assert len(existing_paths) == 0
        assert len(needed_files) == 1

    sp3_dates = [
                 tc.datetime_to_gps_millis(datetime(2017, 3, 14, 12,
                                           tzinfo=timezone.utc)),
                ]
    for gps_millis in sp3_dates:
        existing_paths, needed_files = ed._verify_ephemeris("sp3",
                                                            gps_millis,
                                                            verbose=True)
        assert len(existing_paths) == 0
        assert len(needed_files) == 1

    with pytest.raises(RuntimeError) as excinfo:
        ed._verify_ephemeris("rinex_nav",
                             tc.datetime_to_gps_millis(datetime(2012,12,31,12,
                                                        tzinfo=timezone.utc)))
    assert "rinex nav" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        ed._verify_ephemeris("sp3",
                             tc.datetime_to_gps_millis(datetime(2012,5,24,12,
                                                        tzinfo=timezone.utc)))
    assert "sp3" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        ed._verify_ephemeris("clk",
                             tc.datetime_to_gps_millis(datetime(2012,10,13,12,
                                                        tzinfo=timezone.utc)))
    assert "clk" in str(excinfo.value)

def test_valid_ephemeris(all_ephem_paths,sp3_path,clk_path):
    """Extra tests for full valid coverage.

    Parameters
    ----------
    all_ephem_paths : string
        Location of all unit test ephemeris files.
    sp3_path : string
        String with location for the unit_test sp3 measurements
    clk_path : string
        String with location for the unit_test clk measurements

    """

    date = datetime(2023,3,14,2).date()
    valid, path = ed._valid_ephemeris_in_paths(date,
                                               ["rinex_nav_today"],
                                               file_paths=all_ephem_paths)
    assert valid
    assert os.path.split(path)[-1] == "BRDC00WRD_S_20230730000_01D_MN.rnx"

    # check sp3 and clk short file names
    date = datetime(2021,4,28)
    valid, path = ed._valid_ephemeris_in_paths(date,
                                               ["sp3_rapid_CODE"],
                                               file_paths=[sp3_path,clk_path])
    assert valid
    assert os.path.split(path)[-1] == "grg21553.sp3"
    date = datetime(2021,4,28)
    valid, path = ed._valid_ephemeris_in_paths(date,
                                               ["sp3_rapid_GFZ"],
                                               file_paths=[sp3_path,clk_path])
    assert valid
    assert os.path.split(path)[-1] == "grg21553.sp3"
    date = datetime(2021,4,28)
    valid, path = ed._valid_ephemeris_in_paths(date,
                                               ["clk_rapid_CODE"],
                                               file_paths=[sp3_path,clk_path])
    assert valid
    assert os.path.split(path)[-1] == "grg21553.clk"
    date = datetime(2021,4,28)
    valid, path = ed._valid_ephemeris_in_paths(date,
                                               ["clk_rapid_GFZ"],
                                               file_paths=[sp3_path,clk_path])
    assert valid
    assert os.path.split(path)[-1] == "grg21553.clk"
