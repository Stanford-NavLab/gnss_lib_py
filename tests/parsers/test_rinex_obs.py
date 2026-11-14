"""Tests for the Rinex .o file parser.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "26 July 2023"

import os
from datetime import datetime, timedelta

import pytest
import numpy as np

from conftest import lazy_fixture
from gnss_lib_py.parsers.rinex_obs import RinexObs
from gnss_lib_py.navdata.operations import loop_time

@pytest.fixture(name="rinex_mixed_values")
def fixture_rinex_mixed_values(root_path):
    """Location of Rinex 3 .o file with mixed single and double columns.

    Double columns occur when multiple bands are received for the same
    SV. In this file, there are instances of both, multiple bands and
    single bands being received for different SVs.

    Parameters
    ----------
    root_path : string
        Folder location containing all Rinex 3 .o files for unit tests.

    Returns
    -------
    rinex_mixed : gnss_lib_py.parsers.rinex_obs.RinexObs
        Instance of RinexObs class with mixed values.
    """
    rinex_mixed_path = os.path.join(root_path, 'rinex', 'obs',
                                    'rinex_obs_mixed_types.20o')
    rinex_mixed = RinexObs(rinex_mixed_path)
    return rinex_mixed

@pytest.fixture(name="rinex_single_values")
def fixture_rinex_single_values(root_path):
    """Location of Rinex 3 .o file with measurements on only one band.

    Double columns occur when multiple bands are received for the same
    SV. In this file, there are instances of only single bands being
    received by all SVs.

    Parameters
    ----------
    root_path : string
        Folder location containing all Rinex 3 .o files for unit tests.

    Returns
    -------
    rinex_single : gnss_lib_py.parsers.rinex_obs.RinexObs
        Instance of RinexObs class with single values.
    """
    rinex_single_path = os.path.join(root_path, 'rinex', 'obs',
                                'rinex_obs_single_type_only.22o')
    rinex_single = RinexObs(rinex_single_path)
    return rinex_single


@pytest.fixture(name="mixed_exp_values")
def fixture_mixed_exp_values():
    """List of indices and values for comparison in mixed case.

    Returns
    -------
    mixed_exp_values : list
        List of time instance, row names, gnss_sv_id, signal_type,
        and expected values for mixed case.
    """
    mixed_exp_values = []
    mixed_exp_values.append([0, 'raw_pr_m', 'G02', 'l1', 20832141.28024])
    mixed_exp_values.append([0, 'raw_doppler_hz', 'G06', 'l1', -2091.63326])
    mixed_exp_values.append([0, 'cn0_dbhz', 'G06', 'l5', 20.50023])
    mixed_exp_values.append([2, 'raw_pr_m', 'E15', 'e5a', 23506206.01904])
    mixed_exp_values.append([4, 'raw_doppler_hz', 'R22', 'g1', 2125.66905])
    return mixed_exp_values


@pytest.fixture(name="single_exp_values")
def fixture_single_compare():
    """List of indices and values for comparison in single case.

    Returns
    -------
    single_exp_values : list
        List of time instance, row names, gnss_sv_id and expected values
        for single case.
    """
    single_exp_values = []
    single_exp_values.append([0, 'raw_pr_m', 'G02', 23264262.81328])
    single_exp_values.append([0, 'raw_doppler_hz', 'R20', 1610.80626])
    single_exp_values.append([1, 'raw_pr_m', 'R10', 21137195.69826])
    single_exp_values.append([2, 'cn0_dbhz', 'E07', 43.40027])
    return single_exp_values


def test_rinex_obs_3_load_single(rinex_single_values, single_exp_values):
    """Test that loading works for the single case of Rinex 3 .o files.

    Parameters
    ----------
    rinex_single_values : gnss_lib_py.parsers.rinex_obs.RinexObs
        Instance of RinexObs class with data loaded from appropriate
        file.
    single_exp_values : list
        List of lists containing time instance, gnss_sv_id, and expected
        value for different Rinex use cases.

    """
    count = 0
    for _, _, rinex_frame in loop_time(rinex_single_values,'gps_millis'):

        #For each time case, check that the expected values are correct
        for case in single_exp_values:
            if case[0] == count:
                # Check when the count is the same as the time instance
                sv_case = rinex_frame.where('gnss_sv_id', case[2], 'eq')
                np.testing.assert_almost_equal(sv_case[case[1]], case[3],
                                               decimal=2)
        count += 1


def test_rinex_obs_3_load_mixed(rinex_mixed_values, mixed_exp_values):
    """Test that loading works for cases with double and mixed columns.

    Parameters
    ----------
    rinex_mixed_values : gnss_lib_py.parsers.rinex_obs.RinexObs
        Instance of RinexObs class with data loaded from appropriate
        file.
    mixed_exp_values : list
        List of indices and values to compare against.
    """
    count = 0
    for _, _, rinex_frame in loop_time(rinex_mixed_values,'gps_millis'):
        #For each time case, check that the expected values are correct
        for case in mixed_exp_values:
            if case[0] == count:
                # Check when the count is the same as the time instance
                sv_case = rinex_frame.where('gnss_sv_id', case[2], 'eq')
                sv_signal_case = sv_case.where('signal_type', case[3], 'eq')
                np.testing.assert_almost_equal(sv_signal_case[case[1]],
                                               case[4], decimal=2)
                break
        count += 1


@pytest.fixture(name="sats_per_time_single")
def fixture_sats_per_time_single():
    """List containing number of measurements present at each time step.

    Returns
    -------
    sats_per_time_single : list
        List of number of satellites that are present at each time step.
    """
    sats_per_time_single = [25, 24, 24]
    return sats_per_time_single


@pytest.fixture(name="sats_per_time_mixed")
def fixture_sats_per_time_mixed():
    """List containing number of measurements present at each time step.

    Returns
    -------
    sats_per_time_mixed : list
        List of number of satellites that are present at each time step.
    """
    sats_per_time_mixed = [25, 25, 26, 26, 24]
    return sats_per_time_mixed

@pytest.mark.parametrize("rinex_navdata, time_steps, sats_per_time",
                         [
                          (lazy_fixture("rinex_single_values"), 3,
                           lazy_fixture("sats_per_time_single")),
                          (lazy_fixture("rinex_mixed_values"), 5,
                           lazy_fixture("sats_per_time_mixed"))
                        ])
def test_rinex_obs_3_complete_load(rinex_navdata, time_steps, sats_per_time):
    """Test that all measurements that contain information are loaded.

    Parameters
    ----------
    rinex_navdata : str
        Instance of RinexObs class with data loaded from appropriate
        file.
    time_steps : int
        Total times that have measurements in the observation file.
    sats_per_time : list
        List of number of satellites that are present at each time step.
    """
    assert len(np.unique(rinex_navdata['gps_millis'])) == time_steps, \
        "Measurements for all times were not loaded."
    count = 0
    for _, _, rinex_frame in loop_time(rinex_navdata,'gps_millis'):
        assert len(rinex_frame) == sats_per_time[count], \
        "Measurements for all recorded satellites were not loaded."
        count += 1


def test_rinex_obs_3_fails(rinex_mixed_values):
    """Test for cases when no measurements should exist in RinexObs.

    The cases where no measurements should be loaded are: 1) when the
    SV hasn't been received, and 2) when the SV has been received but
    a measurement is unavailable.

    Parameters
    ----------
    rinex_mixed_values : gnss_lib_py.parsers.rinex_obs.RinexObs
        Instance of RinexObs class with data loaded from file with
        measurements received in both, single and double bands.
    """
    for count , (_, _, rinex_frame) in enumerate(loop_time(rinex_mixed_values,'gps_millis')):
            # SV wasn't received
        sv_not_rx = rinex_frame.where('gnss_sv_id', 'G01', 'eq')
        assert len(sv_not_rx) == 0
        if count == 1:
            sv_not_pseudo = rinex_frame.where('gnss_sv_id', 'G29', 'eq')
            assert len(sv_not_pseudo) == 0

def test_z_tracking(root_path):
    """Test z-tracking support.

    Parameters
    ----------
    root_path : string
        Folder location containing all Rinex 3 .o files for unit tests.

    """
    rinex_path = os.path.join(root_path, 'rinex', 'obs',
                              'z_tracking.rnx')
    rinex_navdata = RinexObs(rinex_path)

    assert len(rinex_navdata) == 210
    assert rinex_navdata.shape == (10,210)

    # expected observation codes come from z_tracking.rnx header
    # header also lists the 3Q measurements for glonass but none exist
    # in this shortened version of the file
    expected_observation_codes = ['1C','1W','2W','2L','5Q','7Q','2P',
                                  '2C','2I','7I']

    parsed_observation_codes = np.unique(rinex_navdata["observation_code"])
    assert sorted(expected_observation_codes) == sorted(parsed_observation_codes)


@pytest.fixture(name="write_time_rinex")
def fixture_write_time_rinex(tmp_path):
    """Write a two-epoch RINEX file with a specified observation time system."""
    def write_file(file_system, time_system, epoch):
        satellite_system = 'G' if file_system == 'M' else file_system
        band = {'G': '1C', 'R': '1C', 'E': '1C', 'J': '1C',
                'C': '2I', 'I': '5A'}[satellite_system]
        seconds = epoch.second + epoch.microsecond / 1e6
        header_time = (f"{epoch.year:6d}{epoch.month:6d}{epoch.day:6d}"
                       f"{epoch.hour:6d}{epoch.minute:6d}{seconds:13.7f}"
                       f"     {time_system:3s}")
        header = [
            (f"     3.04           OBSERVATION DATA    {file_system}",
             "RINEX VERSION / TYPE"),
            ("gnss_lib_py test", "PGM / RUN BY / DATE"),
            (f"{satellite_system}    4 C{band} L{band} D{band} S{band}",
             "SYS / # / OBS TYPES"),
            (header_time, "TIME OF FIRST OBS"),
            ("", "END OF HEADER"),
        ]
        contents = "".join(f"{value:60s}{label}\n" for value, label in header)
        for offset in (0, 1):
            timestamp = epoch + timedelta(seconds=offset)
            seconds = timestamp.second + timestamp.microsecond / 1e6
            contents += f"> {timestamp:%Y %m %d %H %M} {seconds:10.7f}  0  1\n"
            contents += f"{satellite_system}01"
            contents += "".join(f"{value:14.3f}  " for value in
                                [22000000 + offset, 110000000, -1000, 45])
            contents += "\n"
        path = tmp_path / "time_system.rnx"
        path.write_text(contents, encoding="ascii")
        return path
    return write_file


@pytest.mark.parametrize("file_system, time_system, second", [
    ('M', 'GPS', 18), ('M', 'GAL', 18), ('M', 'QZS', 18), ('M', 'IRN', 18),
    ('M', 'BDT', 4), ('M', 'GLO', 0), ('M', 'UTC', 0),
    ('G', '', 18), ('E', '', 18), ('J', '', 18), ('I', '', 18),
    ('C', '', 4), ('R', '', 0),
    ('G', 'GLO', 0), ('G', 'BDT', 4),
])
def test_rinex_obs_time_systems(write_time_rinex, file_system, time_system, second):
    """Equivalent epochs in supported time systems must have equal GPS millis."""
    # RINEX 3.05 section 4.1: UTC 00:00:00.25 is GPS 00:00:18.25 and
    # BDT 00:00:04.25 on 2020-01-01. Use independent calendar arithmetic.
    path = write_time_rinex(file_system, time_system,
                           datetime(2020, 1, 1, 0, 0, second, 250000))
    observations = RinexObs(path)
    expected = (datetime(2020, 1, 1, 0, 0, 18, 250000)
                - datetime(1980, 1, 6)).total_seconds() * 1000
    np.testing.assert_allclose(observations['gps_millis'],
                               [expected, expected + 1000], rtol=0, atol=1e-3)
    np.testing.assert_array_equal(observations['raw_pr_m'], [22000000, 22000001])


@pytest.mark.parametrize("time_system", ['GPS', 'GLO'])
def test_rinex_obs_time_system_leap_boundary(write_time_rinex, time_system):
    """Apply the UTC offset across a leap-second boundary only to GLO epochs."""
    epoch = datetime(2016, 12, 31, 23, 59, 59, 250000)
    path = write_time_rinex('M', time_system, epoch)
    observations = RinexObs(path)
    if time_system == 'GPS':
        expected = [(epoch - datetime(1980, 1, 6)).total_seconds() * 1000]
        expected.append(expected[0] + 1000)
    else:
        # GPS-UTC changes from 17 to 18 seconds at 2017-01-01 UTC.
        expected = [(datetime(2017, 1, 1, 0, 0, second, 250000)
                     - datetime(1980, 1, 6)).total_seconds() * 1000
                    for second in (16, 18)]
    np.testing.assert_allclose(observations['gps_millis'], expected, rtol=0, atol=1e-3)


@pytest.mark.parametrize("time_system", ['', 'BAD'])
def test_rinex_obs_unknown_time_system(write_time_rinex, time_system):
    """Reject missing or unknown mixed-file time systems instead of assuming UTC."""
    path = write_time_rinex('M', time_system, datetime(2020, 1, 1))
    with pytest.raises(ValueError, match="time system"):
        RinexObs(path)


def test_rinex_obs_gps_epoch(rinex_single_values):
    """The existing GPS-time sample must not gain a UTC leap-second offset."""
    expected = (datetime(2022, 7, 1, 18, 50, 30, 444669)
                - datetime(1980, 1, 6)).total_seconds() * 1000
    np.testing.assert_allclose(np.min(rinex_single_values['gps_millis']),
                               expected, rtol=0, atol=1e-3)
