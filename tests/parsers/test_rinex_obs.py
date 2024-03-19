"""Tests for the Rinex .o file parser.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "26 July 2023"

import os

import pytest
import numpy as np
from pytest_lazyfixture import lazy_fixture

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
