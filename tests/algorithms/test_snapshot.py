"""Tests for weighted least squres in snapshot

"""

__authors__ = "B. Collicott, A. Kanhere, D. Knowles"
__date__ = "22 October 2021"

import os
import pytest

import numpy as np
import pandas as pd

from gnss_lib_py.algorithms.snapshot import *
from gnss_lib_py.parsers.android import AndroidDerived
from gnss_lib_py.parsers.measurement import Measurement


# Defining test fixtures
@pytest.fixture(name="tolerance")
def fixture_tolerance():
    """Decimal threshold for test pass/fail criterion."""
    return 7

@pytest.fixture(name="set_user_states")
def fixture_set_user_states():
    """ Set the location and clock bias of the user receiver in Earth-Centered,
    Earth-Fixed coordinates.

    Returns
    -------
    rx_truth_m : np.ndarray
        Truth receiver position in ECEF frame in meters and the
        truth receiver clock bias also in meters in an
        array with shape (4 x 1) and the following order:
        x_rx_m, y_rx_m, z_rx_m, b_rx_m.
    """
    x_rx_m = -2700006.81
    y_rx_m = -4292610.78
    z_rx_m =  3855390.92
    b_rx_m = 12.34

    rx_truth_m = np.array([[x_rx_m, y_rx_m, z_rx_m, b_rx_m]]).T

    return rx_truth_m

@pytest.fixture(name="set_sv_states")
def fixture_set_sv_states():
    """Get position of 4 satellite in ECEF coordinates.

    Returns
    -------
    pos_sv_m : np.ndarray
        Satellite positions in ECEF frame as an array of shape
        [# svs x 3] where the columns contain in order
        x_sv_m, y_sv_m, and z_sv_m.

    References
    ----------
    .. [1] Weiss, M., & Ashby, N. (1999).
       Global Positioning System Receivers and Relativity.

    """
    x_sv_m = np.array([13005878.255, 20451225.952, 20983704.633,
                       13798849.321])
    y_sv_m = np.array([18996947.213, 16359086.310, 15906974.416,
                      -8709113.822])
    z_sv_m = np.array([13246718.721,-4436309.8750, 3486495.546,
                       20959777.407])
    pos_sv_m = np.hstack((x_sv_m.reshape(-1,1),
                          y_sv_m.reshape(-1,1),
                          z_sv_m.reshape(-1,1)))

    return pos_sv_m

# Defining tests
def test_wls(set_user_states, set_sv_states, tolerance):
    """Test snapshot positioning against truth user states.

    Parameters
    ----------
    set_user_states : fixture
        Truth values for user position and clock bias
    set_sv_states : fixture
        Satellite position and clock biases
    tolerance : fixture
        Error threshold for test pass/fail
    """
    rx_truth_m  = set_user_states
    pos_sv_m = set_sv_states

    # Compute noise-free pseudorange measurements
    pos_rx_m = np.tile(rx_truth_m[0:3,:].T, (pos_sv_m.shape[0], 1))

    gt_pr_m = np.linalg.norm(pos_rx_m - pos_sv_m, axis = 1,
                             keepdims = True) + rx_truth_m[3,0]

    rx_est_m = np.zeros((4,1))
    user_fix = wls(rx_est_m, pos_sv_m, gt_pr_m)
    truth_fix = rx_truth_m

    np.testing.assert_array_almost_equal(user_fix, truth_fix,
                                         decimal=tolerance)

def test_wls_stationary(set_user_states, set_sv_states, tolerance):
    """Test WLS positioning against truth user states.

    In these stationary tests, it is only solving for clock bias.

    Parameters
    ----------
    set_user_states : fixture
        Truth values for user position and clock bias
    set_sv_states : fixture
        Satellite position and clock biases
    tolerance : fixture
        Error threshold for test pass/fail
    """
    rx_truth_m  = set_user_states
    pos_sv_m = set_sv_states

    # Compute noise-free pseudorange measurements
    pos_rx_m = np.tile(rx_truth_m[0:3,:].T, (pos_sv_m.shape[0], 1))

    gt_pr_m = np.linalg.norm(pos_rx_m - pos_sv_m, axis = 1,
                             keepdims = True) + rx_truth_m[3,0]

    # check that position doesn't change but clock bias does change
    rx_est_m = np.zeros((4,1))
    user_fix = wls(rx_est_m, pos_sv_m, gt_pr_m, None, True)
    np.testing.assert_array_almost_equal(user_fix[:3], np.zeros((3,1)),
                                         decimal=tolerance)
    assert abs(user_fix[3]) >= 1E5

    # check that correct clock bias is calculated
    rx_est_m = np.zeros((4,1))
    rx_est_m[:3] = rx_truth_m[:3]
    user_fix = wls(rx_est_m, pos_sv_m, gt_pr_m, None, True)
    truth_fix = rx_truth_m

    np.testing.assert_array_almost_equal(user_fix, truth_fix,
                                         decimal=tolerance)

# @pytest.fixture(name="root_path")
# def fixture_root_path():
#     """Location of measurements for unit test
#
#     Returns
#     -------
#     root_path : string
#         Folder location containing measurements
#     """
#     root_path = os.path.dirname(
#                 os.path.dirname(
#                 os.path.dirname(
#                 os.path.realpath(__file__))))
#     root_path = os.path.join(root_path, 'data/unit_test/')
#     return root_path
#
#
# @pytest.fixture(name="derived_path")
# def fixture_derived_path(root_path):
#     """Filepath of Android Derived measurements
#
#     Returns
#     -------
#     derived_path : string
#         Location for the unit_test Android derived measurements
#
#     Notes
#     -----
#     Test data is a subset of the Android Raw Measurement Dataset,
#     particularly the train/2020-05-14-US-MTV-1/Pixel4 trace. The dataset
#     was retrieved from
#     https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data
#
#     References
#     ----------
#     .. [1] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
#         "Android Raw GNSS Measurement Datasets for Precise Positioning."
#         Proceedings of the 33rd International Technical Meeting of the
#         Satellite Division of The Institute of Navigation (ION GNSS+
#         2020). 2020.
#     """
#     derived_path = os.path.join(root_path, 'Pixel4_derived.csv')
#     return derived_path
#
#
# @pytest.fixture(name="derived")
# def fixture_load_derived(derived_path):
#     """Load instance of AndroidDerived
#
#     Parameters
#     ----------
#     derived_path : pytest.fixture
#         String with location of Android derived measurement file
#
#     Returns
#     -------
#     derived : AndroidDerived
#         Instance of AndroidDerived for testing
#     """
#     derived = AndroidDerived(derived_path)
#     return derived
#
#
# root_path = fixture_root_path()
# derived_path = fixture_derived_path(root_path)
# derived = fixture_load_derived(derived_path)
# # solution = solve_wls(derived)
# # print(derived.rows)
#
# test_wls(fixture_set_user_states(), fixture_set_sv_states(), fixture_tolerance())
