"""Tests for weighted least squres in snapshot

"""

__authors__ = "B. Collicott, A. Kanhere, D. Knowles"
__date__ = "22 October 2021"

import os
import warnings

import pytest
import numpy as np

from gnss_lib_py.parsers.google_decimeter import AndroidDerived2021, AndroidDerived2022
from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.algorithms.snapshot import wls, solve_wls


# Defining test fixtures
TEST_REPEAT_COUNT = 10

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

    See reference [1]_ for details.

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
    user_fix = wls(rx_est_m, pos_sv_m, gt_pr_m, sv_rx_time=True)
    truth_fix = rx_truth_m

    np.testing.assert_array_almost_equal(user_fix, truth_fix,
                                         decimal=tolerance)

    # should return warning if only three satellites are given
    with pytest.raises(RuntimeError) as excinfo:
        wls(rx_est_m, pos_sv_m[:3,:], gt_pr_m[:3,:])
    assert "Need at least four satellites" in str(excinfo.value)

@pytest.mark.parametrize('random_noise',
                         np.random.normal(0,20,size=(TEST_REPEAT_COUNT,4,1))
                        )
@pytest.mark.parametrize('tolerance_test',
                        [
                         1E10,
                         1E2,
                         1E1,
                         1E0,
                         1E-5,
                        ])
def test_wls_tolerance(set_user_states, set_sv_states,
                       tolerance_test, random_noise):
    """Test snapshot positioning against truth user states.

    Parameters
    ----------
    set_user_states : fixture
        Truth values for user position and clock bias
    set_sv_states : fixture
        Satellite position and clock biases
    tolerance_test : float
        Tolerance with which to end the wls solver
    random_noise : np.ndarray
        Noise added to ground truth pseudoranges of shape 4x1
    """
    rx_truth_m  = set_user_states
    pos_sv_m = set_sv_states

    # Compute noise-free pseudorange measurements
    pos_rx_m = np.tile(rx_truth_m[0:3,:].T, (pos_sv_m.shape[0], 1))

    gt_pr_m = np.linalg.norm(pos_rx_m - pos_sv_m, axis = 1,
                             keepdims = True) + rx_truth_m[3,0]
    noisy_pr_m = gt_pr_m + random_noise

    rx_est_m = np.zeros((4,1))
    wls(rx_est_m, pos_sv_m, noisy_pr_m, tol=tolerance_test,
                  max_count=np.inf)

@pytest.mark.parametrize('random_noise',
                         np.random.normal(0,20,size=(TEST_REPEAT_COUNT,4,1))
                        )
@pytest.mark.parametrize('count_test',
                        [
                         1,
                         2,
                         10,
                         100,
                         1000,
                        ])
def test_wls_max_count(set_user_states, set_sv_states, count_test,
                       random_noise):
    """Test snapshot positioning against truth user states.

    Parameters
    ----------
    set_user_states : fixture
        Truth values for user position and clock bias
    set_sv_states : fixture
        Satellite position and clock biases
    count_test : int
        Max count for wls solver
    random_noise : np.ndarray
        Noise added to ground truth pseudoranges of shape 4x1

    """
    rx_truth_m  = set_user_states
    pos_sv_m = set_sv_states

    # Compute noise-free pseudorange measurements
    pos_rx_m = np.tile(rx_truth_m[0:3,:].T, (pos_sv_m.shape[0], 1))

    gt_pr_m = np.linalg.norm(pos_rx_m - pos_sv_m, axis = 1,
                             keepdims = True) + rx_truth_m[3,0]
    noisy_pr_m = gt_pr_m + random_noise

    rx_est_m = np.zeros((4,1))

    with warnings.catch_warnings(record=True) as warn:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Trigger a warning.
        wls(rx_est_m, pos_sv_m, noisy_pr_m, tol=-np.inf,
            max_count=count_test)

        # verify RuntimeWarning
        assert len(warn) == 1, "No warning is raised."
        assert issubclass(warn[-1].category, RuntimeWarning)


def test_wls_only_bias(set_user_states, set_sv_states, tolerance):
    """Test WLS positioning against truth user states.

    In these only_bias tests, it is only solving for clock bias.

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
    user_fix = wls(rx_est_m, pos_sv_m, gt_pr_m, None, True, sv_rx_time=True)
    np.testing.assert_array_almost_equal(user_fix[:3], np.zeros((3,1)),
                                         decimal=tolerance)
    assert abs(user_fix[3]) >= 1E5

    # check that correct clock bias is calculated
    rx_est_m = np.zeros((4,1))
    rx_est_m[:3] = rx_truth_m[:3]
    user_fix = wls(rx_est_m, pos_sv_m, gt_pr_m, None, True, sv_rx_time=True)
    truth_fix = rx_truth_m

    np.testing.assert_array_almost_equal(user_fix, truth_fix,
                                         decimal=tolerance)


@pytest.fixture(name="root_path")
def fixture_root_path():
    """Location of measurements for unit test

    Returns
    -------
    root_path : string
        Folder location containing measurements
    """
    root_path = os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
    root_path = os.path.join(root_path, 'data/unit_test/android_2021/')
    return root_path


@pytest.fixture(name="derived_path")
def fixture_derived_path(root_path):
    """Filepath of Android Derived measurements

    Returns
    -------
    derived_path : string
        Location for the unit_test Android derived measurements

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset [2]_,
    particularly the train/2020-05-14-US-MTV-1/Pixel4 trace. The dataset
    was retrieved from
    https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data

    References
    ----------
    .. [2] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.
    """
    derived_path = os.path.join(root_path, 'Pixel4_derived.csv')
    return derived_path


@pytest.fixture(name="derived")
def fixture_load_derived(derived_path):
    """Load instance of AndroidDerived2021

    Parameters
    ----------
    derived_path : pytest.fixture
        String with location of Android derived measurement file

    Returns
    -------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing
    """
    derived = AndroidDerived2021(derived_path)
    return derived


@pytest.fixture(name="derived_2022_path")
def fixture_derived_2022_path(root_path):
    """Filepath of Android Derived 2022 measurements

    Returns
    -------
    derived_2022_path : string
        Location for the unit_test Android 2022 derived measurements

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset [3]_,
    from the 2022 Decimeter Challenge. Particularly, the
    train/2021-04-29-MTV-2/SamsungGalaxyS20Ultra trace. The dataset
    was retrieved from
    https://www.kaggle.com/competitions/smartphone-decimeter-2022/data

    References
    ----------
    .. [3] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.
    """
    derived_2022_path = os.path.join(root_path, '../android_2022/device_gnss.csv')
    return derived_2022_path


@pytest.fixture(name="derived_2022")
def fixture_load_derived_2022(derived_2022_path):
    """Load instance of AndroidDerived2021

    Parameters
    ----------
    derived_2022_path : pytest.fixture
        String with location of Android derived 2022 measurement file

    Returns
    -------
    derived_2022 : AndroidDerived2021
        Instance of AndroidDerived2022 for testing
    """
    derived_2022 = AndroidDerived2022(derived_2022_path)
    return derived_2022

def test_solve_wls(derived):
    """Test that solving for wls doesn't fail

    Parameters
    ----------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.

    """
    state_estimate = solve_wls(derived, sv_rx_time=False)

    # result should be a NavData Class instance
    assert isinstance(state_estimate,type(NavData()))

    # should have four rows
    assert len(state_estimate.rows) == 8

    # should have the following contents
    assert "gps_millis" in state_estimate.rows
    assert "x_rx_wls_m" in state_estimate.rows
    assert "y_rx_wls_m" in state_estimate.rows
    assert "z_rx_wls_m" in state_estimate.rows
    assert "b_rx_wls_m" in state_estimate.rows
    assert "lat_rx_wls_deg" in state_estimate.rows
    assert "lon_rx_wls_deg" in state_estimate.rows
    assert "alt_rx_wls_m" in state_estimate.rows

    # should have the same length as the number of unique timesteps
    assert len(state_estimate) == sum(1 for _ in derived.loop_time("gps_millis"))

    # len(np.unique(derived["gps_millis",:]))

    # test what happens when rows down't exist
    for row_index in ["gps_millis","x_sv_m","y_sv_m","z_sv_m"]:
        derived_no_row = derived.remove(rows=row_index)
        with pytest.raises(KeyError) as excinfo:
            solve_wls(derived_no_row)
        assert row_index in str(excinfo.value)

def test_solve_wls_weights(derived, tolerance):
    """Tests that weights are working for weighted least squares.

    Parameters
    ----------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing
    tolerance : fixture
        Error threshold for test pass/fail
    """

    state_estimate_1 = solve_wls(derived, sv_rx_time=False)
    state_estimate_2 = solve_wls(derived, None, sv_rx_time=False)

    # create new column of unity weights
    derived["unity_weights"] = 1
    state_estimate_3 = solve_wls(derived, "unity_weights",  sv_rx_time=False)

    # all of the above should be the same
    np.testing.assert_array_almost_equal(state_estimate_1.array,
                                         state_estimate_2.array,
                                         decimal=tolerance)
    np.testing.assert_array_almost_equal(state_estimate_1.array,
                                         state_estimate_3.array,
                                         decimal=tolerance)

    state_estimate_4 = solve_wls(derived, "raw_pr_sigma_m",  sv_rx_time=False)
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_almost_equal(state_estimate_1.array,
                                             state_estimate_4.array,
                                             decimal=tolerance)

    #should return error for empty string
    with pytest.raises(TypeError):
        solve_wls(derived, "")

    # should return error for row not in NavData instance
    with pytest.raises(TypeError):
        solve_wls(derived, "special_weights")

@pytest.mark.parametrize('random_noise',
                         np.random.normal(0,20,size=(TEST_REPEAT_COUNT,4,1))
                        )
def test_wls_weights(set_user_states, set_sv_states, tolerance,
                     random_noise):
    """Test snapshot positioning against truth user states.

    Parameters
    ----------
    set_user_states : fixture
        Truth values for user position and clock bias
    set_sv_states : fixture
        Satellite position and clock biases
    tolerance : fixture
        Error threshold for test pass/fail
    random_noise : np.ndarray
        Noise added to ground truth pseudoranges of shape 4x1
    """
    rx_truth_m  = set_user_states
    pos_sv_m = set_sv_states

    # Compute noise-free pseudorange measurements
    pos_rx_m = np.tile(rx_truth_m[0:3,:].T, (pos_sv_m.shape[0], 1))

    gt_pr_m = np.linalg.norm(pos_rx_m - pos_sv_m, axis = 1,
                             keepdims = True) + rx_truth_m[3,0]
    noisy_pr_m = gt_pr_m + random_noise

    rx_est_m = np.zeros((4,1))

    # should work if None is passed
    user_fix_1 = wls(rx_est_m, pos_sv_m, noisy_pr_m, weights=None,
                     sv_rx_time=True)
    user_fix_2 = wls(rx_est_m, pos_sv_m, noisy_pr_m,
                     weights=np.ones((pos_sv_m.shape[0],1)),
                     sv_rx_time=True)
    # both should be unity weights and return the same result
    np.testing.assert_array_almost_equal(user_fix_1,
                                         user_fix_2,
                                         decimal=tolerance)

    # result should be different if different weights are used
    user_fix_3 = wls(rx_est_m, pos_sv_m, noisy_pr_m,
                     weights=np.arange(pos_sv_m.shape[0]).reshape(-1,1),
                     sv_rx_time=True)
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_almost_equal(user_fix_1,
                                             user_fix_3,
                                             decimal=tolerance)

    # should return error for string
    with pytest.raises(TypeError):
        wls(rx_est_m, pos_sv_m, noisy_pr_m,
            weights="")

    # should return error even if string is in NavData instance
    with pytest.raises(TypeError):
        wls(rx_est_m, pos_sv_m, noisy_pr_m,
            weights="raw_pr_sigma_m")

    # should return error even if list
    with pytest.raises(TypeError):
        wls(rx_est_m, pos_sv_m, noisy_pr_m,
            weights=[1.]*pos_sv_m.shape[0])

    # should return error if the weights are not right shape
    with pytest.raises(TypeError):
        wls(rx_est_m, pos_sv_m, noisy_pr_m,
            weights=np.ndarray([]))
    with pytest.raises(TypeError):
        wls(rx_est_m, pos_sv_m, noisy_pr_m,
            weights=np.ones((pos_sv_m.shape[0],pos_sv_m.shape[0])))
    with pytest.raises(TypeError):
        wls(rx_est_m, pos_sv_m, noisy_pr_m,
            weights=np.ones(pos_sv_m.shape[0]+1,1))


def test_solve_wls_bias_only(derived_2022):
    """Tests that bias only WLS estimation works as expected.

    Parameters
    ----------
    derived_2022 : AndroidDerived2022
        Instance of AndroidDerived2022 for testing
    """

    # Solve with receiver positions given
    ecef_rows = ['x_rx_m', 'y_rx_m', 'z_rx_m']
    wls_rows = ['x_rx_wls_m','y_rx_wls_m','z_rx_wls_m']
    time_length = sum(1 for _ in derived_2022.loop_time("gps_millis"))
    input_position = NavData()
    for row in ecef_rows:
        input_position[row] = np.zeros(time_length)
    col = 0
    for _, _, measure_frame in derived_2022.loop_time('gps_millis'):
        for row in ecef_rows:
            input_position[row, col] = measure_frame[row, 0]
        col += 1
    state_estimate = solve_wls(derived_2022, only_bias=True,
                               receiver_state=derived_2022, sv_rx_time=False)
    # Verify that both structures have the same length
    assert len(input_position) == len(state_estimate)
    # Verify that solved positions are the same as input positions
    for rr,row in enumerate(ecef_rows):
        np.testing.assert_almost_equal(input_position[row], state_estimate[wls_rows[rr]])

    assert isinstance(state_estimate,type(NavData()))

    # should have four rows
    assert len(state_estimate.rows) == 8

    # should have the following contents
    assert "gps_millis" in state_estimate.rows
    assert "x_rx_wls_m" in state_estimate.rows
    assert "y_rx_wls_m" in state_estimate.rows
    assert "z_rx_wls_m" in state_estimate.rows
    assert "b_rx_wls_m" in state_estimate.rows
    assert "lat_rx_wls_deg" in state_estimate.rows
    assert "lon_rx_wls_deg" in state_estimate.rows
    assert "alt_rx_wls_m" in state_estimate.rows

    # should have the same length as the number of unique timesteps
    assert len(state_estimate) == time_length

    # Solve without receiver positions given. This should cause a warning
    derived_2022.remove(ecef_rows, inplace=True)
    with pytest.raises(KeyError):
        solve_wls(derived_2022, only_bias=True,
                receiver_state=derived_2022, sv_rx_time=False)

    # check error raised when receiver_state is not present in only_bias
    with pytest.raises(RuntimeError):
        solve_wls(derived_2022, only_bias=True, sv_rx_time=False)


def test_wls_fails(capsys):
    """Test expected fails

    Parameters
    ----------
    capsys : error
        The capsys fixture allows access to stdout/stderr output created
        during test execution.

    """

    pos_sv_m = 5.*np.ones((5,3))
    pos_sv_m[0,0] = np.nan

    wls(np.ones((4,1)),pos_sv_m,np.ones((5,1)))
    captured = capsys.readouterr()
    assert captured.out == "SVD did not converge\n"

def test_solve_wls_fails(derived):
    """Test expected fails

    Parameters
    ----------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing

    """

    navdata = derived.remove(cols=list(range(3,50)) \
                                + list(range(53,len(derived))))

    with pytest.warns(RuntimeWarning) as warns:
        solve_wls(navdata)

    # verify RuntimeWarning
    assert len(warns) == 2

    caught_four_sats = False
    caught_empty_state = False
    for warn in warns:
        assert issubclass(warn.category, RuntimeWarning)
        assert "WLS" in str(warn.message)
        if "four satellites" in str(warn.message):
            caught_four_sats = True
        elif "No valid state" in str(warn.message):
            caught_empty_state = True

    assert caught_four_sats
    assert caught_empty_state


def test_rotation_of_earth_fix(derived_2022):
    """Tests that accounting for Earth's rotation reduces WLS errors.

    Parameters
    ----------
    derived_2022 : AndroidDerived2022
        Instance of AndroidDerived2022 for testing
    """
    google_wls = derived_2022[['x_rx_m', 'y_rx_m', 'z_rx_m']]
    google_wls = np.empty([3, len(np.unique(np.round(derived_2022['gps_millis'], decimals=-2)))])
    for idx, (_, _, frame) in enumerate(derived_2022.loop_time('gps_millis')):
        google_wls[:, idx] = frame[['x_rx_m', 'y_rx_m', 'z_rx_m'], 0]
    derived_2022['wls_weights'] = 1/derived_2022['raw_pr_sigma_m']
    state_with_rotn = solve_wls(derived_2022, weight_type='wls_weights',
                               sv_rx_time=False)
    glp_wls = state_with_rotn[['x_rx_wls_m', 'y_rx_wls_m', 'z_rx_wls_m']]
    # Verify that the mean error between both estimates is less than 10m
    for idx in range(3):
        assert np.mean(np.abs(google_wls[idx, :] - glp_wls[idx, :])) < 30
    state_without_rotn = solve_wls(derived_2022, weight_type='wls_weights',
                                   sv_rx_time=True)
    glp_wls_no_rotn = state_without_rotn[['x_rx_wls_m',
                                          'y_rx_wls_m',
                                          'z_rx_wls_m']]
    for idx in range(3):
        error_rotn = np.mean(np.abs(google_wls[idx, :] - glp_wls[idx, :]))
        error_no_rotn = np.mean(np.abs(google_wls[idx, :] - glp_wls_no_rotn[idx, :]))
        assert error_rotn < error_no_rotn

def test_solve_wls_empty():
    """Test scenario where an empty measurement class is passed in.

    """

    measurements = NavData()
    measurements["gps_millis"] = []
    measurements["x_sv_m"] = []
    measurements["y_sv_m"] = []
    measurements["z_sv_m"] = []
    with pytest.warns(RuntimeWarning) as warns:
        state_estimate = solve_wls(measurements)

    # should have the following contents
    assert "gps_millis" in state_estimate.rows
    assert "x_rx_wls_m" in state_estimate.rows
    assert "y_rx_wls_m" in state_estimate.rows
    assert "z_rx_wls_m" in state_estimate.rows
    assert "b_rx_wls_m" in state_estimate.rows
