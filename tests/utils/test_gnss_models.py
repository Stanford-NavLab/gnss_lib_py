"""Tests for GNSS SV state calculation methods.

"""

__authors__ = "Ashwin Kanhere, Bradley Collicott"
__date__ = "6 Aug 2021"


import pytest
import numpy as np

from numpy.random import default_rng
from pytest_lazyfixture import lazy_fixture

from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.parsers.navdata import NavData
import gnss_lib_py.utils.gnss_models as gnss_models
from gnss_lib_py.utils.sv_models import _extract_pos_vel_arr
from gnss_lib_py.utils.coordinates import LocalCoord


# pylint: disable=protected-access

# Number of time to run meausurement simulation code
TEST_REPEAT_COUNT = 10

#TODO: Where is this used?
T = 0.1


@pytest.fixture(name="iono_params")
def fixture_iono_params():
    """Ionospheric delay parameters for the unit test Android measurements.

    Returns
    -------
    iono_params : np.ndarray
        2x4 (first row, alpha and second row, beta) of ionospheric delay
        parameters.
    """
    iono_params = {"gps":np.array([[0.9313E-08,  0.1490E-07, -0.5960E-07, -0.1192E-06],
                                   [0.8806E+05,  0.4915E+05, -0.1311E+06, -0.3277E+06]])}
    return iono_params


def calculate_state(android_gt, idx):
    """Helper function to create state instance of NavData for tests.

    Parameters
    ----------
    android_gt : gnss_lib_py.parsers.android.AndroidGroundTruth2022
        NavData containing ground truth for Android measurements.
    idx : int
        Index of ground truth for which states are required.

    Returns
    -------
    state : gnss_lib_py.parsers.navdata.NavData
        NavData containing state information for one time instance.
    """

    v_gt_n = android_gt['v_rx_gt_mps', idx]*np.cos(android_gt['heading_rx_gt_rad', idx])
    v_gt_e = android_gt['v_rx_gt_mps', idx]*np.sin(android_gt['heading_rx_gt_rad', idx])
    v_ned = np.array([[v_gt_n],[v_gt_e],[0]])
    llh = np.array([[android_gt['lat_rx_gt_deg', idx]],
                    [android_gt['lon_rx_gt_deg', idx]],
                    [android_gt['alt_rx_gt_m', idx]]])
    local_frame = LocalCoord.from_geodetic(llh)
    vx_ecef = local_frame.ned_to_ecefv(v_ned)

    state = NavData()
    state['x_rx_m'] = android_gt['x_rx_gt_m', idx]
    state['y_rx_m'] = android_gt['y_rx_gt_m', idx]
    state['z_rx_m'] = android_gt['z_rx_gt_m', idx]
    state['vx_rx_mps'] = vx_ecef[0,0]
    state['vy_rx_mps'] = vx_ecef[1,0]
    state['vz_rx_mps'] = vx_ecef[2,0]
    state['b_rx_m'] = 0
    state['b_dot_rx_mps'] = 0
    return state


def test_state_extraction():
    """Test the state extraction code by comparing extracted values to
    values used to set states.
    """
    state = NavData()
    state['x_rx_m'] = 1
    state['y_rx_m'] = 2
    state['z_rx_m'] = 3
    state['vx_rx_mps'] = 4
    state['vy_rx_mps'] = 5
    state['vz_rx_mps'] = 6
    state['b_rx_m'] = 7
    state['b_dot_rx_mps'] = 8
    rx_test, v_test, b_test, b_dot_test = gnss_models._extract_state_variables(state)
    np.testing.assert_almost_equal(rx_test, np.array([[1], [2], [3]]))
    np.testing.assert_almost_equal(v_test, np.array([[4], [5], [6]]))
    np.testing.assert_almost_equal(b_test, 7)
    np.testing.assert_almost_equal(b_dot_test, 8)


def test_pseudorange_corrections(gps_measurement_frames, android_gt, iono_params):
    """Test code for generating pseudorange corrections.

    Parameters
    ----------
    gps_measurement_frames : Dict
        Dictionary containing lists of visible ephemeris parameters,
        received Android measurements and SV states. The lists are
        indexed by discrete time indices.
    android_gt : gnss_lib_py.parsers.android.AndroidGroundTruth2022
        NavData containing ground truth for Android measurements.
    iono_params : np.ndarray
        2x4 (first row, alpha and second row, beta) of ionospheric delay
        parameters.
    """
    android_frames = gps_measurement_frames['android_frames']
    vis_ephems = gps_measurement_frames['vis_ephems']
    sv_states = gps_measurement_frames['sv_states']
    for idx, frame in enumerate(android_frames):
        sort_arg = np.argsort(frame['sv_id'])
        # Get Android Derived states, sorted by SVs
        tropo_delay_sort = frame['tropo_delay_m'][sort_arg]
        iono_delay_sort = frame['iono_delay_m'][sort_arg]
        curr_millis = frame['gps_millis', 0]
        gt_slice_idx = android_gt.argwhere('gps_millis', curr_millis)
        state = calculate_state(android_gt, gt_slice_idx)

        # Test corrections with ephemeris parameters
        est_trp, est_iono = gnss_models.calculate_pseudorange_corr(
            curr_millis, state=state, ephem=vis_ephems[idx],
            iono_params =iono_params)
        np.testing.assert_almost_equal(tropo_delay_sort, est_trp, decimal=0)
        np.testing.assert_almost_equal(iono_delay_sort, est_iono, decimal=0)

        # Test corrections without ephemeris parameters buit SV position
        sv_posvel = sv_states[idx]
        est_trp, est_iono = gnss_models.calculate_pseudorange_corr(
            curr_millis, state=state, sv_posvel=sv_posvel,
            iono_params =iono_params)
        np.testing.assert_almost_equal(tropo_delay_sort, est_trp, decimal=0)
        np.testing.assert_almost_equal(iono_delay_sort, est_iono, decimal=0)

        # Test corrections without ionosphere parameters
        with pytest.warns(RuntimeWarning):
            est_trp, est_iono = gnss_models.calculate_pseudorange_corr(
                curr_millis, state=state, ephem=vis_ephems[idx])
            np.testing.assert_almost_equal(tropo_delay_sort, est_trp, decimal=0)
            # Ionosphere delay should be zero without iono parameters
            np.testing.assert_almost_equal(np.zeros(len(frame)), est_iono, decimal=0)


        # Test corrections without receiver position
        with pytest.warns(RuntimeWarning):
            est_trp, est_iono = gnss_models.calculate_pseudorange_corr(
                curr_millis, ephem=vis_ephems[idx],
                iono_params =iono_params)
            # Ionosphere and troposphere delay should be zero without receiver position
            np.testing.assert_almost_equal(np.zeros(len(frame)), est_trp, decimal=0)
            np.testing.assert_almost_equal(np.zeros(len(frame)), est_iono, decimal=0)


def test_measure_generation(gps_measurement_frames, android_gt):
    """Test code to estimate expected measurements given observables.

    Parameters
    ----------
    gps_measurement_frames : Dict
        Dictionary containing lists of visible ephemeris parameters,
        received Android measurements and SV states. The lists are
        indexed by discrete time indices.
    android_gt : gnss_lib_py.parsers.android.AndroidGroundTruth2022
        NavData containing ground truth for Android measurements.
    """
    android_frames = gps_measurement_frames['android_frames']
    vis_ephems = gps_measurement_frames['vis_ephems']
    sv_states = gps_measurement_frames['sv_states']
    #Running this state only for a single measurement frame
    idx = 0
    curr_millis = android_frames[idx]['gps_millis', 0]
    gt_slice_idx = android_gt.argwhere('gps_millis', curr_millis)
    state = calculate_state(android_gt, gt_slice_idx)
    zero_noise_dict = {}
    zero_noise_dict['prange_sigma'] = 0.
    zero_noise_dict['doppler_sigma'] = 0.
    measurements_eph_sim, sv_posvel_eph = gnss_models.simulate_measures(curr_millis, state,
                        ephem=vis_ephems[idx], rng=default_rng(), noise_dict=zero_noise_dict)
    measurements_sv_sim, sv_posvel_used = gnss_models.simulate_measures(curr_millis, state,
                        sv_posvel=sv_states[idx], noise_dict=zero_noise_dict)
    # Check that the output position is the same as the given position
    input_sv_pos, input_sv_vel = _extract_pos_vel_arr(sv_states[idx])
    out_sv_pos, out_sv_vel = _extract_pos_vel_arr(sv_posvel_used)
    np.testing.assert_almost_equal(input_sv_pos, out_sv_pos)
    np.testing.assert_almost_equal(input_sv_vel, out_sv_vel)

    # Test that the calculated satellite positions are similar as inputs
    eph_sv_pos, eph_sv_vel = _extract_pos_vel_arr(sv_posvel_eph)
    np.testing.assert_almost_equal(input_sv_pos, eph_sv_pos, decimal=-1)
    np.testing.assert_almost_equal(input_sv_vel, eph_sv_vel, decimal=-1)
    # Test that the measurements are similar for similar time but
    # different ways of computing satellite states
    np.testing.assert_almost_equal(measurements_eph_sim['raw_pr_m'], measurements_sv_sim['raw_pr_m'], decimal=-1)
    np.testing.assert_almost_equal(measurements_eph_sim['doppler_hz'], measurements_sv_sim['doppler_hz'], decimal=-1)

    # Test that the measurements are similar for expected model
    measurements_exp_eph, _ = gnss_models.expected_measures(curr_millis, state,
                            ephem=vis_ephems[idx])
    measurements_exp_sv, _ = gnss_models.expected_measures(curr_millis, state,
                            sv_posvel=sv_states[idx])
    np.testing.assert_almost_equal(measurements_exp_eph['est_pr_m'], measurements_exp_sv['est_pr_m'], decimal=-1)
    np.testing.assert_almost_equal(measurements_exp_eph['est_doppler_hz'], measurements_exp_sv['est_doppler_hz'], decimal=-1)

    # Test that the measurements are similar for expected model and zero
    # noise simulated model
    np.testing.assert_almost_equal(measurements_exp_eph['est_pr_m'], measurements_eph_sim['raw_pr_m'], decimal=-1)
    np.testing.assert_almost_equal(measurements_exp_eph['est_doppler_hz'], measurements_eph_sim['doppler_hz'], decimal=-1)

    # Test that the measurements are similar for default noise levels
    # different generators with same seed
    seeded_rng = default_rng(seed=0)
    measurements_first_run_sim, _ = gnss_models.simulate_measures(curr_millis, state,
                        sv_posvel=sv_states[idx], rng=seeded_rng)
    measurements_second_run_sim, _ = gnss_models.simulate_measures(curr_millis, state,
                        sv_posvel=sv_states[idx], rng=seeded_rng)
    np.testing.assert_almost_equal(measurements_first_run_sim['raw_pr_m'], measurements_second_run_sim['raw_pr_m'], decimal=-1)
    np.testing.assert_almost_equal(measurements_first_run_sim['doppler_hz'], measurements_second_run_sim['doppler_hz'], decimal=-1)


@pytest.mark.parametrize('android_measurements',
                         [lazy_fixture("android_gps_l1"),
                        #   lazy_fixture("android_gps_l1_reversed")
                         ])
@pytest.mark.filterwarnings("ignore:.*invalid value encountered in divide.*: RuntimeWarning")
def test_add_measures_wrapper(android_measurements, android_state,
                              ephemeris_path, iono_params, error_tol_dec):
    """Test wrapper that adds SV states to received measurements.

    Parameters
    ----------
    android_measurements : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing L1 measurements for received GPS
        measurements.
    ephemeris_path : string
        The location where ephemeris files are read from or downloaded to
        if they don't exist.
    iono_params : np.ndarray
        2x4 (first row, alpha and second row, beta) of ionospheric delay
        parameters.
    error_tol_dec : Dict
        Dictionary containing decimals for error tolerances in computed
        states. Used for comparing to SV states provided in Android
        Derived measurements.

    """
    corr_rows = ['iono_delay_m', 'tropo_delay_m']
    measure_rows = ['est_pr_m', 'est_doppler_hz']
    sv_rows = ['x_sv_m', 'y_sv_m', 'z_sv_m', \
            'vx_sv_mps', 'vy_sv_mps', 'vz_sv_mps', 'b_sv_m']
    rx_pos_rows = ['x_rx_m', 'y_rx_m', 'z_rx_m']
    rx_vel_rows = ['vx_rx_mps', 'vy_rx_mps', 'vz_rx_mps']
    all_rows = corr_rows + sv_rows
    #TODO: Add pseudoranges and doppler in the above rows
    comparison_states = NavData()
    for row in all_rows:
        comparison_states[row] = android_measurements[row]
    android_measurements.remove(corr_rows, inplace=True)
    state_estimate = solve_wls(android_measurements, only_bias=True,
                               receiver_state=android_measurements)
    android_measurements['vx_rx_mps'] = 0
    android_measurements['vy_rx_mps'] = 0
    android_measurements['vz_rx_mps'] = 0
    android_measurements['b_rx_m'] = np.repeat(state_estimate['b_rx_wls_m'], 7)
    android_measurements['b_dot_rx_mps'] = 0
    measures = gnss_models.add_measures(android_measurements, android_state,
                                        ephemeris_path, iono_params)
    for row in corr_rows:
        # Test that results of SV state and other calculations is correct
        if 'delay' in row:
            np.testing.assert_almost_equal(measures[row],
                                        comparison_states[row],
                                        decimal=error_tol_dec['delay'])

    # Test measurement estimation without given ionosphere parameters

    measures_extract_iono = gnss_models.add_measures(android_measurements,
                                                     android_state,
                                                     ephemeris_path,
                                                     iono_params=None)
    for row in corr_rows:
        np.testing.assert_almost_equal(measures[row], measures_extract_iono[row])
    # Test measurement estimation when SV states are not provided

    android_without_sv = android_measurements.remove(sv_rows)
    measures_without_sv = gnss_models.add_measures(android_without_sv,
                                                   android_state,
                                                   ephemeris_path,
                                                   iono_params)
    for row in all_rows:
        # Test that results of SV state and other calculaations is correct
        if 'sv_mps' in row:
            np.testing.assert_almost_equal(measures[row],
                                        comparison_states[row],
                                        decimal=error_tol_dec['vel'])
        elif 'sv_m' in row:
            np.testing.assert_almost_equal(measures[row],
                                        comparison_states[row],
                                        decimal=error_tol_dec['brd_eph'])
        elif 'delay' in row:
            np.testing.assert_almost_equal(measures[row],
                                        comparison_states[row],
                                        decimal=error_tol_dec['delay'])
        elif row=='b_sv_m':
            np.testing.assert_almost_equal(measures[row],
                                        comparison_states[row],
                                        decimal=error_tol_dec['clock'])

    # Test measure estimation without Rx states in state
    with pytest.raises(KeyError):
        android_state_without_rx = android_state.copy()
        android_state_without_rx.remove('x_rx_m', inplace=True)
        _ = gnss_models.add_measures(measures, android_state_without_rx,
                                     ephemeris_path, iono_params)
    android_state_without_rxv = android_state.remove(rx_vel_rows)
    with pytest.warns(RuntimeWarning):

        _ = gnss_models.add_measures(measures, android_state_without_rxv,
                                    ephemeris_path,iono_params)
    # Check whether correct rows exist for different flags
    measures_no_pseudo = gnss_models.add_measures(android_measurements,
                                                  android_state,
                                                  ephemeris_path,
                                                  iono_params,
                                                  pseudorange=False)
    no_pseudo_rows = corr_rows + ['est_doppler_hz']
    measures_no_pseudo.in_rows(no_pseudo_rows)

    measures_no_doppler = gnss_models.add_measures(android_measurements,
                                                   android_state,
                                                   ephemeris_path,
                                                   iono_params,
                                                   doppler=False)
    no_doppler_rows = corr_rows + ['est_pr_m']
    measures_no_doppler.in_rows(no_doppler_rows)

    measures_no_corr = gnss_models.add_measures(android_measurements,
                                                android_state,
                                                ephemeris_path,
                                                iono_params,
                                                corrections = False)
    measures_no_corr.in_rows(measure_rows)

    measures_only_corr = gnss_models.add_measures(android_measurements,
                                                  android_state,
                                                  ephemeris_path,
                                                  iono_params,
                                                  pseudorange = False,
                                                  doppler = False)
    measures_only_corr.in_rows(corr_rows)
    with pytest.raises(KeyError):
        measures_only_corr.in_rows(measure_rows)
