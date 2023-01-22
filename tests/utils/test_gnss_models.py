"""Tests for GNSS SV state calculation methods.

"""

__authors__ = "Ashwin Kanhere, Bradley Collicott"
__date__ = "6 Aug 2021"


import pytest
import numpy as np

from numpy.random import default_rng

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
    iono_params = np.array([[0.9313E-08,  0.1490E-07, -0.5960E-07, -0.1192E-06],
                        [0.8806E+05,  0.4915E+05, -0.1311E+06, -0.3277E+06]])
    return iono_params


def test_state_extraction():
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
    return state


def test_pseudorange_corrections(gps_measurement_frames, android_gt, iono_params):
    android_frames = gps_measurement_frames['android_frames']
    vis_ephems = gps_measurement_frames['vis_ephems']
    sv_states = gps_measurement_frames['sv_states']
    for idx, frame in enumerate(android_frames):
        sort_arg = np.argsort(frame['sv_id'])
        # Get Android Derived states, sorted by SVs
        tropo_delay_sort = frame['tropo_delay_m'][sort_arg]
        iono_delay_sort = frame['iono_delay_m'][sort_arg]
        clock_corr_sort = frame['b_sv_m'][sort_arg]
        curr_millis = frame['gps_millis', 0]
        gt_slice_idx = android_gt.argwhere('gps_millis', curr_millis)
        x_ecef = android_gt[['x_gt_m', 'y_gt_m', 'z_gt_m'], gt_slice_idx]

        # Test corrections with ephemeris parameters
        est_clk, est_trp, est_iono = gnss_models._calculate_pseudorange_corr(
                                        curr_millis, ephem=vis_ephems[idx],
                                        iono_params =iono_params, rx_ecef = x_ecef)
        np.testing.assert_almost_equal(tropo_delay_sort, est_trp, decimal=0)
        np.testing.assert_almost_equal(iono_delay_sort, est_iono, decimal=0)
        np.testing.assert_almost_equal(clock_corr_sort, est_clk, decimal=0)

        # Test corrections without ephemeris parameters
        sv_posvel = sv_states[idx]
        with pytest.warns(RuntimeWarning):
            est_clk, est_trp, est_iono = gnss_models._calculate_pseudorange_corr(
                                            curr_millis, sv_posvel=sv_posvel,
                                            iono_params =iono_params, rx_ecef = x_ecef)
            np.testing.assert_almost_equal(tropo_delay_sort, est_trp, decimal=0)
            np.testing.assert_almost_equal(iono_delay_sort, est_iono, decimal=0)
            # Clock correction should be zero without epehemeris parameters
            np.testing.assert_almost_equal(np.zeros(len(frame)), est_clk)

        # Test corrections without ionosphere parameters
        with pytest.warns(RuntimeWarning):
            est_clk, est_trp, est_iono = gnss_models._calculate_pseudorange_corr(
                                            curr_millis, ephem=vis_ephems[idx],
                                            rx_ecef = x_ecef)
            np.testing.assert_almost_equal(tropo_delay_sort, est_trp, decimal=0)
            # Ionosphere delay should be zero without iono parameters
            np.testing.assert_almost_equal(np.zeros(len(frame)), est_iono, decimal=0)
            np.testing.assert_almost_equal(clock_corr_sort, est_clk, decimal=0)


        # Test corrections without receiver position
        with pytest.warns(RuntimeWarning):
            est_clk, est_trp, est_iono = gnss_models._calculate_pseudorange_corr(
                                            curr_millis, ephem=vis_ephems[idx],
                                            iono_params =iono_params)
            # Ionosphere and troposphere delay should be zero without receiver position
            np.testing.assert_almost_equal(np.zeros(len(frame)), est_trp, decimal=0)
            np.testing.assert_almost_equal(np.zeros(len(frame)), est_iono, decimal=0)
            np.testing.assert_almost_equal(clock_corr_sort, est_clk, decimal=0)


def calculate_state(android_gt, idx):
    bearing_rad = np.deg2rad(android_gt['BearingDegrees', idx])

    v_gt_n = android_gt['SpeedMps', idx]*np.cos(bearing_rad)
    v_gt_e = android_gt['SpeedMps', idx]*np.sin(bearing_rad)
    v_ned = np.array([[v_gt_n],[v_gt_e],[0]])
    llh = np.array([[android_gt['lat_gt_deg', idx]],
                    [android_gt['lon_gt_deg', idx]],
                    [android_gt['alt_gt_m', idx]]])
    local_frame = LocalCoord.from_geodetic(llh)
    vx_ecef = local_frame.ned_to_ecefv(v_ned)

    state = NavData()
    state['x_rx_m'] = android_gt['x_gt_m', idx]
    state['y_rx_m'] = android_gt['y_gt_m', idx]
    state['z_rx_m'] = android_gt['z_gt_m', idx]
    state['vx_rx_mps'] = vx_ecef[0,0]
    state['vy_rx_mps'] = vx_ecef[1,0]
    state['vz_rx_mps'] = vx_ecef[2,0]
    state['b_rx_m'] = 0
    state['b_dot_rx_mps'] = 0
    return state




def test_measure_generation(gps_measurement_frames, android_gt):
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
    measurements_eph, sv_posvel_eph = gnss_models.simulate_measures(curr_millis, state,
                        ephem=vis_ephems[idx], rng=default_rng(), noise_dict=zero_noise_dict)
    measurements_sv, sv_posvel_used = gnss_models.simulate_measures(curr_millis, state,
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
    np.testing.assert_almost_equal(measurements_eph['prange'], measurements_sv['prange'], decimal=-1)
    np.testing.assert_almost_equal(measurements_eph['doppler'], measurements_sv['doppler'], decimal=-1)

    # Test that the measurements are similar for expected model
    measurements_exp_eph, _ = gnss_models.expected_measures(curr_millis, state,
                            ephem=vis_ephems[idx])
    measurements_exp_sv, _ = gnss_models.expected_measures(curr_millis, state,
                            sv_posvel=sv_states[idx])
    np.testing.assert_almost_equal(measurements_exp_eph['prange'], measurements_exp_sv['prange'], decimal=-1)
    np.testing.assert_almost_equal(measurements_exp_eph['doppler'], measurements_exp_sv['doppler'], decimal=-1)

    # Test that the measurements are similar for expected model and zero
    # noise simulated model
    np.testing.assert_almost_equal(measurements_exp_eph['prange'], measurements_eph['prange'], decimal=-1)
    np.testing.assert_almost_equal(measurements_exp_eph['doppler'], measurements_eph['doppler'], decimal=-1)

    # Test that the measurements are similar for default noise levels
    # different generators with same seed
    seeded_rng = default_rng(seed=0)
    measurements_first_run, _ = gnss_models.simulate_measures(curr_millis, state,
                        sv_posvel=sv_states[idx], rng=seeded_rng)
    measurements_second_run, _ = gnss_models.simulate_measures(curr_millis, state,
                        sv_posvel=sv_states[idx], rng=seeded_rng)
    np.testing.assert_almost_equal(measurements_first_run['prange'], measurements_second_run['prange'], decimal=-1)
    np.testing.assert_almost_equal(measurements_first_run['doppler'], measurements_second_run['doppler'], decimal=-1)



