"""Tests for GNSS SV state calculation methods.

"""

__authors__ = "Ashwin Kanhere, Bradley Collicott"
__date__ = "6 Aug 2021"

import pytest
import numpy as np


from gnss_lib_py.parsers.navdata import NavData
import gnss_lib_py.utils.sv_models as sv_models
# pylint: disable=protected-access

# Number of time to run meausurement simulation code
TEST_REPEAT_COUNT = 10


@pytest.fixture(name="scaling_value")
def fixture_scaling_value():
    scaling_value = np.arange(6).astype(float)
    return scaling_value


@pytest.fixture(name="dummy_pos_vel")
def fixture_dummy_pos_vel(scaling_value):
    dummy_posvel = NavData()
    dummy_posvel['x_sv_m'] = scaling_value
    dummy_posvel['y_sv_m'] = 10.*scaling_value
    dummy_posvel['z_sv_m'] = 100.*scaling_value
    dummy_posvel['vx_sv_mps'] = -scaling_value
    dummy_posvel['vy_sv_mps'] = -10.*scaling_value
    dummy_posvel['vz_sv_mps'] = -100.*scaling_value
    return dummy_posvel


def test_svs_from_elaz():
    el_deg = np.array([0, 0, 45, 60])
    az_deg = np.array([0, 90, 0, 60])
    input_elaz = np.vstack((el_deg, az_deg))

    sin_45 = np.sqrt(1/2)
    cos_45 = sin_45
    sin_60 = np.sin(np.deg2rad(60))
    cos_60 = np.cos(np.deg2rad(60))
    exp_x = np.array([0, 1, 0, cos_60*sin_60])
    exp_y = np.array([1, 0, cos_45, cos_60*cos_60])
    exp_z = np.array([0, 0, sin_45, sin_60])
    unit_vect = np.vstack((exp_x, exp_y, exp_z))
    exp_sats = 20200000*unit_vect/np.linalg.norm(unit_vect, axis=0)
    out_sats = sv_models.svs_from_el_az(input_elaz)
    np.testing.assert_almost_equal(out_sats, exp_sats)


def test_posvel_extract(dummy_pos_vel, scaling_value):
    out_pos, out_vel = sv_models._extract_pos_vel_arr(dummy_pos_vel)
    exp_pos = np.vstack((scaling_value, 10*scaling_value, 100*scaling_value))
    exp_vel = np.vstack((-scaling_value, -10*scaling_value, -100*scaling_value))
    np.testing.assert_almost_equal(out_pos, exp_pos)
    np.testing.assert_almost_equal(out_vel, exp_vel)


def test_del_xyz_range(dummy_pos_vel, scaling_value):
    test_rx_pos = np.zeros([3, 1])
    out_del_xyz, out_range = sv_models._find_delxyz_range(dummy_pos_vel, test_rx_pos)
    exp_del_xyz = np.vstack((scaling_value, 10*scaling_value, 100*scaling_value))
    exp_range = scaling_value*np.linalg.norm([1, 10, 100])
    np.testing.assert_almost_equal(out_del_xyz, exp_del_xyz)
    np.testing.assert_almost_equal(out_range, exp_range)
    with pytest.raises(ValueError):
        test_rx_pos_4d = np.zeros([4,1])
        out_del_xyz, out_range = sv_models._find_delxyz_range(dummy_pos_vel, test_rx_pos_4d)



def test_sv_state_model(gps_measurement_frames, android_gt):
    android_frames = gps_measurement_frames['android_frames']
    vis_ephems = gps_measurement_frames['vis_ephems']
    sv_states = gps_measurement_frames['sv_states']
    for idx, and_sv_posvel in enumerate(sv_states):
        # Get Android Derived states, sorted by SVs
        curr_millis = android_frames[idx]['gps_millis', 0]
        gt_slice_idx = android_gt.argwhere('gps_millis', curr_millis)
        x_ecef = android_gt[['x_gt_m', 'y_gt_m', 'z_gt_m'], gt_slice_idx]


        est_sv_posvel, _, _ = sv_models._find_sv_location(curr_millis, x_ecef, ephem=vis_ephems[idx])
        np.testing.assert_almost_equal(and_sv_posvel[['x_sv_m']], est_sv_posvel['x_sv_m'], decimal=-1)
        np.testing.assert_almost_equal(and_sv_posvel[['y_sv_m']], est_sv_posvel['y_sv_m'], decimal=-1)
        np.testing.assert_almost_equal(and_sv_posvel[['z_sv_m']], est_sv_posvel['z_sv_m'], decimal=-1)
        np.testing.assert_almost_equal(and_sv_posvel[['vx_sv_mps']], est_sv_posvel['vx_sv_mps'], decimal=-1)
        np.testing.assert_almost_equal(and_sv_posvel[['vy_sv_mps']], est_sv_posvel['vy_sv_mps'], decimal=-1)
        np.testing.assert_almost_equal(and_sv_posvel[['vz_sv_mps']], est_sv_posvel['vz_sv_mps'], decimal=-1)


def test_visible_ephem(all_gps_ephem, gps_measurement_frames, android_gt):
    android_frames = gps_measurement_frames['android_frames']
    vis_ephems = gps_measurement_frames['vis_ephems']
    for idx, frame in enumerate(android_frames):
        curr_millis = frame['gps_millis', 0]
        gt_slice_idx = android_gt.argwhere('gps_millis', curr_millis)
        x_ecef = android_gt[['x_gt_m', 'y_gt_m', 'z_gt_m'], gt_slice_idx]
        # Test visible satellite computation with ephemeris
        eph = sv_models._find_visible_ephem(curr_millis, x_ecef, ephem=vis_ephems[idx], el_mask=0.)
        vis_svs = set(eph['sv_id'])
        assert vis_svs == set(vis_ephems[idx]['sv_id'])

        # Test that actually visible satellites are subset of expected satellites
        eph = sv_models._find_visible_ephem(curr_millis, x_ecef, ephem=all_gps_ephem, el_mask=0.)
        vis_svs = set(eph['sv_id'])
        assert set(vis_ephems[idx]['sv_id']).issubset(vis_svs)


def test_visible_sv_posvel(gps_measurement_frames, android_gt):
    android_frames = gps_measurement_frames['android_frames']
    sv_states = gps_measurement_frames['sv_states']
    for idx, sv_posvel in enumerate(sv_states):
        curr_millis = android_frames[idx]['gps_millis', 0]
        gt_slice_idx = android_gt.argwhere('gps_millis', curr_millis)
        x_ecef = android_gt[['x_gt_m', 'y_gt_m', 'z_gt_m'], gt_slice_idx]

        # Test that actually visible satellites are subset of expected satellites
        vis_posvel = sv_models._find_visible_sv_posvel(curr_millis, x_ecef, sv_posvel, el_mask=0.)
        vis_svs = set(vis_posvel['sv_id'])
        assert vis_svs.issubset(set(sv_posvel['sv_id']))

