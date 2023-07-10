"""Tests for GNSS SV state calculation methods.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "21 Mar 2023"

import os

import pytest
import numpy as np
import pandas as pd
from pytest_lazyfixture import lazy_fixture

from gnss_lib_py.parsers.navdata import NavData
import gnss_lib_py.utils.sv_models as sv_models
from gnss_lib_py.parsers.android import AndroidDerived2021
# pylint: disable=protected-access

# Number of time to run meausurement simulation code
TEST_REPEAT_COUNT = 10

# Define the keys relevant for satellite information
SV_KEYS = ['x_sv_m', 'y_sv_m', 'z_sv_m', \
           'vx_sv_mps','vy_sv_mps','vz_sv_mps', \
           'b_sv_m', 'b_dot_sv_mps']

@pytest.fixture(name="scaling_value")
def fixture_scaling_value():
    """Scaling value to test extract_pos_vel function.

    Returns
    -------
    scaling_value : np.ndarray
        Scaling value for testing position and velocity extration code.
    """
    scaling_value = np.arange(6).astype(float)
    return scaling_value


@pytest.fixture(name="dummy_pos_vel")
def fixture_dummy_pos_vel(scaling_value):
    """Fixture to create NavData for testing position, velocity extraction.

    Parameters
    ----------
    scaling_value : np.ndarray
        Linear range for 6 instances of positions and velocities.

    Returns
    -------
    dummy_posvel = gnss_lib_py.parsers.navdata.NavData
        NavData example containing position and velocity.
    """
    dummy_posvel = NavData()
    dummy_posvel['x_sv_m'] = scaling_value
    dummy_posvel['y_sv_m'] = 10.*scaling_value
    dummy_posvel['z_sv_m'] = 100.*scaling_value
    dummy_posvel['vx_sv_mps'] = -scaling_value
    dummy_posvel['vy_sv_mps'] = -10.*scaling_value
    dummy_posvel['vz_sv_mps'] = -100.*scaling_value
    return dummy_posvel


def test_svs_from_elaz():
    """Test that the SVs generated from elevation and azimuth are correct.

    Uses fixed values of the elevation and azimuth and tests them against
    satellite vehicle positions known for extreme angles.
    """
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
    """Test extraction of position and velocity

    Parameters
    ----------
    dummy_pos_vel : gnss_lib_py.parsers.navdata.NavData
        NavData example containing position and velocity.
    scaling_value : np.ndarray
        Linear range for 6 instances of positions and velocities.

    """
    out_pos, out_vel = sv_models._extract_pos_vel_arr(dummy_pos_vel)
    exp_pos = np.vstack((scaling_value, 10*scaling_value, 100*scaling_value))
    exp_vel = np.vstack((-scaling_value, -10*scaling_value, -100*scaling_value))
    np.testing.assert_almost_equal(out_pos, exp_pos)
    np.testing.assert_almost_equal(out_vel, exp_vel)
    #TODO: Put the following statements in the testing file
    assert np.shape(out_pos)[0]==3, "sv_pos: Incorrect shape Expected 3xN"
    assert np.shape(out_vel)[0]==3, "sv_vel: Incorrect shape Expected 3xN"


def test_del_xyz_range(dummy_pos_vel, scaling_value):
    """Test calculation of position difference and range calculations.

    Parameters
    ----------
    dummy_pos_vel : gnss_lib_py.parsers.navdata.NavData
        NavData example containing position and velocity.
    scaling_value : np.ndarray
        Linear range for 6 instances of positions and velocities.

    """
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
    """Test models used to calculate GPS SV positions and velocities.

    Tests models that use broadcast ephemeris parameters to estimate SV
    states for GPS

    Parameters
    ----------
    gps_measurement_frames : Dict
        Dictionary containing NavData instances of ephemeris parameters
        for received satellites, received Android measurements and SV
        states, all corresponding to the same received time frame.
    android_gt : gnss_lib_py.parsers.navdata.NavData
        Ground truth for received measurements.
    """
    android_frames = gps_measurement_frames['android_frames']
    vis_ephems = gps_measurement_frames['vis_ephems']
    sv_states = gps_measurement_frames['sv_states']
    for idx, and_sv_posvel in enumerate(sv_states):
        # Get Android Derived states, sorted by SVs
        curr_millis = android_frames[idx]['gps_millis', 0]
        gt_slice_idx = android_gt.argwhere('gps_millis', curr_millis)
        x_ecef = android_gt[['x_rx_gt_m', 'y_rx_gt_m', 'z_rx_gt_m'], gt_slice_idx]

        est_sv_posvel, _, _ = sv_models.find_sv_location(curr_millis, x_ecef, ephem=vis_ephems[idx])

        np.testing.assert_almost_equal(and_sv_posvel[['x_sv_m']], est_sv_posvel['x_sv_m'], decimal=-1)
        np.testing.assert_almost_equal(and_sv_posvel[['y_sv_m']], est_sv_posvel['y_sv_m'], decimal=-1)
        np.testing.assert_almost_equal(and_sv_posvel[['z_sv_m']], est_sv_posvel['z_sv_m'], decimal=-1)
        np.testing.assert_almost_equal(and_sv_posvel[['vx_sv_mps']], est_sv_posvel['vx_sv_mps'], decimal=-1)
        np.testing.assert_almost_equal(and_sv_posvel[['vy_sv_mps']], est_sv_posvel['vy_sv_mps'], decimal=-1)
        np.testing.assert_almost_equal(and_sv_posvel[['vz_sv_mps']], est_sv_posvel['vz_sv_mps'], decimal=-1)
        np.testing.assert_almost_equal(and_sv_posvel[['b_sv_m']], est_sv_posvel['b_sv_m'], decimal=1)


def test_visible_ephem(all_gps_ephem, gps_measurement_frames, android_gt):
    """Verify process for finding visible satellites.

    Verifies method for computing visible satellites by checking that
    actually received satellites are subset of computed visible satellites
    and that all received satellites are computed as visible.

    Parameters
    ----------

    all_gps_ephem : gnss_lib_py.parsers.navdata.NavData
        Ephemeris parameters for all satellites at the time when measurements
        were received.
    gps_measurement_frames : Dict
        Dictionary containing NavData instances of ephemeris parameters
        for received satellites, received Android measurements and SV
        states, all corresponding to the same received time frame.
    android_gt : gnss_lib_py.parsers.navdata.NavData
        Ground truth for received measurements.
    """
    android_frames = gps_measurement_frames['android_frames']
    vis_ephems = gps_measurement_frames['vis_ephems']
    for idx, frame in enumerate(android_frames):
        curr_millis = frame['gps_millis', 0]
        gt_slice_idx = android_gt.argwhere('gps_millis', curr_millis)
        x_ecef = android_gt[['x_rx_gt_m', 'y_rx_gt_m', 'z_rx_gt_m'], gt_slice_idx]
        # Test visible satellite computation with ephemeris
        eph = sv_models.find_visible_ephem(curr_millis, x_ecef, ephem=vis_ephems[idx], el_mask=0.)
        vis_svs = set(eph['sv_id'])
        assert vis_svs == set(vis_ephems[idx]['sv_id'])

        # Test that actually visible satellites are subset of expected satellites
        eph = sv_models.find_visible_ephem(curr_millis, x_ecef, ephem=all_gps_ephem, el_mask=0.)
        vis_svs = set(eph['sv_id'])
        assert set(vis_ephems[idx]['sv_id']).issubset(vis_svs)


def test_visible_sv_posvel(gps_measurement_frames, android_gt):
    """Test that correct SVs are used for positions for visible satellites.

    Parameters
    ----------
    gps_measurement_frames : Dict
        Dictionary containing NavData instances of ephemeris parameters
        for received satellites, received Android measurements and SV
        states, all corresponding to the same received time frame.
    android_gt : gnss_lib_py.parsers.navdata.NavData
        Ground truth for received measurements.
    """
    android_frames = gps_measurement_frames['android_frames']
    sv_states = gps_measurement_frames['sv_states']
    for idx, sv_posvel in enumerate(sv_states):
        curr_millis = android_frames[idx]['gps_millis', 0]
        gt_slice_idx = android_gt.argwhere('gps_millis', curr_millis)
        x_ecef = android_gt[['x_rx_gt_m', 'y_rx_gt_m', 'z_rx_gt_m'], gt_slice_idx]

        # Test that actually visible satellites are subset of expected satellites
        vis_posvel = sv_models.find_visible_sv_posvel(x_ecef, sv_posvel, el_mask=0.)
        vis_svs = set(vis_posvel['sv_id'])
        assert vis_svs.issubset(set(sv_posvel['sv_id']))

@pytest.mark.parametrize('android_measurements',
                         [lazy_fixture("android_gps_l1"),
                          lazy_fixture("android_gps_l1_reversed")
                         ])
def test_add_sv_state_wrapper(android_measurements, ephemeris_path, error_tol_dec):
    """Test wrapper that adds SV states to received measurements.

    Parameters
    ----------
    android_measurements : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing L1 measurements for received GPS
        measurements.
    ephemeris_path : string
        The location where ephemeris files are read from or downloaded to
        if they don't exist.
    error_tol_dec : Dict
        Dictionary containing decimals for error tolerances in computed
        states. Used for comparing to SV states provided in Android
        Derived measurements.

    """
    true_rows = ['x_sv_m', 'y_sv_m', 'z_sv_m', 'el_sv_deg', 'az_sv_deg',
                 'vx_sv_mps', 'vy_sv_mps', 'vz_sv_mps','iono_delay_m',
                 'tropo_delay_m']
    comparison_states = NavData()
    for row in true_rows:
        comparison_states[row] = android_measurements[row]
    android_measurements.remove(true_rows, inplace=True)
    android_gps_states = sv_models.add_sv_states(android_measurements, ephemeris_path)
    for row in true_rows:
        if 'mps' in row:
            np.testing.assert_almost_equal(android_gps_states[row],
                                        comparison_states[row],
                                        decimal=error_tol_dec['vel'])
        elif 'sv_m' in row:
            np.testing.assert_almost_equal(android_gps_states[row],
                                        comparison_states[row],
                                        decimal=error_tol_dec['pos'])
    # Test position estimation when desired constellations are not in
    # received measurements
    with pytest.warns(RuntimeWarning):
        android_gps_states = sv_models.add_sv_states(android_measurements, ephemeris_path,
                                constellations=['gps', 'glonass'])
    # Testing position estimation without receiver position
    android_measurements.remove(rows=['x_rx_m', 'y_rx_m', 'z_rx_m'], inplace=True)
    android_gps_states = sv_models.add_sv_states(android_measurements, ephemeris_path)
    for row in true_rows:
        if 'mps' in row:
            np.testing.assert_almost_equal(android_gps_states[row],
                                        comparison_states[row],
                                        decimal=error_tol_dec['vel'])
        elif 'sv_m' in row:
            np.testing.assert_almost_equal(android_gps_states[row],
                                        comparison_states[row],
                                        decimal=error_tol_dec['brd_eph'])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_filter_ephemeris_none(android_gps_l1, ephemeris_path):
    """Test the case when _filter_ephemeris_measurements is given None.

    Parameters
    ----------
    android_gps_l1 : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing L1 measurements for received GPS
        measurements.
    ephemeris_path : string
        The location where ephemeris files are read from or downloaded to
        if they don't exist.
    """
    android_subset,_, _ = sv_models._filter_ephemeris_measurements(android_gps_l1,
                                                                constellations=None,
                                                                ephemeris_path=ephemeris_path)
    assert len(android_gps_l1)==len(android_subset)


def test_add_visible_svs_for_trajectory(android_gps_l1, ephemeris_path,
                                        error_tol_dec):
    """
    Test add_visible_svs_for_trajectory wrapper in sv_models

    Parameters
    ----------
    android_gps_l1 : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing L1 measurements for received GPS
        measurements.
    ephemeris_path : string
        The location where ephemeris files are read from or downloaded to
        if they don't exist.
    error_tol_dec : Dict
        Dictionary containing decimals for error tolerances in computed
        states. Used for comparing to SV states provided in Android
        Derived measurements.
    """
    # Create list of times and states from android_gps_l1 into an estimate
    state_estimate = NavData()
    unique_svs = np.unique(android_gps_l1['sv_id'])
    android_sv = android_gps_l1.where("sv_id", unique_svs[0])
    state_estimate['gps_millis'] = android_sv['gps_millis']
    state_estimate['x_rx_m'] = android_sv['x_rx_m']
    state_estimate['y_rx_m'] = android_sv['y_rx_m']
    state_estimate['z_rx_m'] = android_sv['z_rx_m']
    sv_posvel_traj = sv_models.add_visible_svs_for_trajectory(state_estimate,
                                                             ephemeris_path)
    # assert that actually received SVs in the given times are a
    # subset of those considered visible
    true_rows = ['x_sv_m', 'y_sv_m', 'z_sv_m', 'vx_sv_mps', 'vy_sv_mps',
                 'vz_sv_mps']
    for milli, _, measure_frame in android_gps_l1.loop_time("gps_millis",
                                                                 delta_t_decimals=-2):
        se_frame = sv_posvel_traj.where("gps_millis", milli)
        se_svs = set(np.unique(se_frame['sv_id']))
        measure_svs = set(np.unique(measure_frame['sv_id']))
        assert measure_svs.issubset(se_svs)
        # Check that estimated states match
        for sv_id in measure_svs:
            measure_frame_sv = measure_frame.where("sv_id", sv_id)
            se_frame_sv = se_frame.where("sv_id", sv_id)
            for row in true_rows:
                if 'mps' in row:
                    np.testing.assert_almost_equal(se_frame_sv[row],
                                                measure_frame_sv[row],
                                                decimal=error_tol_dec['vel'])
                elif 'sv_m' in row:
                    np.testing.assert_almost_equal(se_frame_sv[row],
                                                measure_frame_sv[row],
                                                decimal=error_tol_dec['brd_eph'])

    # Test same function with None for satellites. No support for non-GPS
    # constellations currently and should raise an error
    with pytest.warns(RuntimeWarning):
        _ = sv_models.add_visible_svs_for_trajectory(state_estimate,
                                                    ephemeris_path,
                                                    constellations=None)

@pytest.fixture(name="navdata_path")
def fixture_navdata_path(root_path):
    """Filepath of Android Derived measurements

    Returns
    -------
    navdata_path : string
        Location for the unit_test Android derived measurements

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset [3]_,
    particularly the train/2021-04-28-US-SJC-1/Pixel4 trace. The dataset
    was retrieved from
    https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data

    References
    ----------
    .. [3] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.
    """
    navdata_path = os.path.join(root_path, "android_2021/Pixel4_derived_clkdiscnt.csv")
    return navdata_path

@pytest.fixture(name="navdata")
def fixture_load_navdata(navdata_path):
    """Load instance of AndroidDerived2021

    Parameters
    ----------
    navdata_path : pytest.fixture
        String with location of Android navdata measurement file

    Returns
    -------
    navdata : AndroidDerived2021
        Instance of AndroidDerived2021 for testing

    """
    navdata = AndroidDerived2021(navdata_path, remove_timing_outliers=False)

    return navdata

@pytest.fixture(name="navdata_gps")
def fixture_load_navdata_gps(navdata):
    """Load GPS instance of AndroidDerived2021

    Parameters
    ----------
    navdata : pytest.fixture
        Instance of AndroidDerived for testing

    Returns
    -------
    navdata_gps : AndroidDerived2021
        Instance of AndroidDerived (GPS) for testing
    """
    navdata_gps = navdata.where("gnss_id", "gps")

    return navdata_gps

@pytest.fixture(name="navdata_gpsl1")
def fixture_load_navdata_gpsl1(navdata):
    """Load GPS instance of AndroidDerived2021

    Parameters
    ----------
    navdata : pytest.fixture
        Instance of AndroidDerived for testing

    Returns
    -------
    navdata_gpsl1 : AndroidDerived2021
        Instance of AndroidDerived (GPS-L1) for testing ephemeris
    """
    navdata_gpsl1 = navdata.where("gnss_id", "gps")
    navdata_gpsl1 = navdata_gpsl1.where('signal_type', 'l1')

    return navdata_gpsl1

@pytest.fixture(name="navdata_glonass")
def fixture_load_navdata_glonass(navdata):
    """Load GLONASS instance of AndroidDerived2021

    Parameters
    ----------
    navdata : pytest.fixture
        Instance of AndroidDerived for testing

    Returns
    -------
    navdata_glonass : AndroidDerived2021
        Instance of AndroidDerived (GLONASS) for testing
    """
    navdata_glonass = navdata.where("gnss_id", "glonass")

    return navdata_glonass

@pytest.fixture(name="navdata_glonassg1")
def fixture_load_navdata_glonassg1(navdata):
    """Load GLONASS instance of AndroidDerived2021

    Parameters
    ----------
    navdata : pytest.fixture
        Instance of AndroidDerived for testing

    Returns
    -------
    navdata_glonassg1 : AndroidDerived2021
        Instance of AndroidDerived (GLONASS-G1) for testing
    """
    navdata_glonassg1 = navdata.where("gnss_id", "glonass")
    navdata_glonassg1 = navdata_glonassg1.where('signal_type', 'g1')

    return navdata_glonassg1

def test_compute_gps_precise_eph(navdata_gps, sp3data, clkdata):
    """Tests that sv_models.single_gnss_from_precise_eph does not fail for GPS

    Notes
    ----------
    The threshold for assertion checks are set heuristically; not applicable if
    input unit test files are changed.

    Parameters
    ----------
    navdata_gps : pytest.fixture
        Instance of the NavData class that depicts android derived dataset
    sp3data : pytest.fixture
        Instance of Sp3 class dictionary
    clkdata : pytest.fixture
        Instance of Clk class dictionary
    """
    navdata_prcs_gps = navdata_gps.copy()
    navdata_prcs_gps = sv_models.single_gnss_from_precise_eph(navdata_prcs_gps, \
                                                        sp3data,
                                                        clkdata,
                                                        verbose=True)


    # Check if the resulting derived is NavData class
    assert isinstance( navdata_prcs_gps, type(NavData()) )

    for sval in SV_KEYS:
        # Check if the resulting navdata class has satellite information
        assert sval in navdata_prcs_gps.rows

        # Check if satellite info from AndroidDerived and sp3 closely resemble
        if (sval == 'x_sv_m') | (sval == 'y_sv_m') | (sval == 'z_sv_m'):
            assert max(abs(navdata_prcs_gps[sval] - navdata_gps[sval])) != 0.0
            assert max(abs(navdata_prcs_gps[sval] - navdata_gps[sval])) < 13e3 #300
        if (sval == 'vx_sv_mps') | (sval == 'vy_sv_mps') | (sval == 'vz_sv_mps'):
            assert max(abs(navdata_prcs_gps[sval] - navdata_gps[sval])) != 0.0
            assert max(abs(navdata_prcs_gps[sval] - navdata_gps[sval])) < 2 #5e-2
        if sval=='b_sv_m':
            assert max(abs(navdata_prcs_gps[sval] - navdata_gps[sval])) != 0.0
            assert max(abs(navdata_prcs_gps[sval] - navdata_gps[sval])) < 15
        if sval=='b_dot_sv_mps':
            assert max(abs(navdata_prcs_gps[sval] - navdata_gps[sval])) != 0.0
            assert max(abs(navdata_prcs_gps[sval] - navdata_gps[sval])) < 5e-3

    # Check if the derived classes are same except for the changed SV_KEYS
    navdata_gps_df = navdata_gps.pandas_df()
    navdata_gps_df = navdata_gps_df.drop(columns = SV_KEYS)

    navdata_prcs_gps_df = navdata_prcs_gps.pandas_df()
    navdata_prcs_gps_df = navdata_prcs_gps_df.drop(columns = SV_KEYS \
                                                    + ["gnss_sv_id"])

    pd.testing.assert_frame_equal(navdata_gps_df.sort_index(axis=1),
                                  navdata_prcs_gps_df.sort_index(axis=1),
                                  check_dtype=False, check_names=True)

def test_compute_glonass_precise_eph(navdata_glonass, sp3data, clkdata):
    """Tests that sv_models.single_gnss_from_precise_eph does not fail for GPS

    Notes
    ----------
    The threshold for assertion checks are set heuristically; not applicable if
    input unit test files are changed.

    Parameters
    ----------
    navdata_glonass : pytest.fixture
        Instance of the NavData class that depicts android derived dataset
    sp3data : pytest.fixture
        Instance of Sp3 class dictionary
    clkdata : pytest.fixture
        Instance of Clk class dictionary
    """
    navdata_prcs_glonass = navdata_glonass.copy()
    navdata_prcs_glonass.remove(rows=SV_KEYS,inplace=True)
    new_navdata = sv_models.single_gnss_from_precise_eph(navdata_prcs_glonass,
                                               sp3data, clkdata)

    # Check if the resulting derived is NavData class
    assert isinstance( new_navdata, type(NavData()) )

    for sval in SV_KEYS:
        # Check if the resulting navdata class has satellite information
        assert sval in new_navdata.rows

        # Check if satellite info from AndroidDerived and sp3 closely resemble
        if (sval == 'x_sv_m') | (sval == 'y_sv_m') | (sval == 'z_sv_m'):
            assert max(abs(new_navdata[sval] - navdata_glonass[sval])) != 0.0
            assert max(abs(new_navdata[sval] - navdata_glonass[sval])) < 13e3 #300
        if (sval == 'vx_sv_mps') | (sval == 'vy_sv_mps') | (sval == 'vz_sv_mps'):
            assert max(abs(new_navdata[sval] - navdata_glonass[sval])) != 0.0
            assert max(abs(new_navdata[sval] - navdata_glonass[sval])) < 2 #5e-2
        if sval=='b_sv_m':
            assert max(abs(new_navdata[sval] - navdata_glonass[sval])) != 0.0
            assert max(abs(new_navdata[sval] - navdata_glonass[sval])) < 15
        if sval=='b_dot_sv_mps':
            assert max(abs(new_navdata[sval] - navdata_glonass[sval])) != 0.0
            assert max(abs(new_navdata[sval] - navdata_glonass[sval])) < 5e-3

    # Check if the derived classes are same except for the changed SV_KEYS
    navdata_glonass_df = navdata_glonass.pandas_df()
    navdata_glonass_df = navdata_glonass_df.drop(columns = SV_KEYS)

    navdata_prcs_glonass_df = new_navdata.pandas_df()
    navdata_prcs_glonass_df = navdata_prcs_glonass_df.drop(columns = SV_KEYS\
                                                    + ["gnss_sv_id"])

    pd.testing.assert_frame_equal(navdata_glonass_df.sort_index(axis=1),
                                  navdata_prcs_glonass_df.sort_index(axis=1),
                                  check_dtype=False, check_names=True)
    # test inplace
    new_navdata = sv_models.single_gnss_from_precise_eph(navdata_prcs_glonass,
                                               sp3data, clkdata,
                                               inplace=True)

    # Check if the resulting derived is NavData class
    assert new_navdata is None

    for sval in SV_KEYS:
        # Check if the resulting navdata class has satellite information
        assert sval in navdata_prcs_glonass.rows

        # Check if satellite info from AndroidDerived and sp3 closely resemble
        if (sval == 'x_sv_m') | (sval == 'y_sv_m') | (sval == 'z_sv_m'):
            assert max(abs(navdata_prcs_glonass[sval] - navdata_glonass[sval])) != 0.0
            assert max(abs(navdata_prcs_glonass[sval] - navdata_glonass[sval])) < 13e3 #300
        if (sval == 'vx_sv_mps') | (sval == 'vy_sv_mps') | (sval == 'vz_sv_mps'):
            assert max(abs(navdata_prcs_glonass[sval] - navdata_glonass[sval])) != 0.0
            assert max(abs(navdata_prcs_glonass[sval] - navdata_glonass[sval])) < 2 #5e-2
        if sval=='b_sv_m':
            assert max(abs(navdata_prcs_glonass[sval] - navdata_glonass[sval])) != 0.0
            assert max(abs(navdata_prcs_glonass[sval] - navdata_glonass[sval])) < 15
        if sval=='b_dot_sv_mps':
            assert max(abs(navdata_prcs_glonass[sval] - navdata_glonass[sval])) != 0.0
            assert max(abs(navdata_prcs_glonass[sval] - navdata_glonass[sval])) < 5e-3

    # Check if the derived classes are same except for the changed SV_KEYS
    navdata_glonass_df = navdata_glonass.pandas_df()
    navdata_glonass_df = navdata_glonass_df.drop(columns = SV_KEYS)

    navdata_prcs_glonass_df = navdata_prcs_glonass.pandas_df()
    navdata_prcs_glonass_df = navdata_prcs_glonass_df.drop(columns = SV_KEYS\
                                                    + ["gnss_sv_id"])

    pd.testing.assert_frame_equal(navdata_glonass_df.sort_index(axis=1),
                                  navdata_prcs_glonass_df.sort_index(axis=1),
                                  check_dtype=False, check_names=True)

def test_gpscheck_sp3_eph(navdata_gpsl1, sp3data, clkdata):
    """Tests that validates GPS satellite 3-D position and velocity

    Parameters
    ----------
    navdata_gpsl1 : pytest.fixture
        Instance of the NavData class that depicts GPS-L1 android derived
        dataset for GPS-only constellation
    sp3data : pytest.fixture
        Instance of Sp3 class dictionary
    clkdata : pytest.fixture
        Instance of Clk class dictionary
    """

    navdata_sp3_result = navdata_gpsl1.copy()
    navdata_sp3_result = sv_models.single_gnss_from_precise_eph(navdata_sp3_result, \
                                                          sp3data, clkdata)
    navdata_eph_result = navdata_gpsl1.copy()
    navdata_eph_result = sv_models.sv_gps_from_brdcst_eph_duplicate(navdata_eph_result)

    for sval in SV_KEYS[0:6]:
        # Check if satellite info from sp3 and eph closely resemble
        # here, the threshold of 300 is set in a heuristic sense; need investigation
        if (sval == 'x_sv_m') | (sval == 'y_sv_m') | (sval == 'z_sv_m'):
            assert max(abs(navdata_sp3_result[sval] - navdata_eph_result[sval])) != 0.0
            assert max(abs(navdata_sp3_result[sval] - navdata_eph_result[sval])) < 4.0
        if (sval == 'vx_sv_mps') | (sval == 'vy_sv_mps') | (sval == 'vz_sv_mps'):
            assert max(abs(navdata_sp3_result[sval] - navdata_eph_result[sval])) != 0.0
            assert max(abs(navdata_sp3_result[sval] - navdata_eph_result[sval])) < 0.015


def test_compute_concat_precise_eph(navdata, sp3_path, clk_path):
    """Tests that add_sv_states_sp3_and_clk does not fail for multi-GNSS

    Notes
    ----------
    The threshold for assertion checks are set heuristically; not applicable if
    input unit test files are changed.

    Parameters
    ----------
    navdata : pytest.fixture
        Instance of AndroidDerived for testing
    sp3_path : string
        String with location for the unit_test sp3 measurements
    clk_path : string
        String with location for the unit_test clk measurements
    """

    gnss_consts = {'gps', 'glonass'}

    navdata_merged = NavData()
    navdata_merged = navdata.where('gnss_id',gnss_consts)

    navdata_prcs_merged = sv_models.add_sv_states_sp3_and_clk(navdata, sp3_path,
                                            clk_path,  verbose = True)

    navdata_prcs_merged = navdata_prcs_merged.where("gnss_id",gnss_consts)

    # Check if the resulting derived is NavData class
    assert isinstance( navdata_prcs_merged, NavData )

    for sval in SV_KEYS:
        # Check if the resulting navdata class has satellite information
        assert sval in navdata_prcs_merged.rows

        # Check if satellite info from AndroidDerived and sp3 closely resemble
        if (sval == 'x_sv_m') | (sval == 'y_sv_m') | (sval == 'z_sv_m'):
            assert max(abs(navdata_prcs_merged[sval] - navdata_merged[sval])) != 0.0
            assert max(abs(navdata_prcs_merged[sval] - navdata_merged[sval])) < 13e3 #300
        if (sval == 'vx_sv_mps') | (sval == 'vy_sv_mps') | (sval == 'vz_sv_mps'):
            assert max(abs(navdata_prcs_merged[sval] - navdata_merged[sval])) != 0.0
            assert max(abs(navdata_prcs_merged[sval] - navdata_merged[sval])) < 2 #5e-2
        if sval=='b_sv_m':
            assert max(abs(navdata_prcs_merged[sval] - navdata_merged[sval])) != 0.0
            assert max(abs(navdata_prcs_merged[sval] - navdata_merged[sval])) < 15
        if sval=='b_dot_sv_mps':
            assert max(abs(navdata_prcs_merged[sval] - navdata_merged[sval])) != 0.0
            assert max(abs(navdata_prcs_merged[sval] - navdata_merged[sval])) < 5e-3

    # Check if the derived classes are same except for the changed SV_KEYS
    navdata_merged_df = navdata_merged.pandas_df()
    navdata_merged_df = navdata_merged_df.drop(columns = SV_KEYS)

    navdata_prcs_merged_df = navdata_prcs_merged.pandas_df()
    navdata_prcs_merged_df = navdata_prcs_merged_df.drop(columns = SV_KEYS + ["gnss_sv_id"])

    pd.testing.assert_frame_equal(navdata_merged_df.sort_index(axis=1),
                                  navdata_prcs_merged_df.sort_index(axis=1),
                                  check_dtype=False, check_names=True)

def test_compute_gps_brdcst_eph(navdata_gpsl1, navdata, navdata_glonassg1):
    """Tests that sv_gps_from_brdcst_eph_duplicate does not fail for GPS

    Parameters
    ----------
    navdata_gpsl1 : pytest.fixture
        Instance of NavData class that depicts GPS-L1 only derived dataset
    navdata : pytest.fixture
        Instance of NavData class that depicts entire android derived dataset
    navdata_glonassg1 : pytest.fixture
        Instance of NavData class that depicts GLONASS-G1 only derived dataset
    """

    # test what happens when extra (multi-GNSS) rows down't exist
    with pytest.raises(RuntimeError) as excinfo:
        navdata_eph = navdata.copy()
        sv_models.sv_gps_from_brdcst_eph_duplicate(navdata_eph, verbose=True)
    assert "Multi-GNSS" in str(excinfo.value)

    # test what happens when invalid (non-GPS) rows down't exist
    with pytest.raises(RuntimeError) as excinfo:
        navdata_glonassg1_eph = navdata_glonassg1.copy()
        sv_models.sv_gps_from_brdcst_eph_duplicate(navdata_glonassg1_eph, verbose=True)
    assert "non-GPS" in str(excinfo.value)

    navdata_gpsl1_eph = navdata_gpsl1.copy()
    navdata_gpsl1_eph = sv_models.sv_gps_from_brdcst_eph_duplicate(navdata_gpsl1_eph,
                                               verbose=True)

    # Check if the resulting derived is NavData class
    assert isinstance( navdata_gpsl1_eph, type(NavData()) )

    for sval in SV_KEYS[0:6]:
        # Check if the resulting navdata class has satellite information
        assert sval in navdata_gpsl1_eph.rows

        # Check if satellite info from AndroidDerived and eph closely resemble
        # here, the threshold of 300 is set in a heuristic sense; need investigation
        if (sval == 'x_sv_m') | (sval == 'y_sv_m') | (sval == 'z_sv_m'):
            assert max(abs(navdata_gpsl1_eph[sval] - navdata_gpsl1[sval])) != 0.0
            assert max(abs(navdata_gpsl1_eph[sval] - navdata_gpsl1[sval])) < 13e3 #300.0
        if (sval == 'vx_sv_mps') | (sval == 'vy_sv_mps') | (sval == 'vz_sv_mps'):
            assert max(abs(navdata_gpsl1_eph[sval] - navdata_gpsl1[sval])) != 0.0
            assert max(abs(navdata_gpsl1_eph[sval] - navdata_gpsl1[sval])) < 2 #5e-2

    # Check if the derived classes are same except for corr_pr_m
    navdata_gpsl1_df = navdata_gpsl1.pandas_df()
    navdata_gpsl1_df = navdata_gpsl1_df.drop(columns = SV_KEYS[0:6])

    navdata_gpsl1_eph_df = navdata_gpsl1_eph.pandas_df()
    navdata_gpsl1_eph_df = navdata_gpsl1_eph_df.drop(columns = SV_KEYS[0:6])

    pd.testing.assert_frame_equal(navdata_gpsl1_df.sort_index(axis=1),
                                  navdata_gpsl1_eph_df.sort_index(axis=1),
                                  check_dtype=False, check_names=True)
