"""Tests for simulating GNSS measurements.

"""

__authors__ = "Ashwin Kanhere, Bradley Collicott"
__date__ = "6 Aug 2021"

import os
import datetime

import pytz
import pytest
import numpy as np
import pandas as pd


from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.parsers.ephemeris import EphemerisManager
import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.coordinates import geodetic_to_ecef
from gnss_lib_py.utils.time_conversions import datetime_to_tow
from gnss_lib_py.utils import sim_gnss

# pylint: disable=protected-access

# Number of time to run meausurement simulation code
TEST_REPEAT_COUNT = 10

#TODO: Where is this used?
T = 0.1

@pytest.fixture(name="android_root_path")
def fixture_root_path():
    """Location of Android Derived 2021 measurements for unit test

    Returns
    -------
    root_path : string
        Folder location containing Android Derived 2021 measurements
    """
    root_path = os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
    root_path = os.path.join(root_path, 'data/unit_test/android_2021')
    return root_path


@pytest.fixture(name="derived_path")
def fixture_derived_path(android_root_path):
    """Filepath of Android Derived 2021 measurements

    Parameters
    ----------
    root_path : string
        Path of testing dataset root path

    Returns
    -------
    derived_path : string
        Location for the unit_test Android derived measurements

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset [1]_,
    particularly the train/2020-05-14-US-MTV-1/Pixel4 trace. The dataset
    was retrieved from
    https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data

    References
    ----------
    .. [1] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.
    """
    derived_path = os.path.join(android_root_path, 'Pixel4_derived.csv')
    return derived_path


def test_sv_posvel(derived_path):
    assert True

def test_tropo_delay(derived_path):
    assert True

def test_iono_delay(derived_path):
    assert True

def test_clock_delay(derived_path):
    assert True


def test_expect_measure(derived_path):
    assert True

def test_simulate_measure(derived_path):
    # use TEST_REPEAT_COUNT here
    assert True


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
    out_sats = sim_gnss.svs_from_el_az(input_elaz)
    np.testing.assert_almost_equal(out_sats, exp_sats)


def test_posvel_extract(dummy_pos_vel, scaling_value):
    out_pos, out_vel = sim_gnss._extract_pos_vel_arr(dummy_pos_vel)
    exp_pos = np.vstack((scaling_value, 10*scaling_value, 100*scaling_value))
    exp_vel = np.vstack((-scaling_value, -10*scaling_value, -100*scaling_value))
    np.testing.assert_almost_equal(out_pos, exp_pos)
    np.testing.assert_almost_equal(out_vel, exp_vel)


def test_del_xyz_range(dummy_pos_vel, scaling_value):
    test_rx_pos = np.zeros([3, 1])
    out_del_xyz, out_range = sim_gnss._find_delxyz_range(dummy_pos_vel, test_rx_pos)
    exp_del_xyz = np.vstack((scaling_value, 10*scaling_value, 100*scaling_value))
    exp_range = scaling_value*np.linalg.norm([1, 10, 100])
    np.testing.assert_almost_equal(out_del_xyz, exp_del_xyz)
    np.testing.assert_almost_equal(out_range, exp_range)


# def timestamp():
#     """Set timestamp for getting satellite positions."""
#     return datetime.datetime(2020, 5, 15, 0, 47, 15, 448796, pytz.UTC)

# def ephem_man():
#     """Create emphemeris manager for GNSS satellites."""
#     parent_directory = os.getcwd()

#     ephemeris_data_directory = os.path.join(parent_directory,
#                                         'data', 'unit_test', 'ephemeris')

#     return EphemerisManager(ephemeris_data_directory)

# def set_rx_ecef():
#     """Set receiver positon in Earth-Centered, Earth-Fixed coordinates."""
#     rx_lla  = np.reshape([37.427112, -122.1764146, 16], [1, 3])
#     rx_ecef = np.reshape(geodetic_to_ecef(rx_lla), [3, 1])
#     return rx_ecef

# def extract_ephem():
#     """Extract satellite ephemeris."""
#     manager = ephem_man()
#     sats = [f"G{sv_num:02d}" for sv_num in range(1,33)]
#     ephemeris = manager.get_ephemeris(timestamp(), sats)
#     return ephemeris

# def simulate_test_measures(delta_time=0):
#     """Set receiver positon in Earth-Centered, Earth-Fixed coordinates.

#     Parameters
#     ----------
#     delta_time : float
#         Delta time forward from time set in timestamp()

#     Returns
#     -------
#     measurements : pd.DataFrame
#         Pseudorange and doppler measurements indexed by satellite SV with
#         Gaussian noise
#     sv_posvel : pd.DataFrame
#         Satellite positions and velocities (same as input if provided)

#     """
#     gpsweek = 2105
#     _, gpstime = datetime_to_tow(timestamp())

#     #NOTE: Calling measures to generate measurements to test measures?
#     measurement, sv_posvel = sim_gnss.simulate_measures(
#         gpsweek, gpstime + delta_time, extract_ephem(), set_rx_ecef(),
#         0., 0., np.zeros([3, 1]))

#     return measurement, sv_posvel

# # Define test fixtures
# @pytest.fixture(name="get_meas")
# def fixture_get_meas():
#     """Set simulated test measurements.

#     Returns
#     -------
#     measurements : pd.DataFrame
#         Pseudorange and doppler measurements indexed by satellite SV with
#         Gaussian noise
#     sv_posvel : pd.DataFrame
#         Satellite positions and velocities (same as input if provided)

#     """
#     measurement, sv_posvel = simulate_test_measures()
#     return measurement, sv_posvel

# @pytest.fixture(name="get_meas_dt")
# def fixture_get_meas_dt():
#     """Set simulated test measurements at t + T.

#     Returns
#     -------
#     measurements : pd.DataFrame
#         Pseudorange and doppler measurements indexed by satellite SV with
#         Gaussian noise
#     sv_posvel : pd.DataFrame
#         Satellite positions and velocities (same as input if provided)

#     """
#     measurement, sv_posvel = simulate_test_measures(delta_time=T)
#     return measurement, sv_posvel

# @pytest.fixture(name="set_xyz")
# def fixture_set_xyz():
#     """Set position and velocity inputs for testing pos/vel extract function.

#     Returns
#     -------
#     pos_array : np.ndarray
#         3x3 position array
#     vel_array : np.ndarray
#         3x3 velocity array
#     times : np.ndarray
#         1x3 velocity array

#     """
#     pos_array = np.array([
#         [100, 200, 300],
#         [400, 500, 600],
#         [700, 800, 900],
#     ])
#     vel_array = np.array([
#         [10, 20, 30],
#         [40, 50, 60],
#         [70, 80, 90],
#     ])
#     times = np.array([[1.], [1.], [1.]])
#     return pos_array, vel_array, times

# @pytest.fixture(name="expected_elaz")
# def fixture_expected_elaz():
#     """Set the expected elevation and azimuth from sample positions.

#     Returns
#     -------
#     expect_elaz : np.ndarray
#         Array containing 6 el/az pairs for testing elaz function

#     """
#     expect_elaz = np.array([[90.0, -90.0, 0.0 ,  0.0 , 0.0, 0.0  ],
#                             [0.0 ,  0.0 , 90.0, -90.0, 0.0, 180.0]]).T
#     return expect_elaz

# @pytest.fixture(name="set_sv_pos")
# def fixture_set_sv_pos():
#     """Set the sample satellite positions for computing elevation and azimuth.

#     Returns
#     -------
#     sv_pos : np.ndarray
#         Array containing 6 satellite x, y, z coordinates

#     """
#     sv_pos = np.zeros([6, 3])
#     sv_pos[0,0] =  consts.A*1.25
#     sv_pos[1,0] =  consts.A*0.75
#     sv_pos[2,0] =  consts.A
#     sv_pos[2,1] =  consts.A
#     sv_pos[3,0] =  consts.A
#     sv_pos[3,1] = -consts.A
#     sv_pos[4,0] =  consts.A
#     sv_pos[4,2] =  consts.A
#     sv_pos[5,0] =  consts.A
#     sv_pos[5,2] = -consts.A
#     return sv_pos

# @pytest.fixture(name="set_rx_pos")
# def fixture_set_rx_pos():
#     """Set the sample reciever position for computing elaz.

#     Returns
#     -------
#     rx_pos : np.ndarray
#         Array containing 6 satellite x, y, z coordinates

#     """
#     rx_pos = np.reshape(np.array([consts.A, 0, 0]), [1, 3])
#     return rx_pos

# # Define tests
# def test_find_elaz(expected_elaz, set_sv_pos, set_rx_pos):
#     """Test receiver to satellite azimuth and elevation calculation.

#     Parameters
#     ----------
#     expected_elaz : fixture
#         Expected elevation and azimuth angles generated by the given satellite
#         and receiver positions.
#     set_sv_pos : fixture
#         Satellite position setter
#     set_rx_pos : fixture
#         Receiver position setter

#     """
#     calc_elaz = sim_gnss.find_elaz(set_rx_pos, set_sv_pos)
#     np.testing.assert_array_almost_equal(expected_elaz, calc_elaz)

# def test_measures_value_range(get_meas):
#     """Test the order of magnitude of simulated pseudorange measurements,
#     simulated doppler measurements, and returned satellite positions.

#     Parameters
#     ----------
#     get_meas : fixture
#         Measurements simulated at the base timestamp

#     """
#     measurements, sv_posvel = get_meas

#     assert np.logical_and(measurements['prange'].values > 20000e3,
#         measurements['prange'].values < 3e7).all(), ("Invalid range of "
#         "pseudorange values")

#     assert np.all(np.abs(measurements['doppler'].values) < 5000), \
#         "Magnitude of doppler values is greater than 5 KHz"

#     assert np.all(np.abs(sv_posvel['x']).values < consts.A + 2e7), \
#     ("Invalid range of ECEF x for satellite position")

#     assert np.all(np.abs(sv_posvel['y']).values < consts.A + 2e7), \
#     ("Invalid range of ECEF y for satellite position")

#     assert np.all(np.abs(sv_posvel['z']).values < consts.A + 2e7), \
#     ("Invalid range of ECEF z for satellite position")

# def test_sv_velocity(get_meas, get_meas_dt):
#     """Test that satellite velocity at an adjacent timestep can be approximated
#     by the difference of satellite positions at adjacent timesteps.

#     Parameters
#     ----------
#     get_meas : fixture
#         Measurements simulated at the base timestamp
#     get_meas_dt : fixture
#         Measurements simulated at the base timestamp + T

#     """
#     _, sv_posvel_prev = get_meas
#     _, sv_posvel_new  = get_meas_dt

#     np.testing.assert_array_almost_equal(
#         (sv_posvel_new['x'] - sv_posvel_prev['x']) / T,
#         sv_posvel_prev['vx'], decimal=1)

#     np.testing.assert_array_almost_equal(
#         (sv_posvel_new['y'] - sv_posvel_prev['y']) / T,
#         sv_posvel_prev['vy'], decimal=1)

#     np.testing.assert_array_almost_equal(
#         (sv_posvel_new['z'] - sv_posvel_prev['z']) / T,
#         sv_posvel_prev['vz'], decimal=1)

# def test_measure_sizes(get_meas):
#     """Test that the size of the simulated measurements is equal to the size
#     of the satellite positions and velocities output by simulate_measures.

#     Parameters
#     ----------
#     get_meas : fixture
#         Measurements simulated at the base timestamp

#     """
#     measurements, sv_posvel = get_meas
#     assert (len(measurements['prange'].index)==len(sv_posvel.index))

# def test_pseudorange_corrections(get_meas, get_meas_dt):
#     """Test pseudorange correction consistency by checking the difference
#     between the expected and corrected pseudorange measurements at t and t + T.

#     Parameters
#     ----------
#     get_meas : fixture
#         Measurements simulated at the base timestamp
#     get_meas_dt : fixture
#         Measurements simulated at the base timestamp + T

#     """
#     #TODO: Alternatively check ranges of corrections/against true values
#     meas_prev, _ = get_meas
#     meas_new , _  = get_meas_dt

#     gpsweek = 2105
#     _, gpstime = datetime_to_tow(timestamp())

#     rx_ecef = np.reshape(set_rx_ecef(), [-1, 3])

#     ephem = extract_ephem()

#     sv_names = (meas_prev.index).tolist()

#     meas_prev_corr = sim_gnss.correct_pseudorange(
#         gpstime, gpsweek, ephem.loc[sv_names,:], meas_prev['prange'], rx_ecef)

#     meas_new_corr  = sim_gnss.correct_pseudorange(
#         gpstime+T, gpsweek, ephem.loc[sv_names,:], meas_new['prange'], rx_ecef)

#     diff_prev = meas_prev_corr - meas_prev['prange']
#     diff_new  = meas_new_corr  - meas_new['prange']
#     np.testing.assert_array_almost_equal(diff_prev, diff_new, decimal=2)
