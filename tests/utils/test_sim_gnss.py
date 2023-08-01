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

from gnss_lib_py.utils import sim_gnss
import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.coordinates import geodetic_to_ecef
from gnss_lib_py.parsers.ephemeris import EphemerisManager
from gnss_lib_py.utils.time_conversions import datetime_to_tow



T = 0.1

def timestamp():
    """Set timestamp for getting satellite positions."""
    return datetime.datetime(2020, 5, 15, 0, 47, 15, 448796, pytz.UTC)

def ephem_man():
    """Create emphemeris manager for GNSS satellites."""
    parent_directory = os.getcwd()

    ephemeris_data_directory = os.path.join(parent_directory,
                                        'data', 'unit_test', 'ephemeris')

    return EphemerisManager(ephemeris_data_directory)

def set_rx_ecef():
    """Set receiver positon in Earth-Centered, Earth-Fixed coordinates."""
    rx_lla  = np.reshape([37.427112, -122.1764146, 16], [1, 3])
    rx_ecef = np.reshape(geodetic_to_ecef(rx_lla), [3, 1])
    return rx_ecef

def extract_ephem():
    """Extract satellite ephemeris."""
    manager = ephem_man()
    sats = [f"G{sv_num:02d}" for sv_num in range(1,33)]
    ephemeris = manager.get_ephemeris(timestamp(), sats)
    return ephemeris

def simulate_test_measures(delta_time=0):
    """Set receiver positon in Earth-Centered, Earth-Fixed coordinates.

    Parameters
    ----------
    delta_time : float
        Delta time forward from time set in timestamp()

    Returns
    -------
    measurements : pd.DataFrame
        Pseudorange and doppler measurements indexed by satellite SV with
        Gaussian noise
    sv_posvel : pd.DataFrame
        Satellite positions and velocities (same as input if provided)

    """
    gpsweek = 2105
    _, gpstime = datetime_to_tow(timestamp())

    #NOTE: Calling measures to generate measurements to test measures?
    measurement, sv_posvel = sim_gnss.simulate_measures(
        gpsweek, gpstime + delta_time, extract_ephem(), set_rx_ecef(),
        0., 0., np.zeros([3, 1]))

    return measurement, sv_posvel

# Define test fixtures
@pytest.fixture(name="get_meas")
def fixture_get_meas():
    """Set simulated test measurements.

    Returns
    -------
    measurements : pd.DataFrame
        Pseudorange and doppler measurements indexed by satellite SV with
        Gaussian noise
    sv_posvel : pd.DataFrame
        Satellite positions and velocities (same as input if provided)

    """
    measurement, sv_posvel = simulate_test_measures()
    return measurement, sv_posvel

@pytest.fixture(name="get_meas_dt")
def fixture_get_meas_dt():
    """Set simulated test measurements at t + T.

    Returns
    -------
    measurements : pd.DataFrame
        Pseudorange and doppler measurements indexed by satellite SV with
        Gaussian noise
    sv_posvel : pd.DataFrame
        Satellite positions and velocities (same as input if provided)

    """
    measurement, sv_posvel = simulate_test_measures(delta_time=T)
    return measurement, sv_posvel

@pytest.fixture(name="set_xyz")
def fixture_set_xyz():
    """Set position and velocity inputs for testing pos/vel extract function.

    Returns
    -------
    pos_array : np.ndarray
        3x3 position array
    vel_array : np.ndarray
        3x3 velocity array
    times : np.ndarray
        1x3 velocity array

    """
    pos_array = np.array([
        [100, 200, 300],
        [400, 500, 600],
        [700, 800, 900],
    ])
    vel_array = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
    ])
    times = np.array([[1.], [1.], [1.]])
    return pos_array, vel_array, times

@pytest.fixture(name="expected_elaz")
def fixture_expected_elaz():
    """Set the expected elevation and azimuth from sample positions.

    Returns
    -------
    expect_elaz : np.ndarray
        Array containing 6 el/az pairs for testing elaz function

    """
    expect_elaz = np.array([[90.0, -90.0, 0.0 ,  0.0 , 0.0, 0.0  ],
                            [0.0 ,  0.0 , 90.0, -90.0, 0.0, 180.0]]).T
    return expect_elaz

@pytest.fixture(name="set_sv_pos")
def fixture_set_sv_pos():
    """Set the sample satellite positions for computing elevation and azimuth.

    Returns
    -------
    sv_pos : np.ndarray
        Array containing 6 satellite x, y, z coordinates

    """
    sv_pos = np.zeros([6, 3])
    sv_pos[0,0] =  consts.A*1.25
    sv_pos[1,0] =  consts.A*0.75
    sv_pos[2,0] =  consts.A
    sv_pos[2,1] =  consts.A
    sv_pos[3,0] =  consts.A
    sv_pos[3,1] = -consts.A
    sv_pos[4,0] =  consts.A
    sv_pos[4,2] =  consts.A
    sv_pos[5,0] =  consts.A
    sv_pos[5,2] = -consts.A
    return sv_pos

@pytest.fixture(name="set_rx_pos")
def fixture_set_rx_pos():
    """Set the sample reciever position for computing elaz.

    Returns
    -------
    rx_pos : np.ndarray
        Array containing 6 satellite x, y, z coordinates

    """
    rx_pos = np.reshape(np.array([consts.A, 0, 0]), [1, 3])
    return rx_pos

# Define tests
def test_extract_xyz(set_xyz):
    """Test the extraction of satellite epoch, position, and velocity from
    Pandas DataFrame to numpy array.

    Parameters
    ----------
    set_xyz : fixture
        Setter for sample x, y, and z position

    """
    pos_array, vel_array, times = set_xyz
    sv_names = np.array(['G01', 'G02', 'G03'])
    comb_array = np.hstack((times, pos_array, vel_array))
    test_posvel = pd.DataFrame(
        data = comb_array,
        index = sv_names,
        columns = ['times', 'x_sv_m', 'y_sv_m', 'z_sv_m',
                   'vx_sv_mps', 'vy_sv_mps', 'vz_sv_mps']
    )
    prns, test_pos, test_vel = sim_gnss._extract_pos_vel_arr(test_posvel)

    np.testing.assert_array_equal(test_pos, pos_array)
    np.testing.assert_array_equal(test_vel, vel_array)
    np.testing.assert_array_equal(prns, np.array([1, 2, 3]))

def test_find_elaz(expected_elaz, set_sv_pos, set_rx_pos):
    """Test receiver to satellite azimuth and elevation calculation.

    Parameters
    ----------
    expected_elaz : fixture
        Expected elevation and azimuth angles generated by the given satellite
        and receiver positions.
    set_sv_pos : fixture
        Satellite position setter
    set_rx_pos : fixture
        Receiver position setter

    """
    calc_elaz = sim_gnss.find_elaz(set_rx_pos, set_sv_pos)
    np.testing.assert_array_almost_equal(expected_elaz, calc_elaz)

def test_measures_value_range(get_meas):
    """Test the order of magnitude of simulated pseudorange measurements,
    simulated doppler measurements, and returned satellite positions.

    Parameters
    ----------
    get_meas : fixture
        Measurements simulated at the base timestamp

    """
    measurements, sv_posvel = get_meas

    assert np.logical_and(measurements['prange'].values > 20000e3,
        measurements['prange'].values < 3e7).all(), ("Invalid range of "
        "pseudorange values")

    assert np.all(np.abs(measurements['doppler'].values) < 5000), \
        "Magnitude of doppler values is greater than 5 KHz"

    assert np.all(np.abs(sv_posvel['x_sv_m']).values < consts.A + 2e7), \
    ("Invalid range of ECEF x for satellite position")

    assert np.all(np.abs(sv_posvel['y_sv_m']).values < consts.A + 2e7), \
    ("Invalid range of ECEF y for satellite position")

    assert np.all(np.abs(sv_posvel['z_sv_m']).values < consts.A + 2e7), \
    ("Invalid range of ECEF z for satellite position")

def test_sv_velocity(get_meas, get_meas_dt):
    """Test that satellite velocity at an adjacent timestep can be approximated
    by the difference of satellite positions at adjacent timesteps.

    Parameters
    ----------
    get_meas : fixture
        Measurements simulated at the base timestamp
    get_meas_dt : fixture
        Measurements simulated at the base timestamp + T

    """
    _, sv_posvel_prev = get_meas
    _, sv_posvel_new  = get_meas_dt

    np.testing.assert_array_almost_equal(
        (sv_posvel_new['x_sv_m'] - sv_posvel_prev['x_sv_m']) / T,
        sv_posvel_prev['vx_sv_mps'], decimal=1)

    np.testing.assert_array_almost_equal(
        (sv_posvel_new['y_sv_m'] - sv_posvel_prev['y_sv_m']) / T,
        sv_posvel_prev['vy_sv_mps'], decimal=1)

    np.testing.assert_array_almost_equal(
        (sv_posvel_new['z_sv_m'] - sv_posvel_prev['z_sv_m']) / T,
        sv_posvel_prev['vz_sv_mps'], decimal=1)

def test_measure_sizes(get_meas):
    """Test that the size of the simulated measurements is equal to the size
    of the satellite positions and velocities output by simulate_measures.

    Parameters
    ----------
    get_meas : fixture
        Measurements simulated at the base timestamp

    """
    measurements, sv_posvel = get_meas
    assert (len(measurements['prange'].index)==len(sv_posvel.index))

def test_pseudorange_corrections(get_meas, get_meas_dt):
    """Test pseudorange correction consistency by checking the difference
    between the expected and corrected pseudorange measurements at t and t + T.

    Parameters
    ----------
    get_meas : fixture
        Measurements simulated at the base timestamp
    get_meas_dt : fixture
        Measurements simulated at the base timestamp + T

    """
    #TODO: Alternatively check ranges of corrections/against true values
    meas_prev, _ = get_meas
    meas_new , _  = get_meas_dt

    gpsweek = 2105
    _, gpstime = datetime_to_tow(timestamp())

    rx_ecef = np.reshape(set_rx_ecef(), [-1, 3])

    ephem = extract_ephem()

    sv_names = (meas_prev.index).tolist()

    meas_prev_corr = sim_gnss.correct_pseudorange(
        gpstime, gpsweek, ephem.where("sv_id",sv_names), meas_prev['prange'], rx_ecef)

    meas_new_corr  = sim_gnss.correct_pseudorange(
        gpstime+T, gpsweek, ephem.where("sv_id",sv_names), meas_new['prange'], rx_ecef)

    diff_prev = meas_prev_corr - meas_prev['prange']
    diff_new  = meas_new_corr  - meas_new['prange']
    np.testing.assert_array_almost_equal(diff_prev, diff_new, decimal=2)
