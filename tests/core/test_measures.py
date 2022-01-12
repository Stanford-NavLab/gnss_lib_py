"""Tests for measures.

"""

__authors__ = "Ashwin Kanhere, Bradley Collicott"
__date__ = "6 Aug 2021"

import os
import datetime

import pytz
import pytest
import numpy as np
import pandas as pd

from gnss_lib_py.core.coordinates import geodetic2ecef
import gnss_lib_py.core.constants as consts
from gnss_lib_py.core import measures
from gnss_lib_py.core.ephemeris import datetime2tow
from gnss_lib_py.core.ephemeris import EphemerisManager



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
    rx_ecef = np.reshape(geodetic2ecef(rx_lla), [3, 1])
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
    sat_posvel : pd.DataFrame
        Satellite positions and velocities (same as input if provided)

    """
    gpsweek = 2105
    _, gpstime = datetime2tow(timestamp())

    #NOTE: Calling measures to generate measurements to test measures?
    measurement, sat_posvel = measures.simulate_measures(
        gpsweek, gpstime + delta_time, extract_ephem(), set_rx_ecef(),
        0., 0., np.zeros([3, 1]))

    return measurement, sat_posvel

# Define test fixtures
@pytest.fixture(name="get_meas")
def fixture_get_meas():
    """Set simulated test measurements.

    Returns
    -------
    measurements : pd.DataFrame
        Pseudorange and doppler measurements indexed by satellite SV with
        Gaussian noise
    sat_posvel : pd.DataFrame
        Satellite positions and velocities (same as input if provided)

    """
    measurement, sat_posvel = simulate_test_measures()
    return measurement, sat_posvel

@pytest.fixture(name="get_meas_dt")
def fixture_get_meas_dt():
    """Set simulated test measurements at t + T.

    Returns
    -------
    measurements : pd.DataFrame
        Pseudorange and doppler measurements indexed by satellite SV with
        Gaussian noise
    sat_posvel : pd.DataFrame
        Satellite positions and velocities (same as input if provided)

    """
    measurement, sat_posvel = simulate_test_measures(delta_time=T)
    return measurement, sat_posvel

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

@pytest.fixture(name="set_sat_pos")
def fixture_set_sat_pos():
    """Set the sample satellite positions for computing elevation and azimuth.

    Returns
    -------
    sat_pos : np.ndarray
        Array containing 6 satellite x, y, z coordinates

    """
    sat_pos = np.zeros([6, 3])
    sat_pos[0,0] =  consts.A*1.25
    sat_pos[1,0] =  consts.A*0.75
    sat_pos[2,0] =  consts.A
    sat_pos[2,1] =  consts.A
    sat_pos[3,0] =  consts.A
    sat_pos[3,1] = -consts.A
    sat_pos[4,0] =  consts.A
    sat_pos[4,2] =  consts.A
    sat_pos[5,0] =  consts.A
    sat_pos[5,2] = -consts.A
    return sat_pos

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
        columns = ['times', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    )
    prns, test_pos, test_vel = measures._extract_pos_vel_arr(test_posvel)

    np.testing.assert_array_equal(test_pos, pos_array)
    np.testing.assert_array_equal(test_vel, vel_array)
    np.testing.assert_array_equal(prns, np.array([1, 2, 3]))

def test_find_elaz(expected_elaz, set_sat_pos, set_rx_pos):
    """Test receiver to satellite azimuth and elevation calculation.

    Parameters
    ----------
    expected_elaz : fixture
        Expected elevation and azimuth angles generated by the given satellite
        and receiver positions.
    set_sat_pos : fixture
        Satellite position setter
    set_rx_pos : fixture
        Receiver position setter

    """
    calc_elaz = measures.find_elaz(set_rx_pos, set_sat_pos)
    #print(calc_elaz)
    np.testing.assert_array_almost_equal(expected_elaz, calc_elaz)

def test_measures_value_range(get_meas):
    """Test the order of magnitude of simulated pseudorange measurements,
    simulated doppler measurements, and returned satellite positions.

    Parameters
    ----------
    get_meas : fixture
        Measurements simulated at the base timestamp

    """
    measurements, sat_posvel = get_meas

    assert np.logical_and(measurements['prange'].values > 20000e3,
        measurements['prange'].values < 3e7).all(), ("Invalid range of "
        "pseudorange values")

    assert np.all(np.abs(measurements['doppler'].values) < 5000), \
        "Magnitude of doppler values is greater than 5 KHz"

    assert np.all(np.abs(sat_posvel['x']).values < consts.A + 2e7), \
    ("Invalid range of ECEF x for satellite position")

    assert np.all(np.abs(sat_posvel['y']).values < consts.A + 2e7), \
    ("Invalid range of ECEF y for satellite position")

    assert np.all(np.abs(sat_posvel['z']).values < consts.A + 2e7), \
    ("Invalid range of ECEF z for satellite position")

def test_sat_velocity(get_meas, get_meas_dt):
    """Test that satellite velocity at an adjacent timestep can be approximated
    by the difference of satellite positions at adjacent timesteps.

    Parameters
    ----------
    get_meas : fixture
        Measurements simulated at the base timestamp
    get_meas_dt : fixture
        Measurements simated at the base timestamp + T

    """
    _, sat_posvel_prev = get_meas
    _, sat_posvel_new  = get_meas_dt

    np.testing.assert_array_almost_equal(
        (sat_posvel_new['x'] - sat_posvel_prev['x']) / T,
        sat_posvel_prev['vx'], decimal=1)

    np.testing.assert_array_almost_equal(
        (sat_posvel_new['y'] - sat_posvel_prev['y']) / T,
        sat_posvel_prev['vy'], decimal=1)

    np.testing.assert_array_almost_equal(
        (sat_posvel_new['z'] - sat_posvel_prev['z']) / T,
        sat_posvel_prev['vz'], decimal=1)

def test_measure_sizes(get_meas):
    """Test that the size of the simulated measurements is equal to the size
    of the satellite positions and velocities output by simulate_measures.

    Parameters
    ----------
    get_meas : fixture
        Measurements simulated at the base timestamp

    """
    measurements, sat_posvel = get_meas
    assert (len(measurements['prange'].index)==len(sat_posvel.index))

def test_pseudorange_corrections(get_meas, get_meas_dt):
    """Test pseudorange correction consistency by checking the difference
    between the expected and corrected pseudorange measurements at t and t + T.

    Parameters
    ----------
    get_meas : fixture
        Measurements simulated at the base timestamp
    get_meas_dt : fixture
        Measurements simated at the base timestamp + T

    """
    #TODO: Alternatively check ranges of corrections/against true values
    meas_prev, _ = get_meas
    meas_new , _  = get_meas_dt

    gpsweek = 2105
    _, gpstime = datetime2tow(timestamp())

    rx_ecef = np.reshape(set_rx_ecef(), [-1, 3])

    ephem = extract_ephem()

    sat_names = (meas_prev.index).tolist()
    print(ephem.loc[sat_names,:])

    meas_prev_corr = measures.correct_pseudorange(
        gpstime, gpsweek, ephem.loc[sat_names,:], meas_prev['prange'], rx_ecef)

    meas_new_corr  = measures.correct_pseudorange(
        gpstime+T, gpsweek, ephem.loc[sat_names,:], meas_new['prange'], rx_ecef)

    diff_prev = meas_prev_corr - meas_prev['prange']
    diff_new  = meas_new_corr  - meas_new['prange']
    np.testing.assert_array_almost_equal(diff_prev, diff_new, decimal=2)
