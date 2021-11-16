"""Tests for measures.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "6 Aug 2021"

import os
import sys
import datetime
# append <path>/gnss_lib_py/gnss_lib_py/ to path
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__)))))

import pytz
import pytest
import numpy as np
import pandas as pd

from gnss_lib_py.core.coordinates import geodetic2ecef
from gnss_lib_py.core.constants import GPSConsts
from gnss_lib_py.core import measures
from gnss_lib_py.core.ephemeris import datetime_to_tow
from gnss_lib_py.core.ephemeris import EphemerisManager


gpsconsts = GPSConsts()
dt = 0.1

def timestamp():
    return datetime.datetime(2020, 5, 15, 0, 47, 15, 448796, pytz.UTC)

def ephem_man():
    parent_directory = os.getcwd()
    src_directory = os.path.join(parent_directory, 'src')

    ephemeris_data_directory = os.path.join(parent_directory, 
                                            'data', 'ephemeris')

    return EphemerisManager(ephemeris_data_directory)

def Rx_ECEF():
    rx_lla  = np.reshape([37.427112, -122.1764146, 16], [1, 3])
    rx_ecef = np.reshape(geodetic2ecef(rx_lla), [3, 1])
    return rx_ecef

def extract_ephem():
    manager = ephem_man()
    sats = ['G'+"%02d"%sv_num for sv_num in range(1, 33)]
    ephemeris = manager.get_ephemeris(timestamp(), sats)
    return ephemeris

def simulate_test_measures(deltaT=0):
    gpsweek = 2105
    _, gpstime = datetime_to_tow(timestamp())

    #NOTE: Calling measures to generate measurements to test measures?
    measurement, sat_posvel = measures.simulate_measures(
        gpsweek, gpstime + deltaT, extract_ephem(), Rx_ECEF(), 
        0., 0., np.zeros([3, 1]))

    return measurement, sat_posvel

# Define test fixtures
@pytest.fixture(name="get_meas")
def fixture_get_meas():
    measurement, sat_posvel = simulate_test_measures()
    return measurement, sat_posvel

@pytest.fixture(name="get_meas_dt")
def fixture_get_meas_dt():
    measurement, sat_posvel = simulate_test_measures(deltaT=dt)
    return measurement, sat_posvel

@pytest.fixture(name="set_XYZ")
def fixture_set_XYZ():
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
    expect_elaz = np.array([[90.0, -90.0, 0.0 ,  0.0 , 0.0, 0.0  ],
                            [0.0 ,  0.0 , 90.0, -90.0, 0.0, 180.0]]).T
    return expect_elaz

@pytest.fixture(name="set_sat_pos")
def fixture_set_sat_pos():
    sat_pos = np.zeros([6, 3])
    sat_pos[0,0] = 1.25*gpsconsts.A
    sat_pos[1,0] = 0.75*gpsconsts.A
    sat_pos[2,0] = gpsconsts.A; sat_pos[2,1]= gpsconsts.A
    sat_pos[3,0] = gpsconsts.A; sat_pos[3,1]= -gpsconsts.A
    sat_pos[4,0] = gpsconsts.A; sat_pos[4,2]= gpsconsts.A
    sat_pos[5,0] = gpsconsts.A; sat_pos[5,2]= -gpsconsts.A
    return sat_pos

@pytest.fixture(name="set_rx_pos")
def fixture_set_rx_pos():
    Rx = np.reshape(np.array([gpsconsts.A, 0, 0]), [1, 3])
    return Rx

# Define tests
def test_extract_XYZ(set_XYZ):
    pos_array, vel_array, times = set_XYZ
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
    calc_elaz = measures.find_elaz(set_rx_pos, set_sat_pos)
    #print(calc_elaz)
    np.testing.assert_array_almost_equal(expected_elaz, calc_elaz)

def test_measures_value_range(get_meas):
    measurements, sat_posvel = get_meas

    assert np.logical_and(measurements['prange'].values > 20000e3,
        measurements['prange'].values < 3e7).all(), ("Invalid range of "
        "pseudorange values")

    assert np.logical_and(measurements['doppler'].values > -5000,
        measurements['doppler'].values < 5000).all(), ("Magnitude of doppler "
        "values is greater than 5 KHz")

    assert np.all(np.abs(sat_posvel['x']).values < gpsconsts.A + 2e7),...
    ("Invalid range of ECEF x for satellite position")

    assert np.all(np.abs(sat_posvel['y']).values < gpsconsts.A + 2e7),...
    ("Invalid range of ECEF y for satellite position")

    assert np.all(np.abs(sat_posvel['z']).values < gpsconsts.A + 2e7),...
    ("Invalid range of ECEF z for satellite position")

def test_sat_velocity(get_meas, get_meas_dt):
    _, sat_posvel_prev = get_meas
    _, sat_posvel_new  = get_meas_dt

    np.testing.assert_array_almost_equal(
        (sat_posvel_new['x'] - sat_posvel_prev['x']) / dt,
        sat_posvel_prev['vx'], decimal=1)

    np.testing.assert_array_almost_equal(
        (sat_posvel_new['y'] - sat_posvel_prev['y']) / dt,
        sat_posvel_prev['vy'], decimal=1)

    np.testing.assert_array_almost_equal(
        (sat_posvel_new['z'] - sat_posvel_prev['z']) / dt,
        sat_posvel_prev['vz'], decimal=1)

def test_measure_sizes(get_meas):
    measurements, sat_posvel = get_meas
    assert (len(measurements['prange'].index)==len(sat_posvel.index))

def test_pseudorange_corrections(get_meas, get_meas_dt):
    meas_prev, _ = get_meas
    meas_new , _  = get_meas_dt

    gpsweek = 2105
    _, gpstime = datetime_to_tow(timestamp())

    rx_ecef = np.reshape(Rx_ECEF(), [-1, 3])

    ephem = extract_ephem()

    sat_names = (meas_prev.index).tolist()
    print(ephem.loc[sat_names,:])

    meas_prev_corr = measures.correct_pseudorange(
        gpstime   , gpsweek, ephem.loc[sat_names,:], meas_prev['prange'], rx_ecef)

    meas_new_corr  = measures.correct_pseudorange(
        gpstime+dt, gpsweek, ephem.loc[sat_names,:], meas_new['prange'] , rx_ecef)

    diff_prev = meas_prev_corr - meas_prev['prange']
    diff_new  = meas_new_corr  - meas_new['prange']

    np.testing.assert_array_almost_equal(diff_prev, diff_new, decimal=2)