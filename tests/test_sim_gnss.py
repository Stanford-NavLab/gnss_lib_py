import pytest
import numpy as np
import pandas as pd
import os, sys
src_directory = os.path.join(os.getcwd(), 'gnss_lib_py')
sys.path.insert(0, src_directory)
# Path of package for pytest
import datetime
import pytz

from gnss_lib_py.core.coordinates import geodetic2ecef
from gnss_lib_py.core.constants import GPSConsts
from gnss_lib_py.core import measures
from gnss_lib_py.core.ephemeris import datetime_to_tow
from gnss_lib_py.core.ephemeris import EphemerisManager


# Defining test fixtures

def timestamp():
    return datetime.datetime(2020, 5, 15, 0, 47, 15, 448796, pytz.UTC)


def ephem_man():
    parent_directory = os.getcwd()
    src_directory = os.path.join(parent_directory, 'src')
    ephemeris_data_directory = os.path.join(parent_directory, 'data', 'ephemeris')
    return EphemerisManager(ephemeris_data_directory)

def Rx_ECEF():
    x_LLA = np.reshape([37.427112, -122.1764146, 16], [1, 3])
    x_ECEF = np.reshape(geodetic2ecef(x_LLA), [3, 1])
    return x_ECEF

def extract_ephem():
    manager = ephem_man()
    sats = ['G'+"%02d"%sv_num for sv_num in range(1, 33)]
    ephemeris = manager.get_ephemeris(timestamp(), sats)
    return ephemeris

def static_test_measures(deltaT=0):
    gpsweek = 2105
    _, gpstime = datetime_to_tow(timestamp())
    measurement, satXYZV = measures.simulate_measures(gpsweek, gpstime + deltaT, extract_ephem(), Rx_ECEF(), 0., 0., np.zeros([3, 1]))
    return measurement, satXYZV


# Defining tests

def test_extract_XYZ():
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
    svs = np.array(['G01', 'G02', 'G03'])
    comb_array = np.hstack((times, pos_array, vel_array))
    test_XYZV = pd.DataFrame(data= comb_array, index=svs, columns=['times', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
    prns, test_pos, test_vel = measures._extract_pos_vel_arr(test_XYZV)
    np.testing.assert_array_equal(test_pos, pos_array)
    np.testing.assert_array_equal(test_vel, vel_array)
    np.testing.assert_array_equal(prns, np.array([1, 2, 3]))



def test_find_elaz():
    gpsconsts = GPSConsts()
    Rx = np.reshape(np.array([gpsconsts.A, 0, 0]), [1, 3])
    test_XYZ = np.zeros([6, 3])
    test_XYZ[0,0] = 1.25*gpsconsts.A
    test_XYZ[1,0] = 0.75*gpsconsts.A
    test_XYZ[2,0] = gpsconsts.A; test_XYZ[2,1]= gpsconsts.A
    test_XYZ[3,0] = gpsconsts.A; test_XYZ[3,1]= -gpsconsts.A
    test_XYZ[4,0] = gpsconsts.A; test_XYZ[4,2]= gpsconsts.A
    test_XYZ[5,0] = gpsconsts.A; test_XYZ[5,2]= -gpsconsts.A
    calc_elaz = measures.find_elaz(Rx, test_XYZ)
    expect_elaz = np.zeros([6, 2])
    expect_elaz[:, 0] = np.array([90, -90, 0, 0, 0, 0])
    expect_elaz[:, 1] = np.array([0., 0., 90, -90, 0, 180])
    np.testing.assert_array_almost_equal(expect_elaz, calc_elaz)


def test_measures_value_range():
    gpsconsts = GPSConsts()
    measurements, satXYZV = static_test_measures()
    assert np.logical_and(measurements['prange'].values > 20000e3, measurements['prange'].values < 3e7).all(), "Invalid range of pseudorange values"
    assert np.logical_and(measurements['doppler'].values > -5000, measurements['doppler'].values < 5000).all(), "Magnitude of doppler values is greater than 5 KHz"
    assert np.all(np.abs(satXYZV['x']).values < gpsconsts.A + 2e7), "Invalid range of ECEF x for satellite position"
    assert np.all(np.abs(satXYZV['y']).values < gpsconsts.A + 2e7), "Invalid range of ECEF y for satellite position"
    assert np.all(np.abs(satXYZV['z']).values < gpsconsts.A + 2e7), "Invalid range of ECEF z for satellite position"

def test_satV():
    deltaT = 0.1
    _, satXYZV_prev = static_test_measures()
    _, satXYZV_curr = static_test_measures(deltaT=deltaT)
    np.testing.assert_array_almost_equal((satXYZV_curr['y'] - satXYZV_prev['y'])/deltaT, satXYZV_prev['vy'], decimal=1)
    np.testing.assert_array_almost_equal((satXYZV_curr['x'] - satXYZV_prev['x'])/deltaT, satXYZV_prev['vx'], decimal=1)
    np.testing.assert_array_almost_equal((satXYZV_curr['z'] - satXYZV_prev['z'])/deltaT, satXYZV_prev['vz'], decimal=1)

def measure_sizes():
    measurements, satXYZV = static_test_measures()
    assert np.logical_and(len(measurements['prange'].index)==len(satXYZV.index))

