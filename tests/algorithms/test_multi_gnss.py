"""Tests for multi_gnss codes.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "20 August 2022"

import os
import pytest

import numpy as np
import pandas as pd

from gnss_lib_py.parsers.android import AndroidDerived
from gnss_lib_py.parsers.precise_ephemerides import parse_sp3, parse_clockfile
from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.algorithms.multi_gnss import compute_sv_sp3clk_gps_glonass, compute_sv_eph_gps
from gnss_lib_py.algorithms.multi_gnss import compute_sp3_snapshot, compute_clk_snapshot
from gnss_lib_py.algorithms.multi_gnss import extract_sp3_func, extract_clk_func
import gnss_lib_py.utils.constants as consts

# Define the number of sats to create arrays for
NUMSATS_GPS = 32
NUMSATS_BEIDOU = 46
NUMSATS_GLONASS = 24
NUMSATS_GALILEO = 36
NUMSATS_QZSS = 3

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
    root_path = os.path.join(root_path, 'data/unit_test/')
    return root_path

@pytest.fixture(name="navdata_path")
def fixture_navdata_path(root_path):
    """Filepath of Android Derived measurements

    Returns
    -------
    navdata_path : string
        Location for the unit_test Android derived measurements

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset [1]_,
    particularly the train/2021-04-28-US-SJC-1/Pixel4 trace. The dataset
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
    navdata_path = os.path.join(root_path, 'Pixel4_derived_another.csv')
    return navdata_path

@pytest.fixture(name="sp3_path")
def fixture_sp3_path(root_path):
    """Filepath of .sp3 measurements

    Returns
    -------
    sp3_path : string
        String with location for the unit_test sp3 measurements

    Notes
    ----------
    (1) Need to shorten the data being loaded for unit tests

    References
    ----------
    .. [1]  https://geodesy.noaa.gov/UFCORS/ Accessed as of August 2, 2022
    """
    sp3_path = os.path.join(root_path, 'grg21553.sp3')
    return sp3_path

@pytest.fixture(name="clk_path")
def fixture_clk_path(root_path):
    """Filepath of .clk measurements

    Returns
    -------
    clk_path : string
        String with location for the unit_test clk measurements

    Notes
    ----------
    (1) Need to shorten the data being loaded for unit tests; 34 MB
    for clock file, can I just cut lines at the end?

    References
    ----------
    .. [1]  https://geodesy.noaa.gov/UFCORS/ Accessed as of August 2, 2022
    """
    clk_path = os.path.join(root_path, 'grg21553.clk')
    return clk_path

@pytest.fixture(name="navdata")
def fixture_load_navdata(navdata_path):
    """Load instance of AndroidDerived

    Parameters
    ----------
    navdata_path : pytest.fixture
        String with location of Android navdata measurement file

    Returns
    -------
    navdata : AndroidDerived
        Instance of AndroidDerived (GPS and GLONASS) for testing

    Notes
    ----------
    (1) Need to find sp3 that can load Beidou and Galileo data,
    maybe TU chemnitz one has all of them.

    """
    navdata_full = AndroidDerived(navdata_path)
    multi_gnss_idxs = np.where( (navdata_full["gnss_id",:] == 1) | \
                                (navdata_full["gnss_id",:] == 3) )[1]
    navdata = navdata_full.copy(cols = multi_gnss_idxs)

    return navdata

@pytest.fixture(name="navdata_gps")
def fixture_load_navdata_gps(navdata_path):
    """Load instance of AndroidDerived

    Parameters
    ----------
    navdata_path : pytest.fixture
        String with location of Android navdata measurement file

    Returns
    -------
    navdata_gps : AndroidDerived
        Instance of AndroidDerived (GPS) for testing

    Notes
    ----------

    """
    navdata_full = AndroidDerived(navdata_path)
    gps_idxs = np.where( (navdata_full["gnss_id",:] == 1) )[1]
    navdata_gps = navdata_full.copy(cols = gps_idxs)

    return navdata_gps

@pytest.fixture(name="sp3data_gps")
def fixture_load_sp3data_gps(sp3_path):
    """Load instance of sp3data_gps

    Parameters
    ----------
    sp3_path : pytest.fixture
        String with location for the unit_test sp3 measurements

    Returns
    -------
    sp3data_gps : Array of Sp3 classes with len == # GPS sats
        Instance of GPS-only Sp3 class array for testing

    Notes
    ----------
    """
    sp3data_gps = parse_sp3(sp3_path, constellation = 'G')

    return sp3data_gps

@pytest.fixture(name="clkdata_gps")
def fixture_load_clkdata_gps(clk_path):
    """Load instance of clkdata_gps

    Parameters
    ----------
    clk_path : pytest.fixture
        String with location for the unit_test clk measurements

    Returns
    -------
    clkdata_gps : Array of Clk classes with len == # GPS sats
        Instance of GPS-only Clk class array for testing

    Notes
    ----------
    """
    clkdata_gps = parse_clockfile(clk_path, constellation = 'G')

    return clkdata_gps

@pytest.fixture(name="sp3data_glonass")
def fixture_load_sp3data_glonass(sp3_path):
    """Load instance of sp3data_glonass

    Parameters
    ----------
    sp3_path : pytest.fixture
        String with location for the unit_test sp3 measurements

    Returns
    -------
    sp3data_glonass : Array of Sp3 classes with len == # GLONASS sats
        Instance of GLONASS-only Sp3 class array for testing

    Notes
    ----------
    (1) does precise ephemerides file require to be navdata format? 
    (2) Any ideas on unit tests for sp3 and clk files
    """
    sp3data_glonass = parse_sp3(sp3_path, constellation = 'R')

    return sp3data_glonass

@pytest.fixture(name="clkdata_glonass")
def fixture_load_clkdata_glonass(clk_path):
    """Load instance of clkdata_glonass

    Parameters
    ----------
    clk_path : pytest.fixture
        String with location for the unit_test clk measurements

    Returns
    -------
    clkdata_glonass : Array of Clk classes with len == # GLONASS sats
        Instance of GLONASS-only Clk class array for testing

    Notes
    ----------
    """
    clkdata_glonass = parse_clockfile(clk_path, constellation = 'R')

    return clkdata_glonass

def test_sp3_eph(navdata_gps, sp3_path, clk_path):
    """Tests that validates the satellite 3-D position and velocity
    from .sp3 and .n files closely resemble each other (GPS-only)

    Parameters
    ----------
    navdata_gps : pytest.fixture
        Instance of the NavData class that depicts android derived
        dataset for GPS-only constellation

    Notes
    ----------
    """

    multi_gnss = {'G': (1, 'GPS_L1') }
    navdata_sp3_result = compute_sv_sp3clk_gps_glonass(navdata_gps, sp3_path, \
                                                       clk_path, multi_gnss = multi_gnss)
    multi_gnss = (1, 'GPS_L1')
    navdata_eph_result = compute_sv_eph_gps(navdata_gps, \
                                            multi_gnss = multi_gnss)

    sv_keys = ['x_sv_m', 'y_sv_m', 'z_sv_m', \
               'vx_sv_mps','vy_sv_mps','vz_sv_mps']

    for sval in sv_keys:
        # Check if satellite info from sp3 and eph closely resemble
        # here, the threshold of 300 is set in a heuristic sense; need investigation
        if (sval == 'x_sv_m') | (sval == 'y_sv_m') | (sval == 'z_sv_m'):
            assert max(abs(navdata_sp3_result[sval][0] - navdata_eph_result[sval][0])) < 4.0
        if (sval == 'vx_sv_mps') | (sval == 'vy_sv_mps') | (sval == 'vz_sv_mps'):
            assert max(abs(navdata_sp3_result[sval][0] - navdata_eph_result[sval][0])) < 0.015

def test_gps_func(sp3data_gps, clkdata_gps):
    """Tests that extract_sp3_func and compute_sp3_snapshot functions
    does not fail for GPS constellation

    Parameters
    ----------
    sp3data_gps : pytest.fixture
        Array of Sp3 classes with len == # GPS sats
    clkdata_gps : pytest.fixture
        Array of Clk classes with len == # GPS sats

    Notes
    ----------
    (1) Do I need to validate satellite velocities as well? If so, ideas?
    (2) Need to think if function vs exact point can have 100s of m error
    """

    for prn in range(1, NUMSATS_GPS+1):
        for sidx, _ in enumerate(sp3data_gps[prn].tym):
            func_satpos = extract_sp3_func(sp3data_gps[prn], sidx, ipos = 10, method='CubicSpline')
            cxtime = sp3data_gps[prn].tym[sidx]
            satpos_sp3, _ = compute_sp3_snapshot(func_satpos, cxtime, \
                                                 hstep = 1e-5, method='CubicSpline')
            satpos_sp3_exact = np.array([ sp3data_gps[prn].xpos[sidx], \
                                          sp3data_gps[prn].ypos[sidx], \
                                          sp3data_gps[prn].zpos[sidx] ])
            assert np.linalg.norm(satpos_sp3-satpos_sp3_exact) < 150.0

        for sidx, _ in enumerate(clkdata_gps[prn].tym):
            func_satbias = extract_clk_func(clkdata_gps[prn], sidx, ipos = 10, method='CubicSpline')
            cxtime = clkdata_gps[prn].tym[sidx]
            satbias_clk, _ = compute_clk_snapshot(func_satbias, cxtime, \
                                                  hstep = 1e-5, method='CubicSpline')
            assert consts.C * np.linalg.norm(satbias_clk - clkdata_gps[prn].clk_bias[sidx]) < 1.0

def test_glonass_func(sp3data_glonass, clkdata_glonass):
    """Tests that extract_sp3_func and compute_sp3_snapshot functions
    does not fail for GLONASS constellation

    Parameters
    ----------
    sp3data_glonass : pytest.fixture
        Array of Sp3 classes with len == # GLONASS sats
    clkdata_glonass : pytest.fixture
        Array of Clk classes with len == # GLONASS sats

    Notes
    ----------
    (1) Do I need to validate satellite velocities as well? If so, ideas!
    (2) Need to think if function vs exact point can have 100s of m error
    """

    for prn in range(1, NUMSATS_GLONASS + 1):
        for sidx, _ in enumerate(sp3data_glonass[prn].tym):
            func_satpos = extract_sp3_func(sp3data_glonass[prn], sidx, \
                                           ipos = 10, method='CubicSpline')
            cxtime = sp3data_glonass[prn].tym[sidx]
            satpos_sp3, _ = compute_sp3_snapshot(func_satpos, cxtime, \
                                                 hstep = 1e-5, method='CubicSpline')
            satpos_sp3_exact = np.array([ sp3data_glonass[prn].xpos[sidx], \
                                          sp3data_glonass[prn].ypos[sidx], \
                                          sp3data_glonass[prn].zpos[sidx] ])
            assert np.linalg.norm(satpos_sp3-satpos_sp3_exact) < 150.0

        for sidx, _ in enumerate(clkdata_glonass[prn].tym):
            func_satbias = extract_clk_func(clkdata_glonass[prn], sidx, \
                                         ipos = 10, method='CubicSpline')
            cxtime = clkdata_glonass[prn].tym[sidx]
            satbias_clk, _ = compute_clk_snapshot(func_satbias, cxtime, \
                                                  hstep = 1e-5, method='CubicSpline')
            assert consts.C * np.linalg.norm(satbias_clk - clkdata_glonass[prn].clk_bias[sidx]) < 1.0

# def test_compute_sv_sp3clk_gps_glonass(navdata, sp3_path, clk_path):
#     """Tests that compute_sv_sp3clk_gps_glonass does not fail

#     Parameters
#     ----------
#     navdata : pytest.fixture
#         Instance of the NavData class that depicts android derived dataset
#     sp3_path : pytest.fixture
#         String with location for the unit_test sp3 measurements
#     clk_path : pytest.fixture
#         String with location for the unit_test clk measurements

#     Notes
#     ----------
#     (1) Need to check if setting a threshold of 300, 5e-2, 15, 5e-3
#     heuristically is acceptable or not; need investigation
#     (2) Need to shorten the data being loaded for unit tests
#     (3) Think if the multi_gnss needs to be anything different

#     """
#     multi_gnss = {'G': (1, 'GPS_L1'), \
#                   'R': (3, 'GLO_G1')}
#     navdata_sp3 = compute_sv_sp3clk_gps_glonass(navdata, sp3_path, \
#                                                 clk_path, multi_gnss)

#     # Check if the resulting derived is NavData class
#     assert isinstance( navdata_sp3, type(NavData()) )

#     sv_keys = ['x_sv_m', 'y_sv_m', 'z_sv_m', 'b_sv_m', \
#                'vx_sv_mps','vy_sv_mps','vz_sv_mps','b_dot_sv_mps']

#     for sval in sv_keys:
#         # Check if the resulting navdata class has satellite information
#         assert sval in navdata_sp3.rows

#         # Check if satellite info from AndroidDerived and sp3 closely resemble
#         # here, the threshold of 300 is set in a heuristic sense; need investigation
#         if (sval == 'x_sv_m') | (sval == 'y_sv_m') | (sval == 'z_sv_m'):
#             assert max(abs(navdata_sp3[sval][0] - navdata[sval][0])) < 300.0
#         if (sval == 'vx_sv_mps') | (sval == 'vy_sv_mps') | (sval == 'vz_sv_mps'):
#             assert max(abs(navdata_sp3[sval][0] - navdata[sval][0])) < 5e-2
#         if sval=='b_sv_m':
#             assert max(abs(navdata_sp3[sval][0] - navdata[sval][0])) < 15
#         if sval=='b_dot_sv_mps':
#             assert max(abs(navdata_sp3[sval][0] - navdata[sval][0])) < 5e-3

#     # Check if the derived classes are same except for corr_pr_m
#     navdata_df = navdata.pandas_df()
#     navdata_df = navdata_df.drop(columns = sv_keys)

#     navdata_sp3_df = navdata_sp3.pandas_df()
#     navdata_sp3_df = navdata_sp3_df.drop(columns = sv_keys)

#     pd.testing.assert_frame_equal(navdata_df.sort_index(axis=1),
#                                   navdata_sp3_df.sort_index(axis=1),
#                                   check_dtype=False, check_names=True)

# def test_compute_sv_eph_gps(navdata, navdata_gps):
#     """Tests that compute_sv_eph_gps does not fail

#     Parameters
#     ----------
#     navdata : pytest.fixture
#         Instance of NavData class that depicts entire android derived dataset
#     navdata : pytest.fixture
#         Instance of NavData class that depicts GPS-only android derived dataset

#     Notes
#     ----------

#     """

#     multi_gnss = (1, 'GPS_L1')

#     # test what happens when rows down't exist
#     with pytest.raises(RuntimeError) as excinfo:
#         compute_sv_eph_gps(navdata, multi_gnss = multi_gnss)
#     assert "multi-GNSS" in str(excinfo.value)

#     navdata_eph = compute_sv_eph_gps(navdata_gps, multi_gnss = multi_gnss)

#     # Check if the resulting derived is NavData class
#     assert isinstance( navdata_eph, type(NavData()) )

#     sv_keys = ['x_sv_m', 'y_sv_m', 'z_sv_m', \
#                'vx_sv_mps','vy_sv_mps','vz_sv_mps']

#     for sval in sv_keys:
#         # Check if the resulting navdata class has satellite information
#         assert sval in navdata_eph.rows

#         # Check if satellite info from AndroidDerived and eph closely resemble
#         # here, the threshold of 300 is set in a heuristic sense; need investigation
#         if (sval == 'x_sv_m') | (sval == 'y_sv_m') | (sval == 'z_sv_m'):
#             assert max(abs(navdata_eph[sval][0] - navdata[sval][0])) < 300.0
#         if (sval == 'vx_sv_mps') | (sval == 'vy_sv_mps') | (sval == 'vz_sv_mps'):
#             assert max(abs(navdata_eph[sval][0] - navdata[sval][0])) < 5e-2

#     # Check if the derived classes are same except for corr_pr_m
#     navdata_df = navdata.pandas_df()
#     navdata_df = navdata_df.drop(columns = sv_keys)

#     navdata_eph_df = navdata_eph.pandas_df()
#     navdata_eph_df = navdata_eph_df.drop(columns = sv_keys)

#     pd.testing.assert_frame_equal(navdata_df.sort_index(axis=1),
#                                   navdata_eph_df.sort_index(axis=1),
#                                   check_dtype=False, check_names=True)
