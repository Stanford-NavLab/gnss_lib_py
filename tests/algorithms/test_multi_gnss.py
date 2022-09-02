"""Tests for multi_gnss codes.
Notes
----------
(1) Probably should test with another android, sp3, clk dataset?

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "20 August 2022"

import os
import pytest

import numpy as np
import pandas as pd

from gnss_lib_py.parsers.android import AndroidDerived2021
from gnss_lib_py.parsers.precise_ephemerides import parse_sp3, parse_clockfile
from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.algorithms.multi_gnss \
        import compute_sv_gnss_from_precise_eph, compute_sv_gps_from_brdcst_eph
from gnss_lib_py.algorithms.multi_gnss import compute_sp3_snapshot, compute_clk_snapshot
from gnss_lib_py.algorithms.multi_gnss import extract_sp3_func, extract_clk_func
import gnss_lib_py.utils.constants as consts

# Define the number of sats to create arrays for
NUMSATS_GPS = 32
NUMSATS_BEIDOU = 46
NUMSATS_GLONASS = 24
NUMSATS_GALILEO = 36
NUMSATS_QZSS = 3

SV_KEYS = ['x_sv_m', 'y_sv_m', 'z_sv_m', \
           'vx_sv_mps','vy_sv_mps','vz_sv_mps', \
           'b_sv_m', 'b_dot_sv_mps']

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
    navdata_path = os.path.join(root_path, "android_2021/Pixel4_derived_SJC_28thApr2021.csv")
    return navdata_path

@pytest.fixture(name="sp3_path")
def fixture_sp3_path(root_path):
    """Filepath of .sp3 measurements

    Returns
    -------
    sp3_path : string
        String with location for the unit_test sp3 measurements

    References
    ----------
    .. [1]  https://geodesy.noaa.gov/UFCORS/ Accessed as of August 2, 2022
    """
    sp3_path = os.path.join(root_path, "precise_ephemeris/grg21553_short.sp3")
    return sp3_path

@pytest.fixture(name="clk_path")
def fixture_clk_path(root_path):
    """Filepath of .clk measurements

    Returns
    -------
    clk_path : string
        String with location for the unit_test clk measurements

    References
    ----------
    .. [1]  https://geodesy.noaa.gov/UFCORS/ Accessed as of August 2, 2022
    """
    clk_path = os.path.join(root_path, "precise_ephemeris/grg21553_short.clk")
    return clk_path

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
        Instance of AndroidDerived2021 (GPS and GLONASS) for testing

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

    Notes
    -------
    (1) Need to add functionality for multiple GNSS constellations in navdata class
    (2) Need to add functionality for printing current keys in navdata class
    (3) Can we force unit test files to be read-only?

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
    """
    sp3data_gps = parse_sp3(sp3_path, constellation = 'gps')

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
    """
    clkdata_gps = parse_clockfile(clk_path, constellation = 'gps')

    return clkdata_gps

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

    """
    sp3data_glonass = parse_sp3(sp3_path, constellation = 'glonass')

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
    """
    clkdata_glonass = parse_clockfile(clk_path, constellation = 'glonass')

    return clkdata_glonass

def test_gpscheck_sp3_eph(navdata_gpsl1, sp3data_gps, clkdata_gps):
    """Tests that validates the satellite 3-D position and velocity
    from .sp3 and .n files closely resemble each other (GPS-only)

    Parameters
    ----------
    navdata_gpsl1 : pytest.fixture
        Instance of the NavData class that depicts GPS-L1 android derived
        dataset for GPS-only constellation
    sp3data_gps : pytest.fixture
        Instance of Sp3 class array associated with any one constellation
        with len == # GPS NUMSATS
    clkdata_gps : pytest.fixture
        Instance of Clk class array associated with any one constellation
        with len == # GPS NUMSATS
    """

    navdata_sp3_result = navdata_gpsl1.copy()
    navdata_sp3_result = compute_sv_gnss_from_precise_eph(navdata_sp3_result, \
                                                          sp3data_gps, clkdata_gps)
    navdata_eph_result = navdata_gpsl1.copy()
    navdata_eph_result = compute_sv_gps_from_brdcst_eph(navdata_eph_result)

    for sval in SV_KEYS[0:6]:
        # Check if satellite info from sp3 and eph closely resemble
        # here, the threshold of 300 is set in a heuristic sense; need investigation
        if (sval == 'x_sv_m') | (sval == 'y_sv_m') | (sval == 'z_sv_m'):
            assert max(abs(navdata_sp3_result[sval] - navdata_eph_result[sval])) != 0.0
            assert max(abs(navdata_sp3_result[sval] - navdata_eph_result[sval])) < 4.0
        if (sval == 'vx_sv_mps') | (sval == 'vy_sv_mps') | (sval == 'vz_sv_mps'):
            assert max(abs(navdata_sp3_result[sval] - navdata_eph_result[sval])) != 0.0
            assert max(abs(navdata_sp3_result[sval] - navdata_eph_result[sval])) < 0.015

def test_gps_sp3_clk_funcs(sp3data_gps, clkdata_gps):
    """Tests that extract_sp3_func and compute_sp3_snapshot functions
    does not fail for GPS constellation

    Parameters
    ----------
    sp3data_gps : pytest.fixture
        Instance of Sp3 class array associated with any one constellation
        with len == # GPS NUMSATS
    clkdata_gps : pytest.fixture
        Instance of Clk class array associated with any one constellation
        with len == # GPS NUMSATS
    """

    for prn in range(1, NUMSATS_GPS+1):
        # Last index interpolation does not work well, so eliminating
        # validation against the last datapoint
        for sidx, _ in enumerate(sp3data_gps[prn].tym[:-1]):
            func_satpos = extract_sp3_func(sp3data_gps[prn], sidx, \
                                           ipos = 10, method='CubicSpline')
            cxtime = sp3data_gps[prn].tym[sidx]
            satpos_sp3, _ = compute_sp3_snapshot(func_satpos, cxtime, \
                                                 hstep = 5e-1, method='CubicSpline')
            satpos_sp3_exact = np.array([ sp3data_gps[prn].xpos[sidx], \
                                          sp3data_gps[prn].ypos[sidx], \
                                          sp3data_gps[prn].zpos[sidx] ])
            assert np.linalg.norm(satpos_sp3 - satpos_sp3_exact) < 1e-6

        # Last index interpolation does not work well, so eliminating
        # validation against the last datapoint
        for sidx, _ in enumerate(clkdata_gps[prn].tym[:-1]):
            func_satbias = extract_clk_func(clkdata_gps[prn], sidx, \
                                            ipos = 10, method='CubicSpline')
            cxtime = clkdata_gps[prn].tym[sidx]
            satbias_clk, _ = compute_clk_snapshot(func_satbias, cxtime, \
                                                  hstep = 5e-1, method='CubicSpline')
            assert consts.C * np.linalg.norm(satbias_clk - \
                                             clkdata_gps[prn].clk_bias[sidx]) < 1e-6

def test_glonass_sp3_clk_funcs(sp3data_glonass, clkdata_glonass):
    """Tests that extract_sp3_func and compute_sp3_snapshot functions
    does not fail for GLONASS constellation

    Parameters
    ----------
    sp3data_glonass : pytest.fixture
        Array of Sp3 classes with len == # GLONASS sats
    clkdata_glonass : pytest.fixture
        Array of Clk classes with len == # GLONASS sats
    """

    for prn in range(1, NUMSATS_GLONASS + 1):
        # Last index interpolation does not work well, so eliminating
        # validation against the last datapoint
        for sidx, _ in enumerate(sp3data_glonass[prn].tym[:-1]):
            func_satpos = extract_sp3_func(sp3data_glonass[prn], sidx, \
                                           ipos = 10, method='CubicSpline')
            cxtime = sp3data_glonass[prn].tym[sidx]
            satpos_sp3, _ = compute_sp3_snapshot(func_satpos, cxtime, \
                                                 hstep = 5e-1, method='CubicSpline')
            satpos_sp3_exact = np.array([ sp3data_glonass[prn].xpos[sidx], \
                                          sp3data_glonass[prn].ypos[sidx], \
                                          sp3data_glonass[prn].zpos[sidx] ])
            assert np.linalg.norm(satpos_sp3 - satpos_sp3_exact) < 1e-6

        # Last index interpolation does not work well, so eliminating
        # validation against the last datapoint
        for sidx, _ in enumerate(clkdata_glonass[prn].tym[:-1]):
            func_satbias = extract_clk_func(clkdata_glonass[prn], sidx, \
                                         ipos = 10, method='CubicSpline')
            cxtime = clkdata_glonass[prn].tym[sidx]
            satbias_clk, _ = compute_clk_snapshot(func_satbias, cxtime, \
                                                  hstep = 5e-1, method='CubicSpline')
            assert consts.C * np.linalg.norm(satbias_clk - \
                                             clkdata_glonass[prn].clk_bias[sidx]) < 1e-6

def test_compute_gps_precise_eph(navdata_gps, sp3data_gps, clkdata_gps):
    """Tests that compute_sv_gnss_from_precise_eph does not fail for GPS

    Parameters
    ----------
    navdata_gps : pytest.fixture
        Instance of the NavData class that depicts android derived dataset
    sp3data_gps : pytest.fixture
        Instance of Sp3 class array associated with any one constellation
        with len == # GPS NUMSATS
    clkdata_gps : pytest.fixture
        Instance of Clk class array associated with any one constellation
        with len == # GPS NUMSATS
    """
    navdata_gps_sp3 = navdata_gps.copy()
    navdata_gps_sp3 = compute_sv_gnss_from_precise_eph(navdata_gps_sp3, \
                                                       sp3data_gps, clkdata_gps)

    # Check if the resulting derived is NavData class
    assert isinstance( navdata_gps_sp3, type(NavData()) )

    for sval in SV_KEYS:
        # Check if the resulting navdata class has satellite information
        assert sval in navdata_gps_sp3.rows

        # Check if satellite info from AndroidDerived and sp3 closely resemble
        # here, the threshold of 300 is set in a heuristic sense; need investigation
        if (sval == 'x_sv_m') | (sval == 'y_sv_m') | (sval == 'z_sv_m'):
            assert max(abs(navdata_gps_sp3[sval] - navdata_gps[sval])) != 0.0
            assert max(abs(navdata_gps_sp3[sval] - navdata_gps[sval])) < 13e3 #300
        if (sval == 'vx_sv_mps') | (sval == 'vy_sv_mps') | (sval == 'vz_sv_mps'):
            assert max(abs(navdata_gps_sp3[sval] - navdata_gps[sval])) != 0.0
            assert max(abs(navdata_gps_sp3[sval] - navdata_gps[sval])) < 2 #5e-2
        if sval=='b_sv_m':
            assert max(abs(navdata_gps_sp3[sval] - navdata_gps[sval])) != 0.0
            assert max(abs(navdata_gps_sp3[sval] - navdata_gps[sval])) < 15
        if sval=='b_dot_sv_mps':
            assert max(abs(navdata_gps_sp3[sval] - navdata_gps[sval])) != 0.0
            assert max(abs(navdata_gps_sp3[sval] - navdata_gps[sval])) < 5e-3

    # Check if the derived classes are same except for corr_pr_m
    navdata_gps_df = navdata_gps.pandas_df()
    navdata_gps_df = navdata_gps_df.drop(columns = SV_KEYS)

    navdata_gps_sp3_df = navdata_gps_sp3.pandas_df()
    navdata_gps_sp3_df = navdata_gps_sp3_df.drop(columns = SV_KEYS)

    pd.testing.assert_frame_equal(navdata_gps_df.sort_index(axis=1),
                                  navdata_gps_sp3_df.sort_index(axis=1),
                                  check_dtype=False, check_names=True)

def test_compute_glonass_precise_eph(navdata_glonass, sp3data_glonass, clkdata_glonass):
    """Tests that compute_sv_gnss_from_precise_eph does not fail for GPS

    Parameters
    ----------
    navdata_glonass : pytest.fixture
        Instance of the NavData class that depicts android derived dataset
    sp3data_glonass : pytest.fixture
        Instance of Sp3 class array associated with any one constellation
        with len == # GLONASS NUMSATS
    clkdata_glonass : pytest.fixture
        Instance of Clk class array associated with any one constellation
        with len == # GLONASS NUMSATS
    """
    navdata_glonass_sp3 = navdata_glonass.copy()
    navdata_glonass_sp3 = \
            compute_sv_gnss_from_precise_eph(navdata_glonass_sp3, \
                                             sp3data_glonass, clkdata_glonass)

    # Check if the resulting derived is NavData class
    assert isinstance( navdata_glonass_sp3, type(NavData()) )

    for sval in SV_KEYS:
        # Check if the resulting navdata class has satellite information
        assert sval in navdata_glonass_sp3.rows

        # Check if satellite info from AndroidDerived and sp3 closely resemble
        # here, the threshold of 300 is set in a heuristic sense; need investigation
        if (sval == 'x_sv_m') | (sval == 'y_sv_m') | (sval == 'z_sv_m'):
            assert max(abs(navdata_glonass_sp3[sval] - navdata_glonass[sval])) != 0.0
            assert max(abs(navdata_glonass_sp3[sval] - navdata_glonass[sval])) < 13e3 #300
        if (sval == 'vx_sv_mps') | (sval == 'vy_sv_mps') | (sval == 'vz_sv_mps'):
            assert max(abs(navdata_glonass_sp3[sval] - navdata_glonass[sval])) != 0.0
            assert max(abs(navdata_glonass_sp3[sval] - navdata_glonass[sval])) < 2 #5e-2
        if sval=='b_sv_m':
            assert max(abs(navdata_glonass_sp3[sval] - navdata_glonass[sval])) != 0.0
            assert max(abs(navdata_glonass_sp3[sval] - navdata_glonass[sval])) < 15
        if sval=='b_dot_sv_mps':
            assert max(abs(navdata_glonass_sp3[sval] - navdata_glonass[sval])) != 0.0
            assert max(abs(navdata_glonass_sp3[sval] - navdata_glonass[sval])) < 5e-3

    # Check if the derived classes are same except for corr_pr_m
    navdata_glonass_df = navdata_glonass.pandas_df()
    navdata_glonass_df = navdata_glonass_df.drop(columns = SV_KEYS)

    navdata_glonass_sp3_df = navdata_glonass_sp3.pandas_df()
    navdata_glonass_sp3_df = navdata_glonass_sp3_df.drop(columns = SV_KEYS)

    pd.testing.assert_frame_equal(navdata_glonass_df.sort_index(axis=1),
                                  navdata_glonass_sp3_df.sort_index(axis=1),
                                  check_dtype=False, check_names=True)

def test_compute_gps_brdcst_eph(navdata_gpsl1, navdata, navdata_glonassg1):
    """Tests that compute_sv_gps_from_brdcst_eph does not fail

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
        compute_sv_gps_from_brdcst_eph(navdata_eph)
    assert "Multi-GNSS" in str(excinfo.value)

    # test what happens when invalid (non-GPS) rows down't exist
    with pytest.raises(RuntimeError) as excinfo:
        navdata_glonassg1_eph = navdata_glonassg1.copy()
        compute_sv_gps_from_brdcst_eph(navdata_glonassg1_eph)
    assert "non-GPS" in str(excinfo.value)

    navdata_gpsl1_eph = navdata_gpsl1.copy()
    navdata_gpsl1_eph = compute_sv_gps_from_brdcst_eph(navdata_gpsl1_eph)

    # Check if the resulting derived is NavData class
    assert isinstance( navdata_gpsl1_eph, type(NavData()) )

    for sval in SV_KEYS[0:6]:
        print( sval, max(abs(navdata_gpsl1[sval] - navdata_gpsl1_eph[sval])) )

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
