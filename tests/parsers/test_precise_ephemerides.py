"""Tests for precise ephemerides data loaders.
"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "25 August 2022"

import os

from datetime import datetime, timezone
import random

import numpy as np
import pytest
import pandas as pd

from gnss_lib_py.parsers.precise_ephemerides import Sp3, Clk
from gnss_lib_py.parsers.precise_ephemerides import parse_sp3, parse_clockfile
from gnss_lib_py.parsers.android import AndroidDerived2021
from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.parsers.precise_ephemerides import single_gnss_from_precise_eph
from gnss_lib_py.parsers.precise_ephemerides import multi_gnss_from_precise_eph
from gnss_lib_py.parsers.precise_ephemerides import sv_gps_from_brdcst_eph
from gnss_lib_py.parsers.precise_ephemerides import sp3_snapshot, clk_snapshot
from gnss_lib_py.parsers.precise_ephemerides import extract_sp3, extract_clk
import gnss_lib_py.utils.constants as consts

# Define the no. of samples to test functions:test_gps_sp3_funcs,
# test_gps_clk_funcs, test_glonass_sp3_funcs, test_glonass_clk_funcs
NUMSAMPLES = 4

# Define the keys relevant for satellite information
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

@pytest.fixture(name="sp3_path")
def fixture_sp3_path(root_path):
    """Filepath of valid .sp3 measurements

    Returns
    -------
    sp3_path : string
        String with location for the unit_test sp3 measurements

    Notes
    -----
    Downloaded the relevant .sp3 files from either CORS website [1]_ or
    CDDIS website [2]_

    References
    ----------
    .. [1]  https://geodesy.noaa.gov/UFCORS/ Accessed as of August 2, 2022
    .. [2]  https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/gnss_mgex.html
            Accessed as of August 2, 2022
    """
    sp3_path = os.path.join(root_path, 'precise_ephemeris/grg21553_short.sp3')
    return sp3_path

@pytest.fixture(name="clk_path")
def fixture_clk_path(root_path):
    """Filepath of valid .clk measurements

    Returns
    -------
    clk_path : string
        String with location for the unit_test clk measurements

    Notes
    -----
    Downloaded the relevant .clk files from either CORS website [1]_ or
    CDDIS website [2]_
    """
    clk_path = os.path.join(root_path, 'precise_ephemeris/grg21553_short.clk')
    return clk_path

@pytest.fixture(name="sp3data")
def fixture_load_sp3data(sp3_path):
    """Load instance of sp3 data.

    Parameters
    ----------
    sp3_path : pytest.fixture
        String with location for the unit_test sp3 measurements

    Returns
    -------
    sp3data : dict
        Instances of Sp3 class for each satellite

    """
    sp3data = parse_sp3(sp3_path)

    return sp3data

@pytest.fixture(name="clkdata")
def fixture_load_clkdata(clk_path):
    """Load instance of clk data.

    Parameters
    ----------
    clk_path : pytest.fixture
        String with location for the unit_test clk measurements

    Returns
    -------
    clkdata : list
        Instances of Clk class for each satellite
    """
    clkdata = parse_clockfile(clk_path)

    return clkdata

@pytest.fixture(name="sp3_path_missing")
def fixture_sp3_path_missing(root_path):
    """Invalid filepath for .sp3 measurements

    Returns
    -------
    sp3_path : string
        String with location for the unit_test sp3 measurements
    """
    sp3_path_missing = os.path.join(root_path, 'precise_ephemeris/grg21553_missing.sp3')
    return sp3_path_missing

@pytest.fixture(name="clk_path_missing")
def fixture_clk_path_missing(root_path):
    """Invalid filepath of .clk measurements

    Returns
    -------
    clk_path : string
        String with location for the unit_test clk measurements
    """
    clk_path_missing = os.path.join(root_path, 'precise_ephemeris/grg21553_missing.clk')
    return clk_path_missing

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

def test_load_sp3data_missing(sp3_path_missing):
    """Load instance of sp3 for GPS constellation from missing file

    Parameters
    ----------
    sp3_path_missing : pytest.fixture
        String with invalid location for unit_test sp3
        measurements
    """
    # Create a sp3 class for each expected satellite
    with pytest.raises(FileNotFoundError):
        parse_sp3(sp3_path_missing)

def test_load_clkdata_missing(clk_path_missing):
    """Load instance of clk for GPS constellation from missing file

    Parameters
    ----------
    clk_path_missing : pytest.fixture
        String with invalid location for unit_test clk
        measurements
    """
    # Create a sp3 class for each expected satellite
    with pytest.raises(FileNotFoundError):
        parse_clockfile(clk_path_missing)

@pytest.fixture(name="sp3_path_nodata")
def fixture_sp3_path_nodata(root_path):
    """Filepath for .sp3 measurements with no data

    Returns
    -------
    sp3_path : string
        String with location for the unit_test sp3 measurements
    """
    sp3_path_nodata = os.path.join(root_path, 'precise_ephemeris/grg21553_nodata.sp3')
    return sp3_path_nodata

@pytest.fixture(name="clk_path_nodata")
def fixture_clk_path_nodata(root_path):
    """Filepath for .clk measurements with no data

    Returns
    -------
    clk_path : string
        String with location for the unit_test clk measurements
    """
    clk_path_nodata = os.path.join(root_path, 'precise_ephemeris/grg21553_nodata.clk')
    return clk_path_nodata

def test_load_sp3data_nodata(sp3_path_nodata):
    """Load sp3 instance from file with no data

    Parameters
    ----------
    sp3_path_nodata : pytest.fixture
        String with no available data for unit_test sp3
        measurements
    """

    sp3data_nodata = parse_sp3(sp3_path_nodata)

    assert len(sp3data_nodata) == 0

def test_load_clkdata_nodata(clk_path_nodata):
    """Load clk instance from file with no data

    Parameters
    ----------
    clk_path_nodata : pytest.fixture
        String with no available data for unit_test clk
        measurements
    """

    clkdata_nodata = parse_clockfile(clk_path_nodata)

    assert len(clkdata_nodata) == 0

@pytest.mark.parametrize('row_name, prn, index, exp_value',
                        [('xpos', 'G01', 2, 13222742.845),
                         ('ypos', 'G12', 5, 9753305.474000001),
                         ('zpos', 'G32', 25, 21728484.688),
                         ('tym', 'G08', 8, 1303670400000.0),
                         ('utc_time', 'G12', 3, datetime(2021, 4, 28, 18, 15, tzinfo=timezone.utc)) ]
                        )
def test_sp3gps_value_check(sp3data, prn, row_name, index, exp_value):
    """Check array of Sp3 entries for GPS against known values

    Parameters
    ----------
    sp3data : pytest.fixture
        Instance of Sp3 class dictionary
    prn : int
        Satellite PRN for test example
    row_name : string
        Row key for test example
    index : int
        Index to query data at
    exp_value : float/datetime
        Expected value at queried indices
    """

    assert sum([1 for key in sp3data if key[0] == 'G']) == 31

    curr_value = getattr(sp3data[prn], row_name)[index]
    np.testing.assert_equal(curr_value, exp_value)

@pytest.mark.parametrize('row_name, prn, index, exp_value',
                        [('xpos', 'R24', 1, 13383401.364),
                         ('ypos', 'R02', 7, 3934479.152),
                         ('zpos', 'R18', 45, 10107376.674999999),
                         ('tym', 'R12', 17, 1303673100000.0),
                         ('utc_time', 'R09', 34, datetime(2021, 4, 28, 20, 50, tzinfo=timezone.utc)) ]
                        )
def test_sp3glonass_value_check(sp3data, prn, row_name, index, exp_value):
    """Check array of Sp3 entries for GLONASS against known values

    Parameters
    ----------
    sp3data : pytest.fixture
        Instance of Sp3 class dictionary
    prn : int
        Satellite PRN for test example
    row_name : string
        Row key for test example
    index : int
        Index to query data at
    exp_value : float/datetime
        Expected value at queried indices
    """

    assert sum([1 for key in sp3data if key[0] == 'R']) == 20

    curr_value = getattr(sp3data[prn], row_name)[index]
    np.testing.assert_equal(curr_value, exp_value)

@pytest.mark.parametrize('row_name, prn, index, exp_value',
                        [('clk_bias', 'G15', 0, -0.00015303409205),
                         ('tym', 'G05', 5, 1303668150000.0),
                         ('utc_time', 'G32', 4, datetime(2021, 4, 28, 18, 2, tzinfo=timezone.utc)) ]
                        )
def test_clkgps_value_check(clkdata, prn, row_name, index, exp_value):
    """Check Clk array entries of GPS constellation against
    known/expected values using test matrix

    Parameters
    ----------
    clkdata : pytest.fixture
        Instance of Clk class dictionary
    prn : int
        Satellite PRN for test example
    row_name : string
        Row key for test example
    index : int
        Index to query data at
    exp_value : float/datetime
        Expected value at queried indices
    """

    assert sum([1 for key in clkdata if key[0] == 'G']) == 31

    curr_value = getattr(clkdata[prn], row_name)[index]
    np.testing.assert_equal(curr_value, exp_value)

@pytest.mark.parametrize('row_name, prn, index, exp_value',
                        [('clk_bias', 'R08', 16, -5.87550990462e-05),
                         ('tym', 'R14', 10, 1303668300000.0),
                         ('utc_time', 'R04', 4, datetime(2021, 4, 28, 18, 2, tzinfo=timezone.utc)) ]
                        )
def test_clkglonass_value_check(clkdata, prn, row_name, index, exp_value):
    """Check array of Sp3 entries for GLONASS against known values

    Parameters
    ----------
    clkdata : pytest.fixture
        Instance of Clk class dictionary
    prn : int
        Satellite PRN for test example
    row_name : string
        Row key for test example
    index : int
        Index to query data at
    exp_value : float/datetime
        Expected value at queried indices
    """

    assert sum([1 for key in clkdata if key[0] == 'R']) == 20

    curr_value = getattr(clkdata[prn], row_name)[index]
    np.testing.assert_equal(curr_value, exp_value)

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
    navdata_sp3_result = single_gnss_from_precise_eph(navdata_sp3_result, \
                                                          sp3data, clkdata)
    navdata_eph_result = navdata_gpsl1.copy()
    navdata_eph_result = sv_gps_from_brdcst_eph(navdata_eph_result)

    for sval in SV_KEYS[0:6]:
        # Check if satellite info from sp3 and eph closely resemble
        # here, the threshold of 300 is set in a heuristic sense; need investigation
        if (sval == 'x_sv_m') | (sval == 'y_sv_m') | (sval == 'z_sv_m'):
            assert max(abs(navdata_sp3_result[sval] - navdata_eph_result[sval])) != 0.0
            assert max(abs(navdata_sp3_result[sval] - navdata_eph_result[sval])) < 4.0
        if (sval == 'vx_sv_mps') | (sval == 'vy_sv_mps') | (sval == 'vz_sv_mps'):
            assert max(abs(navdata_sp3_result[sval] - navdata_eph_result[sval])) != 0.0
            assert max(abs(navdata_sp3_result[sval] - navdata_eph_result[sval])) < 0.015

def test_gps_sp3_funcs(sp3data):
    """Tests extract_sp3, sp3_snapshot for GPS-Sp3

    Notes
    ----------
    Last index interpolation does not work well, so eliminating this index
    while extracting random samples from tym in Sp3

    Parameters
    ----------
    sp3data : pytest.fixture
        Instance of Sp3 class dictionary
    """

    for prn in [key for key in sp3data if key[0] == 'G']:
        if len(sp3data[prn].tym) != 0:
            sp3_subset = random.sample(range(len(sp3data[prn].tym)-1), NUMSAMPLES)
            for sidx, _ in enumerate(sp3_subset):
                func_satpos = extract_sp3(sp3data[prn], sidx, \
                                               ipos = 10, method='CubicSpline',
                                               verbose=True)
                cxtime = sp3data[prn].tym[sidx]
                satpos_sp3, _ = sp3_snapshot(func_satpos, cxtime, \
                                                     hstep = 5e-1, method='CubicSpline')
                satpos_sp3_exact = np.array([ sp3data[prn].xpos[sidx], \
                                              sp3data[prn].ypos[sidx], \
                                              sp3data[prn].zpos[sidx] ])
                assert np.linalg.norm(satpos_sp3 - satpos_sp3_exact) < 1e-6

def test_gps_clk_funcs(clkdata):
    """Tests extract_sp3, sp3_snapshot for GPS-Clk

    Notes
    ----------
    Last index interpolation does not work well, so eliminating this index
    while extracting random samples from tym in Clk

    Parameters
    ----------
    clkdata : pytest.fixture
        Instance of Clk class dictionary
    """
    for prn in [key for key in clkdata if key[0] == 'G']:
        if len(clkdata[prn].tym) != 0:
            clk_subset = random.sample(range(len(clkdata[prn].tym)-1), NUMSAMPLES)
            for sidx, _ in enumerate(clk_subset):
                func_satbias = extract_clk(clkdata[prn], sidx, \
                                                ipos = 10, method='CubicSpline')
                cxtime = clkdata[prn].tym[sidx]
                satbias_clk, _ = clk_snapshot(func_satbias, cxtime, \
                                                      hstep = 5e-1, method='CubicSpline')
                assert consts.C * np.linalg.norm(satbias_clk - \
                                                 clkdata[prn].clk_bias[sidx]) < 1e-6

def test_glonass_sp3_funcs(sp3data):
    """Tests extract_sp3, sp3_snapshot for GLONASS-Sp3

    Notes
    ----------
    Last index interpolation does not work well, so eliminating this index
    while extracting random samples from tym in Sp3

    Parameters
    ----------
    sp3data : pytest.fixture
        Instance of Sp3 class dictionary
    """

    for prn in [key for key in sp3data if key[0] == 'R']:
        if len(sp3data[prn].tym) != 0:
            sp3_subset = random.sample(range(len(sp3data[prn].tym)-1), NUMSAMPLES)
            for sidx, _ in enumerate(sp3_subset):
                func_satpos = extract_sp3(sp3data[prn], sidx, \
                                               ipos = 10, method='CubicSpline')
                cxtime = sp3data[prn].tym[sidx]
                satpos_sp3, _ = sp3_snapshot(func_satpos, cxtime, \
                                                     hstep = 5e-1, method='CubicSpline')
                satpos_sp3_exact = np.array([ sp3data[prn].xpos[sidx], \
                                              sp3data[prn].ypos[sidx], \
                                              sp3data[prn].zpos[sidx] ])
                assert np.linalg.norm(satpos_sp3 - satpos_sp3_exact) < 1e-6

def test_glonass_clk_funcs(clkdata):
    """Tests extract_sp3, sp3_snapshot for GLONASS-Clk


    Notes
    ----------
    Last index interpolation does not work well, so eliminating this index
    while extracting random samples from tym in Clk

    Parameters
    ----------
    clkdata : pytest.fixture
        Instance of Clk class dictionary
    """

    for prn in [key for key in clkdata if key[0] == 'R']:
        if len(clkdata[prn].tym) != 0:
            clk_subset = random.sample(range(len(clkdata[prn].tym)-1), NUMSAMPLES)
            for sidx, _ in enumerate(clk_subset):
                func_satbias = extract_clk(clkdata[prn], sidx, \
                                             ipos = 10, method='CubicSpline',
                                             verbose=True)
                cxtime = clkdata[prn].tym[sidx]
                satbias_clk, _ = clk_snapshot(func_satbias, cxtime, \
                                                      hstep = 5e-1, method='CubicSpline')
                assert consts.C * np.linalg.norm(satbias_clk - \
                                                 clkdata[prn].clk_bias[sidx]) < 1e-6

def test_compute_gps_precise_eph(navdata_gps, sp3data, clkdata):
    """Tests that single_gnss_from_precise_eph does not fail for GPS

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
    navdata_prcs_gps = single_gnss_from_precise_eph(navdata_prcs_gps, \
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
    """Tests that single_gnss_from_precise_eph does not fail for GPS

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
    new_navdata = single_gnss_from_precise_eph(navdata_prcs_glonass,
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
    new_navdata = single_gnss_from_precise_eph(navdata_prcs_glonass,
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

def test_compute_concat_precise_eph(navdata, sp3_path, clk_path):
    """Tests that multi_gnss_from_precise_eph does not fail for multi-GNSS

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

    navdata_prcs_merged = multi_gnss_from_precise_eph(navdata, sp3_path,
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
    """Tests that sv_gps_from_brdcst_eph does not fail for GPS

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
        sv_gps_from_brdcst_eph(navdata_eph, verbose=True)
    assert "Multi-GNSS" in str(excinfo.value)

    # test what happens when invalid (non-GPS) rows down't exist
    with pytest.raises(RuntimeError) as excinfo:
        navdata_glonassg1_eph = navdata_glonassg1.copy()
        sv_gps_from_brdcst_eph(navdata_glonassg1_eph, verbose=True)
    assert "non-GPS" in str(excinfo.value)

    navdata_gpsl1_eph = navdata_gpsl1.copy()
    navdata_gpsl1_eph = sv_gps_from_brdcst_eph(navdata_gpsl1_eph,
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
