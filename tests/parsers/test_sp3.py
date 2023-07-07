"""Tests for precise ephemerides data loaders.
"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "25 August 2022"

import os
import random

import pytest
import numpy as np
from datetime import datetime, timezone


from gnss_lib_py.parsers.sp3 import Sp3

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

@pytest.fixture(name="sp3data")
def fixture_load_sp3data(sp3_path):
    """Load instance of sp3 data.

    Parameters
    ----------
    sp3_path : pytest.fixture
        String with location for the unit_test sp3 measurements

    Returns
    -------
    sp3data : list
        Instance of GPS-only Sp3 class list with len = NUMSATS-GPS
    """
    sp3data = Sp3(sp3_path)

    return sp3data

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
        Sp3(sp3_path_missing)

    # raises exception if input not string or path-like
    with pytest.raises(TypeError):
        Sp3([])

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

def test_load_sp3data_nodata(sp3_path_nodata):
    """Load sp3 instance from file with no data

    Parameters
    ----------
    sp3_path_nodata : pytest.fixture
        String with no available data for unit_test sp3
        measurements
    """

    sp3data_nodata = Sp3(sp3_path_nodata)

    assert len(sp3data_nodata) == 0

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
