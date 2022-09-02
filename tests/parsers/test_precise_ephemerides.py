"""Tests for precise ephemerides data loaders.

Notes
----------
(1) Can probably add unit tests to confirm sp3 and clk
data have the same time instant fields? -> maybe not
necessary, overkill?
"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "25 August 2022"

import os

from datetime import datetime, timezone
import numpy as np
import pytest

from gnss_lib_py.parsers.precise_ephemerides import Sp3, Clk, parse_sp3, parse_clockfile

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
    root_path = os.path.join(root_path, 'data/unit_test/precise_ephemeris/')
    return root_path

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
    sp3_path = os.path.join(root_path, 'grg21553_short.sp3')
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
    clk_path = os.path.join(root_path, 'grg21553_short.clk')
    return clk_path

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

@pytest.fixture(name="sp3_path_missing")
def fixture_sp3_path_missing(root_path):
    """Invalid filepath for .sp3 measurements

    Returns
    -------
    sp3_path : string
        String with location for the unit_test sp3 measurements
    """
    sp3_path_missing = os.path.join(root_path, 'grg21553_missing.sp3')
    return sp3_path_missing

@pytest.fixture(name="clk_path_missing")
def fixture_clk_path_missing(root_path):
    """Invalid filepath of .clk measurements

    Returns
    -------
    clk_path : string
        String with location for the unit_test clk measurements

    References
    ----------
    .. [1]  https://geodesy.noaa.gov/UFCORS/ Accessed as of August 2, 2022
    """
    clk_path_missing = os.path.join(root_path, 'grg21553_missing.clk')
    return clk_path_missing

def test_load_sp3data_gps_missing(sp3_path_missing):
    """Load instance of sp3data_gps_missing

    Parameters
    ----------
    sp3_path_missing : pytest.fixture
        String with invalid location for unit_test sp3
        measurements
    """
    # Create a sp3 class for each expected satellite
    with pytest.raises(OSError):
        _ = parse_sp3(sp3_path_missing, constellation = 'gps')

def test_load_clkdata_gps_missing(clk_path_missing):
    """Load instance of clkdata_gps_missing

    Parameters
    ----------
    clk_path_missing : pytest.fixture
        String with invalid location for unit_test clk
        measurements
    """
    # Create a sp3 class for each expected satellite
    with pytest.raises(OSError):
        _ = parse_clockfile(clk_path_missing, constellation = 'gps')

@pytest.fixture(name="sp3_path_nodata")
def fixture_sp3_path_nodata(root_path):
    """Invalid filepath for .sp3 measurements

    Returns
    -------
    sp3_path : string
        String with location for the unit_test sp3 measurements
    """
    sp3_path_nodata = os.path.join(root_path, 'grg21553_nodata.sp3')
    return sp3_path_nodata

@pytest.fixture(name="clk_path_nodata")
def fixture_clk_path_nodata(root_path):
    """Invalid filepath of .clk measurements

    Returns
    -------
    clk_path : string
        String with location for the unit_test clk measurements

    References
    ----------
    .. [1]  https://geodesy.noaa.gov/UFCORS/ Accessed as of August 2, 2022
    """
    clk_path_nodata = os.path.join(root_path, 'grg21553_nodata.clk')
    return clk_path_nodata

def test_load_sp3data_gps_nodata(sp3_path_nodata):
    """Load instance of sp3data_gps_nodata

    Parameters
    ----------
    sp3_path_nodata : pytest.fixture
        String with no available data for unit_test sp3
        measurements
    """
    # Create a sp3 class for each expected satellite

    sp3data_gps_null = Sp3()
    sp3data_gps_null.const = 'gps'
    with pytest.warns(RuntimeWarning):
        sp3data_gps_nodata = parse_sp3(sp3_path_nodata, constellation = 'gps')
        for prn in np.arange(0, NUMSATS_GPS + 1):
            assert sp3data_gps_nodata[prn].__eq__(sp3data_gps_null)

def test_load_clkdata_gps_nodata(clk_path_nodata):
    """Load instance of clkdata_gps_nodata

    Parameters
    ----------
    clk_path_nodata : pytest.fixture
        String with no available data for unit_test clk
        measurements
    """
    # Create a sp3 class for each expected satellite
    clkdata_gps_null = Clk()
    clkdata_gps_null.const = 'gps'
    with pytest.warns(RuntimeWarning):
        clkdata_gps_nodata = parse_clockfile(clk_path_nodata, constellation = 'gps')
        for prn in np.arange(0, NUMSATS_GPS + 1):
            assert clkdata_gps_nodata[prn].__eq__(clkdata_gps_null)

@pytest.mark.parametrize('row_name, prn, index, exp_value',
                        [('xpos', 1, 2, 13222742.845),
                         ('ypos', 12, 5, 9753305.474000001),
                         ('zpos', 32, 25, 21728484.688),
                         ('tym', 8, 8, 1303670400000.0),
                         ('utc_time', 12, 3, datetime(2021, 4, 28, 18, 15, tzinfo=timezone.utc)) ]
                        )
def test_sp3gps_value_check(sp3data_gps, prn, row_name, index, exp_value):
    """Check Sp3 array entries of GPS constellation
    against known values using test matrix

    Parameters
    ----------
    sp3data_gps : pytest.fixture
        Instance of Sp3 class array for testing
    prn : int
        Satellite PRN number for test example
    row_name : string
        Row key for test example
    index : int
        Column number for test example
    exp_value : float or string
        Known/expected value to be checked against
    """
    assert np.size(sp3data_gps[0].xpos) == 0
    assert np.size(sp3data_gps[0].ypos) == 0
    assert np.size(sp3data_gps[0].zpos) == 0
    assert np.size(sp3data_gps[0].tym) == 0
    assert np.size(sp3data_gps[0].utc_time) == 0
    assert len(sp3data_gps) == NUMSATS_GPS + 1

    curr_value = getattr(sp3data_gps[prn], row_name)[index]
    np.testing.assert_equal(curr_value, exp_value)

@pytest.mark.parametrize('row_name, prn, index, exp_value',
                        [('xpos', 24, 1, 13383401.364),
                         ('ypos', 2, 7, 3934479.152),
                         ('zpos', 18, 45, 10107376.674999999),
                         ('tym', 12, 17, 1303673100000.0),
                         ('utc_time', 9, 34, datetime(2021, 4, 28, 20, 50, tzinfo=timezone.utc)) ]
                        )
def test_sp3glonass_value_check(sp3data_glonass, prn, row_name, index, exp_value):
    """Check Sp3 array entries of GLONASS constellation against
    known/expected values using test matrix

    Parameters
    ----------
    sp3data_glonass : pytest.fixture
        Instance of Sp3 class array for testing
    prn : int
        Satellite PRN number for test example
    row_name : string
        Row key for test example
    index : int
        Column number for test example
    exp_value : float or string
        Known/expected value to be checked against
    """
    assert np.size(sp3data_glonass[0].xpos) == 0
    assert np.size(sp3data_glonass[0].ypos) == 0
    assert np.size(sp3data_glonass[0].zpos) == 0
    assert np.size(sp3data_glonass[0].tym) == 0
    assert np.size(sp3data_glonass[0].utc_time) == 0
    assert len(sp3data_glonass) == NUMSATS_GLONASS + 1

    curr_value = getattr(sp3data_glonass[prn], row_name)[index]
    np.testing.assert_equal(curr_value, exp_value)

@pytest.mark.parametrize('row_name, prn, index, exp_value',
                        [('clk_bias', 15, 0, -0.00015303409205),
                         ('tym', 5, 5, 1303668150000.0),
                         ('utc_time', 32, 4, datetime(2021, 4, 28, 18, 2, tzinfo=timezone.utc)) ]
                        )
def test_clkgps_value_check(clkdata_gps, prn, row_name, index, exp_value):
    """Check Clk array entries of GPS constellation against
    known/expected values using test matrix

    Parameters
    ----------
    clkdata_gps : pytest.fixture
        Instance of Clk class array for testing
    prn : int
        Satellite PRN number for test example
    row_name : string
        Row key for test example
    index : int
        Column number for test example
    exp_value : float or string
        Known/expected value to be checked against
    """
    assert np.size(clkdata_gps[0].tym) == 0
    assert np.size(clkdata_gps[0].clk_bias) == 0
    assert np.size(clkdata_gps[0].utc_time) == 0
    assert len(clkdata_gps) == NUMSATS_GPS + 1

    curr_value = getattr(clkdata_gps[prn], row_name)[index]
    np.testing.assert_equal(curr_value, exp_value)

@pytest.mark.parametrize('row_name, prn, index, exp_value',
                        [('clk_bias', 8, 16, -5.87550990462e-05),
                         ('tym', 14, 10, 1303668300000.0),
                         ('utc_time', 4, 4, datetime(2021, 4, 28, 18, 2, tzinfo=timezone.utc)) ]
                        )
def test_clkglonass_value_check(clkdata_glonass, prn, row_name, index, exp_value):
    """Check Clk array entries of GLONASS constellation against
    known/expected values using test matrix

    Parameters
    ----------
    clkdata_glonass : pytest.fixture
        Instance of Clk class array for testing
    prn : int
        Satellite PRN number for test example
    row_name : string
        Row key for test example
    index : int
        Column number for test example
    exp_value : float or string
        Known/expected value to be checked against
    """
    assert np.size(clkdata_glonass[0].tym) == 0
    assert np.size(clkdata_glonass[0].clk_bias) == 0
    assert np.size(clkdata_glonass[0].utc_time) == 0
    assert len(clkdata_glonass) == NUMSATS_GLONASS + 1

    curr_value = getattr(clkdata_glonass[prn], row_name)[index]
    np.testing.assert_equal(curr_value, exp_value)
