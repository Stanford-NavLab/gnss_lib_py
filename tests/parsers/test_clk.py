"""Tests for precise ephemerides data loaders.
"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "25 August 2022"

import os
import random

import pytest
import numpy as np

from gnss_lib_py.parsers.clk import Clk
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

    References
    ----------
    .. [1]  https://geodesy.noaa.gov/UFCORS/ Accessed as of August 2, 2022
    .. [2]  https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/gnss_mgex.html
            Accessed as of August 2, 2022

    """
    clk_path = os.path.join(root_path, 'clk/grg21553_short.clk')
    return clk_path

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
    clkdata = Clk(clk_path)

    return clkdata

@pytest.fixture(name="clk_path_missing")
def fixture_clk_path_missing(root_path):
    """Invalid filepath of .clk measurements

    Returns
    -------
    clk_path : string
        String with location for the unit_test clk measurements
    """
    clk_path_missing = os.path.join(root_path, 'clk/grg21553_missing.clk')
    return clk_path_missing

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
        Clk(clk_path_missing)

    # raises exception if input not string or path-like
    with pytest.raises(TypeError):
        Clk([])

@pytest.fixture(name="clk_path_nodata")
def fixture_clk_path_nodata(root_path):
    """Filepath for .clk measurements with no data

    Returns
    -------
    clk_path : string
        String with location for the unit_test clk measurements
    """
    clk_path_nodata = os.path.join(root_path, 'clk/grg21553_nodata.clk')
    return clk_path_nodata

def test_load_clkdata_nodata(clk_path_nodata):
    """Load clk instance from file with no data

    Parameters
    ----------
    clk_path_nodata : pytest.fixture
        String with no available data for unit_test clk
        measurements
    """

    clkdata_nodata = Clk(clk_path_nodata)

    assert len(clkdata_nodata) == 0

@pytest.mark.parametrize('row_name, prn, index, exp_value',
                        [('b_sv_m', 'G15', 0, -0.00015303409205*consts.C),
                         ('gps_millis', 'G05', 5, 1303668168000.0),
                         ('b_sv_m', 'R08', 16, -5.87550990462e-05*consts.C),
                         ('gps_millis', 'R14', 10, 1303668318000.0),
                        ]
                        )
def test_clkgps_value_check(clkdata, prn, row_name, index, exp_value):
    """Check Clk array entries of GPS constellation against
    known/expected values using test matrix

    Parameters
    ----------
    clkdata : gnss_lib_py.parsers.clk.Clk
        CLK data.
    prn : int
        Satellite PRN for test example
    row_name : string
        Row key for test example
    index : int
        Index to query data at
    exp_value : float/datetime
        Expected value at queried indices
    """

    assert len(np.unique(clkdata.where("gnss_id","gps")["sv_id"])) == 31
    assert len(np.unique(clkdata.where("gnss_id","glonass")["sv_id"])) == 20

    curr_value = clkdata.where("gnss_sv_id",prn)[row_name][index]
    np.testing.assert_equal(curr_value, exp_value)

def test_gps_clk_funcs(clkdata):
    """Tests extract_clk, clk_snapshot for Clk

    Notes
    ----------
    Last index interpolation does not work well, so eliminating this index
    while extracting random samples from gps_millis in Clk

    Parameters
    ----------
    clkdata : gnss_lib_py.parsers.clk.Clk
        CLK data.

    """


    gnss_sv_ids = np.unique(clkdata["gnss_sv_id"])

    for prn in gnss_sv_ids:
        clkdata_sv = clkdata.where("gnss_sv_id",prn)
        gps_millis_prn = clkdata_sv["gps_millis"]
        clk_subset = random.sample(range(len(gps_millis_prn)-1), NUMSAMPLES)
        for sidx, _ in enumerate(clk_subset):
            func_satbias = clkdata.extract_clk(prn, sidx, \
                                            ipos = 10,
                                            method='CubicSpline',
                                            verbose=True)
            cxtime = gps_millis_prn[sidx]
            satbias_clk, _ = clkdata.clk_snapshot(func_satbias, cxtime, \
                                                  hstep = 5e-1, method='CubicSpline')
            assert np.linalg.norm(satbias_clk - \
                                  clkdata_sv["b_sv_m"][sidx]) < 1e-6
