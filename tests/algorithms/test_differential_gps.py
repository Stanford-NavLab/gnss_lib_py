"""Tests for differential gps codes.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "30 July 2022"

import os
import pytest

import numpy as np
import pandas as pd

from gnss_lib_py.parsers.android import AndroidDerived
from gnss_lib_py.parsers.rinex2 import Rinex2Obs
from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.algorithms.differential_gps import compute_all_dgpscorr

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


@pytest.fixture(name="rinex2_path")
def fixture_rinex2_path(root_path):
    """Filepath of RINEX 2 .o measurements

    Returns
    -------
    rinex2_path : string
        Location for the unit_test RINEX 2 measurements

    References
    ----------
    .. [1]  https://geodesy.noaa.gov/UFCORS/ Accessed as of August 2, 2022
    """
    rinex2_path = os.path.join(root_path, 'slac1180.21o')
    return rinex2_path

@pytest.fixture(name="base")
def fixture_load_rinex2(rinex2_path):
    """Load instance of Rinex2Obs

    Parameters
    ----------
    rinex2_path : pytest.fixture
        String with location of rinex2 measurement file

    Returns
    -------
    rinex2 : Rinex2Obs
        Instance of Rinex2Obs for testing
    """
    rinex2 = Rinex2Obs(rinex2_path)
    return rinex2

@pytest.fixture(name="derived_path")
def fixture_derived_path(root_path):
    """Filepath of Android Derived measurements

    Returns
    -------
    derived_path : string
        Location for the unit_test Android derived measurements

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset [1]_,
    particularly the train/2020-05-14-US-MTV-1/Pixel4 trace. The dataset
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
    derived_path = os.path.join(root_path, 'Pixel4_derived_another.csv')
    return derived_path

@pytest.fixture(name="derived")
def fixture_load_derived(derived_path):
    """Load instance of AndroidDerived

    Parameters
    ----------
    derived_path : pytest.fixture
        String with location of Android derived measurement file

    Returns
    -------
    derived : AndroidDerived
        Instance of AndroidDerived for testing
    """
    derived = AndroidDerived(derived_path)

    return derived

@pytest.fixture(name="derived_gl1")
def fixture_load_derived_gl1(derived_path):
    """Load instance of AndroidDerived

    Parameters
    ----------
    derived_path : pytest.fixture
        String with location of Android derived measurement file

    Returns
    -------
    derived : AndroidDerived
        Instance of AndroidDerived for testing
    """
    derived_data = AndroidDerived(derived_path)
    gl1_idxs = np.where((derived_data["gnss_id",:] == 1) & \
                        (derived_data["signal_type",:] == 'GPS_L1'))[1]
    derived_gl1 = derived_data.copy(cols=gl1_idxs)

    return derived_gl1

def test_compute_all_dgpscorr(derived, derived_gl1, base):
    """Tests that compute_all_dgpscorr does not fail

    Parameters
    ----------
    derived : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class that depicts android derived dataset
    derived : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class that depicts android derived GPS L1
    base : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class that depicts base station dataset

    Notes
    ----------
    (1) Need to add another validation base station dataset here
    (2) Need to shorten the data being loaded for unit tests
    """
    # test what happens when rows down't exist
    derived_no_x_sv_m = derived.remove(rows="x_sv_m")
    with pytest.raises(KeyError) as excinfo:
        solve_wls(derived, base)
    assert "x_sv_m" in str(excinfo.value)

    derived_gl1_result = compute_all_dgpscorr(derived_gl1, base)

    # Check if the resulting derived class has corrected pseudoranges
    assert "corr_pr_m" in derived_gl1_result.rows

    # Check if the resulting derived is NavData class
    assert isinstance( derived_gl1_result, type(NavData()) )

    # Check if the derived classes are same except for corr_pr_m
    derived_gl1_df = derived_gl1.pandas_df()
    derived_gl1_df = derived_gl1_df.drop(columns='corr_pr_m')

    derived_gl1_result_df = derived_gl1_result.pandas_df()
    derived_gl1_result_df = derived_gl1_result_df.drop(columns='corr_pr_m')

    pd.testing.assert_frame_equal(derived_gl1_df.sort_index(axis=1),
                                  derived_gl1_result_df.sort_index(axis=1),
                                  check_dtype=False, check_names=True)

    # Check if corr_pr_m from AndroidDerived and Rinex2Obs closely resemble
    assert max(abs(derived_gl1_result['corr_pr_m'][0] - derived_gl1['corr_pr_m'][0])) < 5.0
