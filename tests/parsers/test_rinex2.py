"""Tests for rinex2 data loaders.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "30 July 2022"

import os
import numpy as np
import pandas as pd
import georinex as gr
import pytest

from gnss_lib_py.parsers.rinex2 import Rinex2Obs
from gnss_lib_py.parsers.navdata import NavData

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


@pytest.fixture(name="pd_df")
def fixture_pd_df(rinex2_path):
    """Load rinex2 measurements into dataframe

    Parameters
    ----------
    rinex2_path : pytest.fixture
        String with filepath to RINEX 2 .o measurement file

    Returns
    -------
    rinex2_df : pd.DataFrame
        Dataframe with RINEX 2 .o measurements
    """
    obs_gnss = gr.load(rinex2_path)
    obs_gnss_df = obs_gnss.to_dataframe()
    rinex2_df = obs_gnss_df.reset_index()
    return rinex2_df


@pytest.fixture(name="rinex2_row_map")
def fixture_inverse_row_map():
    """Map from standard names to rinex2 column names

    Returns
    -------
    inverse_row_map : Dict
        Column names for inverse map of form {standard_name : rinex2_name}
    """
    inverse_col_map = {'raw_pr_m': 'C1',
                       'raw_dp_hz': 'D1',
                       'cn0_dbhz': 'S1'
                       }
    return inverse_col_map


@pytest.fixture(name="rinex2")
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


def test_rinex2_df_equivalence(rinex2, pd_df, rinex2_row_map):
    """Test if naive dataframe and Rinex2Obs contain same data

    Parameters
    ----------
    rinex2 : pytest.fixture
        Instance of Rinex2Obs for testing
    pd_df : pytest.fixture
        pd.DataFrame for testing measurements
    rinex2_row_map : pytest.fixture
        Column map to convert standard to original rinex2 column names

    Notes
    ----------
    (1) Need to figure out how to retain the datetime when passed
    through NavData class

    """
    # Also tests if strings are being converted back correctly
    measure_df = rinex2.pandas_df()
    measure_df.rename(columns=rinex2_row_map, inplace=True)
    measure_df = measure_df.drop(columns='x_gt_m')
    measure_df = measure_df.drop(columns='y_gt_m')
    measure_df = measure_df.drop(columns='z_gt_m')
    measure_df = measure_df.drop(columns='gps_week')
    measure_df = measure_df.drop(columns='gps_tow')
    measure_df = measure_df.drop(columns='sv_id')
    measure_df = measure_df.drop(columns='gnss_id')
    pd.testing.assert_frame_equal(pd_df.sort_index(axis=1),
                                  measure_df.sort_index(axis=1),
                                  check_dtype=False, check_names=True)


@pytest.mark.parametrize('row_name, index, value',
                        [('raw_pr_m', 0, np.asarray([28221722.273], dtype=object)),
                         ('x_gt_m', 13, np.asarray([-2703115.266], dtype=object)),
                         ('y_gt_m', 18, np.asarray([-4291768.344], dtype=object)),
                         ('z_gt_m', 6, np.asarray([3854247.955], dtype=object)),
                         ('cn0_dbhz', 9, np.asarray([37.5], dtype=object)) ]
                        )
def test_rinex2_value_check(rinex2, row_name, index, value):
    """Check Rinex2Obs entries against known values using test matrix

    Parameters
    ----------
    rinex2 : pytest.fixture
        Instance of Rinex2Obs for testing
    row_name : string
        Row key for test example
    index : int
        Column number for test example
    value : float or string
        Known value to be checked against

    """
    np.testing.assert_equal(rinex2[row_name, index], value)


def test_rinex2_get_and_set_num(rinex2):
    """Testing __getitem__ and __setitem__ methods for numerical values

    Parameters
    ----------
    rinex2 : pytest.fixture
        Instance of Rinex2Obs for testing
    """
    key = 'testing123'
    value = np.zeros(len(rinex2))
    rinex2[key] = value
    np.testing.assert_equal(rinex2[key, :], np.reshape(value, [1, -1]))


def test_rinex2_get_and_set_str(rinex2):
    """Testing __getitem__ and __setitem__ methods for string values

    Parameters
    ----------
    rinex2 : pytest.fixture
        Instance of Rinex2Obs for testing
    """
    key = 'testing123_string'
    value = ['word']*len(rinex2)
    rinex2_size = len(rinex2)
    size1 = int(rinex2_size/4)
    size2 = int(rinex2_size/4)
    size3 = (rinex2_size-size1-size2)
    value1 = ['ashwin']*size1
    value2 = ['derek']*size2
    value3 = ['ramya']*size3
    value = np.concatenate((np.asarray(value1, dtype=object), \
                            np.asarray(value2, dtype=object), \
                            np.asarray(value3, dtype=object)))
    rinex2[key] = value

    np.testing.assert_equal(rinex2[key, :], [value])


def test_rinex2_navdata_type(rinex2):
    """Test that all subclasses inherit from NavData

    Parameters
    ----------
    rinex2 : pytest.fixture
        Instance of Rinex2Obs for testing
    """
    isinstance(rinex2, NavData)
    isinstance(rinex2, Rinex2Obs)


def test_rinex2_shape_update(rinex2):
    """Test that shape gets updated after adding a row.

    Parameters
    ----------
    rinex2 : pytest.fixture
        Instance of Rinex2Obs for testing
    """
    old_shape = rinex2.shape
    rinex2["new_row"] = np.ones((old_shape[1]))
    new_shape = rinex2.shape

    # should still have the same number of columns (timesteps)
    np.testing.assert_equal(old_shape[1], new_shape[1])
    # should have added one new row
    np.testing.assert_equal(old_shape[0] + 1, new_shape[0])
