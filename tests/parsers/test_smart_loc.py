"""Tests for TU Chemnitz SmartLoc data loaders.

"""

__authors__ = "Derek Knowles"
__date__ = "09 Aug 2022"

import os

import pytest
import numpy as np
import pandas as pd

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.parsers.smart_loc import SmartLocRaw

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


@pytest.fixture(name="raw_path")
def fixture_raw_path(root_path):
    """Filepath of TU Chemnitz Raw measurements

    Returns
    -------
    raw_path : string
        Location for the unit_test TU Chemnitz rawmeasurements

    Notes
    -----
    Raw measurements come from TU Chemnitz dataset [1]_.

    Dataset is available on their website [2]_.

    References
    ----------
    .. [1] Reisdorf, Pierre, Tim Pfeifer, Julia Bressler, Sven Bauer,
           Peter Weissig, Sven Lange, Gerd Wanielik and Peter Protzel.
           The Problem of Comparable GNSS Results – An Approach for a
           Uniform Dataset with Low-Cost and Reference Data. Vehicular.
           2016.
    .. [2] https://www.tu-chemnitz.de/projekt/smartLoc/gnss_dataset.html.en#Home

    """
    raw_path = os.path.join(root_path, 'tu_chemnitz_berlin_1_raw.csv')
    return raw_path

@pytest.fixture(name="pd_df")
def fixture_pd_df(raw_path):
    """Load TU Chemnitz rawmeasurements into dataframe

    Parameters
    ----------
    raw_path : pytest.fixture
        String with filepath to navdata_raw measurement file

    Returns
    -------
    navdata_raw_df : pd.DataFrame
        Dataframe with TU Chemnitz Raw measurements
    """
    navdata_raw_df = pd.read_csv(raw_path, sep=";")
    return navdata_raw_df


@pytest.fixture(name="navdata_raw")
def fixture_load_navdata_raw(raw_path):
    """Load instance of SmartLocRaw

    Parameters
    ----------
    raw_path : pytest.fixture
        String with location of TU Chemnitz rawmeasurement file

    Returns
    -------
    navdata_raw : SmartLocRaw
        Instance of SmartLocRaw for testing
    """
    navdata_raw = SmartLocRaw(raw_path)
    return navdata_raw


def test_navdata_raw_df_equivalence(navdata_raw, pd_df):
    """Test if naive dataframe and SmartLocRaw contain same data

    Parameters
    ----------
    navdata_raw : pytest.fixture
        Instance of SmartLocRaw for testing
    pd_df : pytest.fixture
        pd.DataFrame for testing measurements

    """
    # Also tests if strings are being converted back correctly
    measure_df = navdata_raw.pandas_df()
    inverse_row_map = {v : k for k,v in navdata_raw._row_map().items()}
    measure_df.rename(columns=inverse_row_map, inplace=True)
    pd_df['GNSS identifier (gnssId) []'] = pd_df['GNSS identifier (gnssId) []'].str.lower()
    pd.testing.assert_frame_equal(pd_df.sort_index(axis=1),
                                  measure_df.sort_index(axis=1),
                                  check_dtype=False, check_names=True)


@pytest.mark.parametrize('row_name, index, value',
                        [('raw_pr_m', 0, 19834597.8712713),
                         ('raw_pr_sigma_m', 9, 40.96),
                         ('gnss_id', 55, np.asarray([['sbas']], dtype=object))]
                        )
def test_navdata_raw_value_check(navdata_raw, row_name, index, value):
    """Check SmartLocRaw entries against known values using test matrix

    Parameters
    ----------
    navdata_raw : pytest.fixture
        Instance of SmartLocRaw for testing
    row_name : string
        Row key for test example
    index : int
        Column number for test example
    value : float or string
        Known value to be checked against

    """
    np.testing.assert_equal(navdata_raw[row_name, index], value)


def test_get_and_set_num(navdata_raw):
    """Testing __getitem__ and __setitem__ methods for numerical values

    Parameters
    ----------
    navdata_raw : pytest.fixture
        Instance of SmartLocRaw for testing
    """
    key = 'testing123'
    value = np.zeros(len(navdata_raw))
    navdata_raw[key] = value
    np.testing.assert_equal(navdata_raw[key, :], np.reshape(value, [1, -1]))


def test_get_and_set_str(navdata_raw):
    """Testing __getitem__ and __setitem__ methods for string values

    Parameters
    ----------
    navdata_raw : pytest.fixture
        Instance of SmartLocRaw for testing
    """
    key = 'testing123_string'
    value = ['word']*len(navdata_raw)
    navdata_raw_size = len(navdata_raw)
    size1 = int(navdata_raw_size/2)
    size2 = (navdata_raw_size-int(navdata_raw_size/2))
    value1 = ['ashwin']*size1
    value2 = ['derek']*size2
    value = np.concatenate((np.asarray(value1, dtype=object), np.asarray(value2, dtype=object)))
    navdata_raw[key] = value

    np.testing.assert_equal(navdata_raw[key, :], [value])

def test_navdata_type(navdata_raw):
    """Test that all subclasses inherit from NavData

    Parameters
    ----------
    navdata_raw : pytest.fixture
        Instance of SmartLocRaw for testing
    """
    isinstance(navdata_raw, NavData)
    isinstance(navdata_raw, SmartLocRaw)


def test_shape_update(navdata_raw):
    """Test that shape gets updated after adding a row.

    Parameters
    ----------
    navdata_raw : pytest.fixture
        Instance of SmartLocRaw for testing
    """
    old_shape = navdata_raw.shape
    navdata_raw["new_row"] = np.ones((old_shape[1]))
    new_shape = navdata_raw.shape

    # should still have the same number of columns (timesteps)
    np.testing.assert_equal(old_shape[1], new_shape[1])
    # should have added one new row
    np.testing.assert_equal(old_shape[0] + 1, new_shape[0])
