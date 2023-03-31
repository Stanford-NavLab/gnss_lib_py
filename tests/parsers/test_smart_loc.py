"""Tests for TU Chemnitz SmartLoc data loaders.

"""

__authors__ = "Derek Knowles"
__date__ = "09 Aug 2022"

import os

import pytest
import numpy as np
import pandas as pd

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.parsers.smart_loc import SmartLocRaw, remove_nlos, \
                                        calculate_gt_ecef, calculate_gt_vel

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
        String with filepath to smartloc_raw measurement file

    Returns
    -------
    smartloc_raw_df : pd.DataFrame
        Dataframe with TU Chemnitz Raw measurements
    """
    smartloc_raw_df = pd.read_csv(raw_path, sep=";")
    return smartloc_raw_df


@pytest.fixture(name="smartloc_raw")
def fixture_load_smartloc_raw(raw_path):
    """Load instance of SmartLocRaw

    Parameters
    ----------
    raw_path : pytest.fixture
        String with location of TU Chemnitz rawmeasurement file

    Returns
    -------
    smartloc_raw : SmartLocRaw
        Instance of SmartLocRaw for testing
    """
    smartloc_raw = SmartLocRaw(raw_path)
    return smartloc_raw


def test_smartloc_raw_df_equivalence(smartloc_raw, pd_df):
    """Test if naive dataframe and SmartLocRaw contain same data

    Parameters
    ----------
    smartloc_raw : pytest.fixture
        Instance of SmartLocRaw for testing
    pd_df : pytest.fixture
        pd.DataFrame for testing measurements

    """
    # Also tests if strings are being converted back correctly
    measure_df = smartloc_raw.pandas_df()
    inverse_row_map = {v : k for k,v in smartloc_raw._row_map().items()}
    measure_df.rename(columns=inverse_row_map, inplace=True)
    measure_df.drop(columns='gps_millis',inplace=True)
    pd_df['GNSS identifier (gnssId) []'] = pd_df['GNSS identifier (gnssId) []'].str.lower()
    pd.testing.assert_frame_equal(pd_df.sort_index(axis=1),
                                  measure_df.sort_index(axis=1),
                                  check_dtype=False, check_names=True)


@pytest.mark.parametrize('row_name, index, value',
                        [('raw_pr_m', 0, 19834597.8712713),
                         ('raw_pr_sigma_m', 9, 40.96),
                         ('gnss_id', 55, 'sbas')]
                        )
def test_smartloc_raw_value_check(smartloc_raw, row_name, index, value):
    """Check SmartLocRaw entries against known values using test matrix

    Parameters
    ----------
    smartloc_raw : pytest.fixture
        Instance of SmartLocRaw for testing
    row_name : string
        Row key for test example
    index : int
        Column number for test example
    value : float or string
        Known value to be checked against

    """
    np.testing.assert_equal(smartloc_raw[row_name, index], value)


def test_get_and_set_num(smartloc_raw):
    """Testing __getitem__ and __setitem__ methods for numerical values

    Parameters
    ----------
    smartloc_raw : pytest.fixture
        Instance of SmartLocRaw for testing
    """
    key = 'testing123'
    value = np.zeros(len(smartloc_raw))
    smartloc_raw[key] = value
    np.testing.assert_equal(smartloc_raw[key, :], value)


def test_get_and_set_str(smartloc_raw):
    """Testing __getitem__ and __setitem__ methods for string values

    Parameters
    ----------
    smartloc_raw : pytest.fixture
        Instance of SmartLocRaw for testing
    """
    key = 'testing123_string'
    value = ['word']*len(smartloc_raw)
    smartloc_raw_size = len(smartloc_raw)
    size1 = int(smartloc_raw_size/2)
    size2 = (smartloc_raw_size-int(smartloc_raw_size/2))
    value1 = ['ashwin']*size1
    value2 = ['derek']*size2
    value = np.concatenate((value1,value2))
    smartloc_raw[key] = value.astype(object)

    np.testing.assert_equal(smartloc_raw[key, :], value)

def test_navdata_type(smartloc_raw):
    """Test that all subclasses inherit from NavData

    Parameters
    ----------
    smartloc_raw : pytest.fixture
        Instance of SmartLocRaw for testing
    """
    isinstance(smartloc_raw, NavData)
    isinstance(smartloc_raw, SmartLocRaw)


def test_shape_update(smartloc_raw):
    """Test that shape gets updated after adding a row.

    Parameters
    ----------
    smartloc_raw : pytest.fixture
        Instance of SmartLocRaw for testing
    """
    old_shape = smartloc_raw.shape
    smartloc_raw["new_row"] = np.ones((old_shape[1]))
    new_shape = smartloc_raw.shape

    # should still have the same number of columns (timesteps)
    np.testing.assert_equal(old_shape[1], new_shape[1])
    # should have added one new row
    np.testing.assert_equal(old_shape[0] + 1, new_shape[0])


def test_nlos_removal(smartloc_raw):
    """Test that NLOS removal reduces length of NavData and that length
    stays same for repeated removals.

    Parameters
    ----------
    smartloc_raw : gnss_lib_py.parsers.smart_loc.SmartLocRaw
        Instance of SmartLocRaw for testing
    """
    first_shape = smartloc_raw.shape
    smartloc_los = remove_nlos(smartloc_raw)
    old_shape = smartloc_los.shape
    smartloc_los_new = remove_nlos(smartloc_los)
    new_shape = smartloc_los_new.shape
    # Ensure that size decreases on removing NLOS satellites
    assert first_shape[1] >= old_shape[1]
    # Ensure that only NLOS satellites remain in the new NavData
    assert np.sum(smartloc_los_new['NLOS (0 == no, 1 == yes, # == No Information)']) \
            == new_shape[1]
    # Assert that no rows are removed when removing NLOS measurements
    np.testing.assert_equal(first_shape[0], old_shape[0])
    np.testing.assert_equal(old_shape, new_shape)


def test_calculate_gt_ecef(smartloc_raw):
    """Test that GT position rows are added and size increases as
    expected when using calculate_gt_ecef.

    Parameters
    ----------
    smartloc_raw : gnss_lib_py.parsers.smart_loc.SmartLocRaw
        Instance of SmartLocRaw for testing
    """
    old_shape = np.asarray(smartloc_raw.shape)
    smartloc_raw = calculate_gt_ecef(smartloc_raw)
    new_shape = np.asarray(smartloc_raw.shape)
    smartloc_raw.in_rows(['x_rx_gt_m', 'y_rx_gt_m', 'z_rx_gt_m'])
    old_shape[0] = old_shape[0] + 3
    np.testing.assert_equal(old_shape, new_shape)


def test_calculate_gt_vel_ecef(smartloc_raw):
    """Test that GT position rows are added and size increases as
    expected when using calculate_gt_vel.

    Parameters
    ----------
    smartloc_raw : gnss_lib_py.parsers.smart_loc.SmartLocRaw
        Instance of SmartLocRaw for testing
    """
    old_shape = np.asarray(smartloc_raw.shape)
    smartloc_raw = calculate_gt_vel(smartloc_raw)
    new_shape = np.asarray(smartloc_raw.shape)
    smartloc_raw.in_rows(['vx_rx_gt_mps',
                          'vy_rx_gt_mps',
                          'vz_rx_gt_mps',
                          'ax_rx_gt_mps2',
                          'ay_rx_gt_mps2',
                          'az_rx_gt_mps2',])
    old_shape[0] = old_shape[0] + 6
    np.testing.assert_equal(old_shape, new_shape)


