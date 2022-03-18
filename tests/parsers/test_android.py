"""Tests for Android data loaders.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "10 Nov 2021"

import os

import numpy as np
import pandas as pd
import pytest

from gnss_lib_py.parsers.android import AndroidDerived, AndroidRawFixes, AndroidRawImu
from gnss_lib_py.parsers.measurement import Measurement
from gnss_lib_py.parsers.android import make_csv


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


@pytest.fixture(name="derived_path")
def fixture_derived_path(root_path):
    """Filepath of Android Derived measurements

    Returns
    -------
    derived_path : string
        Location for the unit_test Android derived measurements

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset,
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
    derived_path = os.path.join(root_path, 'Pixel4_derived.csv')
    return derived_path


@pytest.fixture(name="android_raw_path")
def fixture_raw_path(root_path):
    """Filepath of Android Raw measurements

    Returns
    -------
    raw_path : string
        Location for text log file with Android Raw measurements

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset,
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
    raw_path = os.path.join(root_path, 'Pixel4_GnssLog.txt')
    return raw_path


@pytest.fixture(name="pd_df")
def fixture_pd_df(derived_path):
    """Load Android derived measurements into dataframe

    Parameters
    ----------
    derived_path : pytest.fixture
        String with filepath to derived measurement file

    Returns
    -------
    derived_df : pd.DataFrame
        Dataframe with Android Derived measurements
    """
    derived_df = pd.read_csv(derived_path)
    return derived_df


@pytest.fixture(name="derived_col_map")
def fixture_inverse_col_map():
    """Map from standard names to derived column names

    Returns
    -------
    inverse_col_map : Dict
        Column names for inverse map of form {standard_name : derived_name}
    """
    inverse_col_map = {'trace_name' : 'collectionName',
                       'rx_name' : 'phoneName',
                       'gnss_id' : 'constellationType',
                       'sv_id' : 'svid',
                       'signal_type' : 'signalType',
                       'x_sat_m' : 'xSatPosM',
                       'y_sat_m' : 'ySatPosM',
                       'z_sat_m' : 'zSatPosM',
                       'vx_sat_mps' : 'xSatVelMps',
                       'vy_sat_mps' : 'ySatVelMps',
                       'vz_sat_mps' : 'zSatVelMps',
                       'b_sat_m' : 'satClkBiasM',
                       'b_dot_sat_mps' : 'satClkDriftMps',
                       'raw_pr_m' : 'rawPrM',
                       'raw_pr_sigma_m' : 'rawPrUncM',
                       'intersignal_bias_m' : 'isrbM',
                       'iono_delay_m' : 'ionoDelayM',
                       'tropo_delay_m' : 'tropoDelayM',
                    }
    return inverse_col_map


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


def test_derived_df_equivalence(derived, pd_df, derived_col_map):
    """Test if naive dataframe and AndroidDerived contain same data

    Parameters
    ----------
    derived : pytest.fixture
        Instance of AndroidDerived for testing
    pd_df : pytest.fixture
        pd.DataFrame for testing measurements
    derived_col_map : pytest.fixture
        Column map to convert standard to original derived column names

    """
    # Also tests if strings are being converted back correctly
    measure_df = derived.pandas_df()
    measure_df.rename(columns=derived_col_map, inplace=True)
    measure_df = measure_df.drop(columns='pseudo')
    pd.testing.assert_frame_equal(pd_df, measure_df, check_dtype=False)


@pytest.mark.parametrize('row_name, index, value',
                        [('trace_name', 0, '2020-05-14-US-MTV-1'),
                         ('rx_name', 1, 'Pixel4'),
                         ('vy_sat_mps', 7, 411.162),
                         ('b_dot_sat_mps', 41, -0.003),
                         ('signal_type', 6, 'GLO_G1')]
                        )
def test_derived_value_check(derived, row_name, index, value):
    """Check AndroidDerived entries against known values using test matrix

    Parameters
    ----------
    derived : pytest.fixture
        Instance of AndroidDerived for testing
    row_name : string
        Row key for test example
    index : int
        Column number for test example
    value : float or string
        Known value to be checked against

    """
    # Testing stored values vs their known counterparts
    # String maps have been converted to equivalent integers
    if isinstance(value, str):
        value_str = derived.str_map[row_name][int(derived[row_name, index])]
        assert value == value_str
    else:
        np.testing.assert_equal(derived[row_name, index], value)


def test_get_and_set_num(derived):
    """Testing __getitem__ and __setitem__ methods for numerical values

    Parameters
    ----------
    derived : pytest.fixture
        Instance of AndroidDerived for testing
    """
    key = 'testing123'
    value = np.zeros(len(derived))
    derived[key] = value
    np.testing.assert_equal(derived[key, :], np.reshape(value, [1, -1]))


def test_get_and_set_str(derived):
    """Testing __getitem__ and __setitem__ methods for string values

    Parameters
    ----------
    derived : pytest.fixture
        Instance of AndroidDerived for testing
    """
    key = 'testing123_string'
    value = ['word']*len(derived)
    derived[key] = value
    np.testing.assert_equal(derived[key, :], np.zeros([1, len(derived)]))

def test_set_all(derived):
    """Testing __setitem__ method for all rows simultaneously

    Parameters
    ----------
    derived : pytest.fixture
        Instance of AndroidDerived for testing
    """
    assign_vals = np.zeros(len(derived))
    assign_vals[int(len(derived)/2):] = 1
    num_ones = np.sum(assign_vals==1)

    # choose_rows = [0, 2, 3, 5, 6]
    old_vals = derived.array[:, assign_vals==1]
    derived['all'] = derived[:, assign_vals==1]
    np.testing.assert_equal(len(derived), num_ones)
    np.testing.assert_equal(derived.array[:, :], old_vals)


def test_imu_raw(android_raw_path):
    """Test that AndroidRawImu initialization

    Parameters
    ----------
    android_raw_path : pytest.fixture
        Path to Android Raw measurements text log file
    """
    test_imu = AndroidRawImu(android_raw_path)
    isinstance(test_imu, Measurement)


def test_fix_raw(android_raw_path):
    """Test that AndroidRawImu initialization

    Parameters
    ----------
    android_raw_path : pytest.fixture
        Path to Android Raw measurements text log file
    """
    test_fix = AndroidRawFixes(android_raw_path)
    isinstance(test_fix, Measurement)


def test_measurement_type(derived):
    """Test that all subclasses inherit from Measurement

    Parameters
    ----------
    derived : pytest.fixture
        Instance of AndroidDerived for testing
    """
    isinstance(derived, Measurement)
    isinstance(derived, AndroidDerived)


def test_shape_update(derived):
    """Test that shape gets updated after adding a row.

    Parameters
    ----------
    derived : pytest.fixture
        Instance of AndroidDerived for testing
    """
    old_shape = derived.shape
    derived["new_row"] = np.ones((old_shape[1]))
    new_shape = derived.shape

    # should still have the same number of columns (timesteps)
    np.testing.assert_equal(old_shape[1], new_shape[1])
    # should have added one new row
    np.testing.assert_equal(old_shape[0] + 1, new_shape[0])


#TODO: Add check for equivalence of Raw measurements
@pytest.mark.parametrize('file_type',
                        ['Accel',
                        'Gyro',
                        'Fix'])
def test_csv_equivalence(android_raw_path, root_path, file_type):
    """Test equivalence of loaded measurements and data from split csv

    Parameters
    ----------
    android_raw_path : pytest.fixture
        Path to Android Raw measurements text log file

    root_path : pytest.fixture
        Path to location of all data for Android unit testing

    file_type : string
        Type of measurement to be extracted into csv files

    """
    #NOTE: Times for gyroscope measurements are overridden by accel times
    # and are not checked in this test for any measurement
    no_check = ['utcTimeMillis', 'elapsedRealtimeNanos']
    if file_type=='Accel' or file_type=='Gyro':
        test_measure = AndroidRawImu(android_raw_path)
    elif file_type=='Fix':
        test_measure = AndroidRawFixes(android_raw_path)
    output_directory = os.path.join(root_path, 'csv_test')
    csv_loc = make_csv(android_raw_path, output_directory, file_type)
    test_df = pd.read_csv(csv_loc)
    test_measure = AndroidRawImu(android_raw_path)
    col_map = test_measure._column_map()
    for col_name in test_df.columns:
        if col_name in col_map:
            row_idx = col_map[col_name]
        else:
            row_idx = col_name
        if col_name in no_check or col_name :
            break
        measure_slice = test_measure[row_idx, :]
        df_slice = test_df[col_name].values
        np.testing.assert_almost_equal(measure_slice, df_slice)
    os.remove(csv_loc)
