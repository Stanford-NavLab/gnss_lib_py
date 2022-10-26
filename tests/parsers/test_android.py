"""Tests for Android data loaders.

"""

__authors__ = "Ashwin Kanhere, Derek Knowles"
__date__ = "10 Nov 2021"

import os

import pytest
import numpy as np
import pandas as pd

from gnss_lib_py.parsers import android
from gnss_lib_py.parsers.navdata import NavData

# pylint: disable=protected-access

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
    root_path = os.path.join(root_path, 'data/unit_test/android_2021')
    return root_path


@pytest.fixture(name="derived_path")
def fixture_derived_path(root_path):
    """Filepath of Android Derived measurements

    Parameters
    ----------
    root_path : string
        Path of testing dataset root path

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
    derived_path = os.path.join(root_path, 'Pixel4_derived.csv')
    return derived_path

@pytest.fixture(name="derived_path_xl")
def fixture_derived_path_xl(root_path):
    """Filepath of Android Derived measurements

    Parameters
    ----------
    root_path : string
        Path of testing dataset root path

    Returns
    -------
    derived_path : string
        Location for the unit_test Android derived measurements

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset [6]_,
    particularly the train/2020-05-14-US-MTV-1/Pixel4XL trace. The
    dataset was retrieved from
    https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data

    References
    ----------
    .. [6] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.
    """
    derived_path = os.path.join(root_path, 'Pixel4XL_derived.csv')
    return derived_path


@pytest.fixture(name="android_raw_path")
def fixture_raw_path(root_path):
    """Filepath of Android Raw measurements

    Parameters
    ----------
    root_path : string
        Path of testing dataset root path

    Returns
    -------
    raw_path : string
        Location for text log file with Android Raw measurements

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset [2]_,
    particularly the train/2020-05-14-US-MTV-1/Pixel4 trace. The dataset
    was retrieved from
    https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data

    References
    ----------
    .. [2] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
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


@pytest.fixture(name="derived_row_map")
def fixture_inverse_row_map():
    """Map from standard names to derived column names

    Returns
    -------
    inverse_row_map : Dict
        Column names for inverse map of form
        {standard_name : derived_name}
    """
    inverse_row_map = {v : k for k,v in android.AndroidDerived2021._row_map().items()}
    return inverse_row_map


@pytest.fixture(name="derived")
def fixture_load_derived(derived_path):
    """Load instance of AndroidDerived2021

    Parameters
    ----------
    derived_path : pytest.fixture
        String with location of Android derived measurement file

    Returns
    -------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing
    """
    derived = android.AndroidDerived2021(derived_path)
    return derived

def test_derived_df_equivalence(derived_path, pd_df, derived_row_map):
    """Test if naive dataframe and AndroidDerived2021 contain same data.

    Parameters
    ----------
    derived_path : string
        Location for the unit_test Android 2021 derived measurements.
    pd_df : pytest.fixture
        pd.DataFrame for testing measurements
    derived_row_map : pytest.fixture
        Column map to convert standard to original derived column names.

    """
    # Also tests if strings are being converted back correctly
    derived = android.AndroidDerived2021(derived_path,
                               remove_timing_outliers=False)
    measure_df = derived.pandas_df()
    measure_df.replace({'gnss_id',"gps"},1,inplace=True)
    measure_df.replace({'gnss_id',"glonass"},3,inplace=True)
    measure_df.replace({'gnss_id',"galileo"},6,inplace=True)
    signal_map = {"GPS_L1" : "l1",
                  "GPS_L5" : "l5",
                  "GAL_E1" : "e1",
                  "GAL_E5A" : "e5a",
                  "GLO_G1" : "g1",
                  "QZS_J1" : "j1",
                  "QZS_J5" : "j5",
                  "BDS_B1I" : "b1i",
                  "BDS_B1C" : "b1c",
                  "BDS_B2A" : "b2a",
                 }
    for s_key, s_value in signal_map.items():
        measure_df.replace({'signal_type',s_value},s_key,inplace=True)
    measure_df.rename(columns=derived_row_map, inplace=True)
    measure_df = measure_df.drop(columns='corr_pr_m')
    pd_df = pd_df[pd_df['millisSinceGpsEpoch'] != pd_df.loc[0,'millisSinceGpsEpoch']]
    pd_df.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(pd_df.sort_index(axis=1),
                                  measure_df.sort_index(axis=1),
                                  check_dtype=False, check_names=True)


@pytest.mark.parametrize('row_name, index, value',
                        [('trace_name', 0,
                          np.asarray([['2020-05-14-US-MTV-1']],
                          dtype=object)),
                         ('rx_name', 1,
                          np.asarray([['Pixel4']], dtype=object)),
                         ('vz_sv_mps', 0, 3559.812),
                         ('b_dot_sv_mps', 0, 0.001),
                         ('signal_type', 0,
                          np.asarray([['g1']], dtype=object))]
                        )
def test_derived_value_check(derived, row_name, index, value):
    """Check AndroidDerived2021 entries against known values.

    Parameters
    ----------
    derived : pytest.fixture
        Instance of AndroidDerived2021 for testing
    row_name : string
        Row key for test example
    index : int
        Column number for test example
    value : float or string
        Known value to be checked against

    """
    # Testing stored values vs their known counterparts
    # After filtering for good values, Row 28 is the first row of the
    # dataset because the first time frame is removed.
    # Hardcoded values have been taken from the corresponding row in the
    # csv file
    np.testing.assert_equal(derived[row_name, index], value)


def test_get_and_set_num(derived):
    """Testing __getitem__ and __setitem__ methods for numerical values

    Parameters
    ----------
    derived : pytest.fixture
        Instance of AndroidDerived2021 for testing
    """
    key = 'testing123'
    value = np.zeros(len(derived))
    derived[key] = value
    np.testing.assert_equal(derived[key, :], value)


def test_get_and_set_str(derived):
    """Testing __getitem__ and __setitem__ methods for string values

    Parameters
    ----------
    derived : pytest.fixture
        Instance of AndroidDerived2021 for testing
    """
    key = 'testing123_string'
    value = ['word']*len(derived)
    derived_size = len(derived)
    size1 = int(derived_size/2)
    size2 = (derived_size-int(derived_size/2))
    value1 = ['ashwin']*size1
    value2 = ['derek']*size2
    value = np.concatenate((np.asarray(value1, dtype=object),
                            np.asarray(value2, dtype=object)))
    derived[key] = value

    np.testing.assert_equal(derived[key, :], value)

def test_android_concat(derived, pd_df):
    """Test concat on Android data.

    Parameters
    ----------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing
    pd_df : pytest.fixture
        pd.DataFrame for testing measurements
    """

    # remove first timestamp to match
    pd_df = pd_df[pd_df['millisSinceGpsEpoch'] != pd_df.loc[0,'millisSinceGpsEpoch']]

    # extract and combine gps and glonass data
    gps_data = derived.where("gnss_id","gps")
    glonass_data = derived.where("gnss_id","glonass")
    gps_glonass_navdata = gps_data.concat(glonass_data)
    glonass_gps_navdata = glonass_data.concat(gps_data)

    # combine using pandas
    gps_df = pd_df[pd_df["constellationType"]==1]
    glonass_df = pd_df[pd_df["constellationType"]==3]
    gps_glonass_df = pd.concat((gps_df,glonass_df))
    glonass_gps_df = pd.concat((glonass_df,gps_df))

    for combined_navdata, combined_df in [(gps_glonass_navdata, gps_glonass_df),
                                          (glonass_gps_navdata, glonass_gps_df)]:

        # check a few rows to make sure they're equal
        np.testing.assert_array_equal(combined_navdata["raw_pr_m"],
                                      combined_df["rawPrM"])
        np.testing.assert_array_equal(combined_navdata["raw_pr_sigma_m"],
                                      combined_df["rawPrUncM"])
        np.testing.assert_array_equal(combined_navdata["intersignal_bias_m"],
                                      combined_df["isrbM"])

def test_imu_raw(android_raw_path):
    """Test that AndroidRawImu initialization

    Parameters
    ----------
    android_raw_path : pytest.fixture
        Path to Android Raw measurements text log file
    """
    test_imu = android.AndroidRawImu(android_raw_path)
    isinstance(test_imu, NavData)


def test_fix_raw(android_raw_path):
    """Test that AndroidRawImu initialization

    Parameters
    ----------
    android_raw_path : pytest.fixture
        Path to Android Raw measurements text log file
    """
    test_fix = android.AndroidRawFixes(android_raw_path)
    isinstance(test_fix, NavData)


def test_navdata_type(derived):
    """Test that all subclasses inherit from NavData

    Parameters
    ----------
    derived : pytest.fixture
        Instance of AndroidDerived2021 for testing
    """
    isinstance(derived, NavData)
    isinstance(derived, android.AndroidDerived2021)

def test_timestep_parsing(derived_path_xl):
    """Test that the timesteps contain the same satellites.

    """

    pd_df_xl = pd.read_csv(derived_path_xl)
    derived_xl = android.AndroidDerived2021(derived_path_xl,
                               remove_timing_outliers=False)

    pd_svid_groups = []
    for _, group in pd_df_xl.groupby("millisSinceGpsEpoch"):
        pd_svid_groups.append(group["svid"].tolist())
    pd_svid_groups.pop(0)

    navdata_svid_groups = []
    for _, _, group in derived_xl.loop_time("gps_millis"):
        navdata_svid_groups.append(group["sv_id"].astype(int).tolist())

    assert len(pd_svid_groups) == len(navdata_svid_groups)

    for pd_ids, navdata_ids in zip(pd_svid_groups,navdata_svid_groups):
        assert pd_ids == navdata_ids


def test_shape_update(derived):
    """Test that shape gets updated after adding a row.

    Parameters
    ----------
    derived : pytest.fixture
        Instance of AndroidDerived2021 for testing
    """
    old_shape = derived.shape
    derived["new_row"] = np.ones((old_shape[1]))
    new_shape = derived.shape

    # should still have the same number of columns (timesteps)
    np.testing.assert_equal(old_shape[1], new_shape[1])
    # should have added one new row
    np.testing.assert_equal(old_shape[0] + 1, new_shape[0])

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
        test_measure = android.AndroidRawImu(android_raw_path)
    elif file_type=='Fix':
        test_measure = android.AndroidRawFixes(android_raw_path)
    output_directory = os.path.join(root_path, 'csv_test')
    csv_loc = android.make_csv(android_raw_path, output_directory,
                               file_type)
    test_df = pd.read_csv(csv_loc)
    test_measure = android.AndroidRawImu(android_raw_path)
    row_map = test_measure._row_map()
    for col_name in test_df.columns:
        if col_name in row_map:
            row_idx = row_map[col_name]
        else:
            row_idx = col_name
        if col_name in no_check or col_name :
            break
        measure_slice = test_measure[row_idx, :]
        df_slice = test_df[col_name].values
        np.testing.assert_almost_equal(measure_slice, df_slice)
    os.remove(csv_loc)
    os.rmdir(output_directory)

@pytest.fixture(name="android_gtruth_path")
def fixture_gtruth_path(root_path):
    """Filepath of Android Ground Truth data

    Returns
    -------
    gtruth_path : string
        Location for text log file with Android Ground Truth measurements

    Notes
    -----
    Test data is a subset of the Android Ground Truth Dataset [3]_,
    particularly the train/2020-05-14-US-MTV-1/Pixel4 trace. The dataset
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
    gtruth_path = os.path.join(root_path, 'Pixel4_ground_truth.csv')
    return gtruth_path

@pytest.fixture(name="gtruth")
def fixture_load_gtruth(android_gtruth_path):
    """Load instance of AndroidGroundTruth2021

    Parameters
    ----------
    gtruth_path : pytest.fixture
        String with location of Android ground truth file

    Returns
    -------
    gtruth : AndroidGroundTruth2021
        Instance of AndroidGroundTruth2021 for testing
    """
    gtruth = android.AndroidGroundTruth2021(android_gtruth_path)

    return gtruth

def test_android_gtruth(gtruth):
    """Test AndroidGroundTruth initialization

    Parameters
    ----------
    gtruth : AndroidGroundTruth2021
        Instance of AndroidGroundTruth2021 for testing

    """

    isinstance(gtruth, NavData)
    isinstance(gtruth, android.AndroidGroundTruth2021)

    assert int(gtruth["gps_millis",3]) == 1273529466442


######################################################################
#### Android Derived 2022 Dataset tests
######################################################################

@pytest.fixture(name="root_path_2022")
def fixture_root_path_2022():
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
    root_path = os.path.join(root_path, 'data/unit_test/android_2022')
    return root_path


@pytest.fixture(name="derived_2022_path")
def fixture_derived_2022_path(root_path_2022):
    """Filepath of Android Derived measurements

    Returns
    -------
    derived_path : string
        Location for the unit_test Android derived 2022 measurements

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset [4]_,
    from the 2022 Decimeter Challenge. Particularly, the
    train/2021-04-29-MTV-2/SamsungGalaxyS20Ultra trace. The dataset
    was retrieved from
    https://www.kaggle.com/competitions/smartphone-decimeter-2022/data

    References
    ----------
    .. [4] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.
    """
    derived_path = os.path.join(root_path_2022, 'device_gnss.csv')
    return derived_path


@pytest.fixture(name="gt_2022_path")
def fixture_gt_2022_path(root_path_2022):
    """Filepath of Android ground truth estimates

    Returns
    -------
    derived_path : string
        Location for the unit_test Android ground truth estimates

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset [5]_,
    from the 2022 Decimeter Challenge. Particularly, the
    train/2021-04-29-MTV-2/SamsungGalaxyS20Ultra trace. The dataset
    was retrieved from
    https://www.kaggle.com/competitions/smartphone-decimeter-2022/data

    References
    ----------
    .. [5] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.
    """
    gt_path = os.path.join(root_path_2022, 'ground_truth.csv')
    return gt_path


def test_derived_2022(derived_2022_path):
    """Testing that Android Derived 2022 is created without errors.

    Parameters
    ----------
    derived_2022_path : string
        Location for the unit_test Android derived 2022 measurements
    """
    derived_2022 = android.AndroidDerived2022(derived_2022_path)
    assert isinstance(derived_2022, NavData)


def test_gt_2022(gt_2022_path):
    """Testing that Android ground truth 2022 is created without errors.

    Parameters
    ----------
    gt_2022_path : string
        Location for the unit_test Android ground truth 2022 measurements
    """
    gt_2022 = android.AndroidGroundTruth2022(gt_2022_path)
    assert isinstance(gt_2022, NavData)


def test_gt_alt_nan(root_path_2022):
    """Test Android Ground Truth Loader sets blank altitudes to zero.

    Parameters
    ----------
    root_path_2022 : string
        Location for the files with missing altitude, including the file
        with missing altitude
    """
    gt_2022_nan = os.path.join(root_path_2022, 'alt_nan_ground_truth.csv')
    with pytest.warns(RuntimeWarning):
        gt_2022 = android.AndroidGroundTruth2022(gt_2022_nan)
        np.testing.assert_almost_equal(gt_2022['alt_gt_m'],
                                       np.zeros(len(gt_2022)))

def test_remove_all_data(derived_path_xl):
    """Test what happens when remove_timing_outliers removes all data.

    Parameters
    ----------
    derived_path : string
        Location for the unit_test Android 2021 derived measurements.

    """
    # Also tests if strings are being converted back correctly
    with pytest.warns(RuntimeWarning):
        derived = android.AndroidDerived2021(derived_path_xl,
                                   remove_timing_outliers=True)

    assert derived.shape[1] == 0
