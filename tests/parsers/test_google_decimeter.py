"""Tests for Android data loaders.

"""

__authors__ = "Ashwin Kanhere, Derek Knowles"
__date__ = "10 Nov 2021"

import os

import pytest
import numpy as np
import pandas as pd

from gnss_lib_py.parsers import google_decimeter
from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.algorithms.gnss_filters import solve_gnss_ekf
from gnss_lib_py.navdata.operations import loop_time

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
    root_path = os.path.join(root_path, 'data/unit_test/google_decimeter_2021')
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
    Test data is a subset of the Android Raw Measurement Dataset [2]_,
    particularly the train/2020-05-14-US-MTV-1/Pixel4XL trace. The
    dataset was retrieved from
    https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data

    References
    ----------
    .. [2] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
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
    Test data is a subset of the Android Raw Measurement Dataset [3]_,
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
    inverse_row_map = {v : k for k,v in google_decimeter.AndroidDerived2021._row_map().items()}
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
    derived = google_decimeter.AndroidDerived2021(derived_path)
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
    derived = google_decimeter.AndroidDerived2021(derived_path,
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

def test_navdata_type(derived):
    """Test that all subclasses inherit from NavData

    Parameters
    ----------
    derived : pytest.fixture
        Instance of AndroidDerived2021 for testing
    """
    isinstance(derived, NavData)
    isinstance(derived, google_decimeter.AndroidDerived2021)

def test_timestep_parsing(derived_path_xl):
    """Test that the timesteps contain the same satellites.

    Parameters
    ----------
    derived_path_xl : string
        Location for the unit_test Android derived measurements

    """

    pd_df_xl = pd.read_csv(derived_path_xl)
    derived_xl = google_decimeter.AndroidDerived2021(derived_path_xl,
                               remove_timing_outliers=False)

    pd_svid_groups = []
    for _, group in pd_df_xl.groupby("millisSinceGpsEpoch"):
        pd_svid_groups.append(group["svid"].tolist())
    pd_svid_groups.pop(0)

    navdata_svid_groups = []
    for _, _, group in loop_time(derived_xl,"gps_millis"):
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

@pytest.fixture(name="android_gtruth_path")
def fixture_gtruth_path(root_path):
    """Filepath of Android Ground Truth data

    Returns
    -------
    gtruth_path : string
        Location for text log file with Android Ground Truth measurements

    Notes
    -----
    Test data is a subset of the Android Ground Truth Dataset [4]_,
    particularly the train/2020-05-14-US-MTV-1/Pixel4 trace. The dataset
    was retrieved from
    https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data

    References
    ----------
    .. [4] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
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
    gtruth = google_decimeter.AndroidGroundTruth2021(android_gtruth_path)

    return gtruth

def test_android_gtruth(gtruth):
    """Test AndroidGroundTruth initialization

    Parameters
    ----------
    gtruth : AndroidGroundTruth2021
        Instance of AndroidGroundTruth2021 for testing

    """

    isinstance(gtruth, NavData)
    isinstance(gtruth, google_decimeter.AndroidGroundTruth2021)

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
        Folder location containing 2022 measurements
    """
    root_path = os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
    root_path = os.path.join(root_path, 'data/unit_test/google_decimeter_2022')
    return root_path


@pytest.fixture(name="derived_2022_path")
def fixture_derived_2022_path(root_path_2022):
    """Filepath of Android Derived measurements

    Parameters
    ----------
    root_path_2022 : string
        Folder location containing 2022 measurements

    Returns
    -------
    derived_path : string
        Location for the unit_test Android derived 2022 measurements

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
    derived_path = os.path.join(root_path_2022, 'device_gnss.csv')
    return derived_path


@pytest.fixture(name="gt_2022_path")
def fixture_gt_2022_path(root_path_2022):
    """Filepath of Android ground truth estimates

    Parameters
    ----------
    root_path_2022 : string
        Folder location containing 2022 measurements

    Returns
    -------
    derived_path : string
        Location for the unit_test Android ground truth estimates

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset [6]_,
    from the 2022 Decimeter Challenge. Particularly, the
    train/2021-04-29-MTV-2/SamsungGalaxyS20Ultra trace. The dataset
    was retrieved from
    https://www.kaggle.com/competitions/smartphone-decimeter-2022/data

    References
    ----------
    .. [6] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.
    """
    gt_path = os.path.join(root_path_2022, 'ground_truth.csv')
    return gt_path

@pytest.fixture(name="derived_2022")
def fixture_derived_2022(derived_2022_path):
    """Testing that Android Derived 2022 is created without errors.

    Parameters
    ----------
    derived_2022_path : string
        Location for the unit_test Android derived 2022 measurements

    Returns
    -------
    derived_2022 : gnss_lib_py.parsers.google_decimeter.AndroidDerived2022
        Android Derived 2022 Navdata object for testing.

    """
    derived_2022 = google_decimeter.AndroidDerived2022(derived_2022_path)
    assert isinstance(derived_2022, NavData)
    return derived_2022


def test_derived_state_estimate_ext(derived_2022):
    """Tests the state_estimate extracted as a separate NavData.

    Parameters
    ----------
    derived_2022 : gnss_lib_py.parsers.google_decimeter.AndroidDerived2022
        Android Derived 2022 Navdata object for testing.

    """
    # Test state estimate extraction when velocity and bias terms are
    # unavailable. In this case, an warnings are expected for the velocity
    # and clock rows
    with pytest.warns(RuntimeWarning):
        state_estimate = derived_2022.get_state_estimate()
    assert len(state_estimate)==6, "There should be only 6 unique times"
    time_pos_rows = ['gps_millis', 'x_rx_m', 'y_rx_m', 'z_rx_m']
    state_estimate.in_rows(time_pos_rows)

    # Test extraction of state when velocity and clock rows are available
    vel_clk_rows = ['vx_rx_mps', 'vy_rx_mps', 'vz_rx_mps', 'b_rx_m', 'b_dot_rx_mps']
    for row in vel_clk_rows:
        derived_2022[row] = 0
    state_estimate_vel_clk = derived_2022.get_state_estimate()
    state_estimate_vel_clk.in_rows(time_pos_rows+vel_clk_rows)

    # Test that the first and last extracted values are correct
    for row in time_pos_rows:
        np.testing.assert_almost_equal(derived_2022[row, 0], state_estimate_vel_clk[row, 0])
        np.testing.assert_almost_equal(derived_2022[row, -1], state_estimate_vel_clk[row, -1])


def test_gt_2022(gt_2022_path):
    """Testing that Android ground truth 2022 is created without errors.

    Parameters
    ----------
    gt_2022_path : string
        Location for the unit_test Android ground truth 2022 measurements
    """
    gt_2022 = google_decimeter.AndroidGroundTruth2022(gt_2022_path)
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
        gt_2022 = google_decimeter.AndroidGroundTruth2022(gt_2022_nan)
        np.testing.assert_almost_equal(gt_2022['alt_rx_gt_m'],
                                       np.zeros(len(gt_2022)))

def test_remove_all_data(derived_path_xl):
    """Test what happens when remove_timing_outliers removes all data.

    Parameters
    ----------
    derived_path_xl : string
        Location for the unit_test Android 2021 derived measurements.

    """
    # Also tests if strings are being converted back correctly
    with pytest.warns(RuntimeWarning):
        derived = google_decimeter.AndroidDerived2021(derived_path_xl,
                                   remove_timing_outliers=True)

    assert derived.shape[1] == 0

@pytest.fixture(name="state_estimate")
def test_solve_kaggle_baseline(derived_2022):
    """Testing Kaggle baseline solution.

    Parameters
    ----------
    derived_2022 : gnss_lib_py.parsers.google_decimeter.AndroidDerived2022
        Android derived 2022 measurements

    Returns
    -------
    state_estimate : gnss_lib_py.navdata.navdata.NavData
        Baseline state estimate.
    """

    state_estimate = google_decimeter.solve_kaggle_baseline(derived_2022)

    state_estimate.in_rows(["gps_millis","lat_rx_deg",
                            "lon_rx_deg","alt_rx_m"])

    assert state_estimate.shape[1] == 6

    expected = np.array([1303770943999,1303770944999,1303770945999,
                         1303770946999,1303770947999,1303770948999])
    np.testing.assert_array_equal(state_estimate["gps_millis"],expected)

    return state_estimate

def test_prepare_kaggle_submission(state_estimate):
    """Prepare Kaggle baseline solution.

    Parameters
    ----------
    state_estimate : gnss_lib_py.navdata.navdata.NavData
        Baseline state estimate.

    """

    output = google_decimeter.prepare_kaggle_submission(state_estimate,"test")

    output.in_rows(["tripId","UnixTimeMillis",
                    "LatitudeDegrees","LongitudeDegrees"])

    assert output.shape[1] == 6

    expected = np.array([1619735725999,1619735726999,1619735727999,
                         1619735728999,1619735729999,1619735730999])
    np.testing.assert_array_equal(output["UnixTimeMillis"], expected)

def test_solve_kaggle_dataset(root_path):
    """Test kaggle solver.

    Parameters
    ----------
    root_path : string
        Path of testing dataset root path

    """

    folder_path = os.path.join(root_path,"..","..")
    for solver in [google_decimeter.solve_kaggle_baseline,
                   solve_wls,
                   solve_gnss_ekf,
                  ]:
        for verbose in [True, False]:
            solution = google_decimeter.solve_kaggle_dataset(folder_path, solver,
                                                    verbose)

            solution.in_rows(["tripId","UnixTimeMillis",
                            "LatitudeDegrees","LongitudeDegrees"])

            assert solution.shape[1] == 6

            expected = np.array([1619735725999,1619735726999,
                                 1619735727999,1619735728999,
                                 1619735729999,1619735730999])
            np.testing.assert_array_equal(solution["UnixTimeMillis"],
                                          expected)


######################################################################
#### Android 2023 Dataset tests
######################################################################

@pytest.fixture(name="root_path_2023")
def fixture_root_path_2023():
    """Location of measurements for unit test

    Test data is a subset of the Android Raw Measurement Dataset [7]_,
    from the 2023 Decimeter Challenge.

    References
    ----------
    .. [7] https://www.kaggle.com/competitions/smartphone-decimeter-2023

    Returns
    -------
    root_path : string
        Folder location containing 2023 measurements
    """
    root_path = os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
    root_path = os.path.join(root_path, 'data/unit_test/google_decimeter_2023')
    return root_path

@pytest.fixture(name="derived_2023")
def fixture_derived_2023(root_path_2023):
    """Testing that Android Derived 2023 is created without errors.

    Parameters
    ----------
    root_path_2023 : string
        Folder location containing 2023 measurements

    Returns
    -------
    derived_2023 : gnss_lib_py.parsers.google_decimeter.AndroidDerived2023
        Android Derived 2023 Navdata object for testing.

    """

    derived_2023_path= os.path.join(root_path_2023,
                                    '2023-09-07-18-59-us-ca',
                                    'pixel7pro',
                                    'device_gnss.csv')
    derived_2023 = google_decimeter.AndroidDerived2023(derived_2023_path)
    assert isinstance(derived_2023, NavData)
    return derived_2023

@pytest.fixture(name="ground_truth_2023")
def fixture_ground_truth_2023(root_path_2023):
    """Testing that Ground Truth 2023 is created without errors.

    Parameters
    ----------
    root_path_2023 : string
        Folder location containing 2023 measurements

    Returns
    -------
    ground_truth_2023 : gnss_lib_py.parsers.google_decimeter.AndroidGroundTruth2023
        Ground Truth 2023 Navdata object for testing.

    """

    ground_truth_2023_path= os.path.join(root_path_2023,
                                    '2023-09-07-18-59-us-ca',
                                    'pixel7pro',
                                    'ground_truth.csv')
    ground_truth_2023 = google_decimeter.AndroidGroundTruth2023(ground_truth_2023_path)
    assert isinstance(ground_truth_2023, NavData)
    return ground_truth_2023

def test_derived_2023(derived_2023):
    """Check how the 2023 derived data was parsed.

    Parameters
    ----------
    derived_2023 : gnss_lib_py.parsers.google_decimeter.AndroidDerived2023
        Android Derived 2023 Navdata object for testing.

    """

    assert derived_2023.shape == (58+2,180)

    assert "gps_millis" in derived_2023.rows
    assert "corr_pr_m" in derived_2023.rows

    assert derived_2023["accumulated_delta_range_sigma_m"].dtype == np.float64
    assert derived_2023["accumulated_delta_range_sigma_m",14] == 3.40282346638529E+038

def test_ground_truth_2023(ground_truth_2023):
    """Check how the 2023 ground truth data was parsed.

    Parameters
    ----------
    ground_truth_2023 : gnss_lib_py.parsers.google_decimeter.AndroidGroundTruth2023
        Ground Truth 2023 Navdata object for testing.

    """

    assert ground_truth_2023.shape == (17,5)

    np.testing.assert_array_equal(ground_truth_2023[["SpeedAccuracyMps",
                                                     "BearingAccuracyDegrees",
                                                     "elapsedRealtimeNanos",
                                                     "VerticalAccuracyMeters"]],
                                                     np.ones((4,5))*np.nan,
                                                     )
