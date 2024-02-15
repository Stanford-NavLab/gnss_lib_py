"""Common fixtures for all tests.

"""

__authors__ = "A. Kanhere, D. Knowles"
__date__ = "30 Apr 2022"


import os
import pytest
import numpy as np
import pandas as pd

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.parsers.google_decimeter import AndroidDerived2021, AndroidGroundTruth2021
from gnss_lib_py.parsers.google_decimeter import AndroidDerived2022, AndroidGroundTruth2022
from gnss_lib_py.parsers.rinex_nav import get_time_cropped_rinex
from gnss_lib_py.navdata.operations import loop_time, concat

def pytest_collection_modifyitems(items):
    """Run ephemeris download tests after all other tests.

    The download tests take the longest to run, so save them for the
    end of the testing regime.

    Parameters
    ----------
    items : list
        List of test items.

    Notes
    -----
    Taken from https://stackoverflow.com/a/70759482

    pytest_collection_modifyitems is documented here:
    https://docs.pytest.org/en/latest/_modules/_pytest/hookspec.html#pytest_collection_modifyitems

    """

    module_mapping = {item: item.module.__name__ for item in items}
    download_tests = [
                      "test_ephemeris_downloader",
                      "test_rinex_nav",
                      "test_rinex_obs"
                     ]
    sorted_items = [item for item in items if module_mapping[item] not in download_tests] \
                 + [item for item in items if module_mapping[item] in download_tests]

    items[:] = sorted_items

@pytest.fixture(name="root_path")
def fixture_root_path():
    """Location of unit test directory.

    Returns
    -------
    root_path : string
        Folder location of unit test measurements
    """
    root_path = os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__)))
    root_path = os.path.join(root_path, 'data/unit_test')
    return root_path

@pytest.fixture(name="derived_path")
def fixture_derived_path(root_path):
    """Filepath of Android Derived 2022 measurements

    Parameters
    ----------
    root_path : string
        Folder location of unit test measurements

    Returns
    -------
    derived_path : string
        Location for the unit_test Android derived measurements

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset [1]_,
    and was retrieved from
    https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data

    References
    ----------
    .. [1] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.
    """
    derived_path = os.path.join(root_path, 'google_decimeter_2022',
                                'device_gnss.csv')
    return derived_path


@pytest.fixture(name="gt_2022_path")
def fixture_gt_2022_path(root_path):
    """Filepath of Android Derived 2022 ground truth measurements

    Parameters
    ----------
    root_path : string
        Folder location of unit test measurements

    Returns
    -------
    gt_2022_path : string
        Location for the ground truth of the test Android derived measurements.

    Notes
    -----
    Test data is a subset of the Android Raw Measurement Dataset [2]_,
    and was retrieved from
    https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data

    References
    ----------
    .. [2] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.
    """
    gt_2022_path = os.path.join(root_path, 'google_decimeter_2022',
                            'ground_truth.csv')
    return gt_2022_path


@pytest.fixture(name="ephemeris_path")
def fixture_ephemeris_path(root_path):
    """Path where ephemeris files are downloaded or loaded from.

    Parameters
    ----------
    root_path : string
        Path where all unit testing data is stored.

    Returns
    -------
    ephemeris_path : string
        Path where ephemeris files are to be stored.
    """
    ephemeris_path = os.path.join(root_path)
    return ephemeris_path

@pytest.fixture(name="android_derived")
def fixture_derived(derived_path):
    """Instance of Android Derived measurements, loaded into AndroidDerived2022.

    Parameters
    ----------
    derived_path : string
        Location where file containing measurements is stored.

    Returns
    -------
    derived : gnss_lib_py.parsers.google_decimeter.AndroidDerived2022
        Android Derived measurements for testing
    """
    derived = AndroidDerived2022(derived_path)
    rx_rows_no_pos = ['vx_rx_mps', 'vy_rx_mps', 'vz_rx_mps', 'b_rx_m', 'b_dot_rx_mps']
    for row in rx_rows_no_pos:
        derived[row] = 0
    return derived


@pytest.fixture(name="android_gps_l1")
def fixture_derived_gps_l1(android_derived):
    """Android measurements corresponding to the GPS L1 band.

    Parameters
    ----------
    android_derived : gnss_lib_py.parsers.google_decimeter.AndroidDerived2022
        Android Derived measurements for testing

    Returns
    -------
    android_gps_l1 : gnss_lib_py.parsers.google_decimeter.AndroidDerived2022
        Android Derived measurements containing only entries that
        correspond to the GPS contellation and L1 frequency band.
    """
    android_gps = android_derived.where('gnss_id', "gps", "eq")
    android_gps_l1 = android_gps.where('signal_type', "l1", "eq")
    return android_gps_l1


@pytest.fixture(name="android_gps_l1_reversed")
def fixture_derived_gps_l1_reversed(android_gps_l1):
    """Android GPS L1 measurements with reversed SV IDs in the same time.

    Parameters
    ----------
    android_gps_l1 : gnss_lib_py.parsers.google_decimeter.AndroidDerived2022
        Android Derived measurements containing only entries that
        correspond to the GPS contellation and L1 frequency band.

    Returns
    -------
    android_gps_l1_reversed : gnss_lib_py.parsers.google_decimeter.AndroidDerived2022
        Android GPS L1 measurements with reversed SV IDs in a single time
        instance.
    """
    android_gps_l1_reversed = NavData()
    for _, _, measure_frame in loop_time(android_gps_l1,'gps_millis', delta_t_decimals=-2):
        if len(android_gps_l1_reversed)==0:
            android_gps_l1_reversed = measure_frame
        else:
            android_gps_l1_reversed = concat(android_gps_l1_reversed,measure_frame)
    return android_gps_l1_reversed


@pytest.fixture(name="android_state")
@pytest.mark.filterwarnings("ignore:.*not found*: RuntimeWarning")
def fixture_android_state(android_derived):
    """State estimate corresponding to Android measurements for GPS L1.

    Parameters
    ----------
    android_derived : gnss_lib_py.parsers.google_decimeter.AndroidDerived2022
        Android Derived measurements for testing

    Returns
    -------
    android_state_estimate : gnss_lib_py.navdata.navdata.NavData
        Instance of `NavData` containing `gps_millis` and Rx position
        estimates from Android Derived.
    """
    android_state_estimate = android_derived.get_state_estimate()
    return android_state_estimate


@pytest.fixture(name="android_gt")
def fixture_gt(gt_2022_path):
    """Path to ground truth file for Android Derived measurements.

    Parameters
    ----------
    gt_2022_path : string
        Path where ground truth file is stored.

    Returns
    -------
    android_gt : gnss_lib_py.parsers.google_decimeter.AndroidGroundTruth2022
        NavData containing ground truth for Android measurements.
    """
    android_gt = AndroidGroundTruth2022(gt_2022_path)
    return android_gt


@pytest.fixture(name="all_gps_sats")
def fixture_ephem_all_sats():
    """Returns list of `gnss_sv_id` for all GPS satellites.

    Returns
    -------
    all_gps_sats : list
        List containing 3 character, eg. G02, combination of GNSS and SV
        IDs for all GPS satellites.
    """
    all_gps_sats = [f"G{sv:02}" for sv in range(1,33)]
    return all_gps_sats

@pytest.fixture(name="start_time")
def fixture_start_time(android_gt):
    """Starting time of when measurements were received as a datetime.datetime.

    Parameters
    ----------
    android_gt : gnss_lib_py.parsers.google_decimeter.AndroidGroundTruth2022
        NavData containing ground truth for Android measurements.

    Returns
    -------
    start_time : float
        Time at which measurements were first received in this trace, as
        a datetime.datetime object
    """
    start_time = android_gt['gps_millis', 0]
    return start_time



@pytest.fixture(name="all_gps_ephem")
def fixture_all_gps_ephem(ephemeris_path, start_time, all_gps_sats):
    """Extracts ephemeris parameters for all GPS satellites at start time.

    Parameters
    ----------
    ephemeris_path : string
        Path where ephemeris files are to be stored.
    start_time : float
        Time at which measurements were first received in this trace, as
        a datetime.datetime object
    all_gps_sats : list
        List containing 3 character, eg. G02, combination of GNSS and SV
        IDs for all GPS satellites.

    Returns
    -------
    ephem : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing ephemeris parameters for all GPS
        satellites at the start time for measurement reception.
    """

    ephem = get_time_cropped_rinex(start_time, all_gps_sats,
                                   ephemeris_path)
    return ephem


@pytest.fixture(name="gps_measurement_frames")
def fixture_gps_measurement_frames(all_gps_ephem, android_gps_l1):
    """Parses and separates received measurements, SV states and visible
    ephemeris by time instance.

    Parameters
    ----------
    all_gps_ephem : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing ephemeris parameters for all GPS
        satellites at the start time for measurement reception.
    android_gps_l1 : gnss_lib_py.parsers.google_decimeter.AndroidDerived2022
        Android Derived measurements containing only entries that
        correspond to the GPS contellation and L1 frequency band.

    Returns
    -------
    gps_measurement_frames : Dict
        Dictionary containing lists of visible ephemeris parameters,
        received Android measurements and SV states. The lists are
        indexed by discrete time indices.
    """
    android_frames = loop_time(android_gps_l1,'gps_millis', delta_t_decimals=-2)
    ephems = []
    frames = []
    sv_states = []
    for _, _, frame in android_frames:
        vis_svs = [sv for sv in frame['sv_id']]
        cols = []
        for sv in vis_svs:
            cols.append(all_gps_ephem.argwhere('sv_id', sv))
        vis_ephem = all_gps_ephem.copy(cols = cols)
        sort_arg = np.argsort(frame['sv_id'])

        android_posvel = NavData()
        gnss_id = ["gps" for _ in frame['sv_id'][sort_arg]]
        sv_id = [sv for sv in frame['sv_id'][sort_arg]]
        android_posvel['gnss_id'] = np.asarray(gnss_id, dtype=object)
        android_posvel['sv_id'] = np.asarray(sv_id)
        android_posvel['x_sv_m'] = frame['x_sv_m'][sort_arg]
        android_posvel['y_sv_m'] = frame['y_sv_m'][sort_arg]
        android_posvel['z_sv_m'] = frame['z_sv_m'][sort_arg]
        android_posvel['vx_sv_mps'] = frame['vx_sv_mps'][sort_arg]
        android_posvel['vy_sv_mps'] = frame['vy_sv_mps'][sort_arg]
        android_posvel['vz_sv_mps'] = frame['vz_sv_mps'][sort_arg]
        android_posvel['b_sv_m'] = frame['b_sv_m'][sort_arg]

        frames.append(frame)
        ephems.append(vis_ephem)
        sv_states.append(android_posvel)
    gps_measurement_frames =  {'vis_ephems': ephems, 'android_frames':frames, 'sv_states':sv_states}
    return gps_measurement_frames

@pytest.fixture(name="error_tol_dec")
def fixture_error_tolerances():
    """
    Decimal error tolerances for different measurment and SV state values.

    Returns
    -------
    error_tol : Dict
        Dictionary containing decimal places upto which equivalence is
        checked for position, velocity, atmospheric delays, clock estimation,
        and broadcast ephemeris positions computed by models.
    """
    error_tol = {}
    error_tol['vel'] = 1
    error_tol['pos'] = -1
    error_tol['delay'] = -1
    error_tol['clock'] = 1
    error_tol['brd_eph'] = -2.5
    return error_tol

@pytest.fixture(name="sp3_path")
def fixture_sp3_path(root_path):
    """Filepath of valid .sp3 measurements

    Parameters
    ----------
    root_path : string
        Folder location containing measurements

    Returns
    -------
    sp3_path : string
        String with location for the unit_test sp3 measurements

    Notes
    -----
    Downloaded the relevant .sp3 files from either CORS website [3]_ or
    CDDIS website [4]_

    References
    ----------
    .. [3]  https://geodesy.noaa.gov/UFCORS/ Accessed as of August 2, 2022
    .. [4]  https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/gnss_mgex.html
            Accessed as of August 2, 2022
    """
    sp3_path = os.path.join(root_path, 'sp3/grg21553.sp3')
    return sp3_path

@pytest.fixture(name="clk_path")
def fixture_clk_path(root_path):
    """Filepath of valid .clk measurements

    Parameters
    ----------
    root_path : string
        Folder location containing measurements

    Returns
    -------
    clk_path : string
        String with location for the unit_test clk measurements

    Notes
    -----
    Downloaded the relevant .clk files from either CORS website [5]_ or
    CDDIS website [6]_

    References
    ----------
    .. [5]  https://geodesy.noaa.gov/UFCORS/ Accessed as of August 2, 2022
    .. [6]  https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/gnss_mgex.html
            Accessed as of August 2, 2022

    """
    clk_path = os.path.join(root_path, 'clk/grg21553.clk')
    return clk_path

def fixture_csv_path(csv_filepath):
    """Location of measurements for unit test

    Returns
    -------
    root_path : string
        Folder location containing measurements
    """
    root_path = os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__)))
    root_path = os.path.join(root_path, 'data/unit_test/navdata')

    csv_path = os.path.join(root_path, csv_filepath)

    return csv_path

@pytest.fixture(name="csv_simple")
def fixture_csv_simple():
    """csv with simple format.

    """
    return fixture_csv_path("navdata_test_simple.csv")

@pytest.fixture(name="csv_headless")
def fixture_csv_headless():
    """csv without column names.

    """
    return fixture_csv_path("navdata_test_headless.csv")

@pytest.fixture(name="csv_missing")
def fixture_csv_missing():
    """csv with missing entries.

    """
    return fixture_csv_path("navdata_test_missing.csv")

@pytest.fixture(name="csv_mixed")
def fixture_csv_mixed():
    """csv with mixed data types.

    """
    return fixture_csv_path("navdata_test_mixed.csv")

@pytest.fixture(name="csv_inf")
def fixture_csv_inf():
    """csv with infinity values in numeric columns.

    """
    return fixture_csv_path("navdata_test_inf.csv")

@pytest.fixture(name="csv_nan")
def fixture_csv_nan():
    """csv with NaN values in columns.

    """
    return fixture_csv_path("navdata_test_nan.csv")

@pytest.fixture(name="csv_int_first")
def fixture_csv_int_first():
    """csv where first column are integers.

    """
    return fixture_csv_path("navdata_test_int_first.csv")

@pytest.fixture(name="csv_only_header")
def fixture_csv_only_header():
    """csv where there's no data, only columns.

    """
    return fixture_csv_path("navdata_only_header.csv")

@pytest.fixture(name="csv_dtypes")
def fixture_csv_dtypes():
    """csv made up of different data types.

    """
    return fixture_csv_path("navdata_test_dtypes.csv")

def load_test_dataframe(csv_filepath, header="infer"):
    """Create dataframe test fixture.

    """

    data = pd.read_csv(csv_filepath, header=header)

    return data

@pytest.fixture(name='df_simple')
def fixture_df_simple(csv_simple):
    """df with simple format.

    """
    return load_test_dataframe(csv_simple)

@pytest.fixture(name='df_headless')
def fixture_df_headless(csv_headless):
    """df without column names.

    """
    return load_test_dataframe(csv_headless,None)

@pytest.fixture(name='df_missing')
def fixture_df_missing(csv_missing):
    """df with missing entries.

    """
    return load_test_dataframe(csv_missing)

@pytest.fixture(name='df_mixed')
def fixture_df_mixed(csv_mixed):
    """df with mixed data types.

    """
    return load_test_dataframe(csv_mixed)

@pytest.fixture(name='df_inf')
def fixture_df_inf(csv_inf):
    """df with infinity values in numeric columns.

    """
    return load_test_dataframe(csv_inf)

@pytest.fixture(name='df_nan')
def fixture_df_nan(csv_nan):
    """df with NaN values in columns.

    """
    return load_test_dataframe(csv_nan)

@pytest.fixture(name='df_int_first')
def fixture_df_int_first(csv_int_first):
    """df where first column are integers.

    """
    return load_test_dataframe(csv_int_first)

@pytest.fixture(name='df_only_header')
def fixture_df_only_header(csv_only_header):
    """df where only headers given and no data.

    """
    return load_test_dataframe(csv_only_header)

@pytest.fixture(name="data")
def load_test_navdata(df_simple):
    """Creates a NavData instance from df_simple.

    """
    return NavData(pandas_df=df_simple)

@pytest.fixture(name="numpy_array")
def create_numpy_array():
    """Create np.ndarray test fixture.
    """
    test_array = np.array([[1,2,3,4,5,6],
                            [0.5,0.6,0.7,0.8,-0.001,-0.3],
                            [-3.0,-1.2,-100.,-2.7,-30.,-5],
                            [-543,-234,-986,-123,843,1000],
                            ])
    return test_array

@pytest.fixture(name='add_array')
def fixture_add_array():
    """Array added as additional timesteps to NavData from np.ndarray

    Returns
    -------
    add_array : np.ndarray
        Array that will be added to NavData
    """
    add_array = np.hstack((10*np.ones([4,1]), 11*np.ones([4,1])))
    return add_array

@pytest.fixture(name='add_df')
def fixture_add_dataframe():
    """Pandas DataFrame to be added as additional timesteps to NavData

    Returns
    -------
    add_df : pd.DataFrame
        Dataframe that will be added to NavData
    """
    add_data = {'names': np.asarray(['beta', 'alpha'], dtype=object),
                'integers': np.asarray([-2, 45], dtype=np.int64),
                'floats': np.asarray([1.4, 1.5869]),
                'strings': np.asarray(['glonass', 'beidou'], dtype=object)}
    add_df = pd.DataFrame(data=add_data)
    return add_df

@pytest.fixture(name="derived_2021")
def fixture_derived_2021_path(root_path):
    """Filepath of Android Derived measurements

    Returns
    -------
    derived_2021 : AndroidDerived2021
        Instance of AndroidDerived2021 for testing

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
    derived_path = os.path.join(root_path, 'google_decimeter_2021',
                                'Pixel4_derived.csv')

    derived_2021 = AndroidDerived2021(derived_path)
    return derived_2021

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
    derived_path = os.path.join(root_path, 'google_decimeter_2021',
                                'Pixel4XL_derived.csv')
    return derived_path

@pytest.fixture(name="derived_xl")
def fixture_load_derived_xl(derived_path_xl):
    """Load instance of AndroidDerived2021

    Parameters
    ----------
    derived_path : pytest.fixture
        String with location of Android derived measurement file

    Returns
    -------
    derived_xl : AndroidDerived2021
        Instance of AndroidDerived2021 for testing
    """
    derived_xl = AndroidDerived2021(derived_path_xl,
                                 remove_timing_outliers=False)
    return derived_xl

@pytest.fixture(name="derived_2022_path")
def fixture_derived_2022_path(root_path):
    """Filepath of Android Derived measurements

    Returns
    -------
    root_path : string
        Folder location containing measurements

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
    derived_path = os.path.join(root_path, 'google_decimeter_2022',
                                'device_gnss.csv')
    return derived_path



@pytest.fixture(name="derived_2022")
def fixture_load_derived_2022(derived_2022_path):
    """Load instance of AndroidDerived2022

    Parameters
    ----------
    derived_path : pytest.fixture
    String with location of Android derived measurement file

    Returns
    -------
    derived : AndroidDerived2022
    Instance of AndroidDerived2022 for testing
    """
    derived = AndroidDerived2022(derived_2022_path)
    return derived


@pytest.fixture(name="gtruth")
def fixture_load_gtruth(root_path):
    """Load instance of AndroidGroundTruth2021

    Parameters
    ----------
    root_path : string
        Path of testing dataset root path

    Returns
    -------
    gtruth : AndroidGroundTruth2021
        Instance of AndroidGroundTruth2021 for testing
    """
    gtruth = AndroidGroundTruth2021(os.path.join(root_path,
                                 'google_decimeter_2021',
                                 'Pixel4_ground_truth.csv'))
    return gtruth

@pytest.fixture(name="state_estimate")
def fixture_solve_wls(derived_2021):
    """Fixture of WLS state estimate.

    Parameters
    ----------
    derived_2021 : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.

    Returns
    -------
    state_estimate : gnss_lib_py.navdata.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """
    state_estimate = solve_wls(derived_2021)
    return state_estimate
