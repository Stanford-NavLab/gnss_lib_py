"""Common fixtures required to run tests for GNSS SV state calculation
and measurement correction methods.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "18 Jan 2023"


import os
import pytest
import numpy as np

from gnss_lib_py.navdata.navdata import NavData
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
    """Location of Android Derived 2022 measurements for unit test

    Returns
    -------
    root_path : string
        Folder location of unit test measurements
    """
    root_path = os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
    root_path = os.path.join(root_path, 'data/unit_test')
    return root_path


@pytest.fixture(name="android_root_path")
def fixture_android_root_path(root_path):
    """Location of Android Derived 2022 measurements for unit test

    Parameters
    -------
    root_path : string
        Folder location of unit test measurements

    Returns
    -------
    android_root_path : string
        Folder location containing Android Derived 2022 measurements
    """
    android_root_path = os.path.join(root_path, 'google_decimeter_2022')
    return android_root_path


@pytest.fixture(name="derived_path")
def fixture_derived_path(android_root_path):
    """Filepath of Android Derived 2022 measurements

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
    derived_path = os.path.join(android_root_path, 'device_gnss.csv')
    return derived_path


@pytest.fixture(name="gt_path")
def fixture_gt_path(android_root_path):
    """Filepath of Android Derived 2022 ground truth measurements

    Parameters
    ----------
    android_root_path : string
        Folder location containing Android Derived 2022 measurements

    Returns
    -------
    gt_path : string
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
    gt_path = os.path.join(android_root_path, 'ground_truth.csv')
    return gt_path


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
def fixture_gt(gt_path):
    """Path to ground truth file for Android Derived measurements.

    Parameters
    ----------
    gt_path : string
        Path where ground truth file is stored.

    Returns
    -------
    android_gt : gnss_lib_py.parsers.google_decimeter.AndroidGroundTruth2022
        NavData containing ground truth for Android measurements.
    """
    android_gt = AndroidGroundTruth2022(gt_path)
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
