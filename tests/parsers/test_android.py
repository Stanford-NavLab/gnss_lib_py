"""Tests for Android raw data loaders.

"""

__authors__ = "Derek Knowles"
__date__ = "30 Oct 2023"

import os
import csv
import pathlib

import pytest
import numpy as np
import pandas as pd

from conftest import lazy_fixture
from gnss_lib_py.parsers import android
from gnss_lib_py.parsers.google_decimeter import AndroidDerived2023
from gnss_lib_py.navdata.navdata import NavData

# pylint: disable=protected-access

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
    raw_path = os.path.join(root_path, 'google_decimeter_2021',
                            'Pixel4_GnssLog.txt')
    return raw_path

@pytest.fixture(name="pixel6_raw_path")
def fixture_pixel6_raw_path(root_path):
    """Filepath of Android Raw measurements

    Parameters
    ----------
    root_path : string
        Path of testing dataset root path

    Returns
    -------
    raw_path : string
        Location for text log file with Android Raw measurements

    """
    raw_path = os.path.join(root_path, 'android','measurements',
                            'pixel6.txt')
    return raw_path

@pytest.fixture(name="sensors_raw_path")
def fixture_sensors_raw_path(root_path):
    """Filepath of Android Raw measurements

    Parameters
    ----------
    root_path : string
        Path of testing dataset root path

    Returns
    -------
    raw_path : string
        Location for text log file with Android Raw measurements

    """
    raw_path = os.path.join(root_path, 'android','measurements',
                            'all_sensors.txt')
    return raw_path

@pytest.fixture(name="android_raw_2023_path")
def fixture_raw_2023_path(root_path):
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
    Test data is a subset of the 2023 Google Challenge [3]_.

    References
    ----------
    .. [3] https://www.kaggle.com/competitions/smartphone-decimeter-2023/overview

    """
    raw_path = os.path.join(root_path, 'google_decimeter_2023',
                            '2023-09-07-18-59-us-ca',
                            'pixel7pro',
                            'gnss_log.txt')
    return raw_path

@pytest.fixture(name="android_derived_2023_path")
def fixture_derived_2023_path(root_path):
    """Filepath of Android derived measurements

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
    Test data is a subset of the 2023 Google Challenge [4]_.

    References
    ----------
    .. [4] https://www.kaggle.com/competitions/smartphone-decimeter-2023/overview

    """
    raw_path = os.path.join(root_path, 'google_decimeter_2023',
                            '2023-09-07-18-59-us-ca',
                            'pixel7pro',
                            'device_gnss.csv')
    return raw_path

@pytest.mark.parametrize('sensor_type',
                        [android.AndroidRawMag,
                         android.AndroidRawGyro,
                         android.AndroidRawAccel,
                         android.AndroidRawOrientation,
                        ])
@pytest.mark.parametrize('raw_path',
                        [lazy_fixture("android_raw_path"),
                         lazy_fixture("pixel6_raw_path"),
                         lazy_fixture("sensors_raw_path"),
                        ])
def test_sensor_loaders(raw_path, sensor_type):
    """Test that sensor loaders initialize correctly.

    Parameters
    ----------
    raw_path : pytest.fixture
        Path to Android Raw measurements text log file
    sensor_type : NavData
        Type of NavData object

    """

    test_navdata = sensor_type(input_path = raw_path)
    assert isinstance(test_navdata, NavData)

    test_navdata = sensor_type(pathlib.Path(raw_path))
    assert isinstance(test_navdata, NavData)

    # raises exception if not a file path
    with pytest.raises(FileNotFoundError):
        sensor_type("not_a_file.txt")
    with pytest.raises(FileNotFoundError):
        sensor_type(pathlib.Path("not_a_file.txt"))

    # raises exception if input not string or path-like
    with pytest.raises(TypeError):
        sensor_type([])

@pytest.mark.parametrize('sensor_type',
                        [android.AndroidRawMag,
                         android.AndroidRawGyro,
                         android.AndroidRawAccel,
                         android.AndroidRawOrientation,
                        ])
@pytest.mark.parametrize('raw_path',
                        [
                         lazy_fixture("sensors_raw_path"),
                        ])
def test_sensor_content(raw_path, sensor_type):
    """Test that sensor loaders contain data.

    Parameters
    ----------
    raw_path : pytest.fixture
        Path to Android Raw measurements text log file
    sensor_type : NavData
        Type of NavData object

    """

    test_navdata = sensor_type(input_path = raw_path)

    if sensor_type == android.AndroidRawMag:
        uncalmag = test_navdata[["unix_millis","mag_x_uncal_microt",
                                 "mag_y_uncal_microt","mag_z_uncal_microt",
                                 "mag_bias_x_microt","mag_bias_y_microt",
                                 "mag_bias_z_microt"],0]
        uncalmag_expected = np.array([1699400576748,-54.1436,-88.937996,
                                     -147.3638,-79.950134,-76.57953,
                                     -113.967804])
        np.testing.assert_array_equal(uncalmag,uncalmag_expected)
        mag = test_navdata[["unix_millis","mag_x_microt",
                                 "mag_y_microt","mag_z_microt"],1]
        mag_expected = np.array([1699400576748,-54.1436,-88.937996,
                                -147.3638])
        np.testing.assert_array_equal(mag,mag_expected)
    if sensor_type == android.AndroidRawGyro:
        uncalgyro = test_navdata[["unix_millis","ang_vel_x_uncal_radps",
                                 "ang_vel_y_uncal_radps","ang_vel_z_uncal_radps",
                                 "DriftXRadPerSec","DriftYRadPerSec",
                                 "DriftZRadPerSec"],0]
        uncalgyro_expected = np.array([1699400576750,-0.06261369,
                                      -0.09315695,0.036651913,
                                      -0.0020643917,-0.0038384064,
                                      -0.0013324362])
        np.testing.assert_array_equal(uncalgyro,uncalgyro_expected)
        gyro = test_navdata[["unix_millis","ang_vel_x_radps",
                             "ang_vel_y_radps",
                             "ang_vel_z_radps"],1]
        gyro_expected = np.array([1699400576750,-0.06261369,-0.09315695,0.036651913])
        np.testing.assert_array_equal(gyro,gyro_expected)
    if sensor_type == android.AndroidRawAccel:
        uncalaccel = test_navdata[["unix_millis","acc_x_uncal_mps2",
                                 "acc_y_uncal_mps2","acc_z_uncal_mps2",
                                 "acc_bias_x_mps2","acc_bias_y_mps2",
                                 "acc_bias_z_mps2"],0]
        uncalaccel_expected = np.array([1699400576750,0.17288144,
                                        0.44925246,9.886545,0.065623306,
                                        0.002461203,-0.031848617])
        np.testing.assert_array_equal(uncalaccel,uncalaccel_expected)
        accel = test_navdata[["unix_millis","acc_x_mps2",
                                 "acc_y_mps2","acc_z_mps2"],1]
        accel_expected = np.array([1699400576750,
                                   0.17288144,0.44925246,9.886545])
        np.testing.assert_array_equal(accel,accel_expected)
    if sensor_type == android.AndroidRawOrientation:
        deg = test_navdata[["unix_millis","yaw_rx_deg",
                                 "roll_rx_deg","pitch_rx_deg"]]
        deg_expected = np.array([1699400576750,245.0,0.0,-2.0])
        np.testing.assert_array_equal(deg,deg_expected)

def test_fix_raw(android_raw_path):
    """Test that AndroidRawFixes initialization

    Parameters
    ----------
    android_raw_path : pytest.fixture
        Path to Android Raw measurements text log file
    """
    test_fix = android.AndroidRawFixes(android_raw_path)
    assert isinstance(test_fix, NavData)

    test_fix = android.AndroidRawFixes(pathlib.Path(android_raw_path))
    assert isinstance(test_fix, NavData)

    # raises exception if not a file path
    with pytest.raises(FileNotFoundError):
        android.AndroidRawFixes("not_a_file.txt")
    with pytest.raises(FileNotFoundError):
        android.AndroidRawFixes(pathlib.Path("not_a_file.txt"))

    # raises exception if input not string or path-like
    with pytest.raises(TypeError):
        android.AndroidRawFixes([])

def make_csv(input_path, output_directory, field, show_path=False):
    """Write specific data types from a GNSS android log to a CSV.

    Parameters
    ----------
    input_path : string or path-like
        File location of data file to read.
    output_directory : string
        Directory where new csv file should be created
    field : list of strings
        Type of data to extract. Valid options are either "Raw",
        "Accel", "Gyro", "Mag", or "Fix".
    show_path : bool
        If true, prints output path.

    Returns
    -------
    output_path : string
        New file location of the exported CSV.

    Notes
    -----
    Based off of MATLAB code from Google's gps-measurement-tools
    repository: https://github.com/google/gps-measurement-tools. Compare
    with MakeCsv() in opensource/ReadGnssLogger.m

    """
    if not os.path.isdir(output_directory): #pragma: no cover
        os.makedirs(output_directory)
    output_path = os.path.join(output_directory, field + ".csv")
    with open(output_path, 'w', encoding="utf8") as out_csv:
        writer = csv.writer(out_csv)

        if not isinstance(input_path, (str, os.PathLike)):
            raise TypeError("input_path must be string or path-like")
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path,"file not found")

        with open(input_path, 'r', encoding="utf8") as in_txt:
            for line in in_txt:
                # Comments in the log file
                if line[0] == '#':
                    # Remove initial '#', spaces, trailing newline
                    # and split using commas as delimiter
                    line_data = line[2:].rstrip('\n').replace(" ","").split(",")
                    if line_data[0] == field:
                        writer.writerow(line_data[1:])
                # Data in file
                else:
                    # Remove spaces, trailing newline and split using commas as delimiter
                    line_data = line.rstrip('\n').replace(" ","").split(",")
                    if line_data[0] == field:
                        writer.writerow(line_data[1:])
    if show_path: #pragma: no cover
        print(output_path)

    return output_path

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
    if file_type == 'Accel':
        test_measure = android.AndroidRawAccel(android_raw_path)
    elif file_type == 'Gyro':
        test_measure = android.AndroidRawGyro(android_raw_path)
    elif file_type=='Fix':
        test_measure = android.AndroidRawFixes(android_raw_path)
    output_directory = os.path.join(root_path, 'csv_test')
    csv_loc = make_csv(android_raw_path, output_directory,
                               file_type)
    test_df = pd.read_csv(csv_loc)
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

    # raises exception if not a file path
    with pytest.raises(FileNotFoundError):
        make_csv("", output_directory, file_type)

    # raises exception if input not string or path-like
    with pytest.raises(TypeError):
        make_csv([], output_directory, file_type)

    os.remove(csv_loc)
    for file in os.listdir(output_directory):
        os.remove(os.path.join(output_directory, file))
    os.rmdir(output_directory)

def test_raw_load(android_raw_2023_path, android_derived_2023_path):
    """Test basic loading of android raw file.

    Parameters
    ----------
    android_raw_2023_path : string
        Location for text log file with Android Raw measurements.
    android_derived_2023_path : string
        Location for text log file with Android derived measurements.

    """
    # load derived data
    derived = AndroidDerived2023(input_path=android_derived_2023_path)

    # load raw data
    raw = android.AndroidRawGnss(input_path=android_raw_2023_path,
                                 filter_measurements=False,
                                 remove_rx_b_from_pr=True,
                                 )

    # make sure the same data is contained in both
    assert len(derived) == len(raw)

    equal_rows = ["unix_millis",
                 "gps_millis",
                 "gnss_id",
                 "sv_id",
                 ]
    for row in equal_rows:
        np.testing.assert_array_equal(raw[row],derived[row])

    almost_equal_rows = [
                         ("cn0_dbhz",1E-13),
                         ("raw_pr_sigma_m",1E-13),
                         ("raw_pr_m",29.9),
                        ]
    for row, max_value in almost_equal_rows:
        not_nan_idxs = ~np.isnan(derived[row])
        assert np.max(np.abs(derived[row,not_nan_idxs] - raw[row,not_nan_idxs])) < max_value

def test_raw_filters(root_path):
    """Test all the different measurement filter varieties.

    Parameters
    ----------
    root_path : string
        Path of testing dataset root path

    """

    filter_test_path = os.path.join(root_path, 'android','measurements',
                            'filter_test.txt')

    # load raw data with none removed
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 filter_measurements=False,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    orig_len = len(raw)
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) == orig_len

    # bias_valid filter
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"bias_valid" : False},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) == orig_len
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"bias_valid" : True},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) +2  == orig_len

    # BiasUncertaintyNanos filter
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"bias_uncertainty" : np.inf},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) == orig_len
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"bias_uncertainty" : -np.inf},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) == 0
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"bias_uncertainty" : 40.},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) +1  == orig_len

    # arrival_time filter
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"arrival_time" : False},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) == orig_len
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"arrival_time" : True},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) + 4  == orig_len

    # time_valid filter
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"time_valid" : False},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) == orig_len
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"time_valid" : True},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) + 1  == orig_len

    # state_decoded filter
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"state_decoded" : False},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) == orig_len
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"state_decoded" : True},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) + 1  == orig_len

    # sv_time_uncertainty filter
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"sv_time_uncertainty" : np.inf},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) == orig_len
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"sv_time_uncertainty" : -np.inf},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) == 0
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"sv_time_uncertainty" : 500.},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) +1  == orig_len

    # adr_valid filter
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"adr_valid" : False},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) == orig_len
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"adr_valid" : True},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) + 1  == orig_len

    # sv_time_uncertainty filter
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"adr_uncertainty" : np.inf},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) == orig_len
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"adr_uncertainty" : -np.inf},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) == 0
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 measurement_filters = {"adr_uncertainty" : 15.},
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) +1  == orig_len

    # load data with all removed
    raw = android.AndroidRawGnss(input_path=filter_test_path,
                                 filter_measurements=True,
                                 remove_rx_b_from_pr=False,
                                 verbose=True)
    assert len(raw) + 11 == orig_len
