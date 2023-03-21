"""Common fixtures required to run tests for GNSS SV state calculation
and measurement correction methods.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "18 Jan 2023"


import os
import pytest
import numpy as np

from gnss_lib_py.utils.time_conversions import gps_millis_to_datetime
from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.parsers.android import AndroidDerived2022, AndroidGroundTruth2022
from gnss_lib_py.parsers.ephemeris import EphemerisManager

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

    Returns
    -------
    android_root_path : string
        Folder location containing Android Derived 2022 measurements
    """
    android_root_path = os.path.join(root_path, 'android_2022')
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
    root_path : string
        Path of testing dataset root path

    Returns
    -------
    gt_path : string
        Location for the ground truth of the test Android derived measurements.

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
    gt_path = os.path.join(android_root_path, 'ground_truth.csv')
    return gt_path


@pytest.fixture(name="ephemeris_path")
def fixture_ephemeris_path(root_path):
    ephemeris_path = os.path.join(root_path, 'ephemeris')
    return ephemeris_path



@pytest.fixture(name="android_derived")
def fixture_derived(derived_path):
    derived = AndroidDerived2022(derived_path)
    return derived


@pytest.fixture(name="android_gps_l1")
def fixture_derived_gps_l1(android_derived):
    android_gps = android_derived.where('gnss_id', "gps", "eq")
    android_gps_l1 = android_gps.where('signal_type', "l1", "eq")
    return android_gps_l1


@pytest.fixture(name="android_gt")
def fixture_gt(gt_path):
    android_gt = AndroidGroundTruth2022(gt_path)
    return android_gt


@pytest.fixture(name="ephem_path")
def fixture_ephem_path(root_path):
    ephem_path = os.path.join(root_path, 'ephemeris')
    return ephem_path

@pytest.fixture(name="all_gps_sats")
def fixture_ephem_all_sats():
    all_gps_sats = [f"G{sv:02}" for sv in range(1,33)]
    return all_gps_sats

@pytest.fixture(name="start_time")
def fixture_start_time(android_gt):
    start_time = gps_millis_to_datetime(android_gt['gps_millis', 0])
    return start_time



@pytest.fixture(name="all_gps_ephem")
def fixture_all_gps_ephem(ephem_path, start_time, all_gps_sats):
    ephem_man_nav = EphemerisManager(ephem_path)
    ephem = ephem_man_nav.get_ephemeris(start_time, all_gps_sats)
    return ephem


@pytest.fixture(name="gps_measurement_frames")
def fixture_gps_measurement_frames(all_gps_ephem, android_gps_l1):
    android_frames = android_gps_l1.loop_time('gps_millis', tol_decimals=-2)
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
        sv_id = [f"G{sv:02}" for sv in frame['sv_id'][sort_arg]]
        android_posvel['sv_id'] = np.asarray(sv_id, dtype=object)
        android_posvel['x_sv_m'] = frame['x_sv_m'][sort_arg]
        android_posvel['y_sv_m'] = frame['y_sv_m'][sort_arg]
        android_posvel['z_sv_m'] = frame['z_sv_m'][sort_arg]
        android_posvel['vx_sv_mps'] = frame['vx_sv_mps'][sort_arg]
        android_posvel['vy_sv_mps'] = frame['vy_sv_mps'][sort_arg]
        android_posvel['vz_sv_mps'] = frame['vz_sv_mps'][sort_arg]

        frames.append(frame)
        ephems.append(vis_ephem)
        sv_states.append(android_posvel)
    return {'vis_ephems': ephems, 'android_frames':frames, 'sv_states':sv_states}


@pytest.fixture(name="error_tol_dec")
def fixture_error_tolerances():
    error_tol = {}
    error_tol['vel'] = 1
    error_tol['pos'] = -1
    error_tol['iono'] = -1
    error_tol['tropo'] = -1
    error_tol['clock'] = 1
    error_tol['brd_eph'] = -2.5
    return error_tol
