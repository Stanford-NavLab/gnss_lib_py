"""Tests for visualizations.

"""

__authors__ = "D. Knowles"
__date__ = "22 Jun 2022"

import os

import pytest

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.parsers.google_decimeter import AndroidDerived2021
from gnss_lib_py.parsers.google_decimeter import AndroidDerived2022
from gnss_lib_py.parsers.google_decimeter import AndroidGroundTruth2021

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
    derived_path = os.path.join(root_path, 'google_decimeter_2021',
                                'Pixel4XL_derived.csv')
    return derived_path

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
    root_path = os.path.join(root_path, 'data/unit_test/google_decimeter_2022')
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
    derived = AndroidDerived2021(derived_path)
    return derived

@pytest.fixture(name="derived_xl")
def fixture_load_derived_xl(derived_path_xl):
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
    derived = AndroidDerived2021(derived_path_xl,
                                 remove_timing_outliers=False)
    return derived

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
def fixture_solve_wls(derived):
    """Fixture of WLS state estimate.

    Parameters
    ----------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.

    Returns
    -------
    state_estimate : gnss_lib_py.navdata.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """
    state_estimate = solve_wls(derived)
    return state_estimate
