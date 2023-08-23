"""Tests for visualizations.

"""

__authors__ = "D. Knowles"
__date__ = "22 Jun 2022"

import os
import random

import pytest
import numpy as np
import plotly.graph_objects as go
from pytest_lazyfixture import lazy_fixture
import matplotlib.pyplot as plt

import gnss_lib_py.utils.visualizations as viz
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.parsers.android import AndroidDerived2021
from gnss_lib_py.parsers.android import AndroidDerived2022
from gnss_lib_py.parsers.android import AndroidGroundTruth2021

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
    root_path = os.path.join(root_path, 'data/unit_test/android_2021/')
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
    state_estimate : gnss_lib_py.parsers.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """
    state_estimate = solve_wls(derived)
    return state_estimate

def test_plot_metrics(derived):
    """Test for plotting metrics.

    Parameters
    ----------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.

    """
    test_rows = [
                 "raw_pr_m",
                 ]

    for row in derived.rows:
        if not derived.is_str(row):
            if row in test_rows:
                fig = plt.figure()
                for groupby in ["gnss_id",None]:
                    fig = viz.plot_metric(derived, row,
                                          groupby = groupby,
                                          save=False)
                    viz.close_figures(fig)
        else:
            # string rows should cause a KeyError
            with pytest.raises(KeyError) as excinfo:
                fig = viz.plot_metric(derived, row, save=False)
                viz.close_figures(fig)
            assert "non-numeric row" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        viz.plot_metric(derived, "raw_pr_m", save=True, prefix=1)
    assert "Prefix" in str(excinfo.value)

    for row in derived.rows:
        if not derived.is_str(row):
            if row in test_rows:
                for groupby in ["gnss_id",None]:
                    fig = viz.plot_metric(derived, "raw_pr_m", row,
                                          groupby=groupby, save=False)
                    viz.close_figures(fig)
        else:
            # string rows should cause a KeyError
            with pytest.raises(KeyError) as excinfo:
                fig = viz.plot_metric(derived, "raw_pr_m", row, save=False)
                viz.close_figures(fig)
            with pytest.raises(KeyError) as excinfo:
                fig = viz.plot_metric(derived, row, "raw_pr_m", save=False)
                viz.close_figures(fig)
            assert "non-numeric row" in str(excinfo.value)

    viz.close_figures()

    # test repeating figure and average y
    fig = plt.figure()
    fig = viz.plot_metric(derived, "gps_millis", "raw_pr_m",
                          fig = fig,
                          groupby = "gnss_id",
                          save=False)
    fig = viz.plot_metric(derived, "gps_millis", "raw_pr_m",
                            fig = fig,
                            groupby = "gnss_id",
                            avg_y = True,
                            linestyle="dotted",
                            save=False,
                            )
    viz.close_figures(fig)

    with pytest.raises(TypeError) as excinfo:
        viz.plot_metric(derived, "raw_pr_m", save=True, prefix=1)
    assert "Prefix" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        viz.plot_metric(derived, 'raw_pr_m', row, row, save=False)

    with pytest.raises(TypeError) as excinfo:
        viz.plot_metric("derived", 'raw_pr_m', save=False)
    assert "NavData" in str(excinfo.value)

def test_plot_metrics_by_constellation(derived):
    """Test for plotting metrics by constellation.

    Parameters
    ----------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.

    """

    test_rows = [
                 "raw_pr_m",
                 ]

    for row in derived.rows:
        if not derived.is_str(row):
            if row in test_rows:
                for prefix in ["","test"]:
                    fig = viz.plot_metric_by_constellation(derived, row,
                                               prefix=prefix,save=False)
                    viz.close_figures()
        else:
            # string rows should cause a KeyError
            with pytest.raises(KeyError) as excinfo:
                fig = viz.plot_metric_by_constellation(derived, row,
                                                        save=False)
                viz.close_figures(fig)
            assert "non-numeric row" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        viz.plot_metric_by_constellation(derived, "raw_pr_m", save=True,
                                         prefix=1)
    assert "Prefix" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        viz.plot_metric_by_constellation("derived", "raw_pr_m", save=True)
    assert "NavData" in str(excinfo.value)

    derived_no_gnss_id = derived.remove(rows="gnss_id")
    with pytest.raises(KeyError) as excinfo:
        viz.plot_metric_by_constellation(derived_no_gnss_id, "raw_pr_m",
                                         save=False)
    assert "gnss_id" in str(excinfo.value)

    for optional_row in ["sv_id","signal_type",["sv_id","signal_type"]]:
        derived_partial = derived.remove(rows=optional_row)
        figs = viz.plot_metric_by_constellation(derived_partial,
                                                "raw_pr_m", save=False)
        viz.close_figures(figs)

@pytest.mark.parametrize('navdata',[
                                    # lazy_fixture('derived_2022'),
                                    lazy_fixture('derived'),
                                    ])
def test_plot_skyplot(navdata, state_estimate):
    """Test for plotting skyplot.

    Parameters
    ----------
    navdata : AndroidDerived
        Instance of AndroidDerived for testing.
    state_estimate : gnss_lib_py.parsers.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """

    if isinstance(navdata, AndroidDerived2022):
        state_estimate = navdata.copy(rows=["gps_millis","x_rx_m","y_rx_m","z_rx_m"])

    sv_nan = np.unique(navdata["sv_id"])[0]
    for col_idx, col in enumerate(navdata):
        if col["sv_id"] == sv_nan:
            navdata["x_sv_m",col_idx] = np.nan

    # don't save figures
    fig = viz.plot_skyplot(navdata.copy(), state_estimate, save=False)
    viz.close_figures(fig)

    with pytest.raises(TypeError) as excinfo:
        viz.plot_skyplot(navdata.copy(), state_estimate, save=True, prefix=1)
    assert "Prefix" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        viz.plot_skyplot("derived", "raw_pr_m", save=True)
    assert "NavData" in str(excinfo.value)

    for row in ["x_sv_m","y_sv_m","z_sv_m","gps_millis"]:
        derived_removed = navdata.remove(rows=row)
        with pytest.raises(KeyError) as excinfo:
            viz.plot_skyplot(derived_removed, state_estimate, save=False)
        assert row in str(excinfo.value)

    for row in ["x_rx_m","y_rx_m","z_rx_m"]:
        row_idx = state_estimate.find_wildcard_indexes(row[:4]+'*'+row[4:])[row[:4]+'*'+row[4:]][0]
        state_removed = state_estimate.remove(rows=row_idx)
        with pytest.raises(KeyError) as excinfo:
            viz.plot_skyplot(navdata, state_removed, save=False)
        assert row[:4]+'*'+row[4:] in str(excinfo.value)
        assert "Missing" in str(excinfo.value)

    for row in ["x_rx_m","y_rx_m","z_rx_m"]:
        state_double = state_estimate.copy()
        row_idx = state_estimate.find_wildcard_indexes(row[:4]+'*'+row[4:])[row[:4]+'*'+row[4:]][0]
        state_double[row_idx.replace("rx_","rx_gt_")] = state_double[row_idx]
        with pytest.raises(KeyError) as excinfo:
            viz.plot_skyplot(navdata, state_double, save=False)
        assert row[:4]+'*'+row[4:] in str(excinfo.value)
        assert "More than 1" in str(excinfo.value)

def test_get_label():
    """Test for getting nice labels.

    """

    assert viz._get_label({"signal_type" : "l1"}) == "L1"
    assert viz._get_label({"signal_type" : "g1"}) == "G1"
    assert viz._get_label({"signal_type" : "b1i"}) == "B1i"
    # shouldn't do lowercase 'i' trick if not signal_type
    assert viz._get_label({"random" : "BDS_B1I"}) == "BDS B1I"

    assert viz._get_label({"gnss_id" : "beidou"}) == "BeiDou"
    assert viz._get_label({"gnss_id" : "gps"}) == "GPS"
    assert viz._get_label({"gnss_id" : "galileo"}) == "Galileo"

    assert viz._get_label({"gnss_id" : "galileo",
                           "signal_type" : "b1i"}) == "Galileo B1i"

    assert viz._get_label({"row" : "x_rx_m"}) == "X RX [m]"
    assert viz._get_label({"row" : "lat_rx_deg"}) == "LAT RX [deg]"
    assert viz._get_label({"row" : "vx_sv_mps"}) == "VX SV [m/s]"

    with pytest.raises(TypeError) as excinfo:
        viz._get_label(["should","fail"])
    assert "dictionary" in str(excinfo.value)

def test_sort_gnss_ids():
    """Test sorting GNSS IDs.

    """

    unsorted_ids = ["galileo","beta","beidou","irnss","gps","unknown","glonass",
                "alpha","qzss","sbas"]
    sorted_ids = ["gps","glonass","galileo","beidou","qzss","irnss","sbas",
                  "unknown", "alpha", "beta"]

    assert viz._sort_gnss_ids(unsorted_ids) == sorted_ids
    assert viz._sort_gnss_ids(np.array(unsorted_ids)) == sorted_ids
    assert viz._sort_gnss_ids(set(unsorted_ids)) == sorted_ids
    assert viz._sort_gnss_ids(tuple(unsorted_ids)) == sorted_ids

    for _ in range(100):
        random.shuffle(unsorted_ids)
        assert viz._sort_gnss_ids(unsorted_ids) == sorted_ids
        assert viz._sort_gnss_ids(np.array(unsorted_ids)) == sorted_ids
        assert viz._sort_gnss_ids(set(unsorted_ids)) == sorted_ids
        assert viz._sort_gnss_ids(tuple(unsorted_ids)) == sorted_ids

def test_close_figures_fail():
    """Test expected fail conditions.

    """

    viz.close_figures([])

    with pytest.raises(TypeError) as excinfo:
        viz.close_figures(0.)
    assert "figure" in str(excinfo.value)


def test_plot_map(gtruth, state_estimate):
    """Test for plotting map.

    Parameters
    ----------
    gtruth : AndroidGroundTruth2021
        Instance of AndroidGroundTruth2021 for testing
    state_estimate : gnss_lib_py.parsers.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """

    fig = viz.plot_map(gtruth, state_estimate, save=False)
    assert isinstance(fig, go.Figure)

    figs = viz.plot_map(gtruth, state_estimate,sections=3, save=False)
    for fig in figs:
        assert isinstance(fig, go.Figure)

    with pytest.raises(TypeError) as excinfo:
        viz.plot_map([], state_estimate, save=False)
    assert "NavData" in str(excinfo.value)
    assert "Input" in str(excinfo.value)

    for row in ["lat_rx_wls_deg","lon_rx_wls_deg"]:
        state_removed = state_estimate.remove(rows=row)
        with pytest.raises(KeyError) as excinfo:
            viz.plot_map(gtruth, state_removed, save=False)
        assert row.replace("rx_wls","*") in str(excinfo.value)
        assert "Missing" in str(excinfo.value)

    for row in ["lat_rx_wls_deg","lon_rx_wls_deg"]:
        state_double = state_estimate.copy()
        state_double[row.replace("rx","2")] = state_double[row]
        with pytest.raises(KeyError) as excinfo:
            viz.plot_map(gtruth, state_double, save=False)
        assert row.replace("rx_wls","*") in str(excinfo.value)
        assert "More than 1" in str(excinfo.value)
