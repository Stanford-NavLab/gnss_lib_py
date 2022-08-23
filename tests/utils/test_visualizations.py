"""Tests for visualizations.

"""

__authors__ = "D. Knowles"
__date__ = "22 Jun 2022"

import os
import pytest

import gnss_lib_py.utils.visualizations as viz
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.parsers.android import AndroidDerived2021
from gnss_lib_py.algorithms.residuals import solve_residuals

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
                 "raw_pr_sigma_m",
                 "tropo_delay_m",
                 ]

    for row in derived.rows:
        if not derived.is_str(row):
            if row in test_rows:
                fig = viz.plot_metric(derived, row, save=False)
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
                fig = viz.plot_metric(derived, "raw_pr_m", row, save=False)
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

    with pytest.raises(TypeError) as excinfo:
        viz.plot_metric(derived, "raw_pr_m", save=True, prefix=1)
    assert "Prefix" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        viz.plot_metric(derived, 'raw_pr_m', row, row, save=False)

def test_plot_metrics_by_constellation(derived):
    """Test for plotting metrics.

    Parameters
    ----------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.

    """

    test_rows = [
                 "raw_pr_m",
                 "raw_pr_sigma_m",
                 "tropo_delay_m",
                 ]

    for row in derived.rows:
        if not derived.is_str(row):
            if row in test_rows:
                fig = viz.plot_metric_by_constellation(derived, row, save=False)
                viz.close_figures(fig)
        else:
            # string rows should cause a KeyError
            with pytest.raises(KeyError) as excinfo:
                fig = viz.plot_metric_by_constellation(derived, row, save=False)
                viz.close_figures(fig)
            assert "non-numeric row" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        viz.plot_metric_by_constellation(derived, "raw_pr_m", save=True,
                                         prefix=1)
    assert "Prefix" in str(excinfo.value)

    derived_no_gnss_id = derived.remove(rows="gnss_id")
    with pytest.raises(KeyError) as excinfo:
        viz.plot_metric_by_constellation(derived_no_gnss_id, "raw_pr_m",
                                         save=False)
    assert "gnss_id" in str(excinfo.value)

    derived_no_sv_id = derived.remove(rows="signal_type")
    viz.plot_metric_by_constellation(derived_no_sv_id,
                                     "raw_pr_m", save=False)

    derived_no_signal_type = derived.remove(rows="signal_type")
    viz.plot_metric_by_constellation(derived_no_signal_type,
                                     "raw_pr_m", save=False)


def test_plot_skyplot(derived, state_estimate):
    """Test for plotting skyplot.

    Parameters
    ----------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.
    state_estimate : gnss_lib_py.parsers.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """

    # don't save figures
    fig = viz.plot_skyplot(derived, state_estimate, save=False)
    viz.close_figures(fig)

    with pytest.raises(TypeError) as excinfo:
        viz.plot_skyplot(derived, state_estimate, save=True, prefix=1)
    assert "Prefix" in str(excinfo.value)

    # derived_no_sv_id = derived.remove(rows="sv_id")
    # with pytest.raises(KeyError) as excinfo:
    #     viz.plot_skyplot(derived_no_sv_id, state_estimate, save=False)
    # assert "sv_id" in str(excinfo.value)
    #
    # derived_no_signal_type = derived.remove(rows="signal_type")
    # with pytest.raises(KeyError) as excinfo:
    #     viz.plot_skyplot(derived_no_signal_type, state_estimate, save=False)
    # assert "signal_type" in str(excinfo.value)

    derived_no_x = derived.remove(rows="x_sv_m")
    with pytest.raises(KeyError) as excinfo:
        viz.plot_skyplot(derived_no_x, state_estimate, save=False)
    assert "x_sv_m" in str(excinfo.value)

    derived_no_y = derived.remove(rows="y_sv_m")
    with pytest.raises(KeyError) as excinfo:
        viz.plot_skyplot(derived_no_y, state_estimate, save=False)
    assert "y_sv_m" in str(excinfo.value)

    derived_no_z = derived.remove(rows="z_sv_m")
    with pytest.raises(KeyError) as excinfo:
        viz.plot_skyplot(derived_no_z, state_estimate, save=False)
    assert "z_sv_m" in str(excinfo.value)

    # state_estimate_no_x = state_estimate.remove(rows="x_rx_m")
    # with pytest.raises(KeyError) as excinfo:
    #     viz.plot_skyplot(derived, state_estimate_no_x, save=False)
    # assert "x_rx_m" in str(excinfo.value)
    #
    # state_estimate_no_y = state_estimate.remove(rows="y_rx_m")
    # with pytest.raises(KeyError) as excinfo:
    #     viz.plot_skyplot(derived, state_estimate_no_y, save=False)
    # assert "y_rx_m" in str(excinfo.value)
    #
    # state_estimate_no_z = state_estimate.remove(rows="z_rx_m")
    # with pytest.raises(KeyError) as excinfo:
    #     viz.plot_skyplot(derived, state_estimate_no_z, save=False)
    # assert "z_rx_m" in str(excinfo.value)

def test_plot_residuals(derived, state_estimate):
    """Test for plotting residuals.

    Parameters
    ----------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.
    state_estimate : gnss_lib_py.parsers.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """

    derived_original = derived.copy()

    solve_residuals(derived, state_estimate)

    # don't save figures
    figs = viz.plot_residuals(derived, save=False)
    viz.close_figures(figs)

    # should return KeyError if no residuals row
    with pytest.raises(KeyError) as excinfo:
        viz.plot_residuals(derived_original, save=False)
    assert "residuals missing" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        viz.plot_residuals(derived, save=True, prefix=1)
    assert "Prefix" in str(excinfo.value)

    derived_no_sv_id = derived.remove(rows="sv_id")
    with pytest.raises(KeyError) as excinfo:
        viz.plot_residuals(derived_no_sv_id, save=False)
    assert "sv_id" in str(excinfo.value)

    derived_no_signal_type = derived.remove(rows="signal_type")
    with pytest.raises(KeyError) as excinfo:
        viz.plot_residuals(derived_no_signal_type, save=False)
    assert "signal_type" in str(excinfo.value)

def test_get_signal_label():
    """Test for getting signal labels.

    """

    assert viz.get_signal_label("GPS_L1") == "GPS L1"
    assert viz.get_signal_label("GLO_G1") == "GLO G1"
    assert viz.get_signal_label("BDS_B1I") == "BDS B1i"
