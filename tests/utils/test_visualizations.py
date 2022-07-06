"""Tests for visualizations.

"""

__authors__ = "D. Knowles"
__date__ = "22 Jun 2022"

import os
import pytest

import numpy as np

import gnss_lib_py.utils.visualizations as vis
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.parsers.android import AndroidDerived
from gnss_lib_py.utils.file_operations import close_figures
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

@pytest.fixture(name="state_estimate")
def fixture_solve_wls(derived):
    """Fixture of WLS state estimate.

    Parameters
    ----------
    derived : AndroidDerived
        Instance of AndroidDerived for testing.

    Returns
    -------
    state_estimate : gnss_lib_py.parsers.measurement.Measurement
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the Measurement class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """
    state_estimate = solve_wls(derived)
    return state_estimate

def test_plot_metrics(derived):
    """Test for plotting metrics.

    Parameters
    ----------
    derived : AndroidDerived
        Instance of AndroidDerived for testing.

    """

    test_rows = [
                 "raw_pr_m",
                 "raw_pr_sigma_m",
                 "tropo_delay_m",
                 ]

    for rr, row in enumerate(derived.rows):
        if not derived.str_bool[rr]:
            if row in test_rows:
                fig = vis.plot_metric(derived, row, save=False)
                close_figures(fig)
        else:
            # string rows should cause a KeyError
            with pytest.raises(KeyError) as excinfo:
                fig = vis.plot_metric(derived, row, save=False)
                close_figures(fig)
            assert "non-numeric row" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        vis.plot_metric(derived, "raw_pr_m", save=True, prefix=1)
    assert "Prefix" in str(excinfo.value)

def test_plot_skyplot(derived, state_estimate):
    """Test for plotting skyplot.

    Parameters
    ----------
    derived : AndroidDerived
        Instance of AndroidDerived for testing.
    state_estimate : gnss_lib_py.parsers.measurement.Measurement
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the Measurement class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """

    # don't save figures
    fig = vis.plot_skyplot(derived, state_estimate, save=False)
    close_figures(fig)

    with pytest.raises(TypeError) as excinfo:
        vis.plot_skyplot(derived, state_estimate, save=True, prefix=1)
    assert "Prefix" in str(excinfo.value)


def test_plot_residuals(derived, state_estimate):
    """Test for plotting residuals.

    Parameters
    ----------
    derived : AndroidDerived
        Instance of AndroidDerived for testing.
    state_estimate : gnss_lib_py.parsers.measurement.Measurement
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the Measurement class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """

    derived_original = derived.copy()

    solve_residuals(derived, state_estimate)

    # don't save figures
    figs = vis.plot_residuals(derived, save=False)
    close_figures(figs)

    # should return KeyError if no residuals row
    with pytest.raises(KeyError) as excinfo:
        vis.plot_residuals(derived_original, save=False)
    assert "residuals missing" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        vis.plot_residuals(derived, save=True, prefix=1)
    assert "Prefix" in str(excinfo.value)
