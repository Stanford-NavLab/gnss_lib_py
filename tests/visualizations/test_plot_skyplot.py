"""Tests for visualizations.

"""

__authors__ = "D. Knowles"
__date__ = "22 Jun 2022"

import os

import pytest
import numpy as np
import matplotlib as mpl
from pytest_lazyfixture import lazy_fixture

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import find_wildcard_indexes
from gnss_lib_py.visualizations import style
from gnss_lib_py.visualizations import plot_skyplot
from gnss_lib_py.utils.coordinates import geodetic_to_ecef
from gnss_lib_py.parsers.google_decimeter import AndroidDerived2022

@pytest.mark.parametrize('navdata',[
                                    # lazy_fixture('derived_2022'),
                                    lazy_fixture('derived_2021'),
                                    ])
def test_plot_skyplot(navdata, state_estimate):
    """Test for plotting skyplot.

    Parameters
    ----------
    navdata : AndroidDerived
        Instance of AndroidDerived for testing.
    state_estimate : gnss_lib_py.navdata.navdata.NavData
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
    fig = plot_skyplot.plot_skyplot(navdata.copy(), state_estimate, save=False)
    style.close_figures(fig)

    with pytest.raises(TypeError) as excinfo:
        plot_skyplot.plot_skyplot(navdata.copy(), state_estimate, save=True, prefix=1)
    assert "Prefix" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        plot_skyplot.plot_skyplot("derived", "raw_pr_m", save=True)
    assert "NavData" in str(excinfo.value)

    for row in ["x_sv_m","y_sv_m","z_sv_m","gps_millis"]:
        derived_removed = navdata.remove(rows=row)
        with pytest.raises(KeyError) as excinfo:
            plot_skyplot.plot_skyplot(derived_removed, state_estimate, save=False)
        assert row in str(excinfo.value)

    for row in ["x_rx_m","y_rx_m","z_rx_m"]:
        row_idx = find_wildcard_indexes(state_estimate,row[:4]+'*'+row[4:])[row[:4]+'*'+row[4:]][0]
        state_removed = state_estimate.remove(rows=row_idx)
        with pytest.raises(KeyError) as excinfo:
            plot_skyplot.plot_skyplot(navdata, state_removed, save=False)
        assert row[:4]+'*'+row[4:] in str(excinfo.value)
        assert "Missing" in str(excinfo.value)

    for row in ["x_rx_m","y_rx_m","z_rx_m"]:
        state_double = state_estimate.copy()
        row_idx = find_wildcard_indexes(state_estimate,row[:4]+'*'+row[4:])[row[:4]+'*'+row[4:]][0]
        state_double[row_idx.replace("rx_","rx_gt_")] = state_double[row_idx]
        with pytest.raises(KeyError) as excinfo:
            plot_skyplot.plot_skyplot(navdata, state_double, save=False)
        assert row[:4]+'*'+row[4:] in str(excinfo.value)
        assert "More than 1" in str(excinfo.value)

def test_skyplot_trim(root_path):
    """Test trimming separate time instances for same SV.

    Parameters
    ----------
    root_path : string
        Folder location containing unit test data

    """

    sp3_path = os.path.join(root_path,"vis","sp3_g05.csv")
    sp3 = NavData(csv_path=sp3_path)

    lat, lon, alt = -77.87386688990695, -34.62755517700375, 0.
    x_rx_m, y_rx_m, z_rx_m = geodetic_to_ecef(np.array([[lat,lon,alt]]))[0]
    receiver_state = NavData()
    receiver_state["gps_millis"] = 0.
    receiver_state["x_rx_m"] = x_rx_m
    receiver_state["y_rx_m"] = y_rx_m
    receiver_state["z_rx_m"] = z_rx_m

    fig = plot_skyplot.plot_skyplot(sp3,receiver_state)
    # verify that two line segments were removed. Should be 57 not 59
    # after trimming the two separated ones.
    for child in fig.get_children():
        if isinstance(child,mpl.projections.polar.PolarAxes):
            for grandchild in child.get_children():
                if isinstance(grandchild,mpl.collections.LineCollection):
                    assert len(grandchild.get_array()) == 57
    style.close_figures()

    fig = plot_skyplot.plot_skyplot(sp3,receiver_state,trim_options={"az" : 95.})
    # verify that only one line segment was removed. Should be 58 not 59
    # after trimming the one larger than 95 degrees in azimuth separated ones.
    for child in fig.get_children():
        if isinstance(child,mpl.projections.polar.PolarAxes):
            for grandchild in child.get_children():
                if isinstance(grandchild,mpl.collections.LineCollection):
                    assert len(grandchild.get_array()) == 58
    style.close_figures()

    fig = plot_skyplot.plot_skyplot(sp3,receiver_state,trim_options={"gps_millis" : 3.5E6})
    # verify that only one line segment was removed. Should be 58 not 59
    # after trimming the one larger than 95 degrees in azimuth separated ones.
    for child in fig.get_children():
        if isinstance(child,mpl.projections.polar.PolarAxes):
            for grandchild in child.get_children():
                if isinstance(grandchild,mpl.collections.LineCollection):
                    assert len(grandchild.get_array()) == 57
    style.close_figures()


    with pytest.raises(TypeError) as excinfo:
        plot_skyplot.plot_skyplot(sp3, receiver_state, step=20.1)
    assert "step" in str(excinfo.value)
