"""Tests for visualizations.

"""

__authors__ = "D. Knowles"
__date__ = "22 Jun 2022"

import pytest
import matplotlib.pyplot as plt

import gnss_lib_py.visualizations.style as style
import gnss_lib_py.visualizations.plot_metric as metric

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
                    fig = metric.plot_metric(derived, row,
                                          groupby = groupby,
                                          save=False)
                    style.close_figures(fig)
        else:
            # string rows should cause a KeyError
            with pytest.raises(KeyError) as excinfo:
                fig = metric.plot_metric(derived, row, save=False)
                style.close_figures(fig)
            assert "non-numeric row" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        metric.plot_metric(derived, "raw_pr_m", save=True, prefix=1)
    assert "Prefix" in str(excinfo.value)

    for row in derived.rows:
        if not derived.is_str(row):
            if row in test_rows:
                for groupby in ["gnss_id",None]:
                    fig = metric.plot_metric(derived, "raw_pr_m", row,
                                          groupby=groupby, save=False)
                    style.close_figures(fig)
        else:
            # string rows should cause a KeyError
            with pytest.raises(KeyError) as excinfo:
                fig = metric.plot_metric(derived, "raw_pr_m", row, save=False)
                style.close_figures(fig)
            with pytest.raises(KeyError) as excinfo:
                fig = metric.plot_metric(derived, row, "raw_pr_m", save=False)
                style.close_figures(fig)
            assert "non-numeric row" in str(excinfo.value)

    style.close_figures()

    # test repeating figure and average y
    fig = plt.figure()
    fig = metric.plot_metric(derived, "gps_millis", "raw_pr_m",
                          fig = fig,
                          groupby = "gnss_id",
                          save=False)
    fig = metric.plot_metric(derived, "gps_millis", "raw_pr_m",
                            fig = fig,
                            groupby = "gnss_id",
                            avg_y = True,
                            linestyle="dotted",
                            save=False,
                            )
    style.close_figures(fig)

    with pytest.raises(TypeError) as excinfo:
        metric.plot_metric(derived, "raw_pr_m", save=True, prefix=1)
    assert "Prefix" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        metric.plot_metric(derived, 'raw_pr_m', row, row, save=False)

    with pytest.raises(TypeError) as excinfo:
        metric.plot_metric("derived", 'raw_pr_m', save=False)
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
                    fig = metric.plot_metric_by_constellation(derived, row,
                                               prefix=prefix,save=False)
                    style.close_figures()
        else:
            # string rows should cause a KeyError
            with pytest.raises(KeyError) as excinfo:
                fig = metric.plot_metric_by_constellation(derived, row,
                                                        save=False)
                style.close_figures(fig)
            assert "non-numeric row" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        metric.plot_metric_by_constellation(derived, "raw_pr_m", save=True,
                                         prefix=1)
    assert "Prefix" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        metric.plot_metric_by_constellation("derived", "raw_pr_m", save=True)
    assert "NavData" in str(excinfo.value)

    derived_no_gnss_id = derived.remove(rows="gnss_id")
    with pytest.raises(KeyError) as excinfo:
        metric.plot_metric_by_constellation(derived_no_gnss_id, "raw_pr_m",
                                         save=False)
    assert "gnss_id" in str(excinfo.value)

    for optional_row in ["sv_id","signal_type",["sv_id","signal_type"]]:
        derived_partial = derived.remove(rows=optional_row)
        figs = metric.plot_metric_by_constellation(derived_partial,
                                                "raw_pr_m", save=False)
        style.close_figures(figs)
