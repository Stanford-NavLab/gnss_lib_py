"""Tests for visualizations.

"""

__authors__ = "D. Knowles"
__date__ = "22 Jun 2022"

def test_plot_map(gtruth, state_estimate):
    """Test for plotting map.

    Parameters
    ----------
    gtruth : AndroidGroundTruth2021
        Instance of AndroidGroundTruth2021 for testing
    state_estimate : gnss_lib_py.navdata.navdata.NavData
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
