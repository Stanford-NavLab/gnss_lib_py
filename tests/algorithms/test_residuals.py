"""Tests for residuals

"""

__authors__ = "D. Knowles"
__date__ = "22 Jun 2022"

import os
import pytest

import numpy as np

from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.parsers.google_decimeter import AndroidDerived2021
from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.algorithms.residuals import solve_residuals

def test_residuals_inplace(derived_2021):
    """Test that solving for residuals doesn't fail

    Parameters
    ----------
    derived_2021 : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.

    """

    derived_original = derived_2021.copy()

    state_estimate = solve_wls(derived_2021)

    solve_residuals(derived_2021, state_estimate)

    # result should still be a NavData Class instance
    assert isinstance(derived_2021,type(NavData()))

    # derived should have one more row but same number of cols
    assert len(derived_2021.rows) == len(derived_original.rows) + 1
    assert len(derived_2021) == len(derived_original)

    # derived should include new residuals rows but not its copy
    assert "residuals_m" in derived_2021.rows
    assert "residuals_m" not in derived_original.rows

    assert not np.any(np.isinf(derived_2021["residuals_m"]))

    # max is 47.814594604074955
    assert max(derived_2021["residuals_m"]) < 50.

def test_residuals(derived_2021):
    """Test that solving for residuals doesn't fail

    Parameters
    ----------
    derived_2021 : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.

    """

    state_estimate = solve_wls(derived_2021)

    residuals = solve_residuals(derived_2021, state_estimate, inplace=False)

    # result should still be a NavData Class instance
    assert isinstance(residuals,type(NavData()))

    # derived should have one more row but same number of cols
    for row in ["residuals_m","gps_millis","gnss_id","sv_id","signal_type"]:
        assert row in residuals.rows
    assert len(residuals) == len(derived_2021)

    # derived should not include new residuals row
    assert "residuals_m" not in derived_2021.rows

    assert not np.any(np.isinf(residuals["residuals_m"]))

    # max is 47.814594604074955
    assert max(residuals["residuals_m"]) < 50.

def test_residuals_fails(derived_2021):
    """Test that solving for residuals fails when it should

    Parameters
    ----------
    derived_2021 : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.

    """

    state_estimate = solve_wls(derived_2021)

    for inplace in [True,False]:

        for row in ["gps_millis","corr_pr_m"]:
            derived_removed = derived_2021.remove(rows=row)
            with pytest.raises(KeyError) as excinfo:
                _ = solve_residuals(derived_removed, state_estimate,
                                    inplace=inplace)
            assert row in str(excinfo.value)

        for row in ["gps_millis"]:
            state_estimate_removed = state_estimate.remove(rows=row)
            with pytest.raises(KeyError) as excinfo:
                _ = solve_residuals(derived_2021,
                                    state_estimate_removed,
                                    inplace=inplace)
            assert row in str(excinfo.value)

        for row in ["x_rx_wls_m", "y_rx_wls_m", "z_rx_wls_m", "b_rx_wls_m"]:
            duplicated = state_estimate.copy()
            new_name = row.split("_")
            new_name[2] = "gt"
            new_name = "_".join(new_name)
            error_name = row[:4] + '*' + row[-2:]
            duplicated[new_name] = duplicated[row]
            with pytest.raises(KeyError) as excinfo:
                _ = solve_residuals(derived_2021,
                                    duplicated,
                                    inplace=inplace)
            assert error_name in str(excinfo.value)

        for row in ["x_rx_wls_m", "y_rx_wls_m", "z_rx_wls_m", "b_rx_wls_m"]:
            state_estimate_removed = state_estimate.remove(rows=row)
            error_name = row[:4] + '*' + row[-2:]
            with pytest.raises(KeyError) as excinfo:
                _ = solve_residuals(derived_2021,
                                    state_estimate_removed,
                                    inplace=inplace)
            assert error_name in str(excinfo.value)
