"""Tests for residuals

"""

__authors__ = "D. Knowles"
__date__ = "22 Jun 2022"

import os
import pytest

import numpy as np

from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.parsers.google_decimeter import AndroidDerived2021
from gnss_lib_py.parsers.navdata import NavData
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
    root_path = os.path.join(root_path, 'data/unit_test/google_decimeter_2021')
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

def test_residuals_inplace(derived):
    """Test that solving for residuals doesn't fail

    Parameters
    ----------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.

    """

    derived_original = derived.copy()

    state_estimate = solve_wls(derived)

    solve_residuals(derived, state_estimate)

    # result should still be a NavData Class instance
    assert isinstance(derived,type(NavData()))

    # derived should have one more row but same number of cols
    assert len(derived.rows) == len(derived_original.rows) + 1
    assert len(derived) == len(derived_original)

    # derived should include new residuals rows but not its copy
    assert "residuals_m" in derived.rows
    assert "residuals_m" not in derived_original.rows

    assert not np.any(np.isinf(derived["residuals_m"]))

    # max is 47.814594604074955
    assert max(derived["residuals_m"]) < 50.

def test_residuals(derived):
    """Test that solving for residuals doesn't fail

    Parameters
    ----------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.

    """

    state_estimate = solve_wls(derived)

    residuals = solve_residuals(derived, state_estimate, inplace=False)

    # result should still be a NavData Class instance
    assert isinstance(residuals,type(NavData()))

    # derived should have one more row but same number of cols
    for row in ["residuals_m","gps_millis","gnss_id","sv_id","signal_type"]:
        assert row in residuals.rows
    assert len(residuals) == len(derived)

    # derived should not include new residuals row
    assert "residuals_m" not in derived.rows

    assert not np.any(np.isinf(residuals["residuals_m"]))

    # max is 47.814594604074955
    assert max(residuals["residuals_m"]) < 50.

def test_residuals_fails(derived):
    """Test that solving for residuals fails when it should

    Parameters
    ----------
    derived : AndroidDerived2021
        Instance of AndroidDerived2021 for testing.

    """

    state_estimate = solve_wls(derived)

    for inplace in [True,False]:

        for row in ["gps_millis","corr_pr_m"]:
            derived_removed = derived.remove(rows=row)
            with pytest.raises(KeyError) as excinfo:
                _ = solve_residuals(derived_removed, state_estimate,
                                    inplace=inplace)
            assert row in str(excinfo.value)

        for row in ["gps_millis"]:
            state_estimate_removed = state_estimate.remove(rows=row)
            with pytest.raises(KeyError) as excinfo:
                _ = solve_residuals(derived,
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
                _ = solve_residuals(derived,
                                    duplicated,
                                    inplace=inplace)
            assert error_name in str(excinfo.value)

        for row in ["x_rx_wls_m", "y_rx_wls_m", "z_rx_wls_m", "b_rx_wls_m"]:
            state_estimate_removed = state_estimate.remove(rows=row)
            error_name = row[:4] + '*' + row[-2:]
            with pytest.raises(KeyError) as excinfo:
                _ = solve_residuals(derived,
                                    state_estimate_removed,
                                    inplace=inplace)
            assert error_name in str(excinfo.value)
