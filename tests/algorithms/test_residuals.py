"""Tests for residuals

"""

__authors__ = "D. Knowles"
__date__ = "22 Jun 2022"

import os
import pytest

import numpy as np

from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.parsers.android import AndroidDerived
from gnss_lib_py.parsers.measurement import Measurement
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

def test_residuals(derived):
    """Test that solving for residuals doesn't fail

    Parameters
    ----------
    derived : AndroidDerived
        Instance of AndroidDerived for testing.

    """

    derived_original = derived.copy()

    state_estimate = solve_wls(derived)

    solve_residuals(derived, state_estimate)

    # result should still be a Measurement Class instance
    assert isinstance(derived,type(Measurement()))

    # derived should have one more row but same number of cols
    assert len(derived.rows) == len(derived_original.rows) + 1
    assert len(derived) == len(derived_original)

    # derived should include new residuals rows but not its copy
    assert "residuals" in derived.rows
    assert "residuals" not in derived_original.rows

    assert not np.any(np.isinf(derived["residuals"]))
