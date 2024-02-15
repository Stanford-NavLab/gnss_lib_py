"""Tests for fault detection and exclusion methods.

"""

__authors__ = "D. Knowles"
__date__ = "11 Jul 2023"

import os

import pytest
import numpy as np

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.algorithms.fde import solve_fde, evaluate_fde

@pytest.mark.parametrize('method',
                        [
                         "residual",
                         "edm",
                         "ss"
                        ])
def test_solve_fde(derived_2022, method):
    """Test residual-based FDE.

    Parameters
    ----------
    derived_2022 : AndroidDerived2022
        Instance of AndroidDerived2022 for testing.
    method : string
        Method for fault detection and exclusion.

    """

    # test without removing outliers
    navdata = derived_2022.copy()
    navdata = solve_fde(navdata, method=method)
    assert "fault_" + method in navdata.rows

    # max thresholds shouldn't remove any
    navdata = derived_2022.copy()
    navdata = solve_fde(navdata, threshold=np.inf, method=method)
    assert sum(navdata.where("fault_" + method,1)["fault_" + method]) == 0

    # min threshold should remove most all
    navdata = derived_2022.copy()
    navdata = solve_fde(navdata, threshold=-np.inf, method=method)
    print(sum(navdata.where("fault_" + method,1)["fault_" + method]))
    assert len(navdata.where("fault_" + method,0)) == 24
    num_unknown = len(navdata.where("fault_" + method,2))

    navdata = derived_2022.copy()
    original_length = len(navdata)
    navdata = solve_fde(navdata,
                        threshold=-np.inf,
                        max_faults=1,
                        method=method,
                        remove_outliers=True,
                        verbose=True)
    assert "fault_" + method in navdata.rows
    np.testing.assert_array_equal(np.unique(navdata["fault_" + method]),
                                  np.array([0]))
    assert len(navdata) == original_length - num_unknown - 6

def test_fde_fails(derived_2022):
    """Test that solve_fde fails when it should.

    Parameters
    ----------
    derived_2022 : AndroidDerived2022
        Instance of AndroidDerived2022 for testing.

    """

    with pytest.raises(ValueError) as excinfo:
        solve_fde(derived_2022, method="perfect_method")
    assert "invalid method" in str(excinfo.value)

@pytest.mark.parametrize('method',
                        [
                         "residual",
                         "ss",
                         "edm",
                        ])
def test_evaluate_fde(derived_2022, method):
    """Evaluate FDE methods.

    Parameters
    ----------
    derived_2022 : AndroidDerived2022
        Instance of AndroidDerived2022 for testing.
    method : string
        Method for fault detection and exclusion.

    """

    navdata = derived_2022.copy()
    evaluate_fde(navdata,
                 method=method,
                 fault_truth_row="MultipathIndicator",
                 verbose=True,
                 time_fde=True,
                 )

    if method == "edm":
        evaluate_fde(navdata,
                     method=method,
                     fault_truth_row="MultipathIndicator",
                     verbose=False,
                     time_fde=False,
                     )

def test_edm_breakouts():
    """Test places when EDM FDE should breakout.

    """
    root_path = os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))

    # test case when there should be nothing removed
    csv_path = os.path.join(root_path, 'data','unit_test','fde',
                            'nothing_removed.csv')
    navdata = NavData(csv_path=csv_path)
    solve_fde(navdata,"edm",threshold=0)

    # test case when there are no fault suspects
    csv_path = os.path.join(root_path, 'data','unit_test','fde',
                            'no_suspects.csv')
    navdata = NavData(csv_path=csv_path)
    solve_fde(navdata,"edm",threshold=0)
