"""
Test for metrics, such as DOP calculations.

"""

__authors__ = "Ashwin Kanhere, Daniel Neamati"
__date__ = "7 Feb 2024"

import pytest 
import numpy as np

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.utils.metrics import get_dop, calculate_dop
from gnss_lib_py.navdata.operations import loop_time


def test_simple_dop():
    """A simple set of satellites for DOP calculation."""
    # Create a simple NavData instance
    navdata = NavData()
    
    # Add a few satellites
    navdata['el_sv_deg'] = np.array([0, 0, 45, 45, 90], dtype=float)
    navdata['az_sv_deg'] = np.array([0, 90, 180, 270, 360], dtype=float)

    dop = calculate_dop(navdata)

    # Check the DOP output has all the expected keys
    assert set(dop.keys()) == {'dop_matrix', 
                               'GDOP', 'HDOP', 'VDOP', 'PDOP', 'TDOP'}
    
    # Check the DOP output has the expected values
    sqrt2 = np.sqrt(2)
    expected_dop_matrix = 0.25 * np.array(
        [[(25/3) - 2*sqrt2, (17/3) - 2*sqrt2,  5 + sqrt2,   -5 + sqrt2],
         [(17/3) - 2*sqrt2, (25/3) - 2*sqrt2,  5 + sqrt2,   -5 + sqrt2],
         [5 + sqrt2,         5 + sqrt2,        9 + 4*sqrt2, -5 - 2*sqrt2],
         [-5 + sqrt2,       -5 + sqrt2,       -5 - 2*sqrt2,  5]])
    
    # Assert symmetry
    assert np.allclose(expected_dop_matrix.T, expected_dop_matrix)
    assert np.allclose(dop['dop_matrix'].T, dop['dop_matrix'])
    # Assert matching values
    assert np.allclose(dop['dop_matrix'], expected_dop_matrix)

    # Check the DOP values
    assert np.allclose(dop['GDOP'], np.sqrt(23/3))
    assert np.allclose(dop['HDOP'], np.sqrt((25/6) - sqrt2))
    assert np.allclose(dop['VDOP'], np.sqrt((9/4) + sqrt2))
    assert np.allclose(dop['PDOP'], 0.5 * np.sqrt(77/3))
    assert np.allclose(dop['TDOP'], 0.5 * np.sqrt(5))

    



