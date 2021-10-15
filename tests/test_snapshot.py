import pytest
import numpy as np
import os
import sys
# append <path>/gnss_lib_py/gnss_lib_py/ to path
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
# Path of package for pytest

from gnss_lib_py.core.constants import GPSConsts
from gnss_lib_py.algorithms import snapshot

# Defining test fixtures

@pytest.fixture
def tolerance():
    return 1e-7

def Ru_ECEF():
    x_user = 3678300.0
    y_user = 3678300.0
    z_user = 3678300.0
    bias_clock_user = 10.0
    return x_user, y_user, z_user, bias_clock_user

def Rsv_ECEF():
    x_sv = np.array([13005878.255, 20451225.952, 20983704.633, 13798849.321])
    y_sv = np.array([18996947.213, 16359086.310, 15906974.416, -8709113.822])
    z_sv = np.array([13246718.721, -4436309.875, 3486495.546, 20959777.407])
    bias_clock_sv = 5.0
    return x_sv, y_sv, z_sv, bias_clock_sv

def prange_measurements():
    x_user, y_user, z_user, bias_clock_user = Ru_ECEF()
    x_sv, y_sv, z_sv, bias_clock_sv = Rsv_ECEF()
    prange_measured = np.sqrt((x_user-x_sv)**2 + (y_user-y_sv)**2 +(z_user-z_sv)**2) + bias_clock_user - bias_clock_sv
    return prange_measured
    
# Defining tests

def test_snapshot(tolerance):
    gpsconsts = GPSConsts()
    x_user, y_user, z_user, bias_clock_user = Ru_ECEF()
    x_sv  , y_sv  , z_sv  , bias_clock_sv   = Rsv_ECEF()
    prange = prange_measurements()
    X_fix = snapshot.solvepos(prange, x_sv, y_sv, z_sv, bias_clock_sv)
    X_fix[-1] = X_fix[-1] / 1e6 * gpsconsts.C

    assert all(abs(X_fix - np.array([x_user, y_user, z_user, bias_clock_user])) < tolerance)