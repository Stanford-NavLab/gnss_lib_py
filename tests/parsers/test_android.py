"""Tests for Android data loaders.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "10 Nov 2021"

import os
import sys

import numpy as np
import pandas as pd
import pytest

# append <path>/gnss_lib_py/gnss_lib_py/ to path
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__)))))

from gnss_lib_py.parsers.android import AndroidDerived



@pytest.fixture(name="inpath")
def fixture_inpath():
    root_path = os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
    inpath = os.path.join(root_path, 'data/unit_test/Pixel4_derived.csv')
    return inpath


@pytest.fixture(name="pd_df")
def fixture_pd_df(inpath):
    derived_df = pd.read_csv(inpath)
    return derived_df


@pytest.fixture(name="col_map")
def fixture_inverse_col_map():
    inverse_col_map = {'toe' : 'millisSinceGpsEpoch',
                        'SV' : 'svid',
                        'x' : 'xSatPosM',
                        'y' : 'ySatPosM',
                        'z' : 'zSatPosM',
                        'vx' : 'xSatVelMps',
                        'vy' : 'ySatVelMps',
                        'vz' : 'zSatVelMps',
                        'b' : 'satClkBiasM',
                        'b_dot' : 'satClkDriftMps'
                    }
    return inverse_col_map

@pytest.fixture(name="derived")
def fixture_load_measure(inpath):
    derived = AndroidDerived(inpath)
    return derived


def test_derived_df_equivalence(derived, pd_df, col_map):
    # Also tests if strings are being converted back correctly
    measure_df = derived.pandas_df()
    measure_df.rename(columns=col_map, inplace=True)
    measure_df = measure_df.loc[:, measure_df.columns!='pseudo']
    pd.testing.assert_frame_equal(pd_df, measure_df, check_dtype=False)

@pytest.mark.parametrize('row_name, index, value', 
                        [('collectionName', 0, '2020-05-14-US-MTV-1'),
                         ('phoneName', 1, 'Pixel4'),
                         ('vy', 7, 411.162),
                         ('b_dot', 41, -0.003),
                         ('signalType', 6, 'GLO_G1')]
                        )
def test_derived_value_check(derived, row_name, index, value):
    # Testing stored values vs their known counterparts
    # String maps have been converted to equivalent integers
    if isinstance(value, str):
        value_str = derived.str_map[row_name][int(derived[row_name, index])]
        assert value == value_str
    else:
        np.testing.assert_equal(derived[row_name, index], value)

def test_get_and_set_num(derived):
    key = 'testing123'
    value = np.zeros(len(derived))
    derived[key] = value
    np.testing.assert_equal(derived[key, :], np.reshape(value, [1, -1]))
    
def test_get_and_set_str(derived):
    key = 'testing123_string'
    value = ['word']*len(derived)
    derived[key] = value
    np.testing.assert_equal(derived[key, :], np.zeros([1, len(derived)]))

