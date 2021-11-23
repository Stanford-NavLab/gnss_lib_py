"""Tests for gnss_lib_py.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "6 Aug 2021"

import os
import sys
# append <path>/gnss_lib_py/gnss_lib_py/ to path
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))

import pytest
import numpy as np
import pandas as pd

from gnss_lib_py import __version__
# from gnss_lib_py.io.android import make_gnss_dataframe

"""
Define fixtures as functions that return fixed objects (which can be passed to later tests)
Can be composed from each other (pass fixture as argument to another fixture)
"""
# @pytest.fixture
# def input_log_path():
#     """
#     Returns path to a measurement log file
#     """
#     parent_directory = os.getcwd()
#     print('The parent directory is ', parent_directory)
#     input_filepath = os.path.join(parent_directory, 'data', 'training', '2020-05-14-US-MTV-1', 'Pixel4XLModded_GnssLog.txt')
#     # return os.path.join("data", "dummy_data", "measurement_log.txt")
#     return input_filepath

# def input_log_df(input_log_path):
#     df, _ = make_gnss_dataframe(input_log_path)
#     return df

# def input_log_fix(input_log_path):
#     _, fix = make_gnss_dataframe(input_log_path)
#     return fix

"""
Run tests with the created fixtures
"""

def test_version():
    assert __version__ == '0.1.0'

# def test_logfile_exists(input_log_path):
#     assert os.path.exists(input_log_path)

# def test_log2df(input_log_path):
#     df, fix = make_gnss_dataframe(input_log_path)
#     assert isinstance(df, pd.DataFrame)
#     assert isinstance(fix, pd.DataFrame)
