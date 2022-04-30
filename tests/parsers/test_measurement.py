"""Tests for Measurement class.

"""

__authors__ = "A. Kanhere, D. Knowles"
__date__ = "30 Apr 2022"


import os

import pytest
import numpy as np
import pandas as pd

from gnss_lib_py.parsers.measurement import Measurement

instance = Measurement(csv="")
instance = Measurement()
instance = Measurement(numpy=np.array)

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


@pytest.fixture(name="csv_path")
def fixture_derived_path(root_path):
    """Filepath of CSV file to test measurements


    Returns
    -------
    derived_path : string
        Location for .csv file to test Measurement functionality

    """
    csv_path = os.path.join(root_path, 'measure_test.csv')
    return csv_path


@pytest.fixture(name="pandas_df")
def load_test_dataframe(csv_path):
    """
    """
    df = pd.read_csv(csv_path)
    return df


@pytest.fixture(name="numpy_array")
def create_numpy_array():
    """
    """
    test_array = np.array([[1,2,3,4],
                            [15,16,17,18],
                            [29,30,31.5,32.3],
                            [0.5,0.6,0.7,0.8]
                            ])
    return test_array