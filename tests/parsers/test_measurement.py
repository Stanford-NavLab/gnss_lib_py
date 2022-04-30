"""Tests for Measurement class.

"""

__authors__ = "A. Kanhere, D. Knowles"
__date__ = "30 Apr 2022"


import os

import pytest
import numpy as np
import pandas as pd

from gnss_lib_py.parsers.measurement import Measurement

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
                            [0.5,0.6,0.7,0.8],
                            [-3.0,-1.2,-100.,-2.7],
                            [-543,-234,-986,-123],
                            ])
    return test_array


@pytest.fixture(name="data_csv")
def create_data_csv(csv_path):
    """Create test fixture for Measurement from csv

    Parameters
    ----------
    csv_path : string
        Path to csv file containing data

    """

    return Measurement(csv=csv_path)

def test_init_blank():
    """Test initializing blank Measurement class

    """

    data = Measurement()

def test_init_csv(csv_path):
    """Test initializing Measurement class with csv

    Parameters
    ----------
    csv_path : string
        Path to csv file containing data

    """

    data = Measurement(csv=csv_path)

def test_init_pd(pandas_df):
    """Test initializing Measurement class with pandas dataframe

    Parameters
    ----------
    pd_df : pd.DataFrame
        Pandas DataFrame containing data

    """

    data = Measurement(pandas=pandas_df)


def test_init_np(numpy_array):
    """Test initializing Measurement class with numpy array

    Parameters
    ----------
    np_array : np.ndarray
        Numpy array containing data

    """

    data = Measurement(numpy=numpy_array)
