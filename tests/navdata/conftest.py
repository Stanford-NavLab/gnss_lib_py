"""Fixtures for NavData class.

"""

__authors__ = "A. Kanhere, D. Knowles"
__date__ = "30 Apr 2022"

import os

import pytest
import numpy as np
import pandas as pd

def fixture_csv_path(csv_filepath):
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
    root_path = os.path.join(root_path, 'data/unit_test/navdata')

    csv_path = os.path.join(root_path, csv_filepath)

    return csv_path

@pytest.fixture(name="csv_simple")
def fixture_csv_simple():
    """csv with simple format.

    """
    return fixture_csv_path("navdata_test_simple.csv")

@pytest.fixture(name="csv_headless")
def fixture_csv_headless():
    """csv without column names.

    """
    return fixture_csv_path("navdata_test_headless.csv")

@pytest.fixture(name="csv_missing")
def fixture_csv_missing():
    """csv with missing entries.

    """
    return fixture_csv_path("navdata_test_missing.csv")

@pytest.fixture(name="csv_mixed")
def fixture_csv_mixed():
    """csv with mixed data types.

    """
    return fixture_csv_path("navdata_test_mixed.csv")

@pytest.fixture(name="csv_inf")
def fixture_csv_inf():
    """csv with infinity values in numeric columns.

    """
    return fixture_csv_path("navdata_test_inf.csv")

@pytest.fixture(name="csv_nan")
def fixture_csv_nan():
    """csv with NaN values in columns.

    """
    return fixture_csv_path("navdata_test_nan.csv")

@pytest.fixture(name="csv_int_first")
def fixture_csv_int_first():
    """csv where first column are integers.

    """
    return fixture_csv_path("navdata_test_int_first.csv")

@pytest.fixture(name="csv_only_header")
def fixture_csv_only_header():
    """csv where there's no data, only columns.

    """
    return fixture_csv_path("navdata_only_header.csv")

@pytest.fixture(name="csv_dtypes")
def fixture_csv_dtypes():
    """csv made up of different data types.

    """
    return fixture_csv_path("navdata_test_dtypes.csv")

def load_test_dataframe(csv_filepath, header="infer"):
    """Create dataframe test fixture.

    """

    data = pd.read_csv(csv_filepath, header=header)

    return data

@pytest.fixture(name='df_simple')
def fixture_df_simple(csv_simple):
    """df with simple format.

    """
    return load_test_dataframe(csv_simple)

@pytest.fixture(name='df_headless')
def fixture_df_headless(csv_headless):
    """df without column names.

    """
    return load_test_dataframe(csv_headless,None)

@pytest.fixture(name='df_missing')
def fixture_df_missing(csv_missing):
    """df with missing entries.

    """
    return load_test_dataframe(csv_missing)

@pytest.fixture(name='df_mixed')
def fixture_df_mixed(csv_mixed):
    """df with mixed data types.

    """
    return load_test_dataframe(csv_mixed)

@pytest.fixture(name='df_inf')
def fixture_df_inf(csv_inf):
    """df with infinity values in numeric columns.

    """
    return load_test_dataframe(csv_inf)

@pytest.fixture(name='df_nan')
def fixture_df_nan(csv_nan):
    """df with NaN values in columns.

    """
    return load_test_dataframe(csv_nan)

@pytest.fixture(name='df_int_first')
def fixture_df_int_first(csv_int_first):
    """df where first column are integers.

    """
    return load_test_dataframe(csv_int_first)

@pytest.fixture(name='df_only_header')
def fixture_df_only_header(csv_only_header):
    """df where only headers given and no data.

    """
    return load_test_dataframe(csv_only_header)

@pytest.fixture(name="data")
def load_test_navdata(df_simple):
    """Creates a NavData instance from df_simple.

    """
    return NavData(pandas_df=df_simple)

@pytest.fixture(name="numpy_array")
def create_numpy_array():
    """Create np.ndarray test fixture.
    """
    test_array = np.array([[1,2,3,4,5,6],
                            [0.5,0.6,0.7,0.8,-0.001,-0.3],
                            [-3.0,-1.2,-100.,-2.7,-30.,-5],
                            [-543,-234,-986,-123,843,1000],
                            ])
    return test_array

@pytest.fixture(name='add_array')
def fixture_add_array():
    """Array added as additional timesteps to NavData from np.ndarray

    Returns
    -------
    add_array : np.ndarray
        Array that will be added to NavData
    """
    add_array = np.hstack((10*np.ones([4,1]), 11*np.ones([4,1])))
    return add_array

@pytest.fixture(name='add_df')
def fixture_add_dataframe():
    """Pandas DataFrame to be added as additional timesteps to NavData

    Returns
    -------
    add_df : pd.DataFrame
        Dataframe that will be added to NavData
    """
    add_data = {'names': np.asarray(['beta', 'alpha'], dtype=object),
                'integers': np.asarray([-2, 45], dtype=np.int64),
                'floats': np.asarray([1.4, 1.5869]),
                'strings': np.asarray(['glonass', 'beidou'], dtype=object)}
    add_df = pd.DataFrame(data=add_data)
    return add_df
