"""Tests for Measurement class.

"""

__authors__ = "A. Kanhere, D. Knowles"
__date__ = "30 Apr 2022"


import os

import pytest
import numpy as np
import pandas as pd
from pytest_lazyfixture import lazy_fixture

from gnss_lib_py.parsers.measurement import Measurement


@pytest.fixture(name="csv_path",
                params=["measurement_test_simple.csv",
                        # "measurement_test_mixed.csv",
                        # "measurement_test_headless.csv",
                       ],
                )
def fixture_csv_path(request):
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

    csv_path = os.path.join(root_path, request.param)

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


# @pytest.fixture(name="data")
# def create_data_csv(csv_path):
#     """Create test fixture for Measurement from csv
#
#     Parameters
#     ----------
#     csv_path : string
#         Path to csv file containing data
#
#     """
#
#     return Measurement(csv=csv_path)

@pytest.fixture(name="data")
def create_data_pd(pandas_df):
    """Create test fixture for Measurement from pandas dataframe

    Parameters
    ----------
    pandas_df : pd.DataFrame
        Pandas DataFrame containing data

    """

    return Measurement(pandas_df=pandas_df)

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

    # should work when csv is passed
    data = Measurement(csv_path=csv_path)

    # raises exception if not a file path
    with pytest.raises(OSError):
        data = Measurement(csv_path="")

    # raises exception if input int
    with pytest.raises(TypeError):
        data = Measurement(csv_path=1)

    # raises exception if input float
    with pytest.raises(TypeError):
        data = Measurement(csv_path=1.2)

    # raises exception if input list
    with pytest.raises(TypeError):
        data = Measurement(csv_path=[])

    # raises exception if input numpy ndarray
    with pytest.raises(TypeError):
        data = Measurement(csv_path=np.array([0]))

    # raises exception if input pandas dataframe
    with pytest.raises(TypeError):
        data = Measurement(csv_path=pd.DataFrame([0]))


def test_init_pd(pandas_df):
    """Test initializing Measurement class with pandas dataframe

    Parameters
    ----------
    pandas_df : pd.DataFrame
        Pandas DataFrame containing data

    """

    # should work if pass in pandas dataframe
    data = Measurement(pandas_df=pandas_df)

    # raises exception if input int
    with pytest.raises(TypeError):
        data = Measurement(pandas_df=1)

    # raises exception if input float
    with pytest.raises(TypeError):
        data = Measurement(pandas_df=1.2)

    # raises exception if input string
    with pytest.raises(TypeError):
        data = Measurement(pandas_df="")

    # raises exception if input list
    with pytest.raises(TypeError):
        data = Measurement(pandas_df=[])

    # raises exception if input numpy ndarray
    with pytest.raises(TypeError):
        data = Measurement(pandas_df=np.array([0]))


def test_init_np(numpy_array):
    """Test initializing Measurement class with numpy array

    Parameters
    ----------
    numpy_array : np.ndarray
        Numpy array containing data

    """

    # should work if input numpy ndarray
    data = Measurement(numpy_array=numpy_array)

    # raises exception if input int
    with pytest.raises(TypeError):
        data = Measurement(numpy_array=1)

    # raises exception if input float
    with pytest.raises(TypeError):
        data = Measurement(numpy_array=1.2)

    # raises exception if input string
    with pytest.raises(TypeError):
        data = Measurement(numpy_array="")

    # raises exception if input list
    with pytest.raises(TypeError):
        data = Measurement(numpy_array=[])

    # raises exception if input pandas dataframe
    with pytest.raises(TypeError):
        data = Measurement(numpy_array=pd.DataFrame([0]))


def test_rename(data):
    """Test column renaming functionality.

    Parameters
    ----------
    data : gnss_lib_py.parsers.Measurement
        test data

    """
    print("\n")
    print("arr_dtype:\n",data.arr_dtype)
    print("array:\n",data.array)
    print("map:\n",data.map)
    print("str_map:\n",data.str_map)


@pytest.fixture(name="df_rows")
def return_df_rows(pandas_df):
    """Extract and return rows from the DataFrame for testing

    Parameters
    ----------
    pandas_df : pd.DataFrame
        Dataframe for testing values

    Returns
    -------
    names : np.ndarray
        String entries in 'names' column of the DataFrame
    integers : np.ndarray
        Numeric entries in 'integers' column of the DataFrame
    floats : np.ndarray
        Numeric entries in 'floats' column of the DataFrame
    strings : np.ndarray
        String entries in 'strings' column of the DataFrame
    """
    names = np.asarray(pandas_df['names'].values, dtype=object)
    integers = np.reshape(np.asarray(pandas_df['integers'].values, dtype=np.float64), [1, -1])
    floats = np.reshape(np.asarray(pandas_df['floats'].values, dtype=np.float64), [1, -1])
    strings = np.asarray(pandas_df['strings'].values, dtype=object)
    return [names, integers, floats, strings]


@pytest.fixture(name="integers")
def return_integers(df_rows):
    """Return data corresponding to the integers label from the test data

    Parameters
    ----------
    df_rows : list
        List of rows from the testing data
    
    Returns
    -------
    integers : np.ndarray
        Array of numeric entries in 'integers' label of data
    """
    _, integers, _, _ = df_rows
    return integers

@pytest.fixture(name="floats")
def return_floats(df_rows):
    """Return data corresponding to the floats label from the test data

    Parameters
    ----------
    df_rows : list
        List of rows from the testing data
    
    Returns
    -------
    floats : np.ndarray
        Array of numeric entries in 'floats' label of data
    """
    _, _, floats, _ = df_rows
    return floats


@pytest.fixture(name="strings")
def return_strings(df_rows):
    """Return data corresponding to the strings label from the test data

    Parameters
    ----------
    df_rows : list
        List of rows from the testing data
    
    Returns
    -------
    strings : np.ndarray
        Array of string entries in 'strings' label of data
    """
    _, _, _, strings = df_rows
    return strings


@pytest.fixture(name="int_flt")
def return_int_flt(df_rows):
    """Return data corresponding to the integers and floats label from the test data

    Parameters
    ----------
    df_rows : list
        List of rows from the testing data
    
    Returns
    -------
    int_flt : np.ndarray
        2D array of numeric entries in 'integers' and 'floats' labels of data
    """
    _, integers, floats, _ = df_rows
    int_flt = np.vstack((integers, floats))
    return int_flt


@pytest.fixture(name="nm_str")
def return_nm_str(df_rows):
    """Return data corresponding to the names and strings label from the test data

    Parameters
    ----------
    df_rows : list
        List of rows from the testing data
    
    Returns
    -------
    nm_str : np.ndarray
        2D array of numeric entries in 'names' and 'strings' labels of data
    """
    names, _, _, strings = df_rows
    nm_str = np.vstack((names, strings))
    return nm_str


@pytest.fixture(name="str_nm")
def return_str_nm(df_rows):
    """Return data corresponding to the strings and names label from the test data

    Parameters
    ----------
    df_rows : list
        List of rows from the testing data
    
    Returns
    -------
    str_nm : np.ndarray
        2D array of numeric entries in 'strings' and 'names' labels of data
    """
    names, _, _, strings = df_rows
    str_nm = np.vstack((strings, names))
    return str_nm


@pytest.fixture(name="flt_int_slc")
def return_flt_int_slc(df_rows):
    """Return data corresponding to the names and strings label from the test data

    Parameters
    ----------
    df_rows : list
        List of rows from the testing data

    Returns
    -------
    flt_int_slc : np.ndarray
        2D array of some numeric entries in 'integers' and 'floats' labels of data
    """
    _, integers, floats, _ = df_rows
    flt_int_slc = np.vstack((integers, floats))[:, 3:]
    return flt_int_slc


@pytest.mark.parametrize("index, exp_value",
                        [(slice(1, 3, 1), lazy_fixture('int_flt')),
                        (slice(1, 2, 1), lazy_fixture('integers')),
                        ('integers', lazy_fixture('integers')),
                        (('integers', slice(None, None)), lazy_fixture('integers')),
                        (['integers', 'floats'], lazy_fixture('int_flt')),
                        (('integers', 0), 10.),
                        (('strings', 0), np.asarray([['gps']], dtype=object)),
                        (['names', 'strings'], lazy_fixture('nm_str')),
                        (['strings', 'names'], lazy_fixture('str_nm')),
                        ((['integers', 'floats'], slice(3, None)), lazy_fixture('flt_int_slc')),
                        (1, lazy_fixture('integers')),
                        (slice(None, None))
                        ])
def test_get_item(data, index, exp_value):
    """Test if assigned value is same as original value given for assignment

    Parameters
    ----------
    data : gnss_lib_py.parsers.Measurement
        Data to test getting values from
    index : slice/str/int/tuple
        Index to query data at
    exp_value : np.ndarray
        Expected value at queried indices
    """
    np.testing.assert_array_equal(data[index], exp_value)


@pytest.fixture(name="new_string")
def return_new_string():
    """String to test for value assignment

    Returns
    -------
    new_string : np.ndarray
        String of length 6 to test string assignment
    """
    new_string = np.asarray(['apple', 'banana', 'cherry', 'date', 'pear', 'lime'], dtype=object)
    return new_string

@pytest.fixture(name="new_str_list")
def return_new_str_list(new_string):
    """String to test for value assignment

    Returns
    -------
    new_string_2d : np.ndarray
        String of shape [1,6], expected value after string assignment
    """
    new_string_2d = np.reshape(new_string, [1, -1])
    return new_string_2d

@pytest.fixture(name="subset_str")
def return_subset_str():
    """Subset string to test for value assignment

    Returns
    -------
    subset_string : np.ndarray
        String of length 6 expected value after string assignment
    """
    subset_str = np.asarray(['gps', 'glonass', 'beidou'], dtype=object)
    return subset_str


@pytest.fixture(name="subset_str_list")
def return_subsect_str_list(subset_str):
    """Subset string to test for value assignment

    Returns
    -------
    subset_string_2d : np.ndarray
        String of shape [1,3], expected value after string assignment
    """
    subset_str_2d = np.reshape(subset_str, [1, -1])
    return subset_str_2d

@pytest.mark.parametrize("index, new_value, exp_value",
                        [('new_key', 0, np.zeros([1,6])),
                        ('new_key_1d', np.ones(6), np.ones([1,6])),
                        ('new_key_2d_row', np.ones([1,6]), np.ones([1,6])),
                        ('new_key_2d_col', np.ones([6,1]), np.ones([1,6])),
                        ('new_str_key', lazy_fixture('new_string'), lazy_fixture('new_str_list')),
                        ('integers', 0, np.zeros([1,6])),
                        (1, 7, 7*np.ones([1,6])),
                        ('names', lazy_fixture('new_string'), lazy_fixture('new_str_list')),
                        ((['integers', 'floats'], slice(1, 4)), -10, -10*np.ones([2,3])),
                        (('strings', slice(2, 5)), lazy_fixture('subset_str'), lazy_fixture('subset_str_list')),
                        ])
def test_set_get_item(data, index, new_value, exp_value):
    """Test if assigned values match expected values on getting again

    Parameters
    ----------
    data : gnss_lib_py.parsers.Measurement
        Measurement instance for testing
    index : slice/str/int/tuple
        Index to query data at
    new_value: np.ndarray/int
        Value to assign at query index
    exp_value : np.ndarray
        Expected value at queried indices
    """
    data[index] = new_value
    np.testing.assert_array_equal(data[index], exp_value)

