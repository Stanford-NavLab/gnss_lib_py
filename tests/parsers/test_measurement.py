"""Tests for Measurement class.

"""

__authors__ = "A. Kanhere, D. Knowles"
__date__ = "30 Apr 2022"


import os

import pytest
import numpy as np
import pandas as pd

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

@pytest.fixture(name="string_array")
def string_array_to_set():
    # value1 = ['ashwin']*2
    # value2 = ['derek']*2
    value1 = ['ashwin']*3
    value2 = ['derek']*3
    value = np.concatenate((np.asarray(value1, dtype=object), np.asarray(value2, dtype=object)))
    return value


@pytest.fixture(name="val_array")
def number_array_to_set():
    # value = np.asarray([9,8.5,-15,32.33])
    value = np.array([9,8.5,-15,32.33, 10, 20])
    return value

# @pytest.mark.parametrize("data_type", ["string", "vals"])
# @pytest.mark.parametrize("size",
#                          ['1d',
#                           '2d_row',
#                           '2d_col'])
# def test_set_1d_2d(data, data_type, size, string_array, val_array):
#     if data_type=="string":
#         newvalue = string_array
#         compare_value = np.hstack((np.zeros((1,3)), np.ones((1,3))))
#     elif data_type=="vals":
#         newvalue = val_array
#         compare_value = np.reshape(newvalue, [1, len(data)])
#     if size=='1d':
#         newvalue = np.reshape(newvalue, -1)
#     elif size=='2d_row':
#         newvalue = np.reshape(newvalue, [1, -1])
#     elif size=='2d_col':
#         newvalue = np.reshape(newvalue, [-1, 1])
#     data["testing_key"] = newvalue
#     compare_value = np.reshape(compare_value, [1, len(data)])
#     np.testing.assert_equal(data["testing_key", :], compare_value)

def test_get_item(data, pandas_df):
    names = np.reshape(np.asarray(pandas_df['names'].values, dtype=object), [1, -1])
    integers = np.reshape(np.asarray(pandas_df['integers'].values, dtype=data.arr_dtype), [1, -1])
    floats = np.reshape(np.asarray(pandas_df['floats'].values, dtype=data.arr_dtype), [1, -1])
    strings = np.reshape(np.asarray(pandas_df['strings'].values, dtype=object), [1, -1])
    print(strings)
    print(names)
    strings_names = [strings, names]
    names_strings = [names, strings]
    #Slicing only rows, with multiple rows
    np.testing.assert_equal(data[1:3], np.vstack((integers, floats)))
    #Slicing only rows, with single row
    np.testing.assert_equal(data[1:2], integers)
    #String for row look up only
    np.testing.assert_equal(data['integers'], integers)
    #String for row and slice for column
    np.testing.assert_equal(data['integers', :], integers)
    #List of strings for row look up only
    np.testing.assert_equal(data[['integers', 'floats']], np.vstack((integers, floats)))
    #String for row and int for column look up
    np.testing.assert_equal(data['integers', 0], np.asarray([10.]))
    #String for row and int for column for looking up string entries
    np.testing.assert_equal(data['strings', 0], [np.asarray(['gps'], dtype=object)])
    #Looking up multiple rows with string values
    np.testing.assert_equal(data[['names', 'strings']], names_strings)
    #Looking up multiple rows with string values and order different from original
    np.testing.assert_equal(data[['strings', 'names']],strings_names)
    #List of rows and slice for column
    np.testing.assert_equal(data[['integers', 'floats'], 3:],np.vstack((integers, floats))[:, 3:])

def test_get_set_item(data):
    new_string = np.asarray(['apple', 'banana', 'cherry', 'date', 'pear', 'lime'], dtype=object)
    #Creating new row for value assignment
    data['new_key'] = 0
    np.testing.assert_array_equal(data['new_key'], np.zeros([1, 6]))
    #Creating new row for string assignment
    data['new_str_key'] = new_string
    print(data['new_str_key'])
    np.testing.assert_equal(data['new_str_key'],[new_string])
    #Assigning all values using integer
    data['integers', :] = 0
    np.testing.assert_equal(data['integers'], np.zeros([1, 6]))
    #Modifying string values using row name only
    data['names'] =  new_string
    np.testing.assert_equal(data['names'],[new_string])
    #Testing list of strings for rows and slice for columns
    data[['integers', 'floats'], 1:4]=-10
    np.testing.assert_equal(data[['integers', 'floats'], 1:4], -10*np.ones([2,3]))
    #Testing modifying strings with some existing and some new values
    subset_string = np.asarray(['gps', 'glonass', 'beidou'], dtype=object)
    data['strings', 2:5] = subset_string
    np.testing.assert_equal(data['strings', 2:5],[subset_string])

