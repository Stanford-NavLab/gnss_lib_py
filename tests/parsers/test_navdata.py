"""Tests for NavData class.

"""

__authors__ = "A. Kanhere, D. Knowles"
__date__ = "30 Apr 2022"


import os

import pytest
import itertools
import numpy as np
import pandas as pd
from pytest_lazyfixture import lazy_fixture

from gnss_lib_py.parsers.navdata import NavData

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

def test_init_blank():
    """Test initializing blank NavData class

    """

    data = NavData()

    # NavData should be empty
    assert data.shape == (0,0)

@pytest.mark.parametrize('csv_path',
                        [
                         lazy_fixture("csv_simple"),
                         lazy_fixture("csv_missing"),
                         lazy_fixture("csv_mixed"),
                         lazy_fixture("csv_nan"),
                         lazy_fixture("csv_inf"),
                         lazy_fixture("csv_int_first"),
                        ])
def test_init_csv(csv_path):
    """Test initializing NavData class with csv

    Parameters
    ----------
    csv_path : string
        Path to csv file containing data

    """

    # should work when csv is passed
    data = NavData(csv_path=csv_path)

    # data should contain full csv
    assert data.shape == (4,6)


    # raises exception if not a file path
    with pytest.raises(OSError):
        data = NavData(csv_path="")

    # raises exception if input int
    with pytest.raises(TypeError):
        data = NavData(csv_path=1)

    # raises exception if input float
    with pytest.raises(TypeError):
        data = NavData(csv_path=1.2)

    # raises exception if input list
    with pytest.raises(TypeError):
        data = NavData(csv_path=[])

    # raises exception if input numpy ndarray
    with pytest.raises(TypeError):
        data = NavData(csv_path=np.array([0]))

    # raises exception if input pandas dataframe
    with pytest.raises(TypeError):
        data = NavData(csv_path=pd.DataFrame([0]))

@pytest.mark.parametrize('pandas_df',
                        [
                         lazy_fixture("df_simple"),
                         lazy_fixture("df_missing"),
                         lazy_fixture("df_mixed"),
                         lazy_fixture("df_inf"),
                         lazy_fixture("df_nan"),
                         lazy_fixture("df_int_first"),
                        ])
def test_init_pd(pandas_df):
    """Test initializing NavData class with pandas dataframe

    Parameters
    ----------
    pandas_df : pd.DataFrame
        Pandas DataFrame containing data

    """

    # should work if pass in pandas dataframe
    data = NavData(pandas_df=pandas_df)

    # data should contain full pandas data
    assert data.shape == (4,6)

    # raises exception if input int
    with pytest.raises(TypeError):
        data = NavData(pandas_df=1)

    # raises exception if input float
    with pytest.raises(TypeError):
        data = NavData(pandas_df=1.2)

    # raises exception if input string
    with pytest.raises(TypeError):
        data = NavData(pandas_df="")

    # raises exception if input list
    with pytest.raises(TypeError):
        data = NavData(pandas_df=[])

    # raises exception if input numpy ndarray
    with pytest.raises(TypeError):
        data = NavData(pandas_df=np.array([0]))

def test_init_headless(csv_headless, df_headless):
    """Test that headless csvs and dataframes can be loaded as expected.

    """
    # headless should still work with CSVs with header=None
    data = NavData(csv_path=csv_headless, header=None)
    assert data.shape == (4,6)

    data = NavData(pandas_df=df_headless)
    assert data.shape == (4,6)

    # should fail if you don't add the header=None variable
    data = NavData(csv_path=csv_headless)
    assert data.shape != (4,6)

def test_init_np(numpy_array):
    """Test initializing NavData class with numpy array

    Parameters
    ----------
    numpy_array : np.ndarray
        Numpy array containing data

    """

    # should work if input numpy ndarray
    data = NavData(numpy_array=numpy_array)

    # data should contain full data
    assert data.shape == (4,6)

    # raises exception if input int
    with pytest.raises(TypeError):
        data = NavData(numpy_array=1)

    # raises exception if input float
    with pytest.raises(TypeError):
        data = NavData(numpy_array=1.2)

    # raises exception if input string
    with pytest.raises(TypeError):
        data = NavData(numpy_array="")

    # raises exception if input list
    with pytest.raises(TypeError):
        data = NavData(numpy_array=[])

    # raises exception if input pandas dataframe
    with pytest.raises(TypeError):
        data = NavData(numpy_array=pd.DataFrame([0]))

def test_init_only_header(csv_only_header, csv_simple):
    """Test initializing NavData class with csv with only header

    Parameters
    ----------
    csv_only_header : string
        Path to csv file containing headers, but no data
    csv_simple : string
        Path to csv file headers and data

    """

    # should work when csv is passed
    csv_data = NavData(csv_path=csv_only_header)
    assert csv_data.shape == (4,0)
    # test adding new data to empty NavData with column names
    csv_data.add(csv_path=csv_simple)
    assert csv_data.shape == (4,6)

    # should work when DataFrame is passed
    pd_data = NavData(pandas_df=pd.read_csv(csv_only_header))
    assert pd_data.shape == (4,0)
    # test adding new data to empty NavData with column names
    pd_data.add(pandas_df=pd.read_csv(csv_simple))
    assert pd_data.shape == (4,6)

@pytest.mark.parametrize('pandas_df',
                        [
                         lazy_fixture("df_simple"),
                        ])
def test_rename(pandas_df):
    """Test column renaming functionality.

    Parameters
    ----------
    data : gnss_lib_py.parsers.navdata.NavData
        test data

    """
    data = NavData(pandas_df=pandas_df)

    data.rename({"names": "terms"})
    assert "names" not in data.map
    assert "names" not in data.str_map
    assert "terms" in data.map
    assert "terms" in data.str_map

    data.rename({"floats": "decimals", "integers": "numbers"})
    assert "floats" not in data.map
    assert "floats" not in data.str_map
    assert "integers" not in data.map
    assert "integers" not in data.str_map
    assert "numbers" in data.map
    assert "numbers" in data.str_map
    assert "decimals" in data.map
    assert "decimals" in data.str_map

    # raises exception if input is not string
    with pytest.raises(TypeError):
        data.rename({"names": 0})
    with pytest.raises(TypeError):
        data.rename({"names": 0.8})

    # should raise error if key doesn't exist
    with pytest.raises(KeyError):
        data.rename({"food": "test"})

@pytest.fixture(name="df_rows",
                params=[
                        lazy_fixture("df_simple")
                ])
def return_df_rows(request):
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
    pandas_df = request.param
    names = np.asarray(pandas_df['names'].values, dtype=object)
    integers = np.reshape(np.asarray(pandas_df['integers'].values,
                                     dtype=np.float64), [1, -1])
    floats = np.reshape(np.asarray(pandas_df['floats'].values,
                                   dtype=np.float64), [1, -1])
    strings = np.asarray(pandas_df['strings'].values, dtype=object)
    return [names, integers, floats, strings]


@pytest.fixture(name="integers")
def fixture_integers(df_rows):
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
def fixture_floats(df_rows):
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
def fixture_strings(df_rows):
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
def fixture_int_flt(df_rows):
    """Return data corresponding to the integers and floats.

    Labeled from the test data.

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
def fixture_nm_str(df_rows):
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
def fixture_str_nm(df_rows):
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
def fixture_flt_int_slc(df_rows):
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
                        (('integers', slice(None, None)),
                          lazy_fixture('integers')),
                        (['integers', 'floats'], lazy_fixture('int_flt')),
                        (('integers', 0), 10.),
                        (('strings', 0), np.asarray([['gps']], dtype=object)),
                        (['names', 'strings'], lazy_fixture('nm_str')),
                        (['strings', 'names'], lazy_fixture('str_nm')),
                        ((['integers', 'floats'], slice(3, None)),
                           lazy_fixture('flt_int_slc')),
                        (1, lazy_fixture('integers'))
                        ])
def test_get_item(data, index, exp_value):
    """Test if assigned value is same as original value given for assignment

    Parameters
    ----------
    data : gnss_lib_py.parsers.navdata.NavData
        Data to test getting values from
    index : slice/str/int/tuple
        Index to query data at
    exp_value : np.ndarray
        Expected value at queried indices
    """
    np.testing.assert_array_equal(data[index], np.squeeze(exp_value))


def test_get_all_numpy(numpy_array):
    """Test get all method using slices for NavData

    Parameters
    ----------
    numpy_array : np.ndarray
        Array to initialize NavData
    """
    data = NavData(numpy_array=numpy_array)
    np.testing.assert_array_almost_equal(data[:], numpy_array)
    np.testing.assert_array_almost_equal(data[:, :], numpy_array)



@pytest.fixture(name="new_string")
def fixture_new_string():
    """String to test for value assignment

    Returns
    -------
    new_string : np.ndarray
        String of length 6 to test string assignment
    """
    new_string = np.asarray(['apple', 'banana', 'cherry',
                             'date', 'pear', 'lime'], dtype=object)
    return new_string

@pytest.fixture(name="new_string_str")
def fixture_new_string_string_type():
    """String to test for value assignment

    Returns
    -------
    new_string : np.ndarray
        String of length 6 to test string assignment
    """
    new_string = np.asarray(['pie', 'cake', 'sherbert',
                             'cookies', 'cupcake', 'brownies'],dtype=str)
    return new_string

@pytest.fixture(name="new_string_unicode")
def fixture_new_string_unicode_type():
    """String to test for value assignment

    Returns
    -------
    new_string : np.ndarray
        String of length 6 to test string assignment
    """
    new_string = np.asarray(['red', 'orange', 'yellow',
                             'green', 'blue', 'purple'])
    return new_string


@pytest.fixture(name="new_str_list")
def fixture_new_str_list(new_string):
    """String to test for value assignment

    Returns
    -------
    new_string_2d : np.ndarray
        String of shape [1,6], expected value after string assignment
    """
    new_string_2d = np.reshape(new_string, [1, -1])
    return new_string_2d


@pytest.fixture(name="subset_str")
def fixture_subset_str():
    """Subset string to test for value assignment

    Returns
    -------
    subset_string : np.ndarray
        String of length 6 expected value after string assignment
    """
    subset_str = np.asarray(['gps', 'glonass', 'beidou'], dtype=object)
    return subset_str


@pytest.fixture(name="subset_str_list")
def fixture_subsect_str_list(subset_str):
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
                        ('new_str_key', lazy_fixture('new_string'),
                         lazy_fixture('new_str_list')),
                        ('new_str_key', lazy_fixture('new_string_str'),
                         lazy_fixture('new_string_str')),
                        ('new_str_key', lazy_fixture('new_string_unicode'),
                         lazy_fixture('new_string_unicode')),
                        ('integers', 0, np.zeros([1,6])),
                        (1, 7, 7*np.ones([1,6])),
                        ('names', lazy_fixture('new_string'),
                         lazy_fixture('new_str_list')),
                        ('names', lazy_fixture('new_string_str'),
                         lazy_fixture('new_string_str')),
                        ('names', lazy_fixture('new_string_unicode'),
                         lazy_fixture('new_string_unicode')),
                        ((['integers', 'floats'], slice(1, 4)), -10,
                         -10*np.ones([2,3])),
                        (('strings', slice(2, 5)),
                         lazy_fixture('subset_str'),
                         lazy_fixture('subset_str_list')),
                        ])
def test_set_get_item(data, index, new_value, exp_value):
    """Test if assigned values match expected values on getting again

    Parameters
    ----------
    data : gnss_lib_py.parsers.navdata.NavData
        NavData instance for testing
    index : slice/str/int/tuple
        Index to query data at
    new_value: np.ndarray/int
        Value to assign at query index
    exp_value : np.ndarray
        Expected value at queried indices
    """
    data[index] = new_value
    np.testing.assert_array_equal(data[index], np.squeeze(exp_value))

def test_multi_set(data,new_string):
    """Test setting a numeric row with strings and vice versa.

    Parameters
    ----------
    data : gnss_lib_py.parsers.navdata.NavData
        NavData instance for testing
    new_string : np.ndarray
        String of length 6 to test string assignment

    """
    new_numeric = np.arange(len(data),dtype=float)
    data_temp1 = data.copy()

    # test numerics with input of size (2,6)
    double_numeric_input = np.vstack((new_numeric.reshape(1,-1),
                                        new_numeric.reshape(1,-1) + 10.5))
    data_temp1[["integers","floats"]] = double_numeric_input

    np.testing.assert_array_equal(data_temp1["integers"], new_numeric)
    np.testing.assert_array_equal(data_temp1["floats"], new_numeric+10.5)

    # test strings with input of size (2,6)
    double_string_input = np.vstack((new_string.reshape(1,-1),
                                     new_string.reshape(1,-1)))
    data_temp1[["strings","names"]] = double_string_input

    np.testing.assert_array_equal(data_temp1["strings"], new_string)
    np.testing.assert_array_equal(data_temp1["names"], new_string)

    data_temp2 = data.copy()

    with pytest.raises(ValueError):
        # NavData does not expect values with rows and columns
        # interchanged. Shapes must be set to what the underlying array
        # expects. This (and the following) test verifies that
        # test numerics with input of size (6,2)
        double_numeric_input = np.vstack((new_numeric.reshape(1,-1),
                                        new_numeric.reshape(1,-1))).T
        data_temp2[["integers","floats"]] = double_numeric_input

        np.testing.assert_array_equal(data_temp2["integers"], new_numeric)
        np.testing.assert_array_equal(data_temp2["floats"], new_numeric)

    with pytest.raises(ValueError):
        # test strings with input of size (6,2)
        double_string_input = np.vstack((new_string.reshape(1,-1),
                                        new_string.reshape(1,-1))).T
        data_temp2[["strings","names"]] = double_string_input

        np.testing.assert_array_equal(data_temp2["strings"], new_string)
        np.testing.assert_array_equal(data_temp2["names"], new_string)

def test_set_changing_type(data,new_string):
    """Test setting a numeric row with strings and vice versa.

    Parameters
    ----------
    data : gnss_lib_py.parsers.navdata.NavData
        NavData instance for testing
    new_string : np.ndarray
        String of length 6 to test string assignment

    """

    new_numeric = np.arange(len(data),dtype=float)

    data_temp1 = data.copy()

    # setting strings with strings
    data_temp1["strings"] = new_string
    np.testing.assert_array_equal(data_temp1["strings"], new_string)
    data_temp1["names"] = np.array(new_string,dtype=object)
    np.testing.assert_array_equal(data_temp1["names"], new_string)

    # should raise error trying to set with list of strings
    with pytest.raises(RuntimeError):
        data_temp1["names"] = np.array(new_string).tolist()

    # setting numerics with numerics
    data_temp1["integers"] = new_numeric
    np.testing.assert_array_equal(data_temp1["integers"], new_numeric)
    data_temp1["floats"] = np.array(new_numeric, dtype=object)
    np.testing.assert_array_equal(data_temp1["floats"], new_numeric)

    data_temp2 = data.copy()
    # setting numerics with strings
    data_temp2["integers"] = new_string
    np.testing.assert_array_equal(data_temp2["integers"], new_string)
    data_temp2["floats"] = np.array(new_string,dtype=object)
    np.testing.assert_array_equal(data_temp2["floats"], new_string)

    # should raise error trying to set with list of strings
    with pytest.raises(RuntimeError):
        data_temp1["floats"] = new_string.tolist()

    # setting strings with numerics
    data_temp2["strings"] = new_numeric
    np.testing.assert_array_equal(data_temp2["strings"], new_numeric)
    data_temp2["names"] = np.array(new_numeric, dtype=object)
    np.testing.assert_array_equal(data_temp2["names"], new_numeric)

def test_multi_set_changing_type(data,new_string):
    """Test setting a numeric row with strings and vice versa.

    Parameters
    ----------
    data : gnss_lib_py.parsers.navdata.NavData
        NavData instance for testing
    new_string : np.ndarray
        String of length 6 to test string assignment

    """
    new_numeric = np.arange(len(data),dtype=float)
    data_temp1 = data.copy()

    # test setting strings to numerics with input of size (2,6)
    double_numeric_input = np.vstack((new_numeric.reshape(1,-1),
                                      new_numeric.reshape(1,-1)))
    data_temp1[["strings","names"]] = double_numeric_input

    np.testing.assert_array_equal(data_temp1["strings"], new_numeric)
    np.testing.assert_array_equal(data_temp1["names"], new_numeric)

    # test setting numerics to strings with input of size (2,6)
    double_string_input = np.vstack((new_string.reshape(1,-1),
                                     new_string.reshape(1,-1)))
    data_temp1[["integers","floats"]] = double_string_input

    np.testing.assert_array_equal(data_temp1["integers"], new_string)
    np.testing.assert_array_equal(data_temp1["floats"], new_string)

    data_temp2 = data.copy()

@pytest.mark.parametrize("row_idx",
                        [slice(7, 8),
                        8])
def test_wrong_init_set(data, row_idx):
    empty_data = NavData()
    with pytest.raises(KeyError):
        empty_data[row_idx] = np.zeros([1, 6])

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
                'integers': np.asarray([-2., 45.]),
                'floats': np.asarray([1.4, 1.5869]),
                'strings': np.asarray(['glonass', 'beidou'], dtype=object)}
    add_df = pd.DataFrame(data=add_data)
    return add_df


def test_add_numpy(numpy_array, add_array):
    """Test addition of a numpy array to NavData

    Parameters
    ----------
    numpy_array : np.ndarray
        Array with which NavData instance is initialized
    add_array : np.ndarray
        Array to add to NavData
    """
    data = NavData(numpy_array=numpy_array)
    data.add(numpy_array=add_array)
    new_col_num = np.shape(add_array)[1]
    np.testing.assert_array_equal(data[:, -new_col_num:], add_array)


def test_add_numpy_1d():
    """Test addition of a 1D numpy array to NavData with single row
    """
    data = NavData(numpy_array=np.zeros([1,6]))
    data.add(numpy_array=np.ones(8))
    np.testing.assert_array_equal(data[0, :], np.hstack((np.zeros(6),
                                  np.ones(8))))

    # test adding to empty NavData
    data_empty = NavData()
    data_empty.add(numpy_array=np.ones((8,8)))
    np.testing.assert_array_equal(data_empty[:,:],np.ones((8,8)))

def test_add_csv(df_simple, csv_simple):
    # Create and add to NavData
    data = NavData(csv_path=csv_simple)
    data.add(csv_path=csv_simple)
    data_df = data.pandas_df()
    # Set up dataframe for comparison
    df_types = {'names': object, 'integers': np.float64,
                'floats': np.float64, 'strings': object}
    expected_df = pd.concat((df_simple,df_simple)).reset_index(drop=True)
    expected_df = expected_df.astype(df_types)
    pd.testing.assert_frame_equal(data_df, expected_df,
                                  check_index_type=False)

    # test adding to empty NavData
    data_empty = NavData()
    data_empty.add(csv_path=csv_simple)
    pd.testing.assert_frame_equal(data_empty.pandas_df(),
                                  df_simple.astype(df_types),
                                  check_index_type=False)

def test_add_pandas_df(df_simple, add_df):
    """Test addition of a pd.DataFrame to NavData

    Parameters
    ----------
    df_simple : pd.DataFrame
        pd.DataFrame to initialize NavData with
    add_df : pd.DataFrame
        pd.DataFrame to add to NavData
    """
    data = NavData(pandas_df=df_simple)
    data.add(pandas_df=add_df)
    new_df = data.pandas_df()
    add_row_num = add_df.shape[0]
    subset_df = new_df.iloc[-add_row_num:, :].reset_index(drop=True)
    pd.testing.assert_frame_equal(subset_df, add_df,
                                  check_index_type=False)

    # test adding to empty NavData
    data_empty = NavData()
    data_empty.add(pandas_df=add_df)
    pd.testing.assert_frame_equal(add_df, data_empty.pandas_df(),
                                  check_index_type=False)

@pytest.mark.parametrize("rows",
                        [None,
                        ['names', 'integers', 'floats', 'strings'],
                        np.asarray(['names', 'integers', 'floats',
                                    'strings'], dtype=object),
                        ['names', 'integers'],
                        np.asarray(['names', 'integers'], dtype=object),
                        [0, 1]
                        ])
@pytest.mark.parametrize("cols",
                        [None,
                        np.arange(6),
                        list(np.arange(6)),
                        [0,1],
                        np.asarray([0,1])
                        ])
def test_copy_navdata(data, df_simple, rows, cols):
    """Test methods to subsets and copies of NavData instance

    Parameters
    ----------
    data : gnss_lib_py.parsers.navdata.NavData
        Instance of NavData
    df_simple : pd.DataFrame
        Dataframe that is sliced to compare copies against
    rows : list/np.ndarray
        Rows to keep in copy
    cols : list/np.ndarray
        Columns to keep in copy
    """
    new_data = data.copy(rows=rows, cols=cols)
    new_df = new_data.pandas_df().reset_index(drop=True)
    if rows is None:
        rows = ['names', 'integers', 'floats', 'strings']
    if cols is None:
        cols = np.arange(6)
    if isinstance(rows[0], str):
        subset_df = df_simple.loc[cols, rows]
    else:
        subset_df = df_simple.iloc[cols, rows]
    subset_df = subset_df.reset_index(drop=True)
    pd.testing.assert_frame_equal(new_df, subset_df, check_dtype=False)


@pytest.mark.parametrize("rows",
                        [None,
                        ['names', 'integers'],
                        np.asarray(['names', 'integers'], dtype=object),
                        ('names', 'integers'),
                        [0, 1],
                        [],
                        [6],
                        [0,3,4],
                        ["howdy"],
                        ["names","fake"]
                        ])
@pytest.mark.parametrize("cols",
                        [None,
                        [0,1],
                        np.asarray([0,1]),
                        (0,1),
                        [12],
                        [1,2,6]
                        ])
def test_remove_navdata(data, df_simple, rows, cols):
    """Test method to remove rows and columns from navdata

    Parameters
    ----------
    data : gnss_lib_py.parsers.navdata.NavData
        Instance of NavData
    df_simple : pd.DataFrame
        Dataframe that is sliced to compare copies against
    rows : list/np.ndarray/tuple
        Rows to remove from NavData
    cols : list/np.ndarray/tuple
        Columns to remove from NavData
    """

    all_rows = ['names', 'integers', 'floats', 'strings']

    expect_fail = False
    if rows is not None and len(rows) != 0 and isinstance(rows[0], int):
        if max(rows) >= len(all_rows):
            expect_fail = True
            expect_message = str(max(rows))
    elif not expect_fail and rows is not None:
        for row in rows:
            if row not in all_rows:
                expect_fail = True
                expect_message = row
                break
    if not expect_fail and cols is not None:
        for col in cols:
            if col >= len(df_simple):
                expect_fail = True
                expect_message = str(col)
                break

    if expect_fail:
        with pytest.raises(KeyError) as excinfo:
            new_data = data.remove(rows=rows, cols=cols)
        assert expect_message in str(excinfo.value)
    else:
        new_data = data.remove(rows=rows, cols=cols)
        new_df = new_data.pandas_df().reset_index(drop=True)

        inv_map = {0 : 'names',
                    1 : 'integers',
                    2 : 'floats',
                    3 : 'strings'}
        all_cols = np.arange(6)
        if rows is None:
            rows = []
        if cols is None:
            cols = []
        if len(rows)!=0:
            if not isinstance(rows[0], str):
                int_rows = rows
                rows = []
                for row_idx in int_rows:
                    rows.append(inv_map[row_idx])
        keep_rows = [row for row in all_rows if row not in rows]
        keep_cols = [col for col in all_cols if col not in cols]
        subset_df = df_simple.loc[keep_cols, keep_rows]

        subset_df = subset_df.reset_index(drop=True)
        pd.testing.assert_frame_equal(new_df, subset_df, check_dtype=False)

@pytest.mark.parametrize("rows",
                        [None,
                        ['names', 'integers'],
                        np.asarray(['names', 'integers'], dtype=object),
                        ('names', 'integers'),
                        [0, 1],
                        [],
                        [6],
                        [0,3,4],
                        ["howdy"],
                        ["names","fake"]
                        ])
@pytest.mark.parametrize("cols",
                        [None,
                        [0,1],
                        np.asarray([0,1]),
                        (0,1),
                        [12],
                        [1,2,6]
                        ])
def test_remove_inplace(data, df_simple, rows, cols):
    """Test method to remove rows and columns from navdata

    Parameters
    ----------
    data : gnss_lib_py.parsers.navdata.NavData
        Instance of NavData
    df_simple : pd.DataFrame
        Dataframe that is sliced to compare copies against
    rows : list/np.ndarray
        Rows to remove from NavData
    cols : list/np.ndarray
        Columns to remove from NavData

    """
    all_rows = ['names', 'integers', 'floats', 'strings']

    expect_fail = False
    if rows is not None and len(rows) != 0 and isinstance(rows[0], int):
        if max(rows) >= len(all_rows):
            expect_fail = True
            expect_message = str(max(rows))
    elif not expect_fail and rows is not None:
        for row in rows:
            if row not in all_rows:
                expect_fail = True
                expect_message = row
                break
    if not expect_fail and cols is not None:
        for col in cols:
            if col >= len(df_simple):
                expect_fail = True
                expect_message = str(col)
                break

    if expect_fail:
        with pytest.raises(KeyError) as excinfo:
            new_data = data.remove(rows=rows, cols=cols, inplace=True)
        assert expect_message in str(excinfo.value)
    else:
        new_data = data.copy()
        new_data.remove(rows=rows, cols=cols, inplace=True)
        new_df = new_data.pandas_df().reset_index(drop=True)

        inv_map = {0 : 'names',
                    1 : 'integers',
                    2 : 'floats',
                    3 : 'strings'}
        all_cols = np.arange(6)
        if rows is None:
            rows = []
        if cols is None:
            cols = []
        if len(rows)!=0:
            if not isinstance(rows[0], str):
                int_rows = rows
                rows = []
                for row_idx in int_rows:
                    rows.append(inv_map[row_idx])
        keep_rows = [row for row in all_rows if row not in rows]
        keep_cols = [col for col in all_cols if col not in cols]
        subset_df = df_simple.loc[keep_cols, keep_rows]

        subset_df = subset_df.reset_index(drop=True)
        pd.testing.assert_frame_equal(new_df, subset_df, check_dtype=False)

def test_where_str(csv_simple):
    """Testing implementation of NavData.where for string values

    Parameters
    ----------
    csv_simple : str
        Path to csv file used to create NavData
    """
    data = NavData(csv_path=csv_simple)
    data_small = data.where('strings', 'gps')
    compare_df = data.pandas_df()
    compare_df = compare_df[compare_df['strings']=="gps"].reset_index(drop=True)
    pd.testing.assert_frame_equal(data_small.pandas_df(), compare_df)


def test_where_numbers(csv_simple):
    """Testing implementation of NavData.where for numeric values

    Parameters
    ----------
    csv_simple : str
        Path to csv file used to create NavData
    """
    data = NavData(csv_path=csv_simple)
    conditions = ["eq", "leq", "geq", "greater", "lesser", "between"]
    values = [98, 10, 250, 67, 45, [30, 80]]
    pd_rows = [[4], [0,1], [5], [4, 5], [0, 1], [2, 3]]
    for idx, condition in enumerate(conditions):
        data_small = data.where("integers", values[idx], condition=condition)
        compare_df = data.pandas_df()
        compare_df = compare_df.iloc[pd_rows[idx], :].reset_index(drop=True)
        pd.testing.assert_frame_equal(data_small.pandas_df(), compare_df)


def test_where_errors(csv_simple):
    """Testing error cases for NavData.where

    Parameters
    ----------
    csv_simple : str
        Path to csv file used to create NavData
    """
    data = NavData(csv_path=csv_simple)
    # Test where with multiple rows
    with pytest.raises(NotImplementedError):
        _ = data.where(["integers", "floats"], 10, condition="leq")
    # Test non-equality condition with strings
    with pytest.raises(ValueError):
        _ = data.where("names", "ab", condition="leq")
    with pytest.raises(ValueError):
        _ = data.where("integers", 10, condition="eq_sqrt")

    # Test condition that is not defined


def test_time_looping(csv_simple):
    """Testing implementation to loop over times

    Parameters
    ----------
    csv_simple : str
        path to csv file used to create NavData
    """
    data = NavData(csv_path=csv_simple)
    data['times'] = np.hstack((np.zeros([1, 2]),
                            1.0001*np.ones([1, 1]),
                            1.0003*np.ones([1,1]),
                            1.50004*np.ones([1, 1]),
                            1.499999*np.ones([1,1])))
    compare_df = data.pandas_df()
    count = 0
    for delta_t, measure in data.loop_time('times'):
        if count == 0:
            np.testing.assert_almost_equal(delta_t, 0)
            row_num = [0,1]
        elif count == 1:
            np.testing.assert_almost_equal(delta_t, 1)
            row_num = [2,3]
        elif count == 2:
            np.testing.assert_almost_equal(delta_t, 0.5)
            row_num = [4,5]
        small_df = measure.pandas_df().reset_index(drop=True)
        expected_df = compare_df.iloc[row_num, :].reset_index(drop=True)
        pd.testing.assert_frame_equal(small_df, expected_df,
                                      check_index_type=False)
        count += 1


def test_col_looping(csv_simple):
    """Testing implementation to loop over columns in NavData

    Parameters
    ----------
    csv_simple : str
        path to csv file used to create NavData
    """
    data = NavData(csv_path=csv_simple)
    compare_df = data.pandas_df()
    for idx, col in enumerate(data):
        col_df = col.pandas_df().reset_index(drop=True)
        expected_df = compare_df.iloc[[idx], :].reset_index(drop=True)
        pd.testing.assert_frame_equal(col_df, expected_df,
                                      check_index_type=False)

def test_is_str(df_simple):
    """Test the is_str function.

    Parameters
    ----------
    df_simple : pd.DataFrame
        Simple pd.DataFrame with which to initialize NavData.

    """
    navdata = NavData(pandas_df=df_simple)

    # check on simple dataframe rows
    assert navdata.is_str("names")
    assert not navdata.is_str("integers")
    assert not navdata.is_str("floats")
    assert navdata.is_str("strings")

    # should raise error if key doesn't exist
    with pytest.raises(KeyError):
        navdata.is_str("bananas")

    with pytest.raises(KeyError):
        navdata.is_str(0)

def test_str_navdata(df_simple):
    """Test that the NavData class can be printed without errors

    Parameters
    ----------
    df_simple : pd.DataFrame
        Dataframe with which to construct NavData instance
    """
    navdata = NavData(pandas_df=df_simple)
    navdata_str = str(navdata)
    # Conversion from int to float in DataFrame for consistency
    df_simple = df_simple.astype({'integers': 'float64'})
    df_str = str(df_simple)
    assert navdata_str==df_str

def test_in_rows_single(data):
    """Test the in_rows function.

    Parameters
    ----------
    data : gnss_lib_py.parsers.navdata.NavData
        Instance of NavData

    """

    # should not throw error
    for row in data.rows:
        data.in_rows(row)

    # check removing a single row
    for row in data.rows:
        data_temp = data.copy()
        data_temp = data_temp.remove(rows=[row])

        ## lists
        # check by passing in multiple rows
        with pytest.raises(KeyError) as excinfo:
            data_temp.in_rows(data.rows)
        assert row in str(excinfo.value)
        # check by passing in single row
        with pytest.raises(KeyError) as excinfo:
            data_temp.in_rows([row])
        assert row in str(excinfo.value)

        ## np.ndarrays
        # check by passing in multiple rows
        with pytest.raises(KeyError) as excinfo:
            data_temp.in_rows(np.array(data.rows))
        assert row in str(excinfo.value)
        # check by passing in single row
        with pytest.raises(KeyError) as excinfo:
            data_temp.in_rows(np.array(row))
        assert row in str(excinfo.value)

        ## tuples
        # check by passing in multiple rows
        with pytest.raises(KeyError) as excinfo:
            data_temp.in_rows(tuple(data.rows))
        assert row in str(excinfo.value)
        # check by passing in single row
        with pytest.raises(KeyError) as excinfo:
            data_temp.in_rows((row))
        assert row in str(excinfo.value)

        ## single value
        # check by passing in single row
        with pytest.raises(KeyError) as excinfo:
            data_temp.in_rows(row)
        assert row in str(excinfo.value)

    # None of these should work
    with pytest.raises(KeyError) as excinfo:
        data_temp.in_rows(1)
    assert "in_rows" in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        data_temp.in_rows(1.)
    assert "in_rows" in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        data_temp.in_rows(np.nan)
    assert "in_rows" in str(excinfo.value)

def test_in_rows_multi(data):
    """Test the in_rows function.

    Parameters
    ----------
    data : gnss_lib_py.parsers.navdata.NavData
        Instance of NavData

    """

    for choice in [2,3]:
        for combo_rows in itertools.combinations(data.rows,choice):

            # should pass without error
            data.in_rows(combo_rows)

            # remove multiple rows
            data_temp = data.copy()
            data_temp = data_temp.remove(rows=combo_rows)

            # list
            with pytest.raises(KeyError) as excinfo:
                data_temp.in_rows(list(combo_rows))
            for row in combo_rows:
                assert row in str(excinfo.value)

            # np.ndarray
            with pytest.raises(KeyError) as excinfo:
                data_temp.in_rows(np.array(combo_rows))
            for row in combo_rows:
                assert row in str(excinfo.value)

            # tuple
            with pytest.raises(KeyError) as excinfo:
                data_temp.in_rows(combo_rows)
            for row in combo_rows:
                assert row in str(excinfo.value)
