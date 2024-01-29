"""Tests for NavData class.

"""

__authors__ = "A. Kanhere, D. Knowles"
__date__ = "30 Apr 2022"

import pathlib
import itertools

import pytest
import numpy as np
import pandas as pd
from pytest_lazyfixture import lazy_fixture

from gnss_lib_py.navdata.navdata import NavData

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

    # should work when csv is passed as pathlib object
    data = NavData(csv_path=pathlib.Path(csv_path))
    # data should contain full csv
    assert data.shape == (4,6)


    # raises exception if not a file path
    with pytest.raises(FileNotFoundError):
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


def test_len_rows(data):
    """Test that `len()` and `rows` return correct output.

    Parameters
    ----------
    data : gnss_lib_py.navdata.navdata.NavData
        Simple version of NavData to use for test.
    """
    assert len(data) == 6
    assert data.rows == ['names', 'integers', 'floats', 'strings']


def test_num_cols(data):
    """Test that `num_cols` returns correct output.

    Parameters
    ----------
    data : gnss_lib_py.navdata.navdata.NavData
        Simple version of NavData to use for test."""
    assert data.num_cols == 6



@pytest.mark.parametrize('pandas_df',
                        [
                         lazy_fixture("df_simple"),
                        ])
def test_rename_inplace(pandas_df):
    """Test column renaming functionality.

    Parameters
    ----------
    pandas_df : pd.DataFrame
        Dataframe for testing values

    """
    data = NavData(pandas_df=pandas_df)
    data_temp = data.copy()

    data_temp.rename({"names": "terms"}, inplace=True)
    assert "names" not in data_temp.map
    assert "names" not in data_temp.str_map
    assert "terms" in data_temp.map
    assert "terms" in data_temp.str_map

    data_temp = data.copy()
    data_temp.rename(mapper={"floats": "decimals", "integers": "numbers"},
                inplace = True)
    assert "floats" not in data_temp.map
    assert "floats" not in data_temp.str_map
    assert "integers" not in data_temp.map
    assert "integers" not in data_temp.str_map
    assert "numbers" in data_temp.map
    assert "numbers" in data_temp.str_map
    assert "decimals" in data_temp.map
    assert "decimals" in data_temp.str_map

    # raises exception if input is not string
    data_temp = data.copy()
    with pytest.raises(TypeError):
        data_temp.rename({"names": 0}, inplace=True)
    data_temp = data.copy()
    with pytest.raises(TypeError):
        data_temp.rename({"names": 0.8}, inplace=True)

    # should raise error if key doesn't exist
    data_temp = data.copy()
    with pytest.raises(KeyError):
        data_temp.rename({"food": "test"}, inplace=True)

@pytest.mark.parametrize('pandas_df',
                        [
                         lazy_fixture("df_simple"),
                        ])
def test_rename_new_navdata(pandas_df):
    """Test column renaming functionality.

    Parameters
    ----------
    pandas_df : pd.DataFrame
        Dataframe for testing values

    """
    data = NavData(pandas_df=pandas_df)

    new_navdata = data.rename({"names": "terms"})
    assert "names" not in new_navdata.map
    assert "names" not in new_navdata.str_map
    assert "terms" in new_navdata.map
    assert "terms" in new_navdata.str_map
    # original one shouldn't have changed
    assert "names" in data.map
    assert "names" in data.str_map
    assert "terms" not in data.map
    assert "terms" not in data.str_map

    navdata = data.rename(mapper={"floats": "decimals", "integers": "numbers"})
    assert "floats" not in navdata.map
    assert "floats" not in navdata.str_map
    assert "integers" not in navdata.map
    assert "integers" not in navdata.str_map
    assert "numbers" in navdata.map
    assert "numbers" in navdata.str_map
    assert "decimals" in navdata.map
    assert "decimals" in navdata.str_map
    # original one shouldn't have changed
    assert "floats" in data.map
    assert "floats" in data.str_map
    assert "integers" in data.map
    assert "integers" in data.str_map
    assert "numbers" not in data.map
    assert "numbers" not in data.str_map
    assert "decimals" not in data.map
    assert "decimals" not in data.str_map

    # raises exception if input is not string
    with pytest.raises(TypeError):
        navdata = data.rename({"names": 0})
    with pytest.raises(TypeError):
        navdata = data.rename({"names": 0.8})

    # should raise error if key doesn't exist
    with pytest.raises(KeyError):
        navdata = data.rename({"food": "test"})

def test_replace_fails(df_simple, df_only_header):
    """Test replace renaming functionality.

    Parameters
    ----------
    df_simple : pd.DataFrame
        pd.DataFrame to initialize NavData with.
    df_only_header : pd.DataFrame
        Dataframe with only column names and no data

    """
    data = NavData(pandas_df=df_simple)

    with pytest.raises(TypeError) as excinfo:
        data.replace(mapper=["names","floats"],rows={"names": "terms"})
    assert "mapper" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        data.replace(mapper={"names":"terms"},rows=1.0)
    assert "rows" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        data.replace({"names": "terms"}, inplace=0.)
    assert "inplace" in str(excinfo.value)

    # should raise error if key doesn't exist
    with pytest.raises(KeyError):
        data.replace({"gps":"GPS"},rows={"food": "test"}, inplace=True)

    data = NavData(pandas_df=df_only_header)
    data.replace({"gps":"GPS"},inplace=True)

def test_rename_fails(df_simple, df_only_header):
    """Test column renaming functionality.

    Parameters
    ----------
    df_simple : pd.DataFrame
        pd.DataFrame to initialize NavData with.
    df_only_header : pd.DataFrame
        Dataframe with only column names and no data

    """
    data = NavData(pandas_df=df_simple)

    with pytest.raises(TypeError) as excinfo:
        data.rename(mapper=["names","floats"])
    assert "mapper" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        data.rename(None)
    assert "mapper" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        data.rename(mapper={"names": "terms"}, inplace=0.)
    assert "inplace" in str(excinfo.value)

    data = NavData(pandas_df=df_only_header)
    data.rename({"strings":"words"},inplace=True)
    assert "words" in data.rows
    assert "strings" not in data.rows

@pytest.mark.parametrize('rows',
                        [
                        None,
                        [],
                        ["strings","integers"],
                        ["strings","integers","names","floats"],
                        ["integers","strings"],
                        {},
                        {"strings","integers"},
                        {"strings","integers","names","floats"},
                        {"integers","strings"},
                        (),
                        ("strings","integers"),
                        ("strings","integers","names","floats"),
                        ("integers","strings"),
                        np.array([]),
                        np.array(["strings","integers"]),
                        np.array(["strings","integers","names","floats"]),
                        np.array(["integers","strings"]),
                        ])
def test_replace_mapper_all(df_simple, rows):
    """Test data renaming functionality.

    Parameters
    ----------
    df_simple : pd.DataFrame
        pd.DataFrame to initialize NavData with.
    rows : None or array-like
        Rows for which mapper is implemented.

    """
    data = NavData(pandas_df=df_simple)
    mapper = {"gps":"GPS",
              45 : 46,
              }

    # test that both rows change
    new_navdata = data.replace(mapper, rows=rows)
    np.testing.assert_array_equal(new_navdata["strings"],
                          np.array(["GPS","glonass","galileo","GPS",
                                    "GPS","galileo"]))
    np.testing.assert_array_equal(new_navdata["integers"],
                          np.array([10,2,46,67,98,300]))

    assert new_navdata["names"].dtype == object
    assert np.issubdtype(new_navdata["integers"].dtype, np.integer)
    assert new_navdata["floats"].dtype == np.float64
    assert new_navdata["strings"].dtype == object

    # test that both rows change inplace
    data_temp = data.copy()
    data_temp.replace(mapper, rows=rows, inplace=True)
    np.testing.assert_array_equal(data_temp["strings"],
                          np.array(["GPS","glonass","galileo","GPS",
                                    "GPS","galileo"]))
    np.testing.assert_array_equal(data_temp["integers"],
                          np.array([10,2,46,67,98,300]))

    assert data_temp["names"].dtype == object
    assert np.issubdtype(data_temp["integers"].dtype, np.integer)
    assert data_temp["floats"].dtype == np.float64
    assert data_temp["strings"].dtype == object


@pytest.mark.parametrize('rows',
                        [
                        ["strings","floats","names"],
                        ["floats","names","strings"],
                        ["strings"],
                        {"strings","floats","names"},
                        {"floats","names","strings"},
                        {"strings"},
                        ("strings","floats","names"),
                        ("floats","names","strings"),
                        ("strings"),
                        np.array(["strings","floats","names"]),
                        np.array(["floats","names","strings"]),
                        np.array(["strings"]),
                        ])
def test_replace_mapper_partial(df_simple, rows):
    """Test data renaming functionality.

    Parameters
    ----------
    df_simple : pd.DataFrame
        pd.DataFrame to initialize NavData with.
    rows : array-like
        Rows for which mapper is implemented.

    """
    data = NavData(pandas_df=df_simple)
    mapper = {"gps":"GPS",
              45 : 46,
              }

    # test that only "strings" changes and not "integers"
    new_navdata = data.replace(mapper, rows=rows)
    np.testing.assert_array_equal(new_navdata["strings"],
                          np.array(["GPS","glonass","galileo","GPS",
                                    "GPS","galileo"]))
    np.testing.assert_array_equal(new_navdata["integers"],
                          np.array([10,2,45,67,98,300]))

    assert new_navdata["names"].dtype == object
    assert np.issubdtype(new_navdata["integers"].dtype, np.integer)
    assert new_navdata["floats"].dtype == np.float64
    assert new_navdata["strings"].dtype == object

    # test that both rows change inplace
    data_temp = data.copy()
    data_temp.replace(mapper, rows=rows, inplace=True)
    np.testing.assert_array_equal(data_temp["strings"],
                          np.array(["GPS","glonass","galileo","GPS",
                                    "GPS","galileo"]))
    np.testing.assert_array_equal(data_temp["integers"],
                          np.array([10,2,45,67,98,300]))

    assert data_temp["names"].dtype == object
    assert np.issubdtype(data_temp["integers"].dtype, np.integer)
    assert data_temp["floats"].dtype == np.float64
    assert data_temp["strings"].dtype == object


def test_replace_mapper_type_change(df_simple):
    """Test data renaming functionality with type changes.

    Parameters
    ----------
    df_simple : pd.DataFrame
        pd.DataFrame to initialize NavData with

    """
    data = NavData(pandas_df=df_simple)

    integers_mapper = {2  : "two",
                       10 : "ten",
                       45 : "forty-five",
                       67 : "sixty-seven",
                       98 : "ninety-eight",
                       300: "three-hundred",
                       }
    strings_mapper = {"gps": 1,
                      "galileo" : 2,
                      "glonass" : 3,
                      }

    # rename contents
    data.replace(integers_mapper, inplace=True)
    data.replace(strings_mapper, inplace=True)

    # make sure the rows hold the correct content
    np.testing.assert_array_equal(data["strings"],
                          np.array([1, 3, 2, 1, 1, 2]))
    np.testing.assert_array_equal(data["integers"],
                          np.array(["ten","two","forty-five",
                                    "sixty-seven","ninety-eight",
                                    "three-hundred"]))

    # verify that rows switched types
    assert data["names"].dtype == object
    assert data["integers"].dtype == object
    assert data["floats"].dtype == np.float64
    assert np.issubdtype(data["strings"].dtype, np.integer)

def test_rename_mapper_and_rows(df_simple):
    """Test data renaming functionality with type changes and row names.

    Parameters
    ----------
    df_simple : pd.DataFrame
        pd.DataFrame to initialize NavData with

    """
    data = NavData(pandas_df=df_simple)

    integers_mapper = {2  : "two",
                       10 : "ten",
                       45 : "forty-five",
                       67 : "sixty-seven",
                       98 : "ninety-eight",
                       300: "three-hundred",
                       }
    row_mapper = {"integers" : "number_words"}

    # rename contents
    data.replace(integers_mapper, rows="integers", inplace=True)
    data.rename(row_mapper, inplace=True)

    # make sure the rows hold the correct content
    np.testing.assert_array_equal(data["number_words"],
                          np.array(["ten","two","forty-five",
                                    "sixty-seven","ninety-eight",
                                    "three-hundred"]))

    # verify that rows switched types
    assert data["names"].dtype == object
    assert data["number_words"].dtype == object
    assert data["floats"].dtype == np.float64
    assert data["strings"].dtype == object



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
                        (('integers', 'floats'), lazy_fixture('int_flt')),
                        ((['integers', 'floats'], 0), np.array([10., 0.5])),
                        ((('integers', 'floats'), 0), np.array([10., 0.5])),
                        (('integers', 0), 10.),
                        (('strings', 0), np.asarray([['gps']], dtype=object)),
                        (['names', 'strings'], lazy_fixture('nm_str')),
                        (('names', 'strings'), lazy_fixture('nm_str')),
                        (['strings', 'names'], lazy_fixture('str_nm')),
                        (('strings', 'names'), lazy_fixture('str_nm')),
                        ((['integers', 'floats'], slice(3, None)),
                           lazy_fixture('flt_int_slc')),
                        ((('integers', 'floats'), slice(3, None)),
                           lazy_fixture('flt_int_slc')),
                        (1, lazy_fixture('integers'))
                        ])
def test_get_item(data, index, exp_value):
    """Test if assigned value is same as original value given for assignment

    Parameters
    ----------
    data : gnss_lib_py.navdata.navdata.NavData
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
    data : gnss_lib_py.navdata.navdata.NavData
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
    data : gnss_lib_py.navdata.navdata.NavData
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
    data : gnss_lib_py.navdata.navdata.NavData
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
    data : gnss_lib_py.navdata.navdata.NavData
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

@pytest.mark.parametrize("row_idx",
                        [slice(7, 8),
                        8])
def test_wrong_init_set(row_idx):
    """ Test init with unknown set.

    """
    empty_data = NavData()
    with pytest.raises(KeyError):
        empty_data[row_idx] = np.zeros([1, 6])

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
    data : gnss_lib_py.navdata.navdata.NavData
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
    data : gnss_lib_py.navdata.navdata.NavData
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
    data : gnss_lib_py.navdata.navdata.NavData
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

    data_small = data.where('strings', 'gps','neq')
    compare_df = data.pandas_df()
    compare_df = compare_df[compare_df['strings']!="gps"].reset_index(drop=True)
    pd.testing.assert_frame_equal(data_small.pandas_df(), compare_df)

    #Test equality for cases where there is only one column
    data_single_column = data.where('strings', 'glonass', 'eq')
    data_new = data_single_column.where('strings', 'glonass', 'eq')
    compare_df = data.pandas_df()
    compare_df = compare_df[compare_df['strings']=="glonass"].reset_index(drop=True)
    pd.testing.assert_frame_equal(data_new.pandas_df(), compare_df)

    #Test inequality for cases where there is only one column
    data_new = data_single_column.where('strings', 'gps', 'neq')
    # Both cases should return the same dataframe as before
    pd.testing.assert_frame_equal(data_new.pandas_df(), compare_df)


def test_where_empty(df_simple):
    """Verify empty slices.

    Parameters
    ----------
    df_simple : pd.DataFrame
        Simple pd.DataFrame with which to initialize NavData.

    """
    navdata = NavData(pandas_df=df_simple)
    for row in navdata.rows:
        if navdata.is_str(row):
            # verify where doesn't break on empty call
            subset = navdata.where(row,"not_here!")
            assert subset.shape == (4,0)

@pytest.mark.parametrize('csv_path',
                        [
                         lazy_fixture("csv_missing"),
                         lazy_fixture("csv_nan"),
                        ])
def test_argwhere_nan(csv_path):
    """Test where options on nan values.

    Parameters
    ----------
    csv_path : string
        Path to csv file containing data

    """
    navdata = NavData(csv_path=csv_path)
    pd_df = pd.read_csv(csv_path)

    for row in navdata.rows:
        navdata_where = navdata.argwhere(row,np.nan,"eq")
        pd_where = np.argwhere(pd.isnull(pd_df[row]).to_numpy())
        np.testing.assert_array_equal(navdata_where,np.squeeze(pd_where))

        navdata_where = navdata.argwhere(row,np.nan,"neq")
        pd_where = np.argwhere(~pd.isnull(pd_df[row]).to_numpy())
        np.testing.assert_array_equal(navdata_where,np.squeeze(pd_where))

def test_where_numbers(csv_simple):
    """Testing implementation of NavData.where for numeric values

    Parameters
    ----------
    csv_simple : str
        Path to csv file used to create NavData
    """
    data = NavData(csv_path=csv_simple)
    conditions = ["eq", "neq", "leq", "geq", "greater", "lesser", "between"]
    values = [98, 98, 10, 250, 67, 45, [30, 80]]
    pd_rows = [[4], [0,1,2,3,5], [0,1], [5], [4, 5], [0, 1], [2, 3]]
    for idx, condition in enumerate(conditions):
        data_small = data.where("integers", values[idx], condition=condition)
        compare_df = data.pandas_df()
        compare_df = compare_df.iloc[pd_rows[idx], :].reset_index(drop=True)
        pd.testing.assert_frame_equal(data_small.pandas_df(), compare_df)

    #Test equality for cases where there is only one column
    data_single_column = data.where('integers', 10, 'eq')
    data_new = data_single_column.where('integers', 10, 'eq')
    compare_df = data.pandas_df()
    compare_df = compare_df[compare_df['integers']==10].reset_index(drop=True)
    pd.testing.assert_frame_equal(data_new.pandas_df(), compare_df)

    #Test inequality for cases where there is only one column
    data_new = data_single_column.where('integers', 56, 'neq')
    pd.testing.assert_frame_equal(data_new.pandas_df(), compare_df)

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
    # Test condition that is not defined
    with pytest.raises(ValueError):
        _ = data.where("integers", 10, condition="eq_sqrt")
    # Test passing float in for string check
    with pytest.raises(ValueError):
        _ = data.where("names", 0.342, condition="eq")

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

def test_str_navdata(df_simple, df_only_header):
    """Test that the NavData class can be printed without errors

    Parameters
    ----------
    df_simple : pd.DataFrame
        Dataframe with which to construct NavData instance
    df_only_header : pd.DataFrame
        Dataframe with only column names and no data
    """
    navdata = NavData(pandas_df=df_simple)
    navdata_str = str(navdata)
    df_str = str(df_simple)
    assert navdata_str==df_str

    # make sure print doesn't break if given only headers
    navdata_str = str(NavData(pandas_df=df_only_header))
    df_str = str(df_only_header).replace("DataFrame","NavData")
    df_str = df_str.replace("Columns","Rows")
    assert navdata_str==df_str

    # make sure it doesn't break with empty NavData
    navdata_str = str(NavData())
    df_str = str(pd.DataFrame()).replace("DataFrame","NavData")
    df_str = df_str.replace("Columns","Rows")
    assert navdata_str==df_str

    # test DataFrame with a single row of data
    df_long = pd.DataFrame(np.zeros((200,4)),
                                    columns=["A","B","C","D"])
    navdata = NavData(pandas_df=df_long)
    assert str(navdata) == str(df_long).replace("[200 rows x 4 columns]",
                                                "[4 rows x 200 columns]")


def test_in_rows_single(data):
    """Test the in_rows function.

    Parameters
    ----------
    data : gnss_lib_py.navdata.navdata.NavData
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
    data : gnss_lib_py.navdata.navdata.NavData
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

def test_pandas_df(df_simple, df_only_header):
    """Test that the NavData class can be printed without errors

    Parameters
    ----------
    df_simple : pd.DataFrame
        Dataframe with which to construct NavData instance
    df_only_header : pd.DataFrame
        Dataframe with only column names and no data
    """
    navdata = NavData(pandas_df=df_simple)

    # test simple DataFrame
    pd.testing.assert_frame_equal(navdata.pandas_df().sort_index(axis=1),
                                  df_simple.sort_index(axis=1),
                                  check_names=True, check_dtype=False)

    # test DataFrame with a single row of data
    df_super_simple = pd.DataFrame([["first",1.,3,"fourth"]],
                                    columns=["A","B","C","D"])
    navdata = NavData(pandas_df=df_super_simple)
    pd.testing.assert_frame_equal(navdata.pandas_df().sort_index(axis=1),
                                  df_super_simple.sort_index(axis=1),
                                  check_names=True, check_dtype=False)

    # make sure print doesn't break if given only headers
    navdata = NavData(pandas_df=df_only_header)
    pd.testing.assert_frame_equal(navdata.pandas_df().sort_index(axis=1),
                                  df_only_header.sort_index(axis=1),
                                  check_names=True)

    # make sure it doesn't break with empty NavData
    navdata = NavData()
    pd.testing.assert_frame_equal(navdata.pandas_df().sort_index(axis=1),
                                  pd.DataFrame().sort_index(axis=1),
                                  check_names=True, check_column_type=False)

def test_large_int():
    """Test get/set for large integers.

    """

    test_list = np.array([1e12 + 1,
                          1e12 + 2,
                          1e12 + 3,
                          1e12 + 4,
                          1e12 + 5,
                          ])
    navdata = NavData()
    navdata["numbers"] = test_list

    np.testing.assert_array_equal(navdata["numbers"], test_list)

@pytest.mark.parametrize('csv_path',
                        [
                         lazy_fixture("csv_dtypes"),
                         lazy_fixture("csv_simple"),
                         lazy_fixture("csv_mixed"),
                         lazy_fixture("csv_inf"),
                         lazy_fixture("csv_int_first"),
                        ])
def test_dtypes_casting(csv_path):
    """Test dtypes casting from csv back to csv

    Parameters
    ----------
    csv_path : string
        Path to csv file containing data

    """

    data = NavData(csv_path=csv_path)
    # check reverse casting
    assert data.__str__() == pd.read_csv(csv_path).__str__()


def test_dtypes_changing(csv_dtypes):
    """Test changing dtypes

    Parameters
    ----------
    csv_dtypes : string
        Path to csv file containing multiple data types

    """

    data = NavData(csv_path=csv_dtypes)
    assert data.orig_dtypes["int"] == np.int64
    assert data.orig_dtypes["float"] == np.float64
    assert data.orig_dtypes["datetime"] == object
    assert data.orig_dtypes["string"] == object

    data_changed = data.copy()
    data_changed["int"] = np.array(["an integer no longer!"])
    data_changed["float"] = np.array(["a float no longer!"])
    assert data_changed.orig_dtypes["int"] == object
    assert data_changed.orig_dtypes["float"] == object
    assert data_changed.orig_dtypes["datetime"] == object
    assert data_changed.orig_dtypes["string"] == object

    data_changed = data.copy()
    data_changed["float"] = 1
    data_changed["datetime"] = 2
    data_changed["string"] = 3
    assert data_changed.orig_dtypes["int"] == np.int64
    assert data_changed.orig_dtypes["float"] == np.int64
    assert data_changed.orig_dtypes["datetime"] == np.int64
    assert data_changed.orig_dtypes["string"] == np.int64

    data_changed = data.copy()
    data_changed["int"] = 1.
    data_changed["datetime"] = 2.
    data_changed["string"] = 3.
    assert data_changed.orig_dtypes["int"] == np.float64
    assert data_changed.orig_dtypes["float"] == np.float64
    assert data_changed.orig_dtypes["datetime"] == np.float64
    assert data_changed.orig_dtypes["string"] == np.float64

def test_keep_cols_where(data, df_simple):
    """Test keep columns with where.

    """
    # test for strings
    keep_cols = ['gps', 'glonass']

    data_subset = data.where('strings', keep_cols,
                                        condition="eq")
    df_simple_subset = df_simple.loc[df_simple['strings'].isin(keep_cols), :]

    df_simple_subset = df_simple_subset.reset_index(drop=True)
    pd.testing.assert_frame_equal(data_subset.pandas_df(), df_simple_subset, check_dtype=False)

    # test for floats
    keep_cols = [0.5, 0.45]

    data_subset = data.where('floats', keep_cols,
                                        condition="neq")
    df_simple_subset = df_simple.loc[~df_simple['floats'].isin(keep_cols), :]

    df_simple_subset = df_simple_subset.reset_index(drop=True)
    pd.testing.assert_frame_equal(data_subset.pandas_df(), df_simple_subset, check_dtype=False)
