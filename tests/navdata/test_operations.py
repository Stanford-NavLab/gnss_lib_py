"""Tests for NavData class.

"""

__authors__ = "A. Kanhere, D. Knowles"
__date__ = "30 Apr 2022"

import pytest
import numpy as np
import pandas as pd

import gnss_lib_py.navdata.operations as op
from gnss_lib_py.navdata.navdata import NavData

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
    data = op.concat(data,NavData(numpy_array=add_array),axis=1)
    new_col_num = np.shape(add_array)[1]
    np.testing.assert_array_equal(data[:, -new_col_num:], add_array)

def test_add_numpy_1d():
    """Test addition of a 1D numpy array to NavData with single row
    """
    data = NavData(numpy_array=np.zeros([1,6]))
    data = op.concat(data,NavData(numpy_array=np.ones(8)),axis=1)
    np.testing.assert_array_equal(data[0, :], np.hstack((np.zeros(6),
                                  np.ones(8))))

    # test adding to empty NavData
    data_empty = NavData()
    data_empty = op.concat(data_empty,NavData(numpy_array=np.ones((8,8))),axis=1)
    np.testing.assert_array_equal(data_empty[:,:],np.ones((8,8)))

def test_add_csv(df_simple, csv_simple):
    """Test adding a csv.

    """
    # Create and add to NavData
    data = NavData(csv_path=csv_simple)
    data = op.concat(data,NavData(csv_path=csv_simple),axis=1)
    data_df = data.pandas_df()
    # Set up dataframe for comparison
    df_types = {'names': object, 'integers': np.int64,
                'floats': np.float64, 'strings': object}
    expected_df = pd.concat((df_simple,df_simple)).reset_index(drop=True)
    expected_df = expected_df.astype(df_types)
    pd.testing.assert_frame_equal(data_df.sort_index(axis=1),
                                  expected_df.sort_index(axis=1),
                                  check_index_type=False)

    # test adding to empty NavData
    data_empty = NavData()
    data_empty = op.concat(data_empty,NavData(csv_path=csv_simple),axis=1)
    pd.testing.assert_frame_equal(data_empty.pandas_df().sort_index(axis=1),
                                  df_simple.astype(df_types).sort_index(axis=1),
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
    data = op.concat(data,NavData(pandas_df=add_df),axis=1)
    new_df = data.pandas_df()
    add_row_num = add_df.shape[0]
    subset_df = new_df.iloc[-add_row_num:, :].reset_index(drop=True)
    pd.testing.assert_frame_equal(subset_df.sort_index(axis=1),
                                  add_df.sort_index(axis=1),
                                  check_index_type=False)

    # test adding to empty NavData
    data_empty = NavData()
    data_empty = op.concat(data_empty,NavData(pandas_df=add_df),axis=1)
    pd.testing.assert_frame_equal(add_df.sort_index(axis=1),
                                  data_empty.pandas_df().sort_index(axis=1),
                                  check_index_type=False)

def test_concat(df_simple):
    """Test concat functionaltiy.

    Parameters
    ----------
    df_simple : pd.DataFrame
        Simple pd.DataFrame with which to initialize NavData.

    """

    navdata_1 = NavData(pandas_df=df_simple)
    navdata_2 = navdata_1.copy()
    navdata_2.rename(mapper={"floats": "decimals", "names": "words"},
                    inplace = True)

    # add new columns
    navdata = op.concat(navdata_1,navdata_1.copy())
    assert navdata.shape == (4,12)
    pandas_equiv = pd.concat((df_simple,df_simple),axis=0)
    pandas_equiv.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(pandas_equiv.sort_index(axis=1),
                                  navdata.pandas_df().sort_index(axis=1),
                                  check_index_type=False,
                                  check_dtype=False)

    # add new rows
    navdata = op.concat(navdata_1,navdata_1.copy(),axis=0)
    assert navdata.shape == (8,6)
    mapper = {"names":"names_0",
              "floats":"floats_0",
              "integers":"integers_0",
              "strings":"strings_0"}
    df_simple_2 = df_simple.rename(mapper,axis=1)
    pandas_equiv = pd.concat((df_simple,df_simple_2),axis=1)
    pandas_equiv.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(pandas_equiv.sort_index(axis=1),
                                  navdata.pandas_df().sort_index(axis=1),
                                  check_index_type=False,
                                  check_dtype=False)

    # concatenate empty NavData
    navdata = op.concat(NavData(),navdata_1,axis=1)
    pd.testing.assert_frame_equal(df_simple.sort_index(axis=1),
                                  navdata.pandas_df().sort_index(axis=1),
                                  check_index_type=False,
                                  check_dtype=False)
    navdata = op.concat(navdata_1,NavData(),axis=1)
    pd.testing.assert_frame_equal(df_simple.sort_index(axis=1),
                                  navdata.pandas_df().sort_index(axis=1),
                                  check_index_type=False,
                                  check_dtype=False)

    # test multiple rows with the same name
    navdata_long = navdata_1.copy()
    for count in range(13):
        navdata_long = op.concat(navdata_long,navdata_1,axis=0)
        for word in ["names","integers","floats","strings"]:
            assert word + "_" + str(count) in navdata_long.rows

    # add semi new columns
    navdata = op.concat(navdata_1,navdata_2)
    assert navdata.shape == (6,12)
    assert np.all(np.isnan(navdata["floats"][-6:]))
    assert np.all(navdata["names"][-6:] == np.array([np.nan]).astype(str)[0])
    assert np.all(np.isnan(navdata["decimals"][:6]))
    assert np.all(navdata["words"][:6] == np.array([np.nan]).astype(str)[0])

    # add semi new columns in opposite order
    navdata = op.concat(navdata_2,navdata_1)
    assert navdata.shape == (6,12)
    assert np.all(np.isnan(navdata["floats"][:6]))
    assert np.all(navdata["names"][:6] == np.array([np.nan]).astype(str)[0])
    assert np.all(np.isnan(navdata["decimals"][-6:]))
    assert np.all(navdata["words"][-6:] == np.array([np.nan]).astype(str)[0])

    # add as new rows
    navdata = op.concat(navdata_1,navdata_2,axis=0)
    assert navdata.shape == (8,6)
    mapper = {"names":"words",
              "floats":"decimals",
              "integers":"integers_0",
              "strings":"strings_0"}
    df_simple_2 = df_simple.rename(mapper,axis=1)
    pandas_equiv = pd.concat((df_simple,df_simple_2),axis=1)
    pandas_equiv.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(pandas_equiv.sort_index(axis=1),
                                  navdata.pandas_df().sort_index(axis=1),
                                  check_index_type=False,
                                  check_dtype=False)

    navdata_a = NavData(pandas_df=pd.DataFrame({'a':[0],'b':[1],'c':[2],
                                                'd':[3],'e':[4],'f':[5],
                                                }))
    navdata_b = op.concat(navdata_a,navdata_a.copy(),axis=0)
    assert navdata_b.shape == (12,1)
    navdata_b = op.concat(navdata_a,navdata_a.copy(),axis=1)
    assert navdata_b.shape == (6,2)

def test_concat_fails(df_simple):
    """Test when concat should fail.

    Parameters
    ----------
    df_simple : pd.DataFrame
        Simple pd.DataFrame with which to initialize NavData.

    """

    navdata_1 = NavData(pandas_df=df_simple)

    with pytest.raises(TypeError) as excinfo:
        op.concat(navdata_1,np.array([]))
    assert "concat" in str(excinfo.value)
    assert "NavData" in str(excinfo.value)

    navdata_2 = navdata_1.remove(cols=[0])

    with pytest.raises(RuntimeError) as excinfo:
        op.concat(navdata_1,navdata_2,axis=0)
    assert "same length" in str(excinfo.value)
    assert "concat" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        op.concat(navdata_1,NavData(),axis=0)
    assert "same length" in str(excinfo.value)
    assert "concat" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        op.concat(NavData(),navdata_1,axis=0)
    assert "same length" in str(excinfo.value)
    assert "concat" in str(excinfo.value)
