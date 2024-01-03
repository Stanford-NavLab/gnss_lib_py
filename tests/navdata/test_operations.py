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
    csv_data = op.concat(csv_data,NavData(csv_path=csv_simple),axis=1)
    assert csv_data.shape == (4,6)
    pd.testing.assert_frame_equal(csv_data.pandas_df().sort_index(axis=1),
                                  pd.read_csv(csv_simple).sort_index(axis=1),
                                  check_dtype=False, check_names=True)

    # should work when DataFrame is passed
    pd_data = NavData(pandas_df=pd.read_csv(csv_only_header))
    assert pd_data.shape == (4,0)
    # test adding new data to empty NavData with column names
    pd_data = op.concat(pd_data,NavData(pandas_df=pd.read_csv(csv_simple)),axis=1)
    assert pd_data.shape == (4,6)

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
                            1.49999*np.ones([1,1])))
    compare_df = data.pandas_df()
    count = 0
    # Testing when loop_time finds overlapping times
    for time, delta_t, measure in op.loop_time(data,'times', delta_t_decimals=2):
        if count == 0:
            np.testing.assert_almost_equal(delta_t, 0)
            np.testing.assert_almost_equal(time, 0)
            row_num = [0,1]
        elif count == 1:
            np.testing.assert_almost_equal(delta_t, 1)
            np.testing.assert_almost_equal(time, 1)
            row_num = [2,3]
        elif count == 2:
            np.testing.assert_almost_equal(delta_t, 0.5)
            np.testing.assert_almost_equal(time, 1.5)
            row_num = [4,5]
        small_df = measure.pandas_df().reset_index(drop=True)
        expected_df = compare_df.iloc[row_num, :].reset_index(drop=True)
        pd.testing.assert_frame_equal(small_df, expected_df,
                                      check_index_type=False)
        count += 1

    # Testing for when loop_time finds only unique times
    count = 0
    expected_times = [0., 1.0001, 1.0003, 1.49999, 1.50004]
    for time, _, measure in op.loop_time(data,'times', delta_t_decimals=5):
        np.testing.assert_almost_equal(time, expected_times[count])
        count += 1

def test_sort(data, df_simple):
    """Test sorting function across simple dataframe.

    """

    df_sorted_int = df_simple.sort_values('integers').reset_index(drop=True)
    df_sorted_float = df_simple.sort_values('floats').reset_index(drop=True)
    data_sorted_int = op.sort(data,'integers').pandas_df()
    data_sorted_float = op.sort(data,'floats').pandas_df()
    float_ind = np.argsort(data['floats'])
    data_sorted_ind = op.sort(data,ind=float_ind).pandas_df()
    pd.testing.assert_frame_equal(data_sorted_int, df_sorted_int)
    pd.testing.assert_frame_equal(df_sorted_float, data_sorted_float)
    pd.testing.assert_frame_equal(df_sorted_float, data_sorted_ind)
    # test strings as well:
    df_sorted_names = df_simple.sort_values('names').reset_index(drop=True)
    data_sorted_names = op.sort(data,'names').pandas_df()
    pd.testing.assert_frame_equal(df_sorted_names, data_sorted_names)

    df_sorted_strings = df_simple.sort_values('strings').reset_index(drop=True)
    data_sorted_strings = op.sort(data,'strings').pandas_df()
    pd.testing.assert_frame_equal(df_sorted_strings, data_sorted_strings)

    # Test usecase when descending order is given
    df_sorted_int_des = df_simple.sort_values('integers', ascending=False).reset_index(drop=True)
    data_sorted_int_des = op.sort(data,'integers', ascending=False).pandas_df()
    pd.testing.assert_frame_equal(df_sorted_int_des, data_sorted_int_des)

    # test inplace
    data_sorted_int_des = data.copy()
    op.sort(data_sorted_int_des,'integers', ascending=False, inplace=True)
    data_sorted_int_des = data_sorted_int_des.pandas_df()
    pd.testing.assert_frame_equal(df_sorted_int_des, data_sorted_int_des)

    # Test sorting for only one column
    unsort_navdata_single_col = NavData()
    unsort_navdata_single_col['name'] = np.asarray(['NAVLab'], dtype=object)
    unsort_navdata_single_col['number'] = 1
    unsort_navdata_single_col['weight'] = 100
    sorted_single_col = op.sort(unsort_navdata_single_col)
    pd.testing.assert_frame_equal(sorted_single_col.pandas_df(),
                                unsort_navdata_single_col.pandas_df())

def test_find_wildcard_indexes(data):
    """Tests find_wildcard_indexes

    """

    all_matching = data.rename({"names" : "x_alpha_m",
                                "integers" : "x_beta_m",
                                "floats" : "x_gamma_m",
                                "strings" : "x_zeta_m"})
    expected = ["x_alpha_m","x_beta_m","x_gamma_m","x_zeta_m"]

    indexes = op.find_wildcard_indexes(all_matching,"x_*_m")
    assert indexes["x_*_m"] == expected
    expect_pass_allows = [None,12,4]
    for max_allow in expect_pass_allows:
        indexes = op.find_wildcard_indexes(all_matching,"x_*_m",max_allow)
        assert indexes["x_*_m"] == expected

    expect_fail_allows = [0,-1,3,2,1]
    for max_allow in expect_fail_allows:
        with pytest.raises(KeyError) as excinfo:
            op.find_wildcard_indexes(all_matching,"x_*_m",max_allow)
        assert "More than " + str(max_allow) in str(excinfo.value)
        assert "x_*_m" in str(excinfo.value)

    multi = data.rename({"names" : "x_alpha_m",
                         "integers" : "x_beta_m",
                         "floats" : "y_alpha_deg",
                         "strings" : "x_zeta_deg"})
    expected = {"x_*_m" : ["x_alpha_m","x_beta_m"],
                "y_*_deg" : ["y_alpha_deg"]}

    expect_pass_allows = [None,2,4]
    for max_allow in expect_pass_allows:
        indexes = op.find_wildcard_indexes(multi,["x_*_m","y_*_deg"],
                                               max_allow)
        assert indexes == expected

    expect_pass_allows = [None,2,4]
    for max_allow in expect_pass_allows:
        indexes = op.find_wildcard_indexes(multi,tuple(["x_*_m","y_*_deg"]),
                                                     max_allow)
        assert indexes == expected

    expect_pass_allows = [None,2,4]
    for max_allow in expect_pass_allows:
        indexes = op.find_wildcard_indexes(multi,set(["x_*_m","y_*_deg"]),
                                                     max_allow)
        assert indexes == expected

    expect_pass_allows = [None,2,4]
    for max_allow in expect_pass_allows:
        indexes = op.find_wildcard_indexes(multi,np.array(["x_*_m",
                                                        "y_*_deg"]),
                                                        max_allow)
        assert indexes == expected

    expect_fail_allows = [0,-1,1]
    for max_allow in expect_fail_allows:
        with pytest.raises(KeyError) as excinfo:
            op.find_wildcard_indexes(multi,["x_*_m","y_*_deg"],max_allow)
        assert "More than " + str(max_allow) in str(excinfo.value)
        assert "x_*_m" in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        op.find_wildcard_indexes(multi,["z_*_m"])
    assert "Missing " in str(excinfo.value)
    assert "z_*_m" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        op.find_wildcard_indexes(multi,1.0)
    assert "find_wildcard_indexes " in str(excinfo.value)
    assert "array-like" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        op.find_wildcard_indexes(multi,[1.0])
    assert "wildcards must be strings" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        op.find_wildcard_indexes(multi,"x_*_*")
    assert "One wildcard" in str(excinfo.value)

    incorrect_max_allow = [3.,"hi",[]]
    for max_allow in incorrect_max_allow:
        with pytest.raises(TypeError) as excinfo:
            op.find_wildcard_indexes(multi,"x_*_m",max_allow)
        assert "max_allow" in str(excinfo.value)

def test_find_wildcard_excludes(data):
    """Tests find_wildcard_indexes

    """
    all_matching = data.rename({"names" : "x_alpha_m",
                                "integers" : "x_beta_m",
                                "floats" : "x_gamma_m",
                                "strings" : "x_zeta_m"})

    # no exclusion
    indexes = op.find_wildcard_indexes(all_matching,"x_*_m",excludes=None)
    assert indexes["x_*_m"] == ["x_alpha_m","x_beta_m",
                                "x_gamma_m","x_zeta_m"]
    indexes = op.find_wildcard_indexes(all_matching,"x_*_m",excludes=[None])
    assert indexes["x_*_m"] == ["x_alpha_m","x_beta_m",
                                "x_gamma_m","x_zeta_m"]

    # single exclusion
    indexes = op.find_wildcard_indexes(all_matching,"x_*_m",excludes="x_beta_m")
    assert indexes["x_*_m"] == ["x_alpha_m","x_gamma_m","x_zeta_m"]

    # two exclusion
    indexes = op.find_wildcard_indexes(all_matching,"x_*_m",
                                excludes=[["x_beta_m","x_zeta_m"]])
    assert indexes["x_*_m"] == ["x_alpha_m","x_gamma_m"]

    # all excluded
    with pytest.raises(KeyError) as excinfo:
        op.find_wildcard_indexes(all_matching,"x_*_m",excludes=["x_*_m"])
    assert "Missing " in str(excinfo.value)
    assert "x_*_m" in str(excinfo.value)


    multi = data.rename({"names" : "x_alpha_m",
                         "integers" : "x_beta_m",
                         "floats" : "y_alpha_deg",
                         "strings" : "y_beta_deg"})

    # no exclusion
    indexes = op.find_wildcard_indexes(multi,["x_*_m","y_*_deg"],
                                                excludes=None)
    assert indexes["x_*_m"] == ["x_alpha_m","x_beta_m"]
    assert indexes["y_*_deg"] == ["y_alpha_deg","y_beta_deg"]
    indexes = op.find_wildcard_indexes(multi,["x_*_m","y_*_deg"],
                                                excludes=[None,None])
    assert indexes["x_*_m"] == ["x_alpha_m","x_beta_m"]
    assert indexes["y_*_deg"] == ["y_alpha_deg","y_beta_deg"]

    # single exclusion
    indexes = op.find_wildcard_indexes(multi,["x_*_m","y_*_deg"],
                                                excludes=["x_alpha*",None])
    assert indexes["x_*_m"] == ["x_beta_m"]
    assert indexes["y_*_deg"] == ["y_alpha_deg","y_beta_deg"]

    # double exclusion
    indexes = op.find_wildcard_indexes(multi,["x_*_m","y_*_deg"],
                                                excludes=["x_alpha*","y_beta*"])
    assert indexes["x_*_m"] == ["x_beta_m"]
    assert indexes["y_*_deg"] == ["y_alpha_deg"]

    # must match length
    with pytest.raises(TypeError) as excinfo:
        op.find_wildcard_indexes(multi,["x_*_m","y_*_deg"],
                                    excludes=[None])
    assert "match length" in str(excinfo.value)

    # must match length
    with pytest.raises(TypeError) as excinfo:
        op.find_wildcard_indexes(multi,["x_*_m","y_*_deg"],
                                    excludes={"a":"dictionary"})
    assert "array-like" in str(excinfo.value)
    # must match length
    with pytest.raises(TypeError) as excinfo:
        op.find_wildcard_indexes(multi,["x_*_m","y_*_deg"],
                                    excludes=[None,{"a":"dictionary"}])
    assert "array-like" in str(excinfo.value)
