"""
Test for metrics, such as DOP calculations.

"""

__authors__ = "Ashwin Kanhere, Daniel Neamati"
__date__ = "7 Feb 2024"

import os

import pytest 
from pytest_lazyfixture import lazy_fixture

import numpy as np
import copy

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.utils.metrics import get_dop, calculate_dop
# from gnss_lib_py.navdata.operations import loop_time
from gnss_lib_py.parsers.google_decimeter import AndroidDerived2022



# # FROM test_coordinates.py
# @pytest.fixture(name="derived_2022_path")
# def fixture_derived_2022_path(root_path_2022):
#     """Filepath of Android Derived measurements

#     Returns
#     -------
#     derived_path : string
#         Location for the unit_test Android derived 2022 measurements

#     Notes
#     -----
#     Test data is a subset of the Android Raw Measurement Dataset [4]_,
#     from the 2022 Decimeter Challenge. Particularly, the
#     train/2021-04-29-MTV-2/SamsungGalaxyS20Ultra trace. The dataset
#     was retrieved from
#     https://www.kaggle.com/competitions/smartphone-decimeter-2022/data

#     References
#     ----------
#     .. [4] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
#         "Android Raw GNSS Measurement Datasets for Precise Positioning."
#         Proceedings of the 33rd International Technical Meeting of the
#         Satellite Division of The Institute of Navigation (ION GNSS+
#         2020). 2020.
#     """
#     derived_path = os.path.join(root_path_2022, 'device_gnss.csv')
#     return derived_path


# @pytest.fixture(name="derived_2022")
# def fixture_load_derived_2022(derived_2022_path):
#     """Load instance of AndroidDerived2021

#     Parameters
#     ----------
#     derived_path : pytest.fixture
#     String with location of Android derived measurement file

#     Returns
#     -------
#     derived : AndroidDerived2022
#     Instance of AndroidDerived2022 for testing
#     """
#     derived = AndroidDerived2022(derived_2022_path)
#     return derived


#####################################################################
# NEW FIXTURES AND TESTS
@pytest.fixture(name="simple_sat_scenario")
def fixture_simple_sat_scenario():
    """
    A simple set of satellites for DOP calculation.
    
    """
    # Create a simple NavData instance
    navdata = NavData()
    
    # Add a few satellites
    navdata['gps_millis'] = np.array([0, 0, 0, 0, 0], dtype=int)
    navdata['el_sv_deg'] = np.array([0, 0, 45, 45, 90], dtype=float)
    navdata['az_sv_deg'] = np.array([0, 90, 180, 270, 360], dtype=float)

    return navdata


@pytest.fixture(name="simple_sat_expected_dop")
def fixture_simple_sat_expected_dop():
    """
    The expected dop values for the simple satellite scenario.
    
    """
    # Expected DOP matrix
    sqrt2 = np.sqrt(2)
    expected_dop_matrix = 0.25 * np.array(
        [[(25/3) - 2*sqrt2, (17/3) - 2*sqrt2,  5 + sqrt2,   -5 + sqrt2],
         [(17/3) - 2*sqrt2, (25/3) - 2*sqrt2,  5 + sqrt2,   -5 + sqrt2],
         [5 + sqrt2,         5 + sqrt2,        9 + 4*sqrt2, -5 - 2*sqrt2],
         [-5 + sqrt2,       -5 + sqrt2,       -5 - 2*sqrt2,  5]])
    # Assert symmetry
    np.testing.assert_array_almost_equal(
        expected_dop_matrix.T, expected_dop_matrix)
    
    # Get the rest of the expected DOP values
    expected_dop = {'dop_matrix': expected_dop_matrix,
                    'GDOP': np.sqrt(23/3),
                    'HDOP': np.sqrt((25/6) - sqrt2),
                    'VDOP': np.sqrt((9/4) + sqrt2),
                    'PDOP': 0.5 * np.sqrt(77/3),
                    'TDOP': 0.5 * np.sqrt(5)}
    
    return expected_dop



@pytest.mark.parametrize('navdata, expected_dop',
                        [
                            (lazy_fixture('simple_sat_scenario'), 
                             lazy_fixture('simple_sat_expected_dop'))
                        ])
def test_simple_dop(navdata, expected_dop):
    """
    A simple set of satellites for DOP calculation.
    
    """
    dop_dict = calculate_dop(navdata)

    # Check the DOP output has all the expected keys
    assert dop_dict.keys() == {'dop_matrix', 
                               'GDOP', 'HDOP', 'VDOP', 'PDOP', 'TDOP'}
    
    # Check the DOP output has the expected values
    
    # Assert symmetry
    np.testing.assert_array_almost_equal(
        dop_dict['dop_matrix'].T, dop_dict['dop_matrix'])
    # Assert matching values
    np.testing.assert_array_almost_equal(
        dop_dict['dop_matrix'], expected_dop['dop_matrix'])

    # Check the _DOP values
    np.testing.assert_array_almost_equal(dop_dict['GDOP'], expected_dop['GDOP'])
    np.testing.assert_array_almost_equal(dop_dict['HDOP'], expected_dop['HDOP'])
    np.testing.assert_array_almost_equal(dop_dict['VDOP'], expected_dop['VDOP'])
    np.testing.assert_array_almost_equal(dop_dict['PDOP'], expected_dop['PDOP'])
    np.testing.assert_array_almost_equal(dop_dict['TDOP'], expected_dop['TDOP'])


@pytest.mark.parametrize('navdata, expected_dop',
                        [
                            (lazy_fixture('simple_sat_scenario'),
                             lazy_fixture('simple_sat_expected_dop'))
                        ])
def test_simple_get_dop_default(navdata, expected_dop):
    """
    Test that the get_dop function works correctly forms the DOP navdata under 
    default selection of the dop entries (i.e., HDOP and VDOP only)
    """

    # Perform the function under test
    dop_navdata = get_dop(navdata)

    # Check the DOP NavData instance has the expected keys
    np.testing.assert_equal(
        dop_navdata.rows,
        ['gps_millis', 'HDOP', 'VDOP'],
        f"Rows are {dop_navdata.rows}")

    # Check the DOP NavData instance has the expected values
    assert dop_navdata['gps_millis'] == navdata['gps_millis'][0]
    np.testing.assert_array_almost_equal(dop_navdata['HDOP'], 
                                        expected_dop['HDOP'])
    np.testing.assert_array_almost_equal(dop_navdata['VDOP'],
                                        expected_dop['VDOP'])


@pytest.mark.parametrize('navdata, expected_dop, which_dop',
                        [
                            (lazy_fixture('simple_sat_scenario'),
                             lazy_fixture('simple_sat_expected_dop'),
                             {'GDOP': True, 'HDOP': True, 'VDOP': True, 
                              'PDOP': True, 'TDOP': True, 'dop_matrix': False}), 
                            (lazy_fixture('simple_sat_scenario'),
                             lazy_fixture('simple_sat_expected_dop'),
                             {'GDOP': True, 'HDOP': True, 'VDOP': True, 
                              'PDOP': True, 'TDOP': True, 'dop_matrix': True}),
                            (lazy_fixture('simple_sat_scenario'),
                             lazy_fixture('simple_sat_expected_dop'),
                             {'GDOP': False, 'HDOP': False, 'VDOP': False, 
                              'PDOP': False, 'TDOP': False, 'dop_matrix': True})
                        ])
def test_simple_get_dop(navdata, expected_dop, which_dop):
    """
    Test that the get_dop function works correctly forms the DOP navdata.
    """
    # Make a deep copy of which_dop to ensure it was not editted
    which_dop_deep_copy = copy.deepcopy(which_dop)

    # Perform the function under test
    dop_navdata = get_dop(navdata, **which_dop)

    # Assert that the which_dop deep copy is the same as the original
    

    assert 'gps_millis' in dop_navdata.rows        
    assert dop_navdata['gps_millis'] == navdata['gps_millis'][0]

    base_rows = which_dop

    for base_row in base_rows:
        assert base_row in dop_navdata.rows
        np.testing.assert_array_almost_equal(dop_navdata[base_row],
                                            expected_dop[base_row])
    
    if 'dop_matrix' in dop_navdata:
        # Handle the splatting of the DOP matrix
        dop_labels = ['ee', 'en', 'eu', 'et', 
                            'nn', 'nu', 'nt', 
                                  'uu', 'ut', 
                                        'tt']
        
        for label in dop_labels:
            assert f"dop_{label}" in dop_navdata.rows
        
        rows = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
        cols = (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)
        ind = 0

        for r, c in zip(rows, cols):
            np.testing.assert_array_almost_equal(
                dop_navdata[f"dop_{dop_labels[ind]}"],
                expected_dop['dop_matrix'][r, c])
            ind += 1
            




# @pytest.mark.parametrize('navdata',[
#                                     lazy_fixture('derived_2022'),
#                                     ])
# def test_dop_across_time(navdata):
#     """
#     Test DOP calculation across time is properly added to NAV data.
#     """

#     dop_navdata = get_dop(navdata)

#     # Check that the DOP NavData instance has the expected keys
#     assert all(dop_navdata.rows == ('gps_millis', 'HDOP', 'VDOP'))

