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
from gnss_lib_py.utils.dop import \
    get_dop, calculate_dop, _calculate_enut_matrix


#####################################################################
# Test under a simplified satellite scenario
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


@pytest.fixture(name="simple_sat_expected_enu_unit_vectors")
def fixture_simple_sat_expected_enu_unit_vectors():
    """
    The expected ENU unit vectors for the simple satellite scenario.
    
    """
    # Expected ENU unit vectors
    divsqrt2 = 1 / np.sqrt(2)
    expected_enu_unit_vectors = np.array([[0, 1, 0],
                                          [1, 0, 0],
                                          [0, -divsqrt2, divsqrt2],
                                          [-divsqrt2, 0, divsqrt2],
                                          [0, 0, 1]])
    
    return expected_enu_unit_vectors


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


@pytest.mark.parametrize('navdata, expected_los_vectors',
                    [
                        (lazy_fixture('simple_sat_scenario'), 
                         lazy_fixture('simple_sat_expected_enu_unit_vectors'))
                    ])
def test_simple_enu_unit_vectors(navdata, expected_los_vectors):
    """
    A simple set of satellites for ENU unit vector calculation.
    
    """
    # Check the we get the expected ENUT matrix
    enut_matrix = _calculate_enut_matrix(navdata)

    # First check the shape
    assert enut_matrix.shape[0] == expected_los_vectors.shape[0]
    assert enut_matrix.shape[1] == (expected_los_vectors.shape[1] + 1)
    assert enut_matrix.shape[0] > enut_matrix.shape[1]

    np.testing.assert_array_almost_equal(
        enut_matrix[:, :3], expected_los_vectors)
    np.testing.assert_array_almost_equal(
        enut_matrix[:, 3], np.ones(expected_los_vectors.shape[0]))


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
                            'PDOP': False, 'TDOP': True, 'dop_matrix': False}),
                        (lazy_fixture('simple_sat_scenario'),
                            lazy_fixture('simple_sat_expected_dop'),
                            {'GDOP': False, 'HDOP': False, 'VDOP': False, 
                            'PDOP': False, 'TDOP': False, 'dop_matrix': True}),
                        (lazy_fixture('simple_sat_scenario'),
                            lazy_fixture('simple_sat_expected_dop'),
                            {'GDOP': False, 'HDOP': False, 'VDOP': False, 
                            'PDOP': False, 'TDOP': False, 'dop_matrix': False})
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
    assert which_dop_deep_copy == which_dop, \
        "The `which_dop` dictionaries was editted by the get_dop function."

    assert 'gps_millis' in dop_navdata.rows        
    assert dop_navdata['gps_millis'] == navdata['gps_millis'][0]

    base_rows = which_dop.keys() - {'dop_matrix'}

    for base_row in base_rows:
        if which_dop[base_row]:
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
            
#############################################
# Singularity issues and edge cases

@pytest.fixture(name="singularity_sat_scenario")
def fixture_singularity_sat_scenario():
    """
    A simple set of satellites that will cause a singularity in the DOP matrix.
    
    """
    # Create a simple NavData instance
    navdata = NavData()
    
    # Add a few satellites
    navdata['gps_millis'] = np.array([0, 0, 0, 0, 0], dtype=int)
    navdata['el_sv_deg'] = np.array([30, 30, 30, 30, 30], dtype=int)
    navdata['az_sv_deg'] = np.array([90, 90, 90, 90, 90], dtype=int)

    return navdata


@pytest.fixture(name="too_few_sat_scenario")
def fixture_too_few_sat_scenario():
    """
    A simple set of too few satellites that will cause a singularity in the 
    DOP matrix.
    
    """
    # Create a simple NavData instance
    navdata = NavData()
    
    # Add a few satellites
    navdata['gps_millis'] = np.array([0, 0], dtype=int)
    navdata['el_sv_deg'] = np.array([20, 30], dtype=int)
    navdata['az_sv_deg'] = np.array([30, 60], dtype=int)

    return navdata


@pytest.mark.parametrize('navdata',
                        [
                            lazy_fixture('singularity_sat_scenario'),
                            lazy_fixture('too_few_sat_scenario')
                        ])
def test_singularity_dop(navdata):
    """
    Testing that the singularity error is raised and handled correctly.

    """
    with pytest.raises(np.linalg.LinAlgError):
        enut_matrix = _calculate_enut_matrix(navdata)
        np.linalg.inv(enut_matrix.T @ enut_matrix)

    # Now check that we get all NaNs for the DOP values when we have a 
    # singularity
    dop_dict = calculate_dop(navdata)
    
    # Check the DOP output has all the expected keys
    assert dop_dict.keys() == {'dop_matrix',
                               'GDOP', 'HDOP', 'VDOP', 'PDOP', 'TDOP'}

    # Check these are all NaNs
    for key in dop_dict.keys():
        assert np.all(np.isnan(dop_dict[key]))



#############################################
# Real data tests across time

@pytest.mark.parametrize('navdata',[
                                    lazy_fixture('android_derived')
                                    ])
def test_dop_across_time(navdata):
    """
    Test DOP calculation across time is properly added to NAV data.
    """

    dop_navdata = get_dop(navdata)

    # Check that the DOP NavData instance has the expected keys
    assert dop_navdata.rows == ['gps_millis', 'HDOP', 'VDOP']

    # Check that the DOP NavData is the expected length
    unique_gps_millis = np.unique(navdata['gps_millis'])
    assert len(dop_navdata['gps_millis']) == len(unique_gps_millis)


@pytest.mark.parametrize('navdata, which_dop',
                         [
                (lazy_fixture('android_derived'), 
                 {'GDOP': True, 'HDOP': True, 'VDOP': True, 
                  'PDOP': True, 'TDOP': True, 'dop_matrix': False}),
                (lazy_fixture('android_derived'), 
                 {'GDOP': True, 'HDOP': True, 'VDOP': True, 
                  'PDOP': True, 'TDOP': True, 'dop_matrix': True}),
                (lazy_fixture('android_derived'), 
                 {'GDOP': False, 'HDOP': False, 'VDOP': False, 
                  'PDOP': False, 'TDOP': True, 'dop_matrix': True}),
                (lazy_fixture('android_derived'), 
                 {'GDOP': False, 'HDOP': False, 'VDOP': False, 
                  'PDOP': False, 'TDOP': False, 'dop_matrix': True})
                         ])
def test_dop_across_time_with_selection(navdata, which_dop):
    """
    Test DOP calculation across time is properly added to NAV data.
    """

    # Make a deep copy of which_dop to ensure it was not editted
    which_dop_deep_copy = copy.deepcopy(which_dop)

    # Perform the function under test
    dop_navdata = get_dop(navdata, **which_dop)

    # Assert that the which_dop deep copy is the same as the original
    assert which_dop_deep_copy == which_dop, \
        "The `which_dop` dictionaries was editted by the get_dop function."

    # Check that the DOP NavData instance has the expected keys
    base_rows = which_dop.keys() - {'dop_matrix'}

    for base_row in base_rows:
        if which_dop[base_row]:
            assert base_row in dop_navdata.rows

            # Check no nans
            assert all(np.isfinite(dop_navdata[base_row]))
            # Check no negative values
            assert all(dop_navdata[base_row] >= 0)

    # Check that the DOP NavData is the expected length
    unique_gps_millis = np.unique(navdata['gps_millis'])
    assert len(dop_navdata['gps_millis']) == len(unique_gps_millis)

    if 'dop_matrix' in dop_navdata:
        # Handle the splatting of the DOP matrix
        dop_labels = ['ee', 'en', 'eu', 'et', 
                            'nn', 'nu', 'nt', 
                                  'uu', 'ut', 
                                        'tt']
        
        for label in dop_labels:
            assert f"dop_{label}" in dop_navdata.rows

        if 'GDOP' in dop_navdata.rows:
            np.testing.assert_array_almost_equal(
                dop_navdata['GDOP'], 
                np.sqrt(dop_navdata['dop_ee'] + dop_navdata['dop_nn'] +
                        dop_navdata['dop_uu'] + dop_navdata['dop_tt']))
            
        if 'HDOP' in dop_navdata.rows:
            np.testing.assert_array_almost_equal(
                dop_navdata['HDOP'], 
                np.sqrt(dop_navdata['dop_ee'] + dop_navdata['dop_nn']))
        
        if 'VDOP' in dop_navdata.rows:
            np.testing.assert_array_almost_equal(
                dop_navdata['VDOP'], 
                np.sqrt(dop_navdata['dop_uu']))
            
        if 'PDOP' in dop_navdata.rows:
            np.testing.assert_array_almost_equal(
                dop_navdata['PDOP'], 
                np.sqrt(dop_navdata['dop_ee'] + dop_navdata['dop_nn'] +
                        dop_navdata['dop_uu']))
        
        if 'TDOP' in dop_navdata.rows:
            np.testing.assert_array_almost_equal(
                dop_navdata['TDOP'], 
                np.sqrt(dop_navdata['dop_tt']))

