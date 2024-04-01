"""Tests for Nmea data parser.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "24 Jun, 2023"

import os
import pathlib

import numpy as np
import pytest

from conftest import lazy_fixture
from gnss_lib_py.parsers.nmea import Nmea

# pylint: disable=protected-access

@pytest.fixture(name="nmea_correct_checksum")
def fixture_nmea_file_w_correct_checksum(root_path):
    """Location of NMEA file with correct checksum values.

    The checksum values are at the end of each NMEA sentence.
    This instance of the NMEA file has the correct checksums at the end.

    Parameters
    ----------
    root_path : string
        Folder location containing all NMEA files for unit tests.

    Returns
    -------
    nmea_checksum : string
        Location of NMEA file with correct checksums
    """
    nmea_checksum = os.path.join(root_path, 'nmea', 'nmea_w_correct_checksum.nmea')
    return nmea_checksum

@pytest.fixture(name="nmea_wrong_checksum")
def fixture_nmea_file_w_wrong_checksum(root_path):
    """Location of NMEA file with wrong checksum values.

    The checksum values are at the end of each NMEA sentence.
    This instance of the NMEA file has the wrong checksums at the end for
    the RMC sentences.

    Parameters
    ----------
    root_path : string
        Folder location containing all NMEA files for unit tests.

    Returns
    -------
    nmea_wrong_checksum : string
        Location of NMEA file with wrong checksums
    """
    nmea_wrong_checksum = os.path.join(root_path, 'nmea', 'nmea_w_wrong_checksum.nmea')
    return nmea_wrong_checksum


@pytest.fixture(name="nmea_no_checksum")
def fixture_nmea_file_no_checksum(root_path):
    r"""Location of NMEA file without checksum values.

    The checksum values are at the end of each NMEA sentence (after an \*).
    This instance of the NMEA file has no checksums at the end.

    Parameters
    ----------
    root_path : string
        Folder location containing all NMEA files for unit tests.

    Returns
    -------
    nmea_no_checksum : string
        Location of NMEA file with correct checksums
    """
    nmea_no_checksum = os.path.join(root_path, 'nmea', 'nmea_no_checksum.nmea')
    return nmea_no_checksum


def compare_nmea_values(nmea_instance, row_name, exp_value, eq_decimal):
    """Helper funtion for comparing NMEA objects.

    nmea_instance : gnss_lib_py.parsers.nmea.Nmea
        Nmea instance to check.
    row_name : string
        Row name to check against
    exp_value : str, float, or int
        Expected value of element.
    eq_decimal : int
        Decimal passed into assert_almost_equal after which isn't
        checked for equivalency.

    """
    if isinstance(exp_value, str):
        np.testing.assert_string_equal(str(nmea_instance[row_name, 0]), exp_value)
    else:
        np.testing.assert_almost_equal(nmea_instance[row_name, 0].astype(float),
                                       exp_value, decimal=eq_decimal)

@pytest.mark.parametrize('nmea_file , check',
                        [
                            (lazy_fixture("nmea_correct_checksum"), True),
                            (lazy_fixture("nmea_wrong_checksum"), False),
                            (lazy_fixture("nmea_no_checksum"), False)
                        ])
@pytest.mark.parametrize('row_name , exp_value , eq_decimal',
                         [
                             ('alt_rx_m', 2.7, 1),
                             ('geo_sep_units', 'M', 0),
                             ('lat_rx_deg', 37.423, 3)
                         ])
def test_nmea_loading(nmea_file, check, row_name, exp_value, eq_decimal):
    """Test that the NMEA files are being loaded without any errors.

    Also compares some fixed values to the loaded values to ensure that
    values are not changed when theyu're loaded

    Parameters
    ----------
    nmea_file : string
        Location of the NMEA file to be loaded
    check : bool
        `True` if the checksum at the end of the NMEA sentence should
        be ignored. `False` if the checksum should be checked and lines
        with incorrect checksums will be ignored.
    row_name : string
        Row name to check against
    exp_value : str, float, or int
        Expected value of element.
    eq_decimal : int
        Decimal passed into assert_almost_equal after which isn't
        checked for equivalency.

    """
    # Testing base version of loading with all default parameters
    nmea_navdata = Nmea(nmea_file, check=check)
    compare_nmea_values(nmea_navdata, row_name, exp_value, eq_decimal)

    nmea_navdata = Nmea(pathlib.Path(nmea_file), check=check)

    # raises exception if not a file path
    with pytest.raises(FileNotFoundError):
        Nmea(pathlib.Path("not_a_file.txt"), check=check)
    with pytest.raises(FileNotFoundError):
        Nmea("not_a_file.txt", check=check)

    # raises exception if input not string or path-like
    with pytest.raises(TypeError):
        Nmea([], check=check)

    # Testing loading with raw latitude and longitude preserved
    nmea_raw = Nmea(nmea_file, check=check, keep_raw=True)
    compare_nmea_values(nmea_raw, row_name, exp_value, eq_decimal)

    # Checking that raw values are preserved
    raw_rows = ['lat', 'lat_dir', 'lon', 'lon_dir']
    raw_values = [3725.4168121, 'N', 12205.6498009, 'W']
    for raw_ind, raw_row in enumerate(raw_rows):
        compare_nmea_values(nmea_raw, raw_row, raw_values[raw_ind], eq_decimal=3)

    # Checking that ECEF conversion takes place without any errors
    nmea_ecef = Nmea(nmea_file, check=check, include_ecef=True)
    compare_nmea_values(nmea_ecef, row_name, exp_value, eq_decimal)


def test_failing_checksum(nmea_wrong_checksum):
    """Test behaviour when wrong checksums exist and `check=True`.

    Parameters
    ----------
    nmea_wrong_checksum : str
        Path to NMEA file with incorrect checksums at the end of the RMC
        statements. These
    """
    with pytest.raises(Exception):
        _ = Nmea(nmea_wrong_checksum, check=True)
