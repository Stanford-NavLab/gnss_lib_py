"""Test timing conversions between reference frames.

"""

__authors__ = "Sriramya Bhamidipati, Ashwin Kanhere"
__date__ = "28 Jul 2022"

from datetime import datetime, timedelta, timezone
import pytest

import numpy as np

import gnss_lib_py.utils.time_conversions as tc
from gnss_lib_py.utils.constants import GPS_EPOCH_0

@pytest.fixture(name="check_leapseconds")
def leapseconds_table():
    """Table with reference for when leapseconds were added

    Returns
    -------
    leapseconds_ref_table : np.ndarray
        Array of reference times and leap seconds for validation
    """
    leapseconds_ref_table = np.transpose([[GPS_EPOCH_0, 0],
                                    [datetime(1981, 7, 1, 0, 0, tzinfo=timezone.utc), 1],
                                    [datetime(1982, 7, 1, 0, 0, tzinfo=timezone.utc), 2],
                                    [datetime(1983, 7, 1, 0, 0, tzinfo=timezone.utc), 3],
                                    [datetime(1985, 7, 1, 0, 0, tzinfo=timezone.utc), 4],
                                    [datetime(1988, 1, 1, 0, 0, tzinfo=timezone.utc), 5],
                                    [datetime(1990, 1, 1, 0, 0, tzinfo=timezone.utc), 6],
                                    [datetime(1991, 1, 1, 0, 0, tzinfo=timezone.utc), 7],
                                    [datetime(1992, 7, 1, 0, 0, tzinfo=timezone.utc), 8],
                                    [datetime(1993, 7, 1, 0, 0, tzinfo=timezone.utc), 9],
                                    [datetime(1994, 7, 1, 0, 0, tzinfo=timezone.utc), 10],
                                    [datetime(1996, 1, 1, 0, 0, tzinfo=timezone.utc), 11],
                                    [datetime(1997, 7, 1, 0, 0, tzinfo=timezone.utc), 12],
                                    [datetime(1999, 1, 1, 0, 0, tzinfo=timezone.utc), 13],
                                    [datetime(2006, 1, 1, 0, 0, tzinfo=timezone.utc), 14],
                                    [datetime(2009, 1, 1, 0, 0, tzinfo=timezone.utc), 15],
                                    [datetime(2012, 7, 1, 0, 0, tzinfo=timezone.utc), 16],
                                    [datetime(2015, 7, 1, 0, 0, tzinfo=timezone.utc), 17],
                                    [datetime(2017, 1, 1, 0, 0, tzinfo=timezone.utc), 18]])
    return leapseconds_ref_table

def test_get_leap_seconds(check_leapseconds):
    """Test to validate leap seconds based on input time.
    """
    input_millis = 1000.0 * (datetime(2022, 7, 28, 0, 0, tzinfo=timezone.utc) - tc.GPS_EPOCH_0).total_seconds()
    valseconds = tc.get_leap_seconds(input_millis)
    assert valseconds == 18
    buffer_secs = 3.0
    num_leapsecarray = len(check_leapseconds[0,:])
    for row in range(num_leapsecarray):
        input_datetime = check_leapseconds[0,row] + timedelta(seconds = buffer_secs)
        valdatetime = tc.get_leap_seconds(input_datetime)
        assert valdatetime == check_leapseconds[1,row]
        input_millis = 1000.0 * ( (check_leapseconds[0,row] \
                                 - tc.GPS_EPOCH_0).total_seconds() \
                              + buffer_secs)
        valseconds = tc.get_leap_seconds(input_millis)
        assert valseconds == check_leapseconds[1,row]

#     with pytest.raises(RuntimeError):
#         buffer_secs = 3.0
#         input_datetime = tc.LEAPSECONDS_TABLE[0,0] - timedelta(seconds = buffer_secs)
#         tc.get_leap_seconds(input_datetime)

def test_datetime_to_tow():
    """Test that datetime conversion to GPS or UTC secs does not fail.

    Verified by using an online gps time calculator [1]_.

    References
    ----------
    .. [1] https://www.labsat.co.uk/index.php/en/gps-time-calculator

    """

    input_time = datetime(2022, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
    output_wk, output_tow = tc.datetime_to_tow(input_time,
                                               add_leap_secs = True)
    assert output_wk == 2220
    assert output_tow == 388818.0

    output_wk2, output_tow2 = tc.datetime_to_tow(input_time,
                                                 add_leap_secs = False)
    assert output_wk2 == 2220
    assert (output_tow - output_tow2) == 18.0

#     with pytest.raises(RuntimeError):
#         buffer_secs = 3.0
#         input_datetime = tc.LEAPSECONDS_TABLE[0,0] - timedelta(seconds = buffer_secs)
#         tc.datetime_to_tow(input_datetime)

def test_millis_since_gps_epoch_to_tow():
    """Test millis since gps ecph to time of week.

    Test that conversion from milliseconds since GPS epoch to GPS or
    UTC seconds of the week does not fail.

    Given a UTC time epoch, an online GPS Time converter [2]_ provides
    seconds since GPS epoch while a GPS Time calculator [3]_ gives GPS
    seconds of the week.

    References
    ----------
    .. [2] https://www.andrews.edu/~tzs/timeconv/timeconvert.php?
           Accessed July 28, 2022.
    .. [3] https://www.labsat.co.uk/index.php/en/gps-time-calculator
           Accessed as of July 28, 2022.

   """
    # These two are for 30th june 2016 (1151280017) and leap seconds: 17
    output_wk, output_tow = tc.gps_millis_to_tow(1151280017.0*1000.0, add_leap_secs = False)
    assert output_wk == 1903.0
    assert output_tow == 345617.0

    output_wk2, output_tow2 = tc.gps_millis_to_tow(1151280017.0*1000.0, add_leap_secs = True)
    assert output_wk2 == 1903.0
    assert output_tow2 - output_tow == 17.0

    output_wk3, output_tow3 = tc.gps_millis_to_tow(1303041618.0*1000.0, add_leap_secs = False)
    assert output_wk3 == 2154.0
    assert output_tow3 == 302418.0
