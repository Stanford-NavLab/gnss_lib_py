"""Test timing conversions between reference frames.

"""

__authors__ = "Ashwin Kanhere, Sriramya Bhamidipati"
__date__ = "28 Jul 2022"

from datetime import datetime, timedelta, timezone
from pytz import timezone as py_timezone
import pytest

import numpy as np

from gnss_lib_py.parsers.navdata import NavData
import gnss_lib_py.utils.time_conversions as tc
from gnss_lib_py.utils.constants import GPS_EPOCH_0

# pylint: disable=protected-access

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

    Parameters
    ----------
    check_leapseconds : np.ndarray
        Array of times at which leap seconds changed and changed values
    """
    input_millis = 1000.0 * (datetime(2022, 7, 28, 0, 0, tzinfo=timezone.utc) \
                 - tc.GPS_EPOCH_0).total_seconds()
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

    # Testing that datetime_to_tow raises error for time before start of
    # GPS epoch
    with pytest.raises(RuntimeError):
        input_time = datetime(1900, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        _ = tc.get_leap_seconds(input_time)

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

    # Test equivalent conversion from TOW to datetime
    rev_time = tc.tow_to_datetime(output_wk, output_tow, rem_leap_secs=True)
    assert input_time == rev_time


    output_wk2, output_tow2 = tc.datetime_to_tow(input_time,
                                                 add_leap_secs = False)
    assert output_wk2 == 2220
    assert (output_tow - output_tow2) == 18.0

    # Test equivalent conversion from TOW to datetime
    rev_time_2 = tc.tow_to_datetime(output_wk2, output_tow2, rem_leap_secs=False)
    assert input_time == rev_time_2

    # Testing that datetime_to_tow raises error for time before start of
    # GPS epoch
    with pytest.raises(RuntimeError):
        input_time = datetime(1900, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        _ = tc.datetime_to_tow(input_time, add_leap_secs=True)

def test_millis_since_gps_epoch_to_tow():
    """Test milliseconds since gps epoch to time of week.

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
    input_millis = 1151280017.0*1000.0
    output_wk, output_tow = tc.gps_millis_to_tow(input_millis, add_leap_secs = False)
    assert output_wk == 1903.0
    assert output_tow == 345617.0

    # Testing reverse conversion
    gps_millis = tc.tow_to_gps_millis(output_wk, output_tow)
    assert gps_millis == input_millis

    output_wk2, output_tow2 = tc.gps_millis_to_tow(input_millis, add_leap_secs = True)
    assert output_wk2 == 1903.0
    assert output_tow2 - output_tow == 17.0

    input_millis3 = 1303041618.0*1000.0
    output_wk3, output_tow3 = tc.gps_millis_to_tow(input_millis3, add_leap_secs = False)
    assert output_wk3 == 2154.0
    assert output_tow3 == 302418.0


    # Testing reverse conversion
    gps_millis3 = tc.tow_to_gps_millis(output_wk3, output_tow3)
    assert gps_millis3 == input_millis3


def test_tow_to_unix_millis():
    """Test TOW to milliseconds since Unix epoch and back.

    Given UTC time, milli seconds since the UNIX epoch were calculated
    from an online calculator [4]_. The UTC time was converted to GPS
    time and TOW using another online calculator [5]_.

    References
    ----------
    .. [4] https://currentmillis.com/
           Accessed August 10, 2022.
    .. [5] https://www.labsat.co.uk/index.php/en/gps-time-calculator
           Accessed as of July 28, 2022.

   """
    gps_week = 2222
    gps_tow = 330687.
    exp_unix_millis = 1660161069000.

    out_unix_millis = tc.tow_to_unix_millis(gps_week, gps_tow)
    assert out_unix_millis == exp_unix_millis

    # Testing reverse conversion

    rev_gps_week, rev_tow = tc.unix_millis_to_tow(exp_unix_millis)
    assert gps_week == rev_gps_week
    assert gps_tow == rev_tow


def test_datetime_to_unix_millis():
    """Test UTC datetime to milliseconds since UNIX epoch conversion
    and back

    Datetime to UNIX milliseconds conversion was obtained using an
    online convertor [6]_.

    References
    ----------
    .. [6] https://currentmillis.com/
           Accessed August 10, 2022.

    """
    t_datetime = datetime(2022, 8, 10, 19, 51, 9, tzinfo=timezone.utc)
    exp_unix_millis = 1660161069000.
    out_unix_millis = tc.datetime_to_unix_millis(t_datetime)
    assert exp_unix_millis == out_unix_millis
    # Testing reverse conversion
    t_rev = tc.unix_millis_to_datetime(out_unix_millis)
    assert t_datetime == t_rev


def test_datetime_to_gps_millis():
    """Test UTC datetime to milliseconds since GPS epoch conversion
    and back

    Datetime to GPS milliseconds conversion was obtained using an
    online convertor [7]_.

    References
    ----------
    .. [7] https://www.labsat.co.uk/index.php/en/gps-time-calculator
           Accessed as of July 28, 2022.

    """
    t_datetime = datetime(2022, 8, 10, 19, 51, 9, tzinfo=timezone.utc)
    exp_gps_millis = 1344196287000.
    out_gps_millis = tc.datetime_to_gps_millis(t_datetime)
    assert exp_gps_millis == out_gps_millis
    # Testing reverse conversion
    t_rev = tc.gps_millis_to_datetime(out_gps_millis)
    assert t_datetime == t_rev


def test_gps_unix_millis():
    """Test milliseconds since GPS epoch to milliseconds since UNIX epoch.

    Given UTC time, milliseconds since the UNIX epoch were calculated
    from an online calculator [8]_. The UTC time was converted to seconds
    (and hence milliseconds) since GPS epoch using another online
    calculator [9]_.

    References
    ----------
    .. [8] https://currentmillis.com/
           Accessed August 10, 2022.
    .. [9] https://www.labsat.co.uk/index.php/en/gps-time-calculator
           Accessed as of July 28, 2022.
   """
    unix_millis = 1660161069000.
    exp_gps_millis = 1344196287000.
    out_gps_millis = tc.unix_to_gps_millis(unix_millis)
    assert exp_gps_millis == out_gps_millis
    # Testing reverse conversion
    rev_unix_millis = tc.gps_to_unix_millis(exp_gps_millis)
    assert unix_millis == rev_unix_millis


def test_gps_unix_millis_vect():
    """Test vectorized version of unix_to_gps_millis and gps_to_unix_millis.

    Notes
    -----
    Test based on the test implemented in test_gps_unix_millis
    """

    delta_times = np.arange(10)
    unix_millis_vect = 1660161069000. + delta_times
    exp_gps_millis_vect  = 1344196287000. + delta_times
    out_gps_millis_vect = tc.unix_to_gps_millis(unix_millis_vect)
    np.testing.assert_almost_equal(exp_gps_millis_vect, out_gps_millis_vect)
    # Testing reverse conversion
    rev_unix_millis_vect = tc.gps_to_unix_millis(exp_gps_millis_vect)
    np.testing.assert_almost_equal(unix_millis_vect, rev_unix_millis_vect)


def test_tz_conversion():
    """Checking internal timezone conversions to UTC

    Checks that when timezone information is None or attribute doesn't
    exist, the timezone is changed to UTC.
    Also checks that if time is in non-UTC frame of reference, the time
    is converted to UTC before being returned.

    """
    local_time = datetime(2022, 8, 10, 19, 51, 9)
    exp_utc_time = datetime(2022, 8, 10, 19, 51, 9, tzinfo=timezone.utc)
    with pytest.warns(RuntimeWarning):
        out_utc_time = tc.tzinfo_to_utc(local_time)
        assert exp_utc_time == out_utc_time
    # Check time conversion when timezone other than UTC is given
    us_western = py_timezone('US/Pacific')
    western_time = us_western.localize(datetime(2022, 8, 10, 12, 51, 9))
    out_utc_time = tc.tzinfo_to_utc(western_time)
    assert exp_utc_time == out_utc_time


def test_array_conversions():
    """Test array conversions between time types.

    """
    num_checks = 10 # number of times to check
    datetimes_list = [datetime(np.random.randint(1981,2024),
                               np.random.randint(1,13),
                               np.random.randint(1,29),
                               np.random.randint(0,24),
                               np.random.randint(0,60),
                               np.random.randint(0,60),
                               np.random.randint(0,1000000),
                               tzinfo=timezone.utc
                               ) for d in range(num_checks)]
    datetimes_np = np.array(datetimes_list)

    # Datetime <--> GPS Millis
    gps_millis = tc.datetime_to_gps_millis(datetimes_list)
    assert len(gps_millis) == num_checks
    assert isinstance(gps_millis,np.ndarray)
    assert gps_millis.dtype is np.dtype(np.float64)
    datetimes_back = tc.gps_millis_to_datetime(gps_millis)
    np.testing.assert_array_equal(datetimes_back, datetimes_np)

    gps_millis = tc.datetime_to_gps_millis(datetimes_np)
    assert len(gps_millis) == num_checks
    assert isinstance(gps_millis,np.ndarray)
    assert gps_millis.dtype is np.dtype(np.float64)
    datetimes_back = tc.gps_millis_to_datetime(gps_millis)
    np.testing.assert_array_equal(datetimes_back, datetimes_np)

    # Datetime <--> UNIX Millis
    unix_millis = tc.datetime_to_unix_millis(datetimes_list)
    assert len(unix_millis) == num_checks
    assert isinstance(unix_millis,np.ndarray)
    assert unix_millis.dtype is np.dtype(np.float64)
    datetimes_back = tc.unix_millis_to_datetime(unix_millis)
    np.testing.assert_array_equal(datetimes_back, datetimes_np)

    unix_millis = tc.datetime_to_unix_millis(datetimes_np)
    assert len(unix_millis) == num_checks
    assert isinstance(unix_millis,np.ndarray)
    assert unix_millis.dtype is np.dtype(np.float64)
    datetimes_back = tc.unix_millis_to_datetime(unix_millis)
    np.testing.assert_array_equal(datetimes_back, datetimes_np)

    # Datetime <--> TOW
    gps_week, tow = tc.datetime_to_tow(datetimes_list)
    assert len(gps_week) == num_checks
    assert gps_week.dtype == np.int64
    assert len(tow) == num_checks
    assert isinstance(gps_week,np.ndarray)
    assert isinstance(tow,np.ndarray)
    datetimes_back = tc.tow_to_datetime(gps_week, tow)
    np.testing.assert_array_equal(datetimes_back, datetimes_np)

    gps_week, tow = tc.datetime_to_tow(datetimes_np)
    assert len(gps_week) == num_checks
    assert gps_week.dtype == np.int64
    assert len(tow) == num_checks
    assert isinstance(gps_week,np.ndarray)
    assert isinstance(tow,np.ndarray)
    datetimes_back = tc.tow_to_datetime(gps_week, tow)
    np.testing.assert_array_equal(datetimes_back, datetimes_np)

    # GPS Millis <--> UNIX Millis
    unix_millis = tc.gps_to_unix_millis(gps_millis.tolist())
    assert len(unix_millis) == num_checks
    assert isinstance(unix_millis,np.ndarray)
    assert unix_millis.dtype is np.dtype(np.float64)
    gps_millis_back = tc.unix_to_gps_millis(unix_millis)
    np.testing.assert_array_equal(gps_millis_back, gps_millis)

    unix_millis = tc.gps_to_unix_millis(gps_millis)
    assert len(unix_millis) == num_checks
    assert isinstance(unix_millis,np.ndarray)
    assert unix_millis.dtype is np.dtype(np.float64)
    gps_millis_back = tc.unix_to_gps_millis(unix_millis)
    np.testing.assert_array_equal(gps_millis_back, gps_millis)

    # GPS Millis <--> TOW
    gps_week, tow = tc.gps_millis_to_tow(gps_millis.tolist())
    assert len(gps_week) == num_checks
    assert gps_week.dtype == np.int64
    assert len(tow) == num_checks
    assert isinstance(gps_week,np.ndarray)
    assert isinstance(tow,np.ndarray)
    gps_millis_back = tc.tow_to_gps_millis(gps_week, tow)
    np.testing.assert_array_equal(gps_millis_back, gps_millis)

    gps_week, tow = tc.gps_millis_to_tow(gps_millis)
    assert len(gps_week) == num_checks
    assert gps_week.dtype == np.int64
    assert len(tow) == num_checks
    assert isinstance(gps_week,np.ndarray)
    assert isinstance(tow,np.ndarray)
    gps_millis_back = tc.tow_to_gps_millis(gps_week, tow)
    np.testing.assert_array_equal(gps_millis_back, gps_millis)

    # UNIX Millis <--> TOW
    gps_week, tow = tc.unix_millis_to_tow(unix_millis.tolist())
    assert len(gps_week) == num_checks
    assert gps_week.dtype == np.int64
    assert len(tow) == num_checks
    assert isinstance(gps_week,np.ndarray)
    assert isinstance(tow,np.ndarray)
    unix_millis_back = tc.tow_to_unix_millis(gps_week, tow)
    np.testing.assert_array_equal(unix_millis_back, unix_millis)

    gps_week, tow = tc.unix_millis_to_tow(unix_millis)
    assert len(gps_week) == num_checks
    assert gps_week.dtype == np.int64
    assert len(tow) == num_checks
    assert isinstance(gps_week,np.ndarray)
    assert isinstance(tow,np.ndarray)
    unix_millis_back = tc.tow_to_unix_millis(gps_week, tow)
    np.testing.assert_array_equal(unix_millis_back, unix_millis)

def test_zero_arrays():
    """Test zero array conversions between time types.

    """
    datetimes_list = datetime(np.random.randint(1981,2024),
                              np.random.randint(1,13),
                              np.random.randint(1,29),
                              np.random.randint(0,24),
                              np.random.randint(0,60),
                              np.random.randint(0,60),
                              np.random.randint(0,1000000),
                              tzinfo=timezone.utc
                              )
    datetimes_np = np.array(datetimes_list)

    # Datetime <--> GPS Millis
    gps_millis = tc.datetime_to_gps_millis(datetimes_np)
    datetimes_back = tc.gps_millis_to_datetime(np.array(gps_millis))
    np.testing.assert_array_equal(datetimes_back, datetimes_np)

    # Datetime <--> UNIX Millis
    unix_millis = tc.datetime_to_unix_millis(datetimes_np)
    datetimes_back = tc.unix_millis_to_datetime(np.array(unix_millis))
    np.testing.assert_array_equal(datetimes_back, datetimes_np)

    # Datetime <--> TOW
    gps_week, tow = tc.datetime_to_tow(datetimes_np)
    datetimes_back = tc.tow_to_datetime(np.array(gps_week),
                                            np.array(tow))
    np.testing.assert_array_equal(datetimes_back, datetimes_np)

    # GPS Millis <--> UNIX Millis
    unix_millis = tc.gps_to_unix_millis(np.array(gps_millis))
    gps_millis_back = tc.unix_to_gps_millis(np.array(unix_millis))
    np.testing.assert_array_equal(gps_millis_back, gps_millis)

    # GPS Millis <--> TOW
    gps_week, tow = tc.gps_millis_to_tow(np.array(gps_millis))
    gps_millis_back = tc.tow_to_gps_millis(np.array(gps_week),
                                            np.array(tow))
    np.testing.assert_array_equal(gps_millis_back, gps_millis)

    # UNIX Millis <--> TOW
    gps_week, tow = tc.unix_millis_to_tow(np.array(unix_millis))
    unix_millis_back = tc.tow_to_unix_millis(np.array(gps_week),
                                            np.array(tow))
    np.testing.assert_array_equal(unix_millis_back, unix_millis)
