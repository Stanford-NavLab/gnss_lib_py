"""Test timing conversions between reference frames.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "28 Jul 2022"

from datetime import datetime, timedelta

import gnss_lib_py.utils.time_conversions as tc

def test_get_leap_seconds():
    """Test to validate leap seconds based on input time.

    """
    input_millis = 1000.0 * (datetime(2022, 7, 28, 0, 0) - tc.GPS_EPOCH_0).total_seconds()
    valseconds = tc.get_leap_seconds(input_millis, compare_dtime=False)
    assert valseconds == 18

    buffer_secs = 3.0
    num_leapsecarray = len(tc.LEAPSECONDS_TABLE[0,:])
    for row in range(num_leapsecarray):
        input_datetime = tc.LEAPSECONDS_TABLE[0,row] + timedelta(seconds = buffer_secs)
        valdatetime = tc.get_leap_seconds(input_datetime, compare_dtime=True)
        assert valdatetime == timedelta(seconds=tc.LEAPSECONDS_TABLE[1,row])

        input_millis = 1000.0 * ( (tc.LEAPSECONDS_TABLE[0,row] \
                                 - tc.GPS_EPOCH_0).total_seconds() \
                              + buffer_secs)
        valseconds = tc.get_leap_seconds(input_millis, compare_dtime=False)
        assert valseconds == tc.LEAPSECONDS_TABLE[1,row]

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

    input_time = datetime(2022, 7, 28, 12, 0, 0)
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
    output_wk, output_tow = tc.millis_since_gps_epoch_to_tow(1151280017.0*1000.0, add_leap_secs = False)
    assert output_wk == 1903.0
    assert output_tow == 345617.0

    output_wk2, output_tow2 = tc.millis_since_gps_epoch_to_tow(1151280017.0*1000.0, add_leap_secs = True)
    assert output_wk2 == 1903.0
    assert output_tow2 - output_tow == 17.0

    output_wk3, output_tow3 = tc.millis_since_gps_epoch_to_tow(1303041618.0*1000.0, add_leap_secs = False)
    assert output_wk3 == 2154.0
    assert output_tow3 == 302418.0
