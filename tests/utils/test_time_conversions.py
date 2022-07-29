"""Timing conversions between reference frames.
"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "28 Jul 2022"

from datetime import datetime, timedelta
import pytest 

from gnss_lib_py.utils.time_conversions import get_leap_seconds, millissincegpsepoch_to_tow, datetime_to_tow
from gnss_lib_py.utils.time_conversions import LEAPSECONDS_TABLE, GPSEPOCH0, GPSWEEK0

def test_get_leap_seconds():
    """Test to validate leap seconds based on input time.
    """
    input_millis = 1000.0 * (datetime(2022, 7, 28, 0, 0) - GPSEPOCH0).total_seconds()
    valseconds = get_leap_seconds(input_millis, compare_dtime=False)
    assert valseconds == 18

    buffer_secs = 3.0
    num_leapsecarray = len(LEAPSECONDS_TABLE[0,:])
    for ii in range(num_leapsecarray):
        input_datetime = LEAPSECONDS_TABLE[0,ii] + timedelta(seconds = buffer_secs)
        valdatetime = get_leap_seconds(input_datetime, compare_dtime=True)
        assert valdatetime == timedelta(seconds=LEAPSECONDS_TABLE[1,ii])

        input_millis = 1000.0 * ( (LEAPSECONDS_TABLE[0,ii] - GPSEPOCH0).total_seconds() + buffer_secs)
        valseconds = get_leap_seconds(input_millis, compare_dtime=False)
        assert valseconds == LEAPSECONDS_TABLE[1,ii]

#     with pytest.raises(RuntimeError):
#         buffer_secs = 3.0
#         input_datetime = LEAPSECONDS_TABLE[0,0] - timedelta(seconds = buffer_secs)
#         get_leap_seconds(input_datetime)
        
def test_datetime_to_tow():
    """Test that datetime conversion to GPS or UTC seconds does not fail.
    
    References
    ----------
    .. [1] https://www.labsat.co.uk/index.php/en/gps-time-calculator
    """
    
    input_time = datetime(2022, 7, 28, 12, 0, 0)
    output_wk, output_tow = datetime_to_tow(input_time, add_leap_secs = True)           
    assert output_wk == 2220
    assert output_tow == 388818.0

    output_wk2, output_tow2 = datetime_to_tow(input_time, add_leap_secs = False)           
    assert output_wk2 == 2220
    assert (output_tow - output_tow2) == 18.0

#     with pytest.raises(RuntimeError):
#         buffer_secs = 3.0
#         input_datetime = LEAPSECONDS_TABLE[0,0] - timedelta(seconds = buffer_secs)
#         datetime_to_tow(input_datetime)        

def test_millissincegpsepoch_to_tow(): 
    """Test that conversion from milliseconds since GPS epoch to GPS or UTC seconds of the week does not fail.
    Given a UTC time epoch, [1] provides seconds since GPS epoch while [2] gives GPS seconds of the week
     
    References
    ----------
    .. [1] https://www.andrews.edu/~tzs/timeconv/timeconvert.php? (Accessed as of July 28, 2022)
    .. [2] https://www.labsat.co.uk/index.php/en/gps-time-calculator (Accessed as of July 28, 2022)
   """
    # These two are for 30th june 2016 (1151280017) and leap seconds: 17
    output_wk, output_tow = millissincegpsepoch_to_tow(1151280017.0*1000.0, add_leap_secs = False)
    assert output_wk == 1903.0
    assert output_tow == 345617.0 

    output_wk2, output_tow2 = millissincegpsepoch_to_tow(1151280017.0*1000.0, add_leap_secs = True)
    assert output_wk2 == 1903.0
    assert output_tow2 - output_tow == 17.0

    output_wk3, output_tow3 = millissincegpsepoch_to_tow(1303041618.0*1000.0, add_leap_secs = False)
    assert output_wk3 == 2154.0
    assert output_tow3 == 302418.0
    