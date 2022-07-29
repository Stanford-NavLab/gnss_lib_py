"""Timing conversions between reference frames.
"""

__authors__ = "Shubh Gupta, Ashwin Kanhere, Sriramya Bhamidipati"
__date__ = "28 Jul 2022"

from datetime import datetime, timedelta
import numpy as np

global GPSEPOCH0, GPSWEEK0, LEAPSECONDS_TABLE

# reference datetime that is considered as start GPS epoch 
GPSEPOCH0 = datetime(1980, 1, 6, 0, 0, 0, 0, None)
GPSWEEK0 = 0

# Manually need to add leapSeconds when needed for future ones
# TODO: There is a way to automatically extract leapSeconds
# https://github.com/eggert/tz/blob/master/leap-seconds.list
# https://gist.github.com/zed/92df922103ac9deb1a05
LEAPSECONDS_TABLE = np.transpose([[GPSEPOCH0, 0], 
                                  [datetime(1981, 7, 1, 0, 0), 1],
                                  [datetime(1982, 7, 1, 0, 0), 2], 
                                  [datetime(1983, 7, 1, 0, 0), 3], 
                                  [datetime(1985, 7, 1, 0, 0), 4], 
                                  [datetime(1988, 1, 1, 0, 0), 5], 
                                  [datetime(1990, 1, 1, 0, 0), 6], 
                                  [datetime(1991, 1, 1, 0, 0), 7], 
                                  [datetime(1992, 7, 1, 0, 0), 8], 
                                  [datetime(1993, 7, 1, 0, 0), 9], 
                                  [datetime(1994, 7, 1, 0, 0), 10], 
                                  [datetime(1996, 1, 1, 0, 0), 11], 
                                  [datetime(1997, 7, 1, 0, 0), 12], 
                                  [datetime(1999, 1, 1, 0, 0), 13], 
                                  [datetime(2006, 1, 1, 0, 0), 14], 
                                  [datetime(2009, 1, 1, 0, 0), 15], 
                                  [datetime(2012, 7, 1, 0, 0), 16], 
                                  [datetime(2015, 7, 1, 0, 0), 17], 
                                  [datetime(2017, 1, 1, 0, 0), 18]])

def get_leap_seconds(t, compare_dtime=True):
    """Compute leap seconds to be added in time conversions
    by comparing the time to LEAPSECONDS_TABLE.
    Parameters
    ----------
    t : seconds
        Float object for Time of Clock.
    compare_dtime : Bool
        Flag for whether output is in seconds or datetime/timedelta
    Returns
    -------
    out_leapsecs : float [s] or timedelta [datetime object] 
    """
    if compare_dtime: 
#         if t < GPSEPOCH0: 
#             raise RuntimeError("Need input time after GPS epoch "+ str(GPSEPOCH0))
        for ii in reversed(range(len(LEAPSECONDS_TABLE[0,:]))):
            if t >= LEAPSECONDS_TABLE[0,ii] :
                out_leapsecs = timedelta(seconds = LEAPSECONDS_TABLE[1,ii])
                return out_leapsecs
    else:
#         if t < 0: 
#             raise RuntimeError("Need input time greater than 0")
        for ii in reversed(range(len(LEAPSECONDS_TABLE[0,:]))):
            if t >= 1000*(LEAPSECONDS_TABLE[0,ii] - GPSEPOCH0).total_seconds():
                out_leapsecs = LEAPSECONDS_TABLE[1,ii]
                return out_leapsecs        

def millissincegpsepoch_to_tow(t, add_leap_secs = True):
    """Convert milli seconds since GPS epoch (defined by variable 
    GPSEPOCH0 and GPSWEEK0) to GPS Week and time of week.
    Parameters
    ----------
    t : seconds
        Float object for Time of Clock.
    add_leapSeconds : Bool
        Flag for whether output is in UTC seconds or GPS seconds
    Returns
    -------
    wk : float
        GPS week
    tow : float
        GPS time of week [s]
    """
    wk, tow = divmod(t, 7*86400*1000)
    tow = tow / 1000.0
    if add_leap_secs: 
        out_leapsecs = get_leap_seconds(t, compare_dtime=False)
        print('leapSecs added', out_leapsecs)
        tow = tow + out_leapsecs
        
    return wk, tow

def datetime_to_tow(t, add_leap_secs = True):
    """Convert Python datetime object to GPS Week and time of week.
    Parameters
    ----------
    t : datetime.datetime
        Datetime object for Time of Clock.
    add_leapSeconds : Bool
        Flag for whether output is in UTC seconds or GPS seconds
    Returns
    -------
    wk : float
        GPS week
    tow : float
        GPS time of week [s]
    """
    # DateTime to GPS week and TOW
    #TODO: Not sure if we expect tzinfo or why it has to be changed to none!
#     if t < GPSEPOCH0: 
#         raise RuntimeError("Need input time after GPS epoch "+ str(GPSEPOCH0))
    if hasattr(t, 'tzinfo'):
        t = t.replace(tzinfo=None)
    if add_leap_secs:
        out_leapsecs = get_leap_seconds(t)
        print('leapSecs added', out_leapsecs)
        t = t + out_leapsecs
    wk = (t - GPSEPOCH0).days // 7 + GPSWEEK0
    tow = ((t - GPSEPOCH0) - timedelta((wk - GPSWEEK0) * 7.0)).total_seconds()
    
    return wk, tow     