"""Timing conversions between reference frames.

"""

__authors__ = "Shubh Gupta, Ashwin Kanhere, Sriramya Bhamidipati"
__date__ = "28 Jul 2022"

from datetime import datetime, timedelta

import numpy as np

# reference datetime that is considered as start GPS epoch
GPS_EPOCH_0 = datetime(1980, 1, 6, 0, 0, 0, 0, None)
GPS_WEEK_0 = 0

# Manually need to add leapSeconds when needed for future ones
LEAPSECONDS_TABLE = np.transpose([[GPS_EPOCH_0, 0],
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

def get_leap_seconds(t_secs, compare_dtime=True):
    """Compute leap seconds to be added in time conversions.

    Computed by comparing the time to LEAPSECONDS_TABLE.

    Parameters
    ----------
    t_secs : float or datetime??
        Float object for Time of Clock [s].
    compare_dtime : Bool
        Flag for whether output is in seconds or datetime/timedelta.

    Returns
    -------
    out_leapsecs : float [s] or timedelta [datetime object].

    """
    if compare_dtime:
#         if t < GPS_EPOCH_0:
#             raise RuntimeError("Need input time after GPS epoch "+ str(GPS_EPOCH_0))
        for row in reversed(range(len(LEAPSECONDS_TABLE[0,:]))):
            if t_secs >= LEAPSECONDS_TABLE[0,row] :
                out_leapsecs = timedelta(seconds = LEAPSECONDS_TABLE[1,row])
                return out_leapsecs
    else:
#         if t < 0:
#             raise RuntimeError("Need input time greater than 0")
        for row in reversed(range(len(LEAPSECONDS_TABLE[0,:]))):
            if t_secs >= 1000*(LEAPSECONDS_TABLE[0,row] - GPS_EPOCH_0).total_seconds():
                out_leapsecs = LEAPSECONDS_TABLE[1,row]
                return out_leapsecs

def millis_since_gps_epoch_to_tow(millis, add_leap_secs = True):
    """Convert milliseconds since GPS epoch to GPS week number and time.

    The initial GPS week is defined by the variables GPS_EPOCH_0 and
    GPS_WEEK_0.

    Parameters
    ----------
    millis : float
        Float object for Time of Clock [ms].
    add_leapSeconds : bool
        Flag for whether output is in UTC seconds or GPS seconds.

    Returns
    -------
    wk : float
        GPS week
    tow : float
        GPS time of week [s].

    """
    gps_week, tow = divmod(millis, 7*86400*1000)
    tow = tow / 1000.0
    if add_leap_secs:
        out_leapsecs = get_leap_seconds(millis, compare_dtime=False)
        print('leapSecs added', out_leapsecs)
        tow = tow + out_leapsecs

    return gps_week, tow

def datetime_to_tow(t_datetime, add_leap_secs = True):
    """Convert Python datetime object to GPS Week and time of week.

    Parameters
    ----------
    t : datetime.datetime
        Datetime object for Time of Clock.
    add_leapSeconds : bool
        Flag for whether output is in UTC seconds or GPS seconds.

    Returns
    -------
    wk : float
        GPS week
    tow : float
        GPS time of week [s]

    """
    # DateTime to GPS week and TOW
    #TODO: Not sure if we expect tzinfo or why it has to be changed to none!
#     if t < GPS_EPOCH_0:
#         raise RuntimeError("Need input time after GPS epoch "+ str(GPS_EPOCH_0))
    if hasattr(t_datetime, 'tzinfo'):
        t_datetime = t_datetime.replace(tzinfo=None)
    if add_leap_secs:
        out_leapsecs = get_leap_seconds(t_datetime)
        print('leapSecs added ', out_leapsecs, type(out_leapsecs))
        t_datetime = t_datetime + out_leapsecs
    gps_week = (t_datetime - GPS_EPOCH_0).days // 7 + GPS_WEEK_0

    tow = ((t_datetime - GPS_EPOCH_0) - timedelta((gps_week - GPS_WEEK_0) * 7.0)).total_seconds()

    return gps_week, tow
