"""Timing conversions between reference frames.

"""

__authors__ = "Ashwin Kanhere, Sriramya Bhamidipati, Shubh Gupta"
__date__ = "28 Jul 2022"

from datetime import datetime, timedelta, timezone
import warnings

import numpy as np

from gnss_lib_py.utils.constants import GPS_EPOCH_0, WEEKSEC

# reference datetime that is considered as start of UTC epoch
UNIX_EPOCH_0 = datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)

# Manually need to add leapSeconds when needed for future ones
LEAPSECONDS_TABLE = [datetime(2017, 1, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(2015, 7, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(2012, 7, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(2009, 1, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(2006, 1, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(1999, 1, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(1997, 7, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(1996, 1, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(1994, 7, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(1993, 7, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(1992, 7, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(1991, 1, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(1990, 1, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(1988, 1, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(1985, 7, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(1983, 7, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(1982, 7, 1, 0, 0, tzinfo=timezone.utc),
                     datetime(1981, 7, 1, 0, 0, tzinfo=timezone.utc),
                     GPS_EPOCH_0]

def get_leap_seconds(gps_time):
    """Compute leap seconds to be added in time conversions.

    Computed by comparing the time to LEAPSECONDS_TABLE.

    Parameters
    ----------
    t_secs : float or datetime.datetime
        Time of clock can be float [ms] or datetime.datetime object.

    Returns
    -------
    out_leapsecs : float
        Leap seconds at given time [s].

    """
    if isinstance(gps_time, datetime):
        curr_time = gps_time
    else:
        curr_time = GPS_EPOCH_0 + timedelta(milliseconds=gps_time)
        curr_time = curr_time.replace(tzinfo=timezone.utc)
    curr_time = _check_tzinfo(curr_time)
    if curr_time < GPS_EPOCH_0:
        raise RuntimeError("Need input time after GPS epoch " \
                           + str(GPS_EPOCH_0))
    for row_num, time_frame in enumerate(LEAPSECONDS_TABLE):
        if curr_time >= time_frame:
            out_leapsecs = len(LEAPSECONDS_TABLE)-1 - row_num
            break
    return out_leapsecs

def gps_millis_to_tow(millis, add_leap_secs=False, verbose=False):
    """Convert milliseconds since GPS epoch to GPS week number and time.

    The initial GPS epoch is defined by the variable GPS_EPOCH_0 at
    which the week number is assumed to be 0.

    Parameters
    ----------
    millis : float
        Float object for Time of Clock [ms].
    add_leap_secs : bool
        Flag for whether output is in UTC seconds or GPS seconds.
    verbose : bool
        Flag for whether to print that leapseconds were added.

    Returns
    -------
    gps_week : int
        GPS week.
    tow : float
        GPS time of week [s].

    """
    gps_week, tow = divmod(millis, 7*86400*1000)
    tow = tow / 1000.0
    if add_leap_secs:
        out_leapsecs = get_leap_seconds(millis)
        if verbose: #pragma: no cover
            print('leapSecs added')
        tow = tow + out_leapsecs

    return int(gps_week), tow

def datetime_to_tow(t_datetime, add_leap_secs=True, verbose=False):
    """Convert Python datetime object to GPS Week and time of week.

    Parameters
    ----------
    t : datetime.datetime
        Datetime object for Time of Clock.
    add_leap_secs : bool
        Flag for whether output is in UTC seconds or GPS seconds.
    verbose : bool
        Flag for whether to print that leapseconds were added.

    Returns
    -------
    gps_week : int
        GPS week.
    tow : float
        GPS time of week [s].

    """
    t_datetime = _check_tzinfo(t_datetime)
    if t_datetime < GPS_EPOCH_0:
        raise RuntimeError("Input time must be after GPS epoch " \
                         + str(GPS_EPOCH_0))
    if add_leap_secs:
        out_leapsecs = get_leap_seconds(t_datetime)
        t_datetime = t_datetime + timedelta(seconds=out_leapsecs)
        if verbose: # pragma: no cover
            print("leapSecs added")
    gps_week = (t_datetime - GPS_EPOCH_0).days // 7

    tow = ((t_datetime - GPS_EPOCH_0) - timedelta(gps_week* 7.0)).total_seconds()

    return int(gps_week), tow


def tow_to_datetime(gps_week, tow, rem_leap_secs=True):
    """Convert GPS week and time of week (seconds) to datetime.

    Parameters
    ----------
    gps_week : int
        GPS week.
    tow : float
        GPS time of week [s].
    rem_leap_secs : bool
        Flag on whether to remove leap seconds from given tow.

    Returns
    -------
    t_datetime: datetime.datetime
        Datetime in UTC timezone, with or without leap seconds based on
        flag.
    """
    seconds_since_epoch = WEEKSEC * gps_week + tow
    t_datetime = GPS_EPOCH_0 + timedelta(seconds=seconds_since_epoch)
    if rem_leap_secs:
        leap_secs = get_leap_seconds(t_datetime)
        t_datetime = t_datetime - timedelta(seconds=leap_secs)
    return t_datetime

def tow_to_unix_millis(gps_week, tow):
    """Convert GPS week and time of week (seconds) to UNIX milliseconds.

    Convert GPS week and time of week (seconds) to milliseconds since
    UNIX epoch.
    Leap seconds will always be removed from tow because of offset between
    UTC and GPS clocks.

    Parameters
    ----------
    gps_week : int
        GPS week.
    tow : float
        GPS time of week [s].

    Returns
    -------
    unix_millis: float
        Milliseconds since UNIX epoch (midnight 1/1/1970 UTC).

    """
    #NOTE: Don't need to remove leapseconds here because they're
    # removed in tow_to_datetime
    t_utc = tow_to_datetime(gps_week, tow, rem_leap_secs=True)
    t_utc = t_utc.replace(tzinfo=timezone.utc)
    unix_millis = datetime_to_unix_millis(t_utc)
    return unix_millis


def tow_to_gps_millis(gps_week, tow):
    """Convert GPS week and time of week (seconds) to GPS milliseconds.

    Convert GPS week and time of week (seconds) to milliseconds since
    GPS epoch.
    No leap seconds adjustments are made because both times are in the
    same frame of reference.

    Parameters
    ----------
    gps_week : int
        GPS week.
    tow : float
        GPS time of week [s].

    Returns
    -------
    gps_millis: float
        Milliseconds since GPS epoch
        (midnight 6th January, 1980 UTC with leap seconds).

    """
    millis_since_epoch = 1000*(WEEKSEC * gps_week + tow)
    return millis_since_epoch


def datetime_to_unix_millis(t_datetime):
    """Convert datetime to milliseconds since UNIX Epoch (1/1/1970 UTC).

    If no timezone is specified, assumes UTC as timezone.

    Parameters
    ----------
    t_datetime : datetime.datetime
        UTC time as a datetime object.

    Returns
    -------
    unix_millis : float
        Milliseconds since UNIX Epoch (1/1/1970 UTC).

    """
    t_datetime = _check_tzinfo(t_datetime)
    unix_millis = 1000*(t_datetime - UNIX_EPOCH_0).total_seconds()
    return unix_millis

def datetime_to_gps_millis(t_datetime, add_leap_secs=True):
    """Convert datetime to milliseconds since GPS Epoch.

    GPS Epoch starts at the 6th January 1980.
    If no timezone is specified, assumes UTC as timezone and returns
    milliseconds in GPS time frame of reference by adding leap seconds.
    Milliseconds are not added when the flag add_leap_secs is False.

    Parameters
    ----------
    t_datetime : datetime.datetime
        UTC time as a datetime object.
    add_leap_secs : bool
        Flag for whether output is in UTC seconds or GPS seconds.

    Returns
    -------
    gps_millis : float
        Milliseconds since GPS Epoch (6th January 1980 GPS).

    """
    gps_week, tow = datetime_to_tow(t_datetime, add_leap_secs=add_leap_secs)
    gps_millis = tow_to_gps_millis(gps_week, tow)
    return gps_millis


def unix_millis_to_datetime(unix_millis):
    """Convert milliseconds since UNIX epoch (1/1/1970) to UTC datetime.

    Parameters
    ----------
    unix_millis : float
        Milliseconds that have passed since UTC epoch.

    Returns
    -------
    t_utc : datetime.datetime
        UTC time as a datetime object.
    """
    t_utc = UNIX_EPOCH_0 + timedelta(milliseconds=unix_millis)
    t_utc = t_utc.replace(tzinfo=timezone.utc)
    return t_utc


def unix_millis_to_tow(unix_millis):
    """Convert UNIX milliseconds to GPS week and time of week (seconds).

    UNIX milliseconds are since UNIX epoch (1/1/1970).
    Always adds leap seconds to convert from UTC to GPS time reference.

    Parameters
    ----------
    unix_millis : float
        Milliseconds that have passed since UTC epoch.

    Returns
    -------
    gps_week : int
        GPS week.
    tow : float
        GPS time of week [s].
    """
    t_utc = unix_millis_to_datetime(unix_millis)
    gps_week, tow = datetime_to_tow(t_utc, add_leap_secs=True)
    return int(gps_week), tow


def unix_to_gps_millis(unix_millis, add_leap_secs=True):
    """Convert milliseconds since UNIX epoch (1/1/1970) to GPS millis.

    Adds leap seconds by default but time can be kept in UTC frame by
    setting to False.

    Parameters
    ----------
    unix_millis : float
        Milliseconds that have passed since UTC epoch.
    add_leap_secs : bool
        Flag for whether output is in UTC seconds or GPS seconds.

    Returns
    -------
    gps_millis : float
        Milliseconds since GPS Epoch (6th January 1980 GPS).
    """
    # Add leapseconds should always be true here
    if isinstance(unix_millis, np.ndarray) and len(unix_millis) > 1:
        gps_millis = np.zeros_like(unix_millis)
        for t_idx, unix in enumerate(unix_millis):
            t_utc = unix_millis_to_datetime(unix)
            gps_millis[t_idx] = datetime_to_gps_millis(t_utc, add_leap_secs=add_leap_secs)
    else:
        t_utc = unix_millis_to_datetime(unix_millis)
        gps_millis = datetime_to_gps_millis(t_utc, add_leap_secs=add_leap_secs)
    return gps_millis


def gps_millis_to_datetime(gps_millis, rem_leap_secs=True):
    """Convert milliseconds since GPS epoch to datetime.

    GPS millis is from the start of the GPS in GPS reference.
    The initial GPS epoch is defined by the variable GPS_EPOCH_0 at
    which the week number is assumed to be 0.

    Parameters
    ----------
    millis : float
        Float object for Time of Clock [ms].
    rem_leap_secs : bool
        Flag for whether output is in UTC seconds or GPS seconds.

    Returns
    -------
    t_utc : datetime.datetime
        UTC time as a datetime object
    """
    gps_week, tow = gps_millis_to_tow(gps_millis, add_leap_secs=False)
    t_utc = tow_to_datetime(gps_week, tow, rem_leap_secs=rem_leap_secs)
    return t_utc


def gps_to_unix_millis(gps_millis, rem_leap_secs=True):
    """Convert milliseconds since GPS epoch to UNIX millis.

    GPS millis is from the start of the GPS in GPS reference.
    The initial GPS epoch is defined by the variable GPS_EPOCH_0 at
    which the week number is assumed to be 0.

    Parameters
    ----------
    millis : float
        Float object for Time of Clock [ms].
    rem_leap_secs : bool
        Flag for whether output is in UTC seconds or GPS seconds.

    Returns
    -------
    unix_millis : float
        Milliseconds since UNIX Epoch (1/1/1970 UTC)

    """
    #NOTE: Ensure that one of these methods is always adding/removing
    # leap seconds here
    if isinstance(gps_millis, np.ndarray) and len(gps_millis) > 1:
        unix_millis = np.zeros_like(gps_millis)
        for t_idx, gps in enumerate(gps_millis):
            t_utc = gps_millis_to_datetime(gps, rem_leap_secs=rem_leap_secs)
            unix_millis[t_idx] = datetime_to_unix_millis(t_utc)
    else:
        t_utc = gps_millis_to_datetime(gps_millis, rem_leap_secs=rem_leap_secs)
        unix_millis = datetime_to_unix_millis(t_utc)
    return unix_millis

def _check_tzinfo(t_datetime):
    """Raises warning if time doesn't have timezone and converts to UTC.

    Parameters
    ----------
    t_datetime : datetime.datetime
        Datetime object with no timezone, UTC timezone, or local
        timezone.

    Returns
    -------
    t_datetime: datetime.datetime
        Datetime object in UTC, added if no timezone was given and
        converted to UTC if datetime was in local timezone.

    """
    if not hasattr(t_datetime, 'tzinfo') or not t_datetime.tzinfo:
        warnings.warn("No time zone info found in datetime, assuming UTC",\
                        RuntimeWarning)
        t_datetime = t_datetime.replace(tzinfo=timezone.utc)
    if t_datetime.tzinfo != timezone.utc:
        t_datetime = t_datetime.astimezone(timezone.utc)
    return t_datetime
