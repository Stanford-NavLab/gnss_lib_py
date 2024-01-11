"""Timing conversions between reference frames for times.

Frame options are described in detail at on the documentation website:
https://gnss-lib-py.readthedocs.io/en/latest/reference/reference.html#timing-conventions

Frame options include:
    - gps_millis : GPS milliseconds
    - unix_millis : UNIX milliseconds
    - tow : Time of week which includes GPS week and time of week in secs
    - datetime : Time assumed to be in UTC timezone

"""

__authors__ = "Derek Knowles, Ashwin Kanhere, Sriramya Bhamidipati, Shubh Gupta"
__date__ = "10 May, 2023"

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
    gps_time : float or datetime.datetime
        Time of clock can be float [ms] or datetime.datetime object.

    Returns
    -------
    out_leapsecs : float
        Leap seconds at given time [s].

    """
    if isinstance(gps_time, datetime):
        curr_time = gps_time
    else:
        curr_time = GPS_EPOCH_0 + timedelta(milliseconds=float(gps_time))
        curr_time = curr_time.replace(tzinfo=timezone.utc)
    curr_time = tzinfo_to_utc(curr_time)
    if curr_time < GPS_EPOCH_0:
        raise RuntimeError("Need input time after GPS epoch " \
                           + str(GPS_EPOCH_0))
    for row_num, time_frame in enumerate(LEAPSECONDS_TABLE):
        if curr_time >= time_frame:
            out_leapsecs = len(LEAPSECONDS_TABLE)-1 - row_num
            break
    return out_leapsecs


def gps_millis_to_tow(millis):
    """Convert milliseconds since GPS epoch to GPS week number and time.

    The initial GPS epoch is defined by the variable GPS_EPOCH_0 at
    which the week number is assumed to be 0.

    Both of these times are in the GPS time frame of reference and as a
    result, leap seconds do not need to be accounted for here.

    Parameters
    ----------
    millis : float or array-like of floats
        Float object for Time of Clock [ms].

    Returns
    -------
    gps_weeks : int or np.ndarray
        GPS week. Either `int` or `np.ndarray` with `dtype = int`.
    tows : float or np.ndarray
        GPS time of week [s]. Either `float` or `np.ndarray` with
        `dtype = float`.

    """

    if np.issubdtype(type(millis), np.integer) \
        or np.issubdtype(type(millis), float):
        millis = [millis]
    if isinstance(millis,np.ndarray) \
        and len(np.atleast_1d(millis)) == 1:
        millis = [millis.item()]

    gps_weeks = []
    tows = []

    for milli in millis:
        gps_week, tow = divmod(milli, 7*86400*1000)
        tow = tow / 1000.0

        gps_weeks.append(np.int64(gps_week))
        tows.append(tow)

    gps_weeks = np.squeeze(np.array(gps_weeks, dtype=np.int64))
    tows = np.squeeze(np.array(tows))
    return gps_weeks, tows


def datetime_to_tow(t_datetimes):
    """Convert Python datetime object to GPS Week and time of week.

    For the `gnss_lib_py` convention, except for specific applications,
    we assume that the time recorded by the `datetime.datetime` object
    is UTC time. As a result, on converting to TOW, we add leap seconds.

    Parameters
    ----------
    t_datetimes : datetime.datetime or array-like of datetime.datetime
        Datetime object for Time of Clock, assumed to be in UTC time frame.

    Returns
    -------
    gps_weeks : int or np.ndarray
        GPS week. Either `int` or `np.ndarray` with `dtype = int`.
    tows : float or np.ndarray
        GPS time of week [s]. Either `float` or `np.ndarray` with
        `dtype = float`.

    """
    if isinstance(t_datetimes,datetime):
        t_datetimes = [t_datetimes]
    if isinstance(t_datetimes,np.ndarray) \
        and len(np.atleast_1d(t_datetimes)) == 1:
        t_datetimes = [t_datetimes.item()]

    gps_weeks = []
    tows = []

    for t_datetime in t_datetimes:
        t_datetime = tzinfo_to_utc(t_datetime)
        if t_datetime < GPS_EPOCH_0:
            raise RuntimeError("Input time must be after GPS epoch " \
                             + str(GPS_EPOCH_0))
        out_leapsecs = get_leap_seconds(t_datetime)
        t_datetime = t_datetime + timedelta(seconds=out_leapsecs)
        gps_week = (t_datetime - GPS_EPOCH_0).days // 7

        tow = ((t_datetime - GPS_EPOCH_0) - timedelta(gps_week* 7.0)).total_seconds()

        gps_weeks.append(np.int64(gps_week))
        tows.append(tow)

    gps_weeks = np.squeeze(np.array(gps_weeks, dtype=np.int64))
    tows = np.squeeze(np.array(tows))
    return gps_weeks, tows


def tow_to_datetime(gps_weeks, tows):
    """Convert GPS week and time of week (seconds) to datetime.

    Because we assume that `datetime.datetime` objects are in UTC time,
    leap seconds are removed from the given TOW.

    Parameters
    ----------
    gps_weeks : int or array-like of ints
        GPS week.
    tows : float or array-like of floats
        GPS time of week [s].

    Returns
    -------
    t_datetimes: datetime.datetime or np.ndarray<datetime.datetime>
        Datetime in UTC timezone, with or without leap seconds based on
        flag.
        If single gps_weeks, tows is given, output is single
        datetime.datetime instance and `np.ndarray` of `datetime.datetime`
        if multiple inputs are given.
    """

    if np.issubdtype(type(gps_weeks), np.integer):
        gps_weeks = [gps_weeks]
    if np.issubdtype(type(tows), np.integer) \
        or np.issubdtype(type(tows), float):
        tows = [tows]
    if isinstance(gps_weeks,np.ndarray) \
        and len(np.atleast_1d(gps_weeks)) == 1:
        gps_weeks = [gps_weeks.item()]
    if isinstance(tows,np.ndarray) \
        and len(np.atleast_1d(tows)) == 1:
        tows = [tows.item()]

    t_datetimes = []

    for t_idx, gps_week in enumerate(gps_weeks):
        tow = tows[t_idx]

        seconds_since_epoch = WEEKSEC * gps_week + tow
        t_datetime = GPS_EPOCH_0 + timedelta(seconds=seconds_since_epoch)
        leap_secs = get_leap_seconds(t_datetime)
        t_datetime = t_datetime - timedelta(seconds=leap_secs)

        t_datetimes.append(t_datetime)

    # np.squeeze(np.array(t_datetimes)) not used because other conversions
    # relying on this conversion break on doing so
    if len(t_datetimes) == 1:
        return t_datetimes[0]
    return np.array(t_datetimes)


def tow_to_unix_millis(gps_weeks, tows):
    """Convert GPS week and time of week (seconds) to UNIX milliseconds.

    Convert GPS week and time of week (seconds) to milliseconds since
    UNIX epoch.
    Leap seconds will always be removed from tow because GPS millis is a
    continuous time reference while unix millis adjust for leap seconds.

    Parameters
    ----------
    gps_weeks : int or array-like of int
        GPS week.
    tows : float or array-like of int
        GPS time of week [s].

    Returns
    -------
    unix_millis: float or np.ndarray
        Milliseconds since UNIX epoch (midnight 1/1/1970 UTC). Either
        `float` or `np.ndarray` with `dtype = float`.


    """
    #NOTE: Don't need to remove leapseconds here because they're
    # removed in tow_to_datetime

    if np.issubdtype(type(gps_weeks), np.integer):
        gps_weeks = [gps_weeks]
    if np.issubdtype(type(tows), np.integer) \
        or np.issubdtype(type(tows), float):
        tows = [tows]
    if isinstance(gps_weeks,np.ndarray) \
        and len(np.atleast_1d(gps_weeks)) == 1:
        gps_weeks = [gps_weeks.item()]
    if isinstance(tows,np.ndarray) \
        and len(np.atleast_1d(tows)) == 1:
        tows = [tows.item()]

    unix_millis = []

    for t_idx, gps_week in enumerate(gps_weeks):
        tow = tows[t_idx]

        t_utc = tow_to_datetime(gps_week, tow)
        t_utc = t_utc.replace(tzinfo=timezone.utc)
        unix_milli = datetime_to_unix_millis(t_utc)
        unix_millis.append(unix_milli)

    unix_millis = np.squeeze(np.array(unix_millis))
    return unix_millis


def tow_to_gps_millis(gps_week, tow):
    """Convert GPS week and time of week (seconds) to GPS milliseconds.

    Convert GPS week and time of week (seconds) to milliseconds since
    GPS epoch.
    No leap seconds adjustments are made because both times are in the
    same frame of reference.

    Parameters
    ----------
    gps_week : int or array-like of int
        GPS week.
    tow : float or array-like of floats
        GPS time of week [s].

    Returns
    -------
    gps_millis: float or np.ndarray
        Milliseconds since GPS epoch
        (midnight 6th January, 1980 UTC with leap seconds). Either
        `float` or `np.ndarray` with `dtype = float`.


    """
    gps_millis = 1000*(WEEKSEC * gps_week + tow)

    gps_millis = np.squeeze(np.asarray(gps_millis, dtype=np.float64))
    return gps_millis


def datetime_to_unix_millis(t_datetimes):
    """Convert datetime to milliseconds since UNIX Epoch (1/1/1970 UTC).

    If no timezone is specified, assumes UTC as timezone. This function
    does not add leapseconds to the datetime to get unix millis.

    Parameters
    ----------
    t_datetimes : datetime.datetime or array-like of datetime.datetime
        UTC time as a datetime object.

    Returns
    -------
    unix_millis : float or np.ndarray
        Milliseconds since UNIX Epoch (1/1/1970 UTC). Either `float` or
        `np.ndarray` with `dtype = float`.


    """

    if isinstance(t_datetimes,datetime):
        t_datetimes = [t_datetimes]
    if isinstance(t_datetimes,np.ndarray) \
        and len(np.atleast_1d(t_datetimes)) == 1:
        t_datetimes = [t_datetimes.item()]

    unix_millis_list = []

    for t_datetime in t_datetimes:
        t_datetime = tzinfo_to_utc(t_datetime)
        unix_millis = 1000*(t_datetime - UNIX_EPOCH_0).total_seconds()
        unix_millis_list.append(unix_millis)

    unix_millis_list = np.squeeze(np.asarray(unix_millis_list,
                                             dtype=np.float64))
    return unix_millis_list


def datetime_to_gps_millis(t_datetimes):
    """Convert datetime to milliseconds since GPS Epoch.

    GPS Epoch starts at the 6th January 1980.
    If no timezone is specified, assumes UTC as timezone and returns
    milliseconds in GPS time frame of reference by adding leap seconds.
    Leap seconds are always added because UTC time is adjusted for leap
    seconds while GPS milliseconds are not.

    Parameters
    ----------
    t_datetime : datetime.datetime or array-like of datetime.datetime
        UTC time as a datetime object.

    Returns
    -------
    gps_millis : float or np.ndarray
        Milliseconds since GPS Epoch (6th January 1980 GPS). Either
        `float` or `np.ndarray` with `dtype = float`.


    """
    gps_weeks, tows = datetime_to_tow(t_datetimes)
    gps_millis = tow_to_gps_millis(gps_weeks, tows)
    return gps_millis


def unix_millis_to_datetime(unix_millis):
    """Convert milliseconds since UNIX epoch (1/1/1970) to UTC datetime.

    Parameters
    ----------
    unix_millis : float or int or np.ndarray
        Milliseconds that have passed since UTC epoch. `np.ndarray` of
        `float` or `int` to be passed in for cases with multiple values.

    Returns
    -------
    t_utc : datetime.datetime or np.ndarray
        UTC time as a datetime object. Either `datetime.datetime` or
        `np.ndarray` with `dtype = datetime.datetime`.
    """
    if np.issubdtype(type(unix_millis), np.integer):
        unix_millis = float(unix_millis)
    if np.issubdtype(type(unix_millis), np.integer) \
        or np.issubdtype(type(unix_millis), float):
        unix_millis = [unix_millis]
    if isinstance(unix_millis,np.ndarray) \
        and len(np.atleast_1d(unix_millis)) == 1:
        unix_millis = [unix_millis.item()]

    t_utcs = []

    for unix_milli in unix_millis:
        t_utc = UNIX_EPOCH_0 + timedelta(milliseconds=unix_milli)
        t_utc = t_utc.replace(tzinfo=timezone.utc)
        t_utcs.append(t_utc)

    if len(t_utcs) == 1:
        return t_utcs[0]
    return np.array(t_utcs)


def unix_millis_to_tow(unix_millis):
    """Convert UNIX milliseconds to GPS week and time of week (seconds).

    UNIX milliseconds are since UNIX epoch (1/1/1970).
    Always adds leap seconds to convert from UTC to GPS time reference.

    Parameters
    ----------
    unix_millis : float or array-like of float
        Milliseconds that have passed since UTC epoch.

    Returns
    -------
    gps_week : int or np.ndarray
        GPS week. Either `int` or `np.ndarray` with `dtype = int`.
    tow : float or np.ndarray
        GPS time of week [s]. Either `float` or `np.ndarray` with
        `dtype = float`.
    """

    t_utc = unix_millis_to_datetime(unix_millis)
    gps_week, tow = datetime_to_tow(t_utc)
    return np.int64(gps_week), tow


def unix_to_gps_millis(unix_millis):
    """Convert milliseconds since UNIX epoch (1/1/1970) to GPS millis.

    Adds leap seconds by default but time can be kept in UTC frame by
    setting to False.

    Parameters
    ----------
    unix_millis : float or array-like of float
        Milliseconds that have passed since UTC epoch.

    Returns
    -------
    gps_millis : int or np.ndarray
        Milliseconds since GPS Epoch (6th January 1980 GPS). Either
        `int` or `np.ndarray` with `dtype = int`.
    """
    # Add leapseconds should always be true here
    if isinstance(unix_millis, np.ndarray) \
        and len(np.atleast_1d(unix_millis)) > 1:
        gps_millis = np.zeros_like(unix_millis)
        for t_idx, unix in enumerate(unix_millis):
            t_utc = unix_millis_to_datetime(unix)
            gps_millis[t_idx] = datetime_to_gps_millis(t_utc)
        gps_millis = gps_millis.astype(np.float64)
    else:
        t_utc = unix_millis_to_datetime(unix_millis)
        gps_millis = np.float64(datetime_to_gps_millis(t_utc))
    return gps_millis


def gps_millis_to_datetime(gps_millis):
    """Convert milliseconds since GPS epoch to datetime.

    GPS millis is from the start of the GPS in GPS reference.
    The initial GPS epoch is defined by the variable GPS_EPOCH_0 at
    which the week number is assumed to be 0.

    The :code:`datetime` instances are assumed to be in UTC time and leap
    seconds are removed from the gps_millis as a result.

    Parameters
    ----------
    gps_millis : float or array-like of float
        Float object for Time of Clock [ms].

    Returns
    -------
    t_utc : datetime.datetime or np.ndarray
        UTC time as a datetime object. Either `datetime.datetime` or
        `np.ndarray` with `dtype = datetime.datetime`.
    """
    gps_week, tow = gps_millis_to_tow(gps_millis)
    t_utc = tow_to_datetime(gps_week, tow)
    return t_utc


def gps_to_unix_millis(gps_millis):
    """Convert milliseconds since GPS epoch to UNIX millis.

    GPS millis is from the start of the GPS in GPS reference.
    The initial GPS epoch is defined by the variable GPS_EPOCH_0 at
    which the week number is assumed to be 0.

    Leap seconds are removed from gps_millis because of the difference
    between how GPS and Unix time handle milliseconds.

    Parameters
    ----------
    gps_millis : float or array-like of float
        Float object for Time of Clock [ms].

    Returns
    -------
    unix_millis : float or np.ndarray
        Milliseconds since UNIX Epoch (1/1/1970 UTC). Either `float`
        or `np.ndarray` with `dtype = float`.

    """
    #NOTE: Ensure that one of these methods is always adding/removing
    # leap seconds here
    if isinstance(gps_millis, np.ndarray) \
        and len(np.atleast_1d(gps_millis)) > 1:
        unix_millis = np.zeros_like(gps_millis)
        for t_idx, gps in enumerate(gps_millis):
            t_utc = gps_millis_to_datetime(gps)
            unix_millis[t_idx] = datetime_to_unix_millis(t_utc)
        gps_millis = gps_millis.astype(np.float64)
    else:
        t_utc = gps_millis_to_datetime(gps_millis)
        unix_millis = np.float64(datetime_to_unix_millis(t_utc))
    return unix_millis


def gps_datetime_to_gps_millis(t_gps):
    """Convert datetime in GPS time of reference to milliseconds since GPS Epoch.

    GPS Epoch starts at the 6th January 1980.
    This function assumes that the input datetime is in the GPS time
    frame of reference and converts that to GPS milliseconds.

    Parameters
    ----------
    t_gps : datetime.datetime or array-like of datetime.datetime
        GPS time as a datetime object.

    Returns
    -------
    gps_millis : float or np.ndarray
        Milliseconds since GPS Epoch (6th January 1980 GPS). Either
        `float` or `np.ndarray` with `dtype = float`.


    """
    if isinstance(t_gps,datetime):
        t_gps = [t_gps]
    if isinstance(t_gps,np.ndarray) \
        and len(np.atleast_1d(t_gps)) == 1:
        t_gps = [t_gps.item()]
    gps_millis = []
    for t_datetime in t_gps:
        gps_milli = (t_datetime - GPS_EPOCH_0).total_seconds()*1000
        gps_millis.append(gps_milli)
    gps_millis = np.squeeze(np.asarray(gps_millis))
    return gps_millis



def tzinfo_to_utc(t_datetime):
    """Raises warning if time doesn't have timezone and converts to UTC.

    If datetime object is offset-naive, then this function will
    interpret the timezone as UTC and add the appropriate timezone info.

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
    if not hasattr(t_datetime, 'tzinfo') or t_datetime.tzinfo is None:
        warnings.warn("No time zone info found in datetime, assuming UTC",\
                        RuntimeWarning)
        t_datetime = t_datetime.replace(tzinfo=timezone.utc)
    if t_datetime.tzinfo != timezone.utc:
        t_datetime = t_datetime.astimezone(timezone.utc)
    return t_datetime
