"""Timing conversions between reference frames.

"""

__authors__ = "S. Gupta, A. Kanhere"
__date__ = "25 Jul 2022"

from datetime import datetime, timedelta


def datetime_to_tow(t, convert_gps=True):
    """Convert Python datetime object to GPS Week and time of week.

    Parameters
    ----------
    t : datetime.datetime
        Datetime object for Time of Clock.

    convert_gps : Bool
        Flag for whether output is in UTC seconds or GPS seconds

    Returns
    -------
    wk : float
        GPS week

    tow : float
        GPS time of week [s]

    """
    # DateTime to GPS week and TOW
    if hasattr(t, 'tzinfo'):
        t = t.replace(tzinfo=None)
    if convert_gps:
        utc_2_gps = timedelta(seconds=18)
        #TODO: Move to ephemeris and use leapseconds attribute from ephemeris files
        t = t + utc_2_gps
    wk_ref = datetime(2014, 2, 16, 0, 0, 0, 0, None)
    refwk = 1780
    wk = (t - wk_ref).days // 7 + refwk
    tow = ((t - wk_ref) - timedelta((wk - refwk) * 7.0)).total_seconds()
    return wk, tow