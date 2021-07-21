########################################################################
# Author(s):    Shubh Gupta, Ashwin Kanhere
# Date:         16 July 2021
# Desc:         Utility functions to convert datetime objects to GNSS 
#               relevant terms
########################################################################

from datetime import datetime


def datetime_to_tow(t, convert_gps=True):
    """Shubh got from somwhere (need to determine)
    """
    """
    Convert a Python datetime object to GPS Week and Time Of Week.
    Does *not* convert from UTC to GPST.
    Fractional seconds are supported.
    Parameters
    ----------
    t : datetime
      A time to be converted, on the GPST timescale.
    mod1024 : bool, optional
      If True (default), the week number will be output in 10-bit form.
    Returns
    convert_gps: bool, optional
        If True (default), UTC time in seconds is converted to GPS time in seconds
    -------
    week, tow : tuple (int, float)
      The GPS week number and time-of-week.
    """
    # DateTime to GPS week and TOW
    if hasattr(t, 'tzinfo'):
        t = t.replace(tzinfo=None)
    if convert_gps:
        utc_2_gps = datetime.timedelta(seconds=18)
        t = t + utc_2_gps
    wk_ref = datetime.datetime(2014, 2, 16, 0, 0, 0, 0, None)
    refwk = 1780
    wk = (t - wk_ref).days // 7 + refwk
    tow = ((t - wk_ref) - datetime.timedelta((wk - refwk) * 7.0)).total_seconds()
    return wk, tow