########################################################################
# Author(s):    Shubh Gupta, Ashwin Kanhere
# Date:         16 July 2021
# Desc:         Utility functions to convert datetime objects to GNSS
#               relevant terms
########################################################################

from datetime import datetime


def datetime_to_tow(t, convert_gps=True):
    """Convert Python datetime object to GPS Week and time of week

    Parameters
    ----------
    t : datetime.datetime
      Datetime object for Time of Clock

    convert_gps : Bool
      Flag for whether output is in UTC seconds or GPS seconds

    Returns
    -------
    wk : float
      GPS week

    tow : float
      GPS time of week in seconds

    """
    # DateTime to GPS week and TOW
    if hasattr(t, 'tzinfo'):
        t = t.replace(tzinfo=None)
    if convert_gps:
        utc_2_gps = datetime.timedelta(seconds=18)
        #TODO: Move to ephemeris and use leapseconds attribute from ephemeris files
        t = t + utc_2_gps
    wk_ref = datetime.datetime(2014, 2, 16, 0, 0, 0, 0, None)
    refwk = 1780
    wk = (t - wk_ref).days // 7 + refwk
    tow = ((t - wk_ref) - datetime.timedelta((wk - refwk) * 7.0)).total_seconds()
    return wk, tow
