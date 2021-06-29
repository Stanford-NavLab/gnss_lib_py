import numpy as np
import math
import datetime

pi = math.pi
# Generate points in a circle
def PointsInCircum(r, n=100):
    return np.array([[math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r] for x in range(0, n+1)])


def sats_from_el_az(elaz_deg):
    assert np.shape(elaz_deg)[1] == 2, "elaz_deg should be a Nx2 array"
    el = np.deg2rad(elaz_deg[:, 0])
    az = np.deg2rad(elaz_deg[:,1])
    unit_vect = np.zeros([3, np.shape(elaz_deg)[0]])
    unit_vect[0, :] = np.sin(az)*np.cos(el)
    unit_vect[1, :] = np.cos(az)*np.cos(el)
    unit_vect[2, :] = np.sin(el)
    sats_ned = 20200000*unit_vect
    return sats_ned.T


def datetime_to_tow(t, convert_gps=True):
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
    if t.tzinfo:
      t = t.replace(tzinfo=None)
    if convert_gps:
        utc_2_gps = datetime.timedelta(seconds=18)
        t = t + utc_2_gps
    wk_ref = datetime.datetime(2014, 2, 16, 0, 0, 0, 0, None)
    refwk = 1780
    wk = (t - wk_ref).days // 7 + refwk
    tow = ((t - wk_ref) - datetime.timedelta((wk - refwk) * 7.0)).total_seconds()
    return wk, tow

if __name__=='__main__':
    print('Testing functions in utils.py')
    elaz_deg = np.array([[0, 0], [90, 0], [0,90], [90, 90], [45, 45]])
    sats = sats_from_el_az(elaz_deg)
    print('The satellites are \n')
    np.set_printoptions(suppress=True)
    for i in range(np.shape(sats)[1]):
        print(sats[:,i])
