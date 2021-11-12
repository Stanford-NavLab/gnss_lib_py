"""Methods for simulating / computing expected GNSS measurements.

This module contains functions necessary for generating expected measurements
and simulating measurements for GNSS pseudorange and doppler based on receiver
position and satellite ephemeris.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "16 July 2021"

import os
import sys
import numpy as np
import pandas as pd
from numpy.random import default_rng

# append <path>/gnss_lib_py/gnss_lib_py/ to path
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
from core.constants import GPSConsts
from core.coordinates import ecef2geodetic

# TODO: Check if any of the functions are sorting the dataframe w.r.t SV while 
# processing the measurements


def _extract_pos_vel_arr(satXYZV):
    """Extract satellite positions and velocities into numpy arrays

    Parameters
    ----------
    satXYZV : pd.DataFrame
        Dataframe with satellite states

    Returns
    -------
    prns : List
        Satellite PRNs in input DataFrame

    satXYZ : ndarray
        ECEF satellite positions

    satV : ndarray
        ECEF satellite x, y and z velocities
    """
    prns   = [int(prn[1:]) for prn in satXYZV.index]
    satXYZ = satXYZV.filter(['x', 'y', 'z'])
    satV   = satXYZV.filter(['vx', 'vy', 'vz'])
    satXYZ = satXYZ.to_numpy()
    satV   = satV.to_numpy()
    return prns, satXYZ, satV
    # TODO: Remove prns from function output if not needed

def simulate_measures(gpsweek, gpstime, ephem, pos, bias, b_dot, vel, 
                      prange_sigma = 6., doppler_sigma=0.1, satXYZV=None):
    """Simulate GNSS pseudoranges and doppler measurements given receiver states

    Measurements are simulated by adding Gaussian noise to measurements expected
    based on the receiver states

    Parameters
    ----------
    gpsweek : int
        Week in GPS calendar

    gpstime : float
        GPS time of the week for simulate measurements [s]

    ephem : pd.DataFrame
        DataFrame containing all satellite ephemeris parameters for gpsweek and
        gpstime

    pos : ndarray
        1x3 Receiver 3D ECEF position [m]

    bias : float
        Receiver clock bais [m]

    b_dot : float
        Receiver clock drift [m/s]

    vel : ndarray
        1x3 Receiver 3D ECEF velocity

    prange_sigma : float
        Standard deviation of Gaussian error in simulated pseduranges

    doppler_sigma : float
        Standard deviation of Gaussian error in simulated doppler measurements

    satXYZV : pd.DataFrame
        Precomputed positions of satellites (if available)

    Returns
    -------
    measurements : pd.DataFrame
        Pseudorange and doppler measurements indexed by satellite SV with
        Gaussian noise

    satXYZV : pd.DataFrame
        Satellite positions and velocities (same as input if provided)

    """
    # TODO: Modify to work with input satellite positions
    # TODO: Add assertions/error handling for sizes of position, bias, b_dot and
    # velocity arrays
    ephem = _find_visible_sats(gpsweek, gpstime, pos, ephem)
    measurements, satXYZV = expected_measures(gpsweek, gpstime, ephem, pos,
                                              bias, b_dot, vel, satXYZV)
    M   = len(measurements.index)
    rng = default_rng()

    measurements['prange']  = (measurements['prange']
        + prange_sigma *rng.standard_normal(M))

    measurements['doppler'] = (measurements['doppler']
        + doppler_sigma*rng.standard_normal(M))

    return measurements, satXYZV

def expected_measures(gpsweek, gpstime, ephem, pos,
                      bias, b_dot, vel, satXYZV=None):
    """Compute expected pseudoranges and doppler measurements given receiver
    states

    Parameters
    ----------
    gpsweek : int
        Week in GPS calendar

    gpstime : float
        GPS time of the week for simulate measurements [s]

    ephem : pd.DataFrame
        DataFrame containing all satellite ephemeris parameters for gpsweek and
        gpstime

    pos : ndarray
        1x3 Receiver 3D ECEF position [m]

    bias : float
        Receiver clock bais [m]

    b_dot : float
        Receiver clock drift [m/s]

    vel : ndarray
        1x3 Receiver 3D ECEF velocity

    satXYZV : pd.DataFrame
        Precomputed positions of satellites (if available)

    Returns
    -------
    measurements : pd.DataFrame
        Expected pseudorange and doppler measurements indexed by satellite SV

    satXYZV : pd.DataFrame
        Satellite positions and velocities (same as input if provided)
    """
    # NOTE: When using saved data, pass saved DataFrame with ephemeris in ephem
    # and satellite positions in satXYZV
    # TODO: Modify this function to use PRNS from measurement in addition to
    # gpstime from measurement
    pos = np.reshape(pos, [1, 3])
    vel = np.reshape(vel, [1, 3])
    gpsconsts = GPSConsts()
    satXYZV, delXYZ, true_range = _find_sat_location(gpsweek, gpstime, ephem, 
                                                     pos, satXYZV)
    # satXYZ, satV, delXYZ are both Nx3
    _, satXYZ, satV = _extract_pos_vel_arr(satXYZV)

    # Obtain corrected pseudoranges and add receiver clock bias to them
    prange = true_range + bias
    # prange = (correct_pseudorange(gpstime, gpsweek, ephem, true_range, 
    #                              np.reshape(pos, [-1, 3])) + bias)
    # TODO: Correction should be applied to the modelled pseudoranges, not
    # received. Check with textbook + data
    # TODO: Add corrections instead of returning corrected pseudoranges

    # Obtain difference of velocity between satellite and receiver
    delV = satV - np.tile(np.reshape(vel, 3), [len(ephem), 1])
    prange_rate = np.sum(delV*delXYZ, axis=1)/true_range + b_dot
    doppler = -(gpsconsts.F1/gpsconsts.C) * (prange_rate)
    # doppler = pd.DataFrame(doppler, index=prange.index.copy())
    measurements = pd.DataFrame(np.column_stack((prange, doppler)), 
                                index=satXYZV.index, 
                                columns=['prange', 'doppler'])
    return measurements, satXYZV


def _find_visible_sats(gpsweek, gpstime, Rx_ECEF, ephem, el_mask=5.):
    """Trim input ephemeris to keep only visible SVs

    Parameters
    ----------
    gpsweek : int
        Week in GPS calendar

    gpstime : float
        GPS time of the week for simulate measurements [s]

    Rx_ECEF : ndarray
        1x3 row Rx ECEF position vector [m]

    ephem : pd.DataFrame
        DataFrame containing all satellite ephemeris parameters for gpsweek and
        gpstime

    el_mask : float
        Minimum elevation of returned satellites

    Returns
    -------
    eph : pd.DataFrame
        Ephemeris parameters of visible satellites

    """
    gpsconsts = GPSConsts()
    # Find positions adn velocities of all satellites
    approx_XYZV = FindSat(ephem, gpstime - gpsconsts.T_TRANS, gpsweek)
    # Find elevation and azimuth angles for all satellites
    _, approx_XYZ, _ = _extract_pos_vel_arr(approx_XYZV)
    approx_elaz = find_elaz(np.reshape(Rx_ECEF, [1, 3]), approx_XYZ)
    # Keep attributes of only those satellites which are visible
    keep_ind = approx_elaz[:,0] > el_mask
    # prns = approx_XYZV.index.to_numpy()[keep_ind]
    # TODO: Remove above statement if superfluous
    # TODO: Check that a copy of the ephemeris is being generated, also if it is
    # needed
    eph = ephem.loc[keep_ind, :]
    return eph


def _find_sat_location(gpsweek, gpstime, ephem, pos, satXYZV=None):
    """Return satellite positions, difference from Rx position and ranges

    Parameters
    ----------
    gpsweek : int
        Week in GPS calendar

    gpstime : float
        GPS time of the week for simulate measurements [s]

    ephem : pd.DataFrame
        DataFrame containing all satellite ephemeris parameters for gpsweek and
        gpstime

    pos : ndarray
        1x3 Receiver 3D ECEF position [m]

    satXYZV : pd.DataFrame
        Precomputed positions of satellites (if available)

    Returns
    -------
    satXYVZ : pd.DataFrame
        Satellite position and velocities (same if input)

    delXYZ : ndarray
        Difference between satellite positions and receiver position

    true_range : ndarray
        Distance between satellite and receiver positions

    """
    gpsconsts = GPSConsts()
    pos = np.reshape(pos, [1, 3])
    if satXYZV is None:
        satellites = len(ephem.index)
        satXYZV = FindSat(ephem, gpstime - gpsconsts.T_TRANS, gpsweek)
        delXYZ, true_range = _find_delxyz_range(satXYZV, pos, satellites)
        tcorr = true_range/gpsconsts.C
        # Find satellite locations at (a more accurate) time of transmission
        satXYZV = FindSat(ephem, gpstime-tcorr, gpsweek)
    else:
        satellites = len(satXYZV.index)
    delXYZ, true_range = _find_delxyz_range(satXYZV, pos, satellites)
    tcorr = true_range/gpsconsts.C
    # Corrections for the rotation of the Earth during transmission
    _, satXYZ, satV = _extract_pos_vel_arr(satXYZV)
    delX = gpsconsts.OMEGAEDOT*satXYZV['x'] * tcorr
    delY = gpsconsts.OMEGAEDOT*satXYZV['y'] * tcorr
    satXYZV['x'] = satXYZV['x'] + delX
    satXYZV['y'] = satXYZV['y'] + delY
    return satXYZV, delXYZ, true_range


def _find_delxyz_range(satXYZV, pos, satellites):
    """Return difference of satellite and Rx positions and range between them

    Parameters
    ----------
    satXYVZ : pd.DataFrame
        Satellite position and velocities

    pos : ndarray
        1x3 Receiver 3D ECEF position [m]

    satellites : int
        Number of satellites in satXYZV

    Returns
    -------
    delXYZ : ndarray
        Difference between satellite positions and receiver position

    true_range : ndarray
        Distance between satellite and receiver positions
    """
    # Repeating computation in find_sat_location
    #NOTE: Input is from satellite finding in AE 456 code
    pos = np.reshape(pos, [1, 3])
    if np.size(pos)!=3:
        raise ValueError('Position is not in XYZ')
    _, satXYZ, _ = _extract_pos_vel_arr(satXYZV)
    delXYZ = satXYZ - np.tile(np.reshape(pos, [-1, 3]), (satellites, 1))
    true_range = np.linalg.norm(delXYZ, axis=1)
    return delXYZ, true_range


def FindSat(ephem, times, gpsweek):
    """Compute position and velocities for all satellites in ephemeris file
    given time of clock

    Parameters
    ----------
    ephem : pd.DataFrame
        DataFrame containing ephemeris parameters of satellies for which states
        are required

    times : ndarray
        GPS time of the week at which positions are required [s]

    gpsweek : int
        Week of GPS calendar corresponding to time of clock

    Returns
    -------
    satXYZV : pd.DataFrame
        DataFrame indexed by satellite SV containing positions and velocities

    Notes
    -----
    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    Satellite velocity calculations based on algorithms introduced in [1]_.

    References
    ----------
    ..  [1] B. F. Thompson, S. W. Lewis, S. A. Brown, and T. M. Scott,
        “Computing GPS satellite velocity and acceleration from the broadcast
        navigation message,” NAVIGATION, vol. 66, no. 4, pp. 769–779, Dec. 2019,
        doi: 10.1002/navi.342.

    """
    # Satloc contains both positions and velocities.
    #TODO: Look into combining this method with the ones in read_rinex.py
    #TODO: Clean up this code

    # Load in GPS constants
    gpsconsts = GPSConsts()

    # Extract parameters
    c_is = ephem['C_is']
    c_ic = ephem['C_ic']
    c_rs = ephem['C_rs']
    c_rc = ephem['C_rc']
    c_uc = ephem['C_uc']
    c_us = ephem['C_us']
    M_0  = ephem['M_0']
    dN   = ephem['deltaN']

    e        = ephem['e']     # eccentricity
    omega    = ephem['omega'] # argument of perigee
    omega_0  = ephem['Omega_0']
    sqrt_sma = ephem['sqrtA'] # sqrt of semi-major axis
    sma      = sqrt_sma**2      # semi-major axis

    sqrt_mu_A = np.sqrt(gpsconsts.MUEARTH) * sqrt_sma**-3 # mean angular motion
    gpsweek_diff = np.mod(gpsweek,1024) - np.mod(ephem['GPSWeek'],1024)*604800.

    # if np.size(times_all)==1:
    #     times_all = times_all*np.ones(len(ephem))
    # else:
    #     times_all = np.reshape(times_all, len(ephem))
    # times = times_all
    satXYZV = pd.DataFrame()
    satXYZV.loc[:,'sv'] = ephem.index
    satXYZV.set_index('sv', inplace=True)
    #TODO: Check if 'dt' or 'times' should be stored in the final DataFrame
    satXYZV.loc[:,'times'] = times

    dt = times - ephem['t_oe'] + gpsweek_diff

    # Calculate the mean anomaly with corrections
    Mcorr = dN * dt
    M = M_0 + (sqrt_mu_A * dt) + Mcorr

    # Compute the eccentric anomaly from mean anomaly using the Newton-Raphson
    # method to solve for E in:
    #  f(E) = M - E + e * sin(E) = 0
    E = M
    for i in np.arange(0,10):
        f    = M - E + e * np.sin(E)
        dfdE = e*np.cos(E) - 1.
        dE   = -f / dfdE
        E    = E + dE

    cos_E = np.cos(E)
    sin_E = np.sin(E)
    e_cos_E = (1 - e*cos_E)

    # Calculate the true anomaly from the eccentric anomaly
    sinnu = np.sqrt(1 - e**2) * (sin_E/e_cos_E)
    cosnu = (cos_E-e) / e_cos_E
    nu    = np.arctan2(sinnu, cosnu)

    # Calcualte the argument of latitude iteratively
    phi0 = nu + omega
    phi  = phi0
    for i in range(5):
        cos2phi = np.cos(2.*phi)
        sin2phi = np.sin(2.*phi)
        phiCorr = c_uc * cos2phi + c_us * sin2phi
        phi = phi0 + phiCorr

    # Calculate the longitude of ascending node with correction
    OmegaCorr = ephem['OmegaDot'] * dt

    # Also correct for the rotation since the beginning of the GPS week for
    # which the Omega0 is defined.  Correct for GPS week rollovers.
    omega = omega_0 - (gpsconsts.OMEGAEDOT*(times + gpsweek_diff)) + OmegaCorr

    # Calculate orbital radius with correction
    rCorr = c_rc * cos2phi + c_rs * sin2phi
    r     = sma*e_cos_E + rCorr

    ############################################
    ######  Lines added for velocity (1)  ######
    ############################################
    dE   = (sqrt_mu_A + dN) / e_cos_E
    dphi = np.sqrt(1 - e**2)*dE / e_cos_E
    # Changed from the paper
    dr   = (sma * e * dE * sin_E) + 2*(c_rs*cos2phi - c_rc*sin2phi)*dphi

    # Calculate the inclination with correction
    iCorr = c_ic*cos2phi + c_is*sin2phi + ephem['IDOT']*dt
    i = ephem['i_0'] + iCorr

    ############################################
    ######  Lines added for velocity (2)  ######
    ############################################
    di = 2*(c_is*cos2phi - c_ic*sin2phi)*dphi + ephem['IDOT']

    # Find the position in the orbital plane
    xp = r*np.cos(phi)
    yp = r*np.sin(phi)

    ############################################
    ######  Lines added for velocity (3)  ######
    ############################################
    du = (1 + 2*(c_us * cos2phi - c_uc*sin2phi))*dphi
    dxp = dr*np.cos(phi) - r*np.sin(phi)*du
    dyp = dr*np.sin(phi) + r*np.cos(phi)*du
    # Find satellite position in ECEF coordinates
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)

    satXYZV.loc[:,'x'] = xp*cos_omega - yp*cos_i*sin_omega
    satXYZV.loc[:,'y'] = xp*sin_omega + yp*cos_i*cos_omega
    satXYZV.loc[:,'z'] = yp*sin_i
    # TODO: Add satellite clock bias here using the 'clock corrections' not to
    # be used but compared against SP3 and Android data

    ############################################
    ######  Lines added for velocity (4)  ######
    ############################################
    dOmega = ephem['OmegaDot'] - gpsconsts.OMEGAEDOT
    satXYZV.loc[:,'vx'] = (dxp*cos_omega
                         - dyp*cos_i*sin_omega
                         + yp*sin_omega*sin_i*di
                         - (xp*sin_omega + yp*cos_i*cos_omega)*dOmega)

    satXYZV.loc[:,'vy'] = (dxp * sin_omega
                         + dyp * cos_i * cos_omega
                         - yp  * sin_i * cos_omega * di
                         + (xp * cos_omega - (yp * cos_i * sin_omega))*dOmega)

    satXYZV.loc[:,'vz'] = dyp*sin_i + yp*cos_i*di
    
    return satXYZV


def correct_pseudorange(gpstime, gpsweek, ephem, pr_meas, rx=[[None]]):
    """Incorporate corrections in measurements

    Incorporate clock corrections (relativistic, drift), tropospheric and
    ionospheric clock corrections

    Parameters
    ----------
    gpstime : float
        Time of clock in seconds of the week

    gpsweek : int
        GPS week for time of clock

    ephem : pd.DataFrame
        Satellite ephemeris parameters for measurement SVs

    pr_meas : ndarray
        Ranging measurements from satellites [m]

    rx : ndarray
        1x3 array of ECEF Rx position [m]

    Returns
    -------
    prCorr : ndarray
        Array of corrected pseudorange measurements [m]

    Notes
    -----
    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    """
    # TODO: Incorporate satellite clock rate changes into doppler measurements
    # TODO: Change default of rx to an array of None with size
    # TODO: Change the sign for corrections to what will be added to expected
    # measurements
    # TODO: Return corrections instead of corrected measurements 

    # Load GPS Constants
    gpsconsts = GPSConsts()

    # Extract parameters
    M_0  = ephem['M_0']
    dN   = ephem['deltaN']

    e        = ephem['e']     # eccentricity
    sqrt_sma = ephem['sqrtA'] # sqrt of semi-major axis

    sqrt_mu_A = np.sqrt(gpsconsts.MUEARTH) * sqrt_sma**-3 # mean angular motion

    # Make sure gpstime and gpsweek are arrays
    if not isinstance(gpstime, np.ndarray):
        gpstime = np.array(gpstime)
    if not isinstance(gpsweek, np.ndarray):
        gpsweek = np.array(gpsweek)

    # Initialize the correction array
    prCorr = np.zeros_like(pr_meas)
    dt = gpstime - ephem['t_oe']
    if np.abs(dt).any() > 302400:
        dt = dt-np.sign(dt)*604800

    # Calculate the mean anomaly with corrections
    Mcorr = dN * dt
    M     = M_0 + (sqrt_mu_A * dt) + Mcorr

    # Compute the eccentric anomaly from mean anomaly using the Newton-Raphson
    # method
    # to solve for E in:
    #  f(E) = M - E + e * sin(E) = 0
    E = M
    for i in np.arange(0,10):
        f = M - E + e * np.sin(E)
        dfdE = e*np.cos(E) - 1.
        dE = -f / dfdE
        E = E + dE

    sin_E = np.sin(E)

    # Determine pseudorange corrections due to satellite clock corrections.
    # Calculate time offset from satellite reference time
    timeOffset = gpstime - ephem['t_oc']
    if np.abs(timeOffset).any() > 302400:
        timeOffset = timeOffset-np.sign(timeOffset)*604800

    # Calculate clock corrections from the polynomial
    # corrPolynomial = ephem.af0
    #                + ephem.af1*timeOffset
    #                + ephem.af2*timeOffset**2
    corrPolynomial = (ephem['SVclockBias']
                    + ephem['SVclockDrift']*timeOffset
                    + ephem['SVclockDriftRate']*timeOffset**2)

    # Calcualte the relativistic clock correction
    corrRelativistic = gpsconsts.F * e * sqrt_sma * sin_E

    # Calculate the total clock correction including the Tgd term
    clockCorr = (corrPolynomial - ephem['TGD'] + corrRelativistic)

    # NOTE: Removed ionospheric delay calculation here

    # calculate clock psuedorange correction
    prCorr = pr_meas + clockCorr*gpsconsts.C

    if rx[0][0] != None: # TODO: Reference using 2D array slicing
        # Calculate the tropospheric delays
        tropoDelay = calculate_tropo_delay(gpstime, gpsweek, ephem, rx)
        # Calculate total pseudorange correction
        prCorr -= tropoDelay*gpsconsts.C

    if isinstance(prCorr, pd.Series):
        prCorr = prCorr.to_numpy(dtype=float)

    # fill nans (fix for non-GPS satellites)
    prCorr = np.where(np.isnan(prCorr), pr_meas, prCorr)

    return prCorr


def calculate_tropo_delay(gpstime, gpsweek, ephem, rx_loc):
    """Calculate tropospheric delay

    Parameters
    ----------
    gpstime : float
        Time of clock in seconds of the week

    gpsweek : int
        GPS week for time of clock

    ephem : pd.DataFrame
        Satellite ephemeris parameters for measurement SVs

    rx_loc : ndarray
        1x3 array of ECEF Rx position [m]

    Returns
    -------
    tropo_delay : ndarray
        Tropospheric corrections to pseudorange measurements

    Notes
    -----
    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    """
    # Load gpsconstants
    gpsconsts = GPSConsts()

    # Make sure things are arrays
    if not isinstance(gpstime, np.ndarray):
        gpstime = np.array(gpstime)
    if not isinstance(gpsweek, np.ndarray):
        gpsweek = np.array(gpsweek)

    # Determine the satellite locations
    satXYZV = FindSat(ephem, gpstime, gpsweek)
    _, satXYZ, _ = _extract_pos_vel_arr(satXYZV)

    # compute elevation and azimuth
    el_az = find_elaz(rx_loc, satXYZ)
    el_r  = np.deg2rad(el_az[:,0])

    # Calculate the WGS-84 latitude/longitude of the receiver
    wgs = ecef2geodetic(rx_loc)
    height = wgs[:,2]

    # Force height to be positive
    ind = np.argwhere(height < 0).flatten()
    if len(ind) > 0:
        height[ind] = 0

    # Calculate the delay
    # TODO: Store these numbers somewhere, we should know where they're from -BC
    c_1 = 2.47
    c_2 = 0.0121
    c_3 = 1.33e-4
    tropo_delay = c_1/(np.sin(el_r)+c_2) * np.exp(-height*c_3)/gpsconsts.C

    return tropo_delay


##########################################################
### Code taken from Makela's Python code as is follows ###
##########################################################


def find_elaz(Rx, Sats):
    """Calculate the elevation and azimuth from a single receiver to multiple
    satellites.

    Parameters
    ----------
    Rx : ndarray
        1x3 vector containing [X, Y, Z] coordinate of receiver
    Sats : ndarray
        Nx3 array  containing [X, Y, Z] coordinates of satellites

    Returns
    -------
    elaz : ndarray
        Nx2 array containing the elevation and azimuth from the
        receiver to the requested satellites. Elevation and azimuth are
        given in decimal degrees.

    Notes
    -----
    Code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    """

    # check for 1D case:
    dim = len(Rx.shape)
    if dim == 1:
        Rx = np.reshape(Rx,(1,3))

    dim = len(Sats.shape)
    if dim == 1:
        Sats = np.reshape(Sats,(1,3))

    # Convert the receiver location to WGS84
    lla = ecef2geodetic(Rx)
    assert np.shape(lla)==(1,3)

    # Create variables with the latitude and longitude in radians
    lat = np.deg2rad(lla[0,0])
    lon = np.deg2rad(lla[0,1])

    # Create the 3 x 3 transform matrix from ECEF to VEN
    cos_lon = np.cos(lon)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    sin_lat = np.sin(lat)
    VEN = np.array([[ cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat],
                    [-sin_lon        ,  cos_lon        , 0.     ],
                    [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat]])

    # Replicate the Rx array to be the same size as the satellite array
    Rx_array = np.ones_like(Sats) * Rx

    # Calculate the pseudorange for each satellite
    p = Sats - Rx_array

    # Calculate the length of this vector
    n = np.array([np.sqrt(p[:,0]**2 + p[:,1]**2 + p[:,2]**2)])

    # Create the normalized unit vector
    p = p / (np.ones_like(p) * n.T)

    # Perform the transform of the normalized psueodrange from ECEF to VEN
    p_VEN = np.dot(VEN, p.T)

    # Calculate elevation and azimuth in degrees
    ea = np.zeros([Sats.shape[0],2])
    ea[:,0] = np.rad2deg((np.pi/2. - np.arccos(p_VEN[0,:])))
    ea[:,1] = np.rad2deg(np.arctan2(p_VEN[1,:],p_VEN[2,:]))

    return ea
