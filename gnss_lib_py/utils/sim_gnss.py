"""Generates expected measurements and simulates pseudoranges.

Functions to generate expected measurements and to simulate pseudoranges
and doppler for GPS satellites.

"""

__authors__ = "Ashwin Kanhere, Bradley Collicott"
__date__ = "16 July 2021"

import numpy as np
import pandas as pd
from numpy.random import default_rng

import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.coordinates import ecef_to_geodetic, ecef_to_el_az
# TODO: Check if any of the functions are sorting the dataframe w.r.t SV while
# processing the measurements


def sats_from_el_az(elaz_deg):
    """Generate NED satellite positions at given elevation and azimuth.

    Given elevation and azimuth angles are with respect to the receiver.
    Generated satellites are in the NED frame of reference with the receiver
    position as the origin.

    Parameters
    ----------
    elaz_deg : np.ndarray
        Nx2 array of elevation and azimuth angles [degrees]

    Returns
    -------
    sats_ned : np.ndarray
        Nx3 satellite NED positions, simulated at a distance of 20,200 km
    """
    assert np.shape(elaz_deg)[1] == 2, "elaz_deg should be a Nx2 array"
    el = np.deg2rad(elaz_deg[:, 0])
    az = np.deg2rad(elaz_deg[:,1])
    unit_vect = np.zeros([3, np.shape(elaz_deg)[0]])
    unit_vect[0, :] = np.sin(az)*np.cos(el)
    unit_vect[1, :] = np.cos(az)*np.cos(el)
    unit_vect[2, :] = np.sin(el)
    sats_ned = np.transpose(20200000*unit_vect)
    return sats_ned


def _extract_pos_vel_arr(sv_posvel):
    """Extract satellite positions and velocities into numpy arrays.

    Parameters
    ----------
    sv_posvel : pd.DataFrame
        Dataframe with satellite states

    Returns
    -------
    prns : List
        Satellite PRNs in input DataFrame
    sv_pos : ndarray
        ECEF satellite positions
    sv_vel : ndarray
        ECEF satellite x, y and z velocities
    """
    prns   = [int(prn[1:]) for prn in sv_posvel.index]
    sv_pos = sv_posvel.filter(['x', 'y', 'z'])
    sv_vel   = sv_posvel.filter(['vx', 'vy', 'vz'])
    sv_pos = sv_pos.to_numpy()
    sv_vel   = sv_vel.to_numpy()
    return prns, sv_pos, sv_vel
    # TODO: Remove prns from function output if not needed

def simulate_measures(gpsweek, gpstime, ephem, pos, bias, b_dot, vel,
                      prange_sigma = 6., doppler_sigma=0.1, sv_posvel=None):
    """Simulate GNSS pseudoranges and doppler measurements given receiver state.

    Measurements are simulated by adding Gaussian noise to measurements expected
    based on the receiver states.

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
    sv_posvel : pd.DataFrame
        Precomputed positions of satellites (if available)

    Returns
    -------
    measurements : pd.DataFrame
        Pseudorange and doppler measurements indexed by satellite SV with
        Gaussian noise
    sv_posvel : pd.DataFrame
        Satellite positions and velocities (same as input if provided)

    """
    #TODO: Modify to work with input satellite positions
    #TODO: Add assertions/error handling for sizes of position, bias, b_dot and
    # velocity arrays
    #TODO: Modify to use single state vector instead of multiple inputs
    #TODO: Modify to use single dictionary with uncertainty values
    ephem = _find_visible_sats(gpsweek, gpstime, pos, ephem)
    measurements, sv_posvel = expected_measures(gpsweek, gpstime, ephem, pos,
                                              bias, b_dot, vel, sv_posvel)
    num_sats   = len(measurements.index)
    rng = default_rng()

    measurements['prange']  = (measurements['prange']
        + prange_sigma *rng.standard_normal(num_sats))

    measurements['doppler'] = (measurements['doppler']
        + doppler_sigma*rng.standard_normal(num_sats))

    return measurements, sv_posvel

def expected_measures(gpsweek, gpstime, ephem, pos,
                      bias, b_dot, vel, sv_posvel=None):
    """Compute expected pseudoranges and doppler measurements given receiver
    states.

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
    sv_posvel : pd.DataFrame
        Precomputed positions of satellites (if available)

    Returns
    -------
    measurements : pd.DataFrame
        Expected pseudorange and doppler measurements indexed by satellite SV
    sv_posvel : pd.DataFrame
        Satellite positions and velocities (same as input if provided)
    """
    # NOTE: When using saved data, pass saved DataFrame with ephemeris in ephem
    # and satellite positions in sv_posvel
    # TODO: Modify this function to use PRNS from measurement in addition to
    # gpstime from measurement
    pos = np.reshape(pos, [1, 3])
    vel = np.reshape(vel, [1, 3])
    sv_posvel, del_pos, true_range = _find_sv_location(gpsweek, gpstime,
                                                         ephem, pos, sv_posvel)
    # sv_pos, sv_vel, del_pos are both Nx3
    _, _, sv_vel = _extract_pos_vel_arr(sv_posvel)

    # Obtain corrected pseudoranges and add receiver clock bias to them
    prange = true_range + bias
    # prange = (correct_pseudorange(gpstime, gpsweek, ephem, true_range,
    #                              np.reshape(pos, [-1, 3])) + bias)
    # TODO: Correction should be applied to the received pseudoranges, not
    # modelled/expected pseudorange -- per discussion in meeting on 11/12
    # TODO: Add corrections instead of returning corrected pseudoranges

    # Obtain difference of velocity between satellite and receiver

    del_vel = sv_vel - np.tile(np.reshape(vel, 3), [len(ephem), 1])
    prange_rate = np.sum(del_vel*del_pos, axis=1)/true_range + b_dot
    doppler = -(consts.F1/consts.C) * (prange_rate)

    # doppler = pd.DataFrame(doppler, index=prange.index.copy())
    measurements = pd.DataFrame(np.column_stack((prange, doppler)),
                                index=sv_posvel.index,
                                columns=['prange', 'doppler'])
    return measurements, sv_posvel


def _find_visible_sats(gpsweek, gpstime, rx_ecef, ephem, el_mask=5.):
    """Trim input ephemeris to keep only visible SVs.

    Parameters
    ----------
    gpsweek : int
        Week in GPS calendar
    gpstime : float
        GPS time of the week for simulate measurements [s]
    rx_ecef : ndarray
        1x3 row rx_pos ECEF position vector [m]
    ephem  pd.DataFrame
        DataFrame containing all satellite ephemeris parameters for gpsweek and
        gpstime
    el_mask : float
        Minimum elevation of returned satellites

    Returns
    -------
    eph : pd.DataFrame
        Ephemeris parameters of visible satellites

    """

    # Find positions and velocities of all satellites
    approx_posvel = find_sat(ephem, gpstime - consts.T_TRANS, gpsweek)

    # Find elevation and azimuth angles for all satellites
    _, approx_pos, _ = _extract_pos_vel_arr(approx_posvel)
    approx_el_az = ecef_to_el_az(np.reshape(rx_ecef, [1, 3]), approx_pos)
    # Keep attributes of only those satellites which are visible
    keep_ind = approx_el_az[:,0] > el_mask
    # prns = approx_posvel.index.to_numpy()[keep_ind]
    # TODO: Remove above statement if superfluous
    # TODO: Check that a copy of the ephemeris is being generated, also if it is
    # needed
    eph = ephem.loc[keep_ind, :]
    return eph


def _find_sv_location(gpsweek, gpstime, ephem, pos, sv_posvel=None):
    """Return satellite positions, difference from rx_pos position and ranges.

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
    sv_posvel : pd.DataFrame
        Precomputed positions of satellites (if available)

    Returns
    -------
    sv_posvel : pd.DataFrame
        Satellite position and velocities (same if input)
    del_pos : ndarray
        Difference between satellite positions and receiver position
    true_range : ndarray
        Distance between satellite and receiver positions

    """
    pos = np.reshape(pos, [1, 3])
    if sv_posvel is None:
        satellites = len(ephem.index)
        sv_posvel = find_sat(ephem, gpstime - consts.T_TRANS, gpsweek)
        del_pos, true_range = _find_delxyz_range(sv_posvel, pos, satellites)
        t_corr = true_range/consts.C

        # Find satellite locations at (a more accurate) time of transmission
        sv_posvel = find_sat(ephem, gpstime-t_corr, gpsweek)
    else:
        satellites = len(sv_posvel.index)
    del_pos, true_range = _find_delxyz_range(sv_posvel, pos, satellites)
    t_corr = true_range/consts.C
    # Corrections for the rotation of the Earth during transmission
    # _, sv_pos, sv_vel = _extract_pos_vel_arr(sv_posvel)
    del_x = consts.OMEGA_E_DOT*sv_posvel['x'] * t_corr
    del_y = consts.OMEGA_E_DOT*sv_posvel['y'] * t_corr
    sv_posvel['x'] = sv_posvel['x'] + del_x
    sv_posvel['y'] = sv_posvel['y'] + del_y
    return sv_posvel, del_pos, true_range



def _find_delxyz_range(sv_posvel, pos, satellites):
    """Return difference of satellite and rx_pos positions and range between them.

    Parameters
    ----------
    sv_posvel : pd.DataFrame
        Satellite position and velocities
    pos : ndarray
        1x3 Receiver 3D ECEF position [m]
    satellites : int
        Number of satellites in sv_posvel

    Returns
    -------
    del_pos : ndarray
        Difference between satellite positions and receiver position
    true_range : ndarray
        Distance between satellite and receiver positions
    """
    # Repeating computation in find_sv_location
    #NOTE: Input is from satellite finding in AE 456 code
    pos = np.reshape(pos, [1, 3])
    if np.size(pos)!=3:
        raise ValueError('Position is not in XYZ')
    _, sv_pos, _ = _extract_pos_vel_arr(sv_posvel)
    del_pos = sv_pos - np.tile(np.reshape(pos, [-1, 3]), (satellites, 1))
    true_range = np.linalg.norm(del_pos, axis=1)
    return del_pos, true_range


def find_sat(ephem, times, gpsweek):
    """Compute position and velocities for all satellites in ephemeris file
    given time of clock.

    Parameters
    ----------
    ephem : pd.DataFrame
        DataFrame containing ephemeris parameters of satellites for which states
        are required
    times : ndarray
        GPS time of the week at which positions are required [s]
    gpsweek : int
        Week of GPS calendar corresponding to time of clock

    Returns
    -------
    sv_posvel : pd.DataFrame
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

    # Extract parameters
    c_is = ephem['C_is']
    c_ic = ephem['C_ic']
    c_rs = ephem['C_rs']
    c_rc = ephem['C_rc']
    c_uc = ephem['C_uc']
    c_us = ephem['C_us']
    M_0  = ephem['M_0']
    dN   = ephem['deltaN']

    ecc        = ephem['e']     # eccentricity
    omega    = ephem['omega'] # argument of perigee
    omega_0  = ephem['Omega_0']
    sqrt_sma = ephem['sqrtA'] # sqrt of semi-major axis
    sma      = sqrt_sma**2      # semi-major axis

    sqrt_mu_A = np.sqrt(consts.MU_EARTH) * sqrt_sma**-3 # mean angular motion
    gpsweek_diff = (np.mod(gpsweek,1024) - np.mod(ephem['GPSWeek'],1024))*604800.

    # if np.size(times_all)==1:
    #     times_all = times_all*np.ones(len(ephem))
    # else:
    #     times_all = np.reshape(times_all, len(ephem))
    # times = times_all
    sv_posvel = pd.DataFrame()
    sv_posvel.loc[:,'sv'] = ephem.index
    sv_posvel.set_index('sv', inplace=True)
    #TODO: Check if 'dt' or 'times' should be stored in the final DataFrame
    sv_posvel.loc[:,'times'] = times

    dt = times - ephem['t_oe'] + gpsweek_diff

    # Calculate the mean anomaly with corrections
    M_corr = dN * dt
    M = M_0 + (sqrt_mu_A * dt) + M_corr

    # Compute Eccentric Anomaly
    E = _compute_eccentric_anomoly(M, ecc, tol=1e-5)

    cos_E   = np.cos(E)
    sin_E   = np.sin(E)
    e_cos_E = (1 - ecc*cos_E)

    # Calculate the true anomaly from the eccentric anomaly
    sin_nu = np.sqrt(1 - ecc**2) * (sin_E/e_cos_E)
    cos_nu = (cos_E-ecc) / e_cos_E
    nu     = np.arctan2(sin_nu, cos_nu)

    # Calcualte the argument of latitude iteratively
    phi_0 = nu + omega
    phi   = phi_0
    for i in range(5):
        cos_to_phi = np.cos(2.*phi)
        sin_to_phi = np.sin(2.*phi)
        phi_corr = c_uc * cos_to_phi + c_us * sin_to_phi
        phi = phi_0 + phi_corr

    # Calculate the longitude of ascending node with correction
    omega_corr = ephem['OmegaDot'] * dt

    # Also correct for the rotation since the beginning of the GPS week for which the Omega0 is
    # defined.  Correct for GPS week rollovers.

    # Also correct for the rotation since the beginning of the GPS week for
    # which the Omega0 is defined.  Correct for GPS week rollovers.
    omega = omega_0 - (consts.OMEGA_E_DOT*(times + gpsweek_diff)) + omega_corr

    # Calculate orbital radius with correction
    r_corr = c_rc * cos_to_phi + c_rs * sin_to_phi
    r      = sma*e_cos_E + r_corr

    ############################################
    ######  Lines added for velocity (1)  ######
    ############################################
    dE   = (sqrt_mu_A + dN) / e_cos_E
    dphi = np.sqrt(1 - ecc**2)*dE / e_cos_E
    # Changed from the paper
    dr   = (sma * ecc * dE * sin_E) + 2*(c_rs*cos_to_phi - c_rc*sin_to_phi)*dphi

    # Calculate the inclination with correction
    i_corr = c_ic*cos_to_phi + c_is*sin_to_phi + ephem['IDOT']*dt
    i = ephem['i_0'] + i_corr

    ############################################
    ######  Lines added for velocity (2)  ######
    ############################################
    di = 2*(c_is*cos_to_phi - c_ic*sin_to_phi)*dphi + ephem['IDOT']

    # Find the position in the orbital plane
    xp = r*np.cos(phi)
    yp = r*np.sin(phi)

    ############################################
    ######  Lines added for velocity (3)  ######
    ############################################
    du = (1 + 2*(c_us * cos_to_phi - c_uc*sin_to_phi))*dphi
    dxp = dr*np.cos(phi) - r*np.sin(phi)*du
    dyp = dr*np.sin(phi) + r*np.cos(phi)*du
    # Find satellite position in ECEF coordinates
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)

    sv_posvel.loc[:,'x'] = xp*cos_omega - yp*cos_i*sin_omega
    sv_posvel.loc[:,'y'] = xp*sin_omega + yp*cos_i*cos_omega
    sv_posvel.loc[:,'z'] = yp*sin_i
    # TODO: Add satellite clock bias here using the 'clock corrections' not to
    # be used but compared against SP3 and Android data

    ############################################
    ######  Lines added for velocity (4)  ######
    ############################################
    omega_dot = ephem['OmegaDot'] - consts.OMEGA_E_DOT
    sv_posvel.loc[:,'vx'] = (dxp * cos_omega
                         - dyp * cos_i*sin_omega
                         + yp  * sin_omega*sin_i*di
                         - (xp * sin_omega + yp*cos_i*cos_omega)*omega_dot)

    sv_posvel.loc[:,'vy'] = (dxp * sin_omega
                         + dyp * cos_i * cos_omega
                         - yp  * sin_i * cos_omega * di
                         + (xp * cos_omega - (yp*cos_i*sin_omega)) * omega_dot)

    sv_posvel.loc[:,'vz'] = dyp*sin_i + yp*cos_i*di
    return sv_posvel


def correct_pseudorange(gpstime, gpsweek, ephem, pr_meas, rx_ecef=[[None]]):
    """Incorporate corrections in measurements.

    Incorporate clock corrections (relativistic, drift), tropospheric and
    ionospheric clock corrections.

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
    rx_ecef : ndarray
        1x3 array of ECEF rx_pos position [m]

    Returns
    -------
    pr_corr : ndarray
        Array of corrected pseudorange measurements [m]

    Notes
    -----
    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    """
    # TODO: Incorporate satellite clock rate changes into the doppler measurements
    # TODO: Change default of rx to an array of None with size
    # TODO: Change the sign for corrections to what will be added to expected measurements
    # TODO: Return corrections instead of corrected measurements

    # Extract parameters
    # M_0  = ephem['M_0']
    # dN   = ephem['deltaN']

    e        = ephem['e']     # eccentricity
    sqrt_sma = ephem['sqrtA'] # sqrt of semi-major axis

    sqrt_mu_A = np.sqrt(consts.MU_EARTH) * sqrt_sma**-3 # mean angular motion

    # Make sure gpstime and gpsweek are arrays
    if not isinstance(gpstime, np.ndarray):
        gpstime = np.array(gpstime)
    if not isinstance(gpsweek, np.ndarray):
        gpsweek = np.array(gpsweek)

    # Initialize the correction array
    pr_corr = pr_meas

    dt = gpstime - ephem['t_oe']
    if np.abs(dt).any() > 302400:
        dt = dt - np.sign(dt)*604800

    # Calculate the mean anomaly with corrections
    M_corr = ephem['deltaN'] * dt
    M      = ephem['M_0'] + (sqrt_mu_A * dt) + M_corr

    # Compute Eccentric Anomaly
    E = _compute_eccentric_anomoly(M, e, tol=1e-5)

    # Determine pseudorange corrections due to satellite clock corrections.
    # Calculate time offset from satellite reference time
    t_offset = gpstime - ephem['t_oc']
    if np.abs(t_offset).any() > 302400:
        t_offset = t_offset-np.sign(t_offset)*604800

    # Calculate clock corrections from the polynomial
    # corr_polynomial = ephem.af0
    #                 + ephem.af1*t_offset
    #                 + ephem.af2*t_offset**2
    corr_polynomial = (ephem['SVclockBias']
                     + ephem['SVclockDrift']*t_offset
                     + ephem['SVclockDriftRate']*t_offset**2)

    # Calcualte the relativistic clock correction
    corr_relativistic = consts.F * e * sqrt_sma * np.sin(E)

    # Calculate the total clock correction including the Tgd term
    clk_corr = (corr_polynomial - ephem['TGD'] + corr_relativistic)

    # NOTE: Removed ionospheric delay calculation here

    # calculate clock pseudorange correction
    pr_corr +=  clk_corr*consts.C

    if rx_ecef[0][0] is not None: # TODO: Reference using 2D array slicing
        # Calculate the tropospheric delays
        tropo_delay = calculate_tropo_delay(gpstime, gpsweek, ephem, rx_ecef)
        # Calculate total pseudorange correction
        pr_corr -= tropo_delay*consts.C

    if isinstance(pr_corr, pd.Series):
        pr_corr = pr_corr.to_numpy(dtype=float)

    # fill nans (fix for non-GPS satellites)
    pr_corr = np.where(np.isnan(pr_corr), pr_meas, pr_corr)

    return pr_corr


def calculate_tropo_delay(gpstime, gpsweek, ephem, rx_ecef):
    """Calculate tropospheric delay

    Parameters
    ----------
    gpstime : float
        Time of clock in seconds of the week
    gpsweek : int
        GPS week for time of clock
    ephem : pd.DataFrame
        Satellite ephemeris parameters for measurement SVs
    rx_ecef : ndarray
        1x3 array of ECEF rx_pos position [m]

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

    # Make sure things are arrays
    if not isinstance(gpstime, np.ndarray):
        gpstime = np.array(gpstime)
    if not isinstance(gpsweek, np.ndarray):
        gpsweek = np.array(gpsweek)

    # Determine the satellite locations
    sv_posvel = find_sat(ephem, gpstime, gpsweek)
    _, sv_pos, _ = _extract_pos_vel_arr(sv_posvel)

    # compute elevation and azimuth
    el_az = ecef_to_el_az(rx_ecef, sv_pos)
    el_r  = np.deg2rad(el_az[:,0])

    # Calculate the WGS-84 latitude/longitude of the receiver
    rx_lla = ecef_to_geodetic(rx_ecef)
    height = rx_lla[:,2]

    # Force height to be positive
    ind = np.argwhere(height < 0).flatten()
    if len(ind) > 0:
        height[ind] = 0

    # Calculate the delay
    # TODO: Store these numbers somewhere, we should know where they're from -BC
    c_1 = 2.47
    c_2 = 0.0121
    c_3 = 1.33e-4
    tropo_delay = c_1/(np.sin(el_r)+c_2) * np.exp(-height*c_3)/consts.C

    return tropo_delay

def _compute_eccentric_anomoly(M, e, tol=1e-5, max_iter=10):
    """Compute the eccentric anomaly from mean anomaly using the Newton-Raphson
    method using equation: f(E) = M - E + e * sin(E) = 0.

    Parameters
    ----------
    M : pd.DataFrame
        Mean Anomaly of GNSS satellite orbits
    e : pd.DataFrame
        Eccentricity of GNSS satellite orbits
    tol : float
        Tolerance for Newton-Raphson convergence
    max_iter : int
        Maximum number of iterations for Newton-Raphson

    Returns
    -------
    E : pd.DataFrame
        Eccentric Anomaly of GNSS satellite orbits

    """
    E = M
    for _ in np.arange(0, max_iter):
        f    = M - E + e * np.sin(E)
        dfdE = e*np.cos(E) - 1.
        dE   = -f / dfdE
        E    = E + dE

    if any(dE.iloc[:] > tol):
        print("Eccentric Anomaly may not have converged: dE = ", dE)

    return E
