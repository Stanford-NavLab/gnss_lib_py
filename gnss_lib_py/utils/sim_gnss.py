"""Generates expected measurements and simulates pseudoranges.

Functions to generate expected measurements and to simulate pseudoranges
and doppler for GPS satellites.

"""

__authors__ = "Ashwin Kanhere, Bradley Collicott"
__date__ = "26 May 2022"

import numpy as np
import pandas as pd
from numpy.random import default_rng

import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.coordinates import ecef_to_geodetic, ecef_to_el_az
from gnss_lib_py.parsers.navdata import NavData
# TODO: Check if any of the functions are sorting the dataframe w.r.t SV while
# processing the measurements


def svs_from_el_az(elaz_deg):
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
    svs_ned : np.ndarray
        Nx3 satellite NED positions, simulated at a distance of 20,200 km
    """
    assert np.shape(elaz_deg)[0] == 2, "elaz_deg should be a 2xN array"
    el_deg = np.deg2rad(elaz_deg[0, :])
    az_deg = np.deg2rad(elaz_deg[1, :])
    unit_vect = np.zeros([3, np.shape(elaz_deg)[1]])
    unit_vect[0, :] = np.sin(az_deg)*np.cos(el_deg)
    unit_vect[1, :] = np.cos(az_deg)*np.cos(el_deg)
    unit_vect[2, :] = np.sin(el_deg)
    svs_ned = 20200000*unit_vect
    return svs_ned


def _extract_pos_vel_arr(sv_posvel):
    """Extract satellite positions and velocities into numpy arrays.

    Parameters
    ----------
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        NavData containing satellite states

    Returns
    -------
    sv_pos : np.ndarray
        ECEF satellite positions 3xN [m]
    sv_vel : np.ndarray
        ECEF satellite x, y and z velocities 3xN [m]
    """
    sv_pos = sv_posvel[['x_sv_m', 'y_sv_m', 'z_sv_m']]
    sv_vel   = sv_posvel[['vx_sv_mps', 'vy_sv_mps', 'vz_sv_mps']]
    assert np.shape(sv_pos)[0]==3, "sv_pos: Incorrect shape Expected 3xN"
    assert np.shape(sv_vel)[0]==3, "sv_vel: Incorrect shape Expected 3xN"
    return sv_pos, sv_vel

def simulate_measures(gps_week, gps_tow, ephem, pos, bias, b_dot, vel,
                      noise_dict={}, sv_posvel=None):
    #TODO: Migrate codebase to Measurement
    """Simulate GNSS pseudoranges and doppler measurements given receiver state.

    Measurements are simulated by adding Gaussian noise to measurements expected
    based on the receiver states.

    Parameters
    ----------
    gps_week : int
        GPS week at which measurements and positions are needed
    gps_tow : float
        GPS time of week for corresponding GPS week at which
        measurements are needed
    ephem : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing satellite ephemeris parameters for a
        particular time of ephemeris
    pos : np.ndarray
        1x3 Receiver 3D ECEF position [m]
    bias : float
        Receiver clock bais [m]
    b_dot : float
        Receiver clock drift [m/s]
    vel : np.ndarray
        1x3 Receiver 3D ECEF velocity
    noise_dict : dict
        Dictionary with pseudorange ('prange_sigma') and doppler noise
        ('doppler_sigma') standard deviation values in [m] and [m/s]
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Precomputed positions of satellites, set to None if not available

    Returns
    -------
    measurements : gnss_lib_py.parsers.navdata.NavData
        Pseudorange and doppler measurements with satellite SV,
        Gaussian noise is added to expected measurements for simulation
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Satellite positions and velocities (same as input if provided)

    """

    #TODO: Modify to work with input satellite positions
    #TODO: Add assertions/error handling for sizes of position, bias, b_dot and
    # velocity arrays
    #TODO: Modify to use single state vector instead of multiple inputs, using *args maybe
    #TODO: Modify to use single dictionary with uncertainty values
    #TODO: Add an option to change elevation mask
    #TODO: Change the default noise values
    #TODO: Add a condition to check if gps_week and gps_tow are in ephem.
    # If not, convert gps_millis to use these values
    print('ephem type in simulated_measures', type(ephem))
    ephem = _find_visible_svs(gps_week, gps_tow, pos, ephem)
    #TODO: Add elevation mask option here
    measurements, sv_posvel = expected_measures(gps_week, gps_tow, ephem, pos,
                                                bias, b_dot, vel, sv_posvel)
    num_svs   = len(measurements)
    rng = default_rng()

    noise_dict.setdefault('prange_sigma', 6.)
    noise_dict.setdefault('doppler_sigma', 1.)

    measurements['prange']  = (measurements['prange']
        + noise_dict['prange_sigma'] *rng.standard_normal(num_svs))

    measurements['doppler'] = (measurements['doppler']
        + noise_dict['doppler_sigma']*rng.standard_normal(num_svs))
    print('measurements type in ', type(measurements))
    print('sv_posvel type in simulate_measures', type(sv_posvel))
    return measurements, sv_posvel

def expected_measures(gps_week, gps_tow, ephem, pos, bias, b_dot, vel, sv_posvel=None):
    """Compute expected pseudoranges and doppler measurements given receiver
    states.

    Parameters
    ----------
    gps_week : int
        GPS week at which measurements and positions are needed
    gps_tow : float
        GPS time of week for corresponding GPS week at which
        measurements are needed
    ephem : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing satellite ephemeris parameters for a
        particular time of ephemeris
    pos : np.ndarray
        3x1 Receiver 3D ECEF position [m]
    bias : float
        Receiver clock bais [m]
    b_dot : float
        Receiver clock drift [m/s]
    vel : np.ndarray
        3x1 Receiver 3D ECEF velocity
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Precomputed positions of satellites (if available)

    Returns
    -------
    measurements : gnss_lib_py.parsers.navdata.NavData
        Pseudorange and doppler measurements with satellite SV,
        Gaussian noise is added to expected measurements for simulation
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Satellite positions and velocities (same as input if provided)
    """
    # NOTE: When using saved data, pass saved DataFrame with ephemeris in ephem
    # and satellite positions in sv_posvel
    # TODO: Modify this function to use PRNS from measurement in addition to
    # gps_tow from measurement
    pos = np.reshape(pos, [3, 1])
    vel = np.reshape(vel, [3, 1])
    sv_posvel, del_pos, true_range = _find_sv_location(gps_week, gps_tow,
                                                         ephem, pos, sv_posvel)
    # sv_pos, sv_vel, del_pos are both Nx3
    _, sv_vel = _extract_pos_vel_arr(sv_posvel)

    # Obtain corrected pseudoranges and add receiver clock bias to them
    prange = true_range
    prange += bias
    # prange = (correct_pseudorange(gps_week, gps_tow ephem, true_range,
    #                              np.reshape(pos, [-1, 3])) + bias)
    # TODO: Correction should be applied to the received pseudoranges, not
    # modelled/expected pseudorange -- per discussion in meeting on 11/12
    # TODO: Add corrections instead of returning corrected pseudoranges

    # Obtain difference of velocity between satellite and receiver

    del_vel = sv_vel - np.tile(np.reshape(vel, [3,1]), [1, len(sv_posvel)])
    prange_rate = np.sum(del_vel*del_pos, axis=0)/true_range
    prange_rate += b_dot
    doppler = -(consts.F1/consts.C) * (prange_rate)
    #TODO: Delete old lines of code
    # doppler = pd.DataFrame(doppler, index=prange.index.copy())
    # measurements = pd.DataFrame(np.column_stack((prange, doppler)),
    #                             index=sv_posvel.index,
    #                             columns=['prange', 'doppler'])
    measurements = NavData()
    measurements['prange'] = prange
    measurements['doppler'] = doppler
    return measurements, sv_posvel


def _find_visible_svs(gps_week, gps_tow, rx_ecef, ephem, el_mask=5.):
    """Trim input ephemeris to keep only visible SVs.

    Parameters
    ----------
    gps_week : int
        Week in GPS calendar
    gps_tow : float
        GPS time of the week for simulate measurements [s]
    rx_ecef : np.ndarray
        1x3 row rx_pos ECEF position vector [m]
    ephem  gnss_lib_py.parsers.navdata.NavData
        NavData instance containing satellite ephemeris parameters
        including gps_week and gps_tow
    el_mask : float
        Minimum elevation of returned satellites

    Returns
    -------
    eph : gnss_lib_py.parsers.navdata.NavData
        Ephemeris parameters of visible satellites

    """

    # Find positions and velocities of all satellites
    approx_posvel = find_sat(ephem, gps_tow - consts.T_TRANS, gps_week)

    # Find elevation and azimuth angles for all satellites
    approx_pos, _ = _extract_pos_vel_arr(approx_posvel)
    approx_el_az = ecef_to_el_az(np.reshape(rx_ecef, [3, 1]), approx_pos)
    # Keep attributes of only those satellites which are visible
    keep_ind = approx_el_az[0,:] > el_mask
    # prns = approx_posvel.index.to_numpy()[keep_ind]
    # TODO: Remove above statement if superfluous
    # TODO: Check that a copy of the ephemeris is being generated, also if it is
    # needed
    eph = ephem.copy(cols=np.nonzero(keep_ind))
    return eph


def _find_sv_location(gps_week, gps_tow, ephem, pos, sv_posvel=None):
    #TODO: Migrate docstring to using Measurement
    #TODO: Migrate codebase to Measurement
    """Return satellite positions, difference from rx_pos position and ranges.

    Parameters
    ----------
    gps_week : int
        Week in GPS calendar
    gps_tow : float
        GPS time of the week for simulate measurements [s]
    ephem : gnss_lib_py.parsers.navdata.NavData
        DataFrame containing all satellite ephemeris parameters for gps_week and
        gps_tow
    pos : np.ndarray
        1x3 Receiver 3D ECEF position [m]
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Precomputed positions of satellites, use None if not available

    Returns
    -------
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Satellite position and velocities (same if input)
    del_pos : np.ndarray
        Difference between satellite positions and receiver position
    true_range : np.ndarray
        Distance between satellite and receiver positions

    """
    pos = np.reshape(pos, [3, 1])
    if sv_posvel is None:
        satellites = len(ephem)
        sv_posvel = find_sat(ephem, gps_tow - consts.T_TRANS, gps_week)
        del_pos, true_range = _find_delxyz_range(sv_posvel, pos, satellites)
        t_corr = true_range/consts.C

        # Find satellite locations at (a more accurate) time of transmission
        sv_posvel = find_sat(ephem, gps_tow-t_corr, gps_week)
    else:
        satellites = len(sv_posvel)
    del_pos, true_range = _find_delxyz_range(sv_posvel, pos, satellites)
    t_corr = true_range/consts.C
    # Corrections for the rotation of the Earth during transmission
    # sv_pos, sv_vel = _extract_pos_vel_arr(sv_posvel)
    del_x = consts.OMEGA_E_DOT*sv_posvel['x_sv_m'] * t_corr
    del_y = consts.OMEGA_E_DOT*sv_posvel['y_sv_m'] * t_corr
    sv_posvel['x_sv_m'] = sv_posvel['x_sv_m'] + del_x
    sv_posvel['y_sv_m'] = sv_posvel['y_sv_m'] + del_y

    return sv_posvel, del_pos, true_range



def _find_delxyz_range(sv_posvel, pos, satellites=None):
    """Return difference of satellite and rx_pos positions and range between them.

    Parameters
    ----------
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Satellite position and velocities
    pos : np.ndarray
        3x1 Receiver 3D ECEF position [m]
    satellites : int
        Number of satellites in sv_posvel

    Returns
    -------
    del_pos : np.ndarray
        Difference between satellite positions and receiver position
    true_range : np.ndarray
        Distance between satellite and receiver positions
    """
    # Repeating computation in find_sv_location
    #NOTE: Input is from satellite finding in AE 456 code
    #TODO: Do we need satellites or is it enough to use len(sv_posvel)
    pos = np.reshape(pos, [3, 1])
    if satellites is None:
        satellites = len(sv_posvel)
    if np.size(pos)!=3:
        raise ValueError(f'Position not 3D, has size {np.size(pos)}')
    sv_pos, _ = _extract_pos_vel_arr(sv_posvel)
    del_pos = sv_pos - np.tile(pos, (1, satellites))
    true_range = np.linalg.norm(del_pos, axis=0)
    return del_pos, true_range


def find_sat(ephem, times, gpsweek):
    """Compute position and velocities for all satellites in ephemeris file
    given time of clock.

    Parameters
    ----------
    ephem : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing ephemeris parameters of satellites
        for which states are required
    times : np.ndarray
        GPS time of the week at which positions are required [s]
    gpsweek : int
        Week of GPS calendar corresponding to time of clock

    Returns
    -------
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        NavData containing satellite positions, velocities, corresponding
        time and SV number

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
    gpsweek_diff = (np.mod(gpsweek,1024) - np.mod(ephem['gps_week'],1024))*604800.

    # if np.size(times_all)==1:
    #     times_all = times_all*np.ones(len(ephem))
    # else:
    #     times_all = np.reshape(times_all, len(ephem))
    # times = times_all
    sv_posvel = NavData()
    sv_posvel['sv_id'] = ephem['sv_id']
    # sv_posvel.set_index('sv', inplace=True)
    # print(times)
    #TODO: How do you deal with multiple times here?
    # Deal with times being a single value or a vector with the same
    # length as the ephemeris
    # print(times.shape)
    sv_posvel['times'] = times

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

    sv_posvel['x_sv_m'] = xp*cos_omega - yp*cos_i*sin_omega
    sv_posvel['y_sv_m'] = xp*sin_omega + yp*cos_i*cos_omega
    sv_posvel['z_sv_m'] = yp*sin_i
    # TODO: Add satellite clock bias here using the 'clock corrections' not to
    # be used but compared against SP3 and Android data

    ############################################
    ######  Lines added for velocity (4)  ######
    ############################################
    omega_dot = ephem['OmegaDot'] - consts.OMEGA_E_DOT
    sv_posvel['vx_sv_mps'] = (dxp * cos_omega
                         - dyp * cos_i*sin_omega
                         + yp  * sin_omega*sin_i*di
                         - (xp * sin_omega + yp*cos_i*cos_omega)*omega_dot)

    sv_posvel['vy_sv_mps'] = (dxp * sin_omega
                         + dyp * cos_i * cos_omega
                         - yp  * sin_i * cos_omega * di
                         + (xp * cos_omega - (yp*cos_i*sin_omega)) * omega_dot)

    sv_posvel['vz_sv_mps'] = dyp*sin_i + yp*cos_i*di

    return sv_posvel


def correct_pseudorange(gps_week, gps_tow, ephem, pr_meas, rx_ecef=[[None]]):
    #TODO: Migrate docstring to using Measurement
    #TODO: Migrate codebase to Measurement
    """Incorporate corrections in measurements.

    Incorporate clock corrections (relativistic, drift), tropospheric and
    ionospheric clock corrections.

    Parameters
    ----------
    gps_week : int
        GPS week for time of clock
    gps_tow : float
        Time of clock in seconds of the week
    ephem : pd.DataFrame
        Satellite ephemeris parameters for measurement SVs
    pr_meas : np.ndarray
        Ranging measurements from satellites [m]
    rx_ecef : np.ndarray
        1x3 array of ECEF rx_pos position [m]

    Returns
    -------
    pr_corr : np.ndarray
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

    print('ephem type in correct_pseudorange is', type(ephem))
    #TODO: Change function to extract visible satellites for given
    # position or corresponding to received measurements

    assert len(pr_meas)==len(ephem), "In correct pseudorange, ephemeris must be only for visible satellites"

    # Make sure gps_tow and gpsweek are arrays
    if not isinstance(gps_tow, np.ndarray):
        gps_tow = np.array(gps_tow)
    if not isinstance(gps_week, np.ndarray):
        gps_week = np.array(gps_week)

    # Initialize the correction array
    pr_corr = pr_meas



    # NOTE: Removed ionospheric delay calculation here

    # calculate clock pseudorange correction
    print('pr_corr', pr_corr.shape, pr_corr)
    print('clk_corr', clk_corr.shape, clk_corr)
    pr_corr +=  clk_corr*consts.C

    if rx_ecef[0][0] is not None:
        # Calculate the tropospheric delays
        tropo_delay = _calculate_tropo_delay(gps_tow, gps_week, ephem, rx_ecef)
        # Calculate total pseudorange correction
        pr_corr -= tropo_delay*consts.C

    #TODO: Change this following statement
    if isinstance(pr_corr, pd.Series):
        pr_corr = pr_corr.to_numpy(dtype=float)

    # fill nans (fix for non-GPS satellites)
    pr_corr = np.where(np.isnan(pr_corr), pr_meas, pr_corr)

    print('pr_corr type in correct_pseudorange is', type(pr_corr))

    return pr_corr


def _calculate_clock_delay(gps_tow, gps_week, ephem):

    e        = ephem['e']     # eccentricity
    sqrt_sma = ephem['sqrtA'] # sqrt of semi-major axis

    sqrt_mu_A = np.sqrt(consts.MU_EARTH) * sqrt_sma**-3 # mean angular motion

    dt = gps_tow - ephem['t_oe']
    if np.abs(dt).any() > 302400:
        dt = dt - np.sign(dt)*604800

    # Calculate the mean anomaly with corrections
    M_corr = ephem['deltaN'] * dt
    M      = ephem['M_0'] + (sqrt_mu_A * dt) + M_corr

    # Compute Eccentric Anomaly
    E = _compute_eccentric_anomoly(M, e, tol=1e-5)

    # Determine pseudorange corrections due to satellite clock corrections.
    # Calculate time offset from satellite reference time
    t_offset = gps_tow - ephem['t_oc']
    if np.abs(t_offset).any() > 302400:
        t_offset = t_offset-np.sign(t_offset)*604800

    # Calculate clock corrections from the polynomial corrections in
    # broadcast message
    corr_polynomial = (ephem['SVclockBias']
                     + ephem['SVclockDrift']*t_offset
                     + ephem['SVclockDriftRate']*t_offset**2)

    # Calcualte the relativistic clock correction
    corr_relativistic = consts.F * e * sqrt_sma * np.sin(E)

    # Calculate the total clock correction including the Tgd term
    clk_corr = (corr_polynomial - ephem['TGD'] + corr_relativistic)

    return clk_corr, corr_polynomial, corr_relativistic


def _calculate_tropo_delay(gps_tow, gpsweek, ephem, rx_ecef):
    #TODO: Migrate codebase to Measurement
    """Calculate tropospheric delay

    Parameters
    ----------
    gps_tow : float
        Time of clock in seconds of the week
    gpsweek : int
        GPS week for time of clock
    ephem : gnss_lib_py.parsers.measurement.Measurement
        Satellite ephemeris parameters for measurement SVs
    rx_ecef : np.ndarray
        1x3 array of ECEF rx_pos position [m]

    Returns
    -------
    tropo_delay : np.ndarray
        Tropospheric corrections to pseudorange measurements

    Notes
    -----
    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    """

    print('ephem.type in _calculate_tropo_delay', type(ephem))

    # Make sure things are arrays
    if not isinstance(gps_tow, np.ndarray):
        gps_tow = np.array(gps_tow)
    if not isinstance(gpsweek, np.ndarray):
        gpsweek = np.array(gpsweek)

    # Determine the satellite locations
    sv_posvel = find_sat(ephem, gps_tow, gpsweek)
    sv_pos, _ = _extract_pos_vel_arr(sv_posvel)

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

    print('tropo_delay type in _calculate_tropo_delay', type(tropo_delay))
    return tropo_delay

def _compute_eccentric_anomoly(mean_anom, ecc, tol=1e-5, max_iter=10):
    """Compute the eccentric anomaly from mean anomaly using the Newton-Raphson
    method using equation: f(E) = M - E + e * sin(E) = 0.

    Parameters
    ----------
    mean_anom : np.ndarray
        Mean Anomaly of GNSS satellite orbits
    ecc : np.ndarray
        Eccentricity of GNSS satellite orbits
    tol : float
        Tolerance for Newton-Raphson convergence
    max_iter : int
        Maximum number of iterations for Newton-Raphson

    Returns
    -------
    ecc_anom : np.ndarray
        Eccentric Anomaly of GNSS satellite orbits

    """
    ecc_anom = mean_anom
    for _ in np.arange(0, max_iter):
        fun = mean_anom - ecc_anom + ecc * np.sin(ecc_anom)
        df_decc_anom = ecc*np.cos(ecc_anom) - 1.
        delta_ecc_anom   = -fun / df_decc_anom
        ecc_anom    = ecc_anom + delta_ecc_anom

    if np.any(delta_ecc_anom > tol):
        raise RuntimeWarning("Eccentric Anomaly may not have converged" \
                            + f"after {max_iter} steps. : dE = {delta_ecc_anom}")
    return ecc_anom
