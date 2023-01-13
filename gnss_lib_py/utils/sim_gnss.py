"""Generates expected measurements and simulates pseudoranges.

Functions to generate expected measurements and to simulate pseudoranges
and doppler for GPS satellites.

"""

__authors__ = "Ashwin Kanhere, Bradley Collicott"
__date__ = "26 May 2022"

import warnings

import numpy as np
from numpy.random import default_rng

import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.coordinates import ecef_to_geodetic, ecef_to_el_az
from gnss_lib_py.parsers.navdata import NavData


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


def simulate_measures(gps_week, gps_tow, state, noise_dict=None, ephem=None,
                      sv_posvel=None, rng=None, el_mask=5.):
    """Simulate GNSS pseudoranges and doppler measurements given receiver state.

    Measurements are simulated by finding satellites visible from the
    given position (in state), computing the expected pseudorange and
    doppler measurements for those satellites, corresponding to the given
    state, and adding Gaussian noise to these expected measurements.

    Parameters
    ----------
    gps_week : int
        GPS week at which measurements and positions are needed
    gps_tow : float
        GPS time of week for corresponding GPS week at which
        measurements are needed
    state : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing state i.e. 3D position, 3D velocity,
        receiver clock bias and receiver clock drift rate at which
        measurements have to be simulated.
        Must be a single state (single column)
    noise_dict : dict
        Dictionary with pseudorange ('prange_sigma') and doppler noise
        ('doppler_sigma') standard deviation values in [m] and [m/s].
        If None, uses default values `prange_sigma=6` and
        `doppler_sigma=1`.
    ephem : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing satellite ephemeris parameters for a
        particular time of ephemeris. Use None if not available and using
        SV positions directly instead
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Precomputed positions of satellites, set to None if not available
    rng : np.random.Generator
        A random number generator for sampling random noise values
    el_mask: float
        The elevation mask above which satellites are considered visible
        from the given receiver position. Only visible sate

    Returns
    -------
    measurements : gnss_lib_py.parsers.navdata.NavData
        Pseudorange (label: `prange`) and doppler (label: `doppler`)
        measurements with satellite SV. Gaussian noise is added to
        expected measurements to simulate stochasticity
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Satellite positions and velocities (same as input if provided)

    """
    #TODO: Verify the default noise value for doppler range
    #Handle default values
    if rng is None:
        rng = default_rng()

    if noise_dict is None:
        noise_dict = {}
        noise_dict['prange_sigma'] = 6.
        noise_dict['doppler_sigma'] = 1.

    pos, _, _, _ = _extract_state_variables(state)


    ephem = _find_visible_svs(gps_week, gps_tow, pos, ephem, el_mask=el_mask)
    measurements, sv_posvel = expected_measures(gps_week, gps_tow, state,
                                                ephem, sv_posvel)
    num_svs   = len(measurements)


    measurements['prange']  = (measurements['prange']
        + noise_dict['prange_sigma'] *rng.standard_normal(num_svs))

    measurements['doppler'] = (measurements['doppler']
        + noise_dict['doppler_sigma']*rng.standard_normal(num_svs))

    return measurements, sv_posvel


def expected_measures(gps_week, gps_tow, state, ephem=None, sv_posvel=None):
    """Compute expected pseudoranges and doppler measurements given receiver
    states.

    Parameters
    ----------
    gps_week : int
        GPS week at which measurements and positions are needed
    gps_tow : float
        GPS time of week for corresponding GPS week at which
        measurements are needed
    state : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing state i.e. 3D position, 3D velocity,
        receiver clock bias and receiver clock drift rate at which
        measurements have to be simulated.
        Must be a single state (single column)
    ephem : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing satellite ephemeris parameters for a
        particular time of ephemeris, use None if not available and
        using position directly
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Precomputed positions of satellites (if available)

    Returns
    -------
    measurements : gnss_lib_py.parsers.navdata.NavData
        Pseudorange (label: `prange`) and doppler (label: `doppler`)
        measurements with satellite SV. Also contains SVs and gps_tow at
        which the measurements are simulated
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Satellite positions and velocities (same as input if provided)
    """
    # and satellite positions in sv_posvel
    rx_ecef, rx_v_ecef, clk_bias, clk_drift = _extract_state_variables(state)
    sv_posvel, del_pos, true_range = _find_sv_location(gps_week, gps_tow,
                                                         ephem, rx_ecef, sv_posvel)
    # sv_pos, sv_vel, del_pos are both Nx3
    _, sv_vel = _extract_pos_vel_arr(sv_posvel)

    # Obtain corrected pseudoranges and add receiver clock bias to them
    prange = true_range
    prange += clk_bias
    # prange = (correct_pseudorange(gps_week, gps_tow ephem, true_range,
    #                              np.reshape(pos, [-1, 3])) + bias)
    # TODO: Correction should be applied to the received pseudoranges, not
    # modelled/expected pseudorange -- per discussion in meeting on 11/12
    # TODO: Add corrections instead of returning corrected pseudoranges
    #TODO: Update to use gps_millis in satellite positions and measurements
    # instead of gps_tow
    # Obtain difference of velocity between satellite and receiver

    del_vel = sv_vel - np.tile(np.reshape(rx_v_ecef, [3,1]), [1, len(sv_posvel)])
    prange_rate = np.sum(del_vel*del_pos, axis=0)/true_range
    prange_rate += clk_drift
    doppler = -(consts.F1/consts.C) * (prange_rate)
    measurements = NavData()
    measurements['prange'] = prange
    measurements['doppler'] = doppler
    return measurements, sv_posvel


def find_sat(gps_tow, gps_week, ephem):
    """Compute position and velocities for all satellites in ephemeris file
    given time of clock.

    Parameters
    ----------
    gps_tow : np.ndarray
        GPS time of the week at which positions are required [s]
    gps_week : int
        Week of GPS calendar corresponding to time of clock
    ephem : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing ephemeris parameters of satellites
        for which states are required

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

    More details on the algorithm used to compute satellite positions
    from broadcast navigation message can be found in [1]_.

    Satellite velocity calculations based on algorithms introduced in [2]_.

    References
    ----------
    ..  [1] Misra, P. and Enge, P,
        "Global Positioning System: Signals, Measurements, and Performance."
        2nd Edition, Ganga-Jamuna Press, 2006.
    ..  [2] B. F. Thompson, S. W. Lewis, S. A. Brown, and T. M. Scott,
        “Computing GPS satellite velocity and acceleration from the broadcast
        navigation message,” NAVIGATION, vol. 66, no. 4, pp. 769–779, Dec. 2019,
        doi: 10.1002/navi.342.

    """

    # Extract parameters

    c_is = ephem['C_is']
    c_ic = ephem['C_ic']
    c_rs = ephem['C_rs']
    c_rc = ephem['C_rc']
    c_uc = ephem['C_uc']
    c_us = ephem['C_us']
    delta_n   = ephem['deltaN']

    ecc        = ephem['e']     # eccentricity
    omega    = ephem['omega'] # argument of perigee
    omega_0  = ephem['Omega_0']
    sqrt_sma = ephem['sqrtA'] # sqrt of semi-major axis
    sma      = sqrt_sma**2      # semi-major axis

    sqrt_mu_a = np.sqrt(consts.MU_EARTH) * sqrt_sma**-3 # mean angular motion
    gpsweek_diff = (np.mod(gps_week,1024) - np.mod(ephem['gps_week'],1024))*604800.

    #TODO: See if these statements need to be removed
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
    sv_posvel['times'] = gps_tow
    #TODO: Update to add gps_millis instead of gps_tow

    delta_t = gps_tow - ephem['t_oe'] + gpsweek_diff

    # Calculate the mean anomaly with corrections
    ecc_anom = _compute_eccentric_anomaly(ephem, gps_tow, gps_week)

    cos_e   = np.cos(ecc_anom)
    sin_e   = np.sin(ecc_anom)
    e_cos_e = (1 - ecc*cos_e)

    # Calculate the true anomaly from the eccentric anomaly
    sin_nu = np.sqrt(1 - ecc**2) * (sin_e/e_cos_e)
    cos_nu = (cos_e-ecc) / e_cos_e
    nu_rad     = np.arctan2(sin_nu, cos_nu)

    # Calcualte the argument of latitude iteratively
    phi_0 = nu_rad + omega
    phi   = phi_0
    for incl in range(5):
        cos_to_phi = np.cos(2.*phi)
        sin_to_phi = np.sin(2.*phi)
        phi_corr = c_uc * cos_to_phi + c_us * sin_to_phi
        phi = phi_0 + phi_corr

    # Calculate the longitude of ascending node with correction
    omega_corr = ephem['OmegaDot'] * delta_t

    # Also correct for the rotation since the beginning of the GPS week for which the Omega0 is
    # defined.  Correct for GPS week rollovers.

    # Also correct for the rotation since the beginning of the GPS week for
    # which the Omega0 is defined.  Correct for GPS week rollovers.
    omega = omega_0 - (consts.OMEGA_E_DOT*(gps_tow + gpsweek_diff)) + omega_corr

    # Calculate orbital radius with correction
    r_corr = c_rc * cos_to_phi + c_rs * sin_to_phi
    orb_radius      = sma*e_cos_e + r_corr

    ############################################
    ######  Lines added for velocity (1)  ######
    ############################################
    #TODO: Factorize out into an internal function for calculating
    # satellite velocities
    delta_e   = (sqrt_mu_a + delta_n) / e_cos_e
    dphi = np.sqrt(1 - ecc**2)*delta_e / e_cos_e
    # Changed from the paper
    delta_r   = (sma * ecc * delta_e * sin_e) + 2*(c_rs*cos_to_phi - c_rc*sin_to_phi)*dphi

    # Calculate the inclination with correction
    i_corr = c_ic*cos_to_phi + c_is*sin_to_phi + ephem['IDOT']*delta_t
    incl = ephem['i_0'] + i_corr

    ############################################
    ######  Lines added for velocity (2)  ######
    ############################################
    delta_i = 2*(c_is*cos_to_phi - c_ic*sin_to_phi)*dphi + ephem['IDOT']

    # Find the position in the orbital plane
    x_plane = orb_radius*np.cos(phi)
    y_plane = orb_radius*np.sin(phi)

    ############################################
    ######  Lines added for velocity (3)  ######
    ############################################
    delta_u = (1 + 2*(c_us * cos_to_phi - c_uc*sin_to_phi))*dphi
    dxp = delta_r*np.cos(phi) - orb_radius*np.sin(phi)*delta_u
    dyp = delta_r*np.sin(phi) + orb_radius*np.cos(phi)*delta_u
    # Find satellite position in ECEF coordinates
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    cos_i = np.cos(incl)
    sin_i = np.sin(incl)

    sv_posvel['x_sv_m'] = x_plane*cos_omega - y_plane*cos_i*sin_omega
    sv_posvel['y_sv_m'] = x_plane*sin_omega + y_plane*cos_i*cos_omega
    sv_posvel['z_sv_m'] = y_plane*sin_i
    # TODO: Add satellite clock bias here using the 'clock corrections' not to
    # be used but compared against SP3 and Android data

    ############################################
    ######  Lines added for velocity (4)  ######
    ############################################
    omega_dot = ephem['OmegaDot'] - consts.OMEGA_E_DOT
    sv_posvel['vx_sv_mps'] = (dxp * cos_omega
                         - dyp * cos_i*sin_omega
                         + y_plane  * sin_omega*sin_i*delta_i
                         - (x_plane * sin_omega + y_plane*cos_i*cos_omega)*omega_dot)

    sv_posvel['vy_sv_mps'] = (dxp * sin_omega
                         + dyp * cos_i * cos_omega
                         - y_plane  * sin_i * cos_omega * delta_i
                         + (x_plane * cos_omega - (y_plane*cos_i*sin_omega)) * omega_dot)

    sv_posvel['vz_sv_mps'] = dyp*sin_i + y_plane*cos_i*delta_i

    return sv_posvel


def _extract_pos_vel_arr(sv_posvel):
    """Extract satellite positions and velocities into numpy arrays.

    Parameters
    ----------
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        NavData containing satellite position and velocity states

    Returns
    -------
    sv_pos : np.ndarray
        ECEF satellite x, y and z positions 3xN [m]
    sv_vel : np.ndarray
        ECEF satellite x, y and z velocities 3xN [m]
    """
    sv_pos = sv_posvel[['x_sv_m', 'y_sv_m', 'z_sv_m']]
    sv_vel   = sv_posvel[['vx_sv_mps', 'vy_sv_mps', 'vz_sv_mps']]
    #TODO: Put the following statements in the testing file
    assert np.shape(sv_pos)[0]==3, "sv_pos: Incorrect shape Expected 3xN"
    assert np.shape(sv_vel)[0]==3, "sv_vel: Incorrect shape Expected 3xN"
    return sv_pos, sv_vel


def _extract_state_variables(state):
    """Extract position, velocity and clock bias terms from state

    Parameters
    ----------
    state : gnss_lib_py.parsers.navdata.NavData
        NavData containing state values i.e. 3D position, 3D velocity,
        receiver clock bias and receiver clock drift rate at which
        measurements will be simulated

    Returns
    -------
    rx_ecef : np.ndarray
        3x1 Receiver 3D ECEF position [m]
    rx_v_ecef : np.ndarray
        3x1 Receiver 3D ECEF velocity
    bias : float
        Receiver clock bais [m]
    b_dot : float
        Receiver clock drift [m/s]

    """
    assert len(state)==1, "Only single state accepted for GNSS simulation"
    rx_ecef = np.reshape(state[['x_rx_m', 'y_rx_m', 'z_rx_m']], [3,1])
    rx_v_ecef = np.reshape(state[['vx_rx_mps', 'vy_rx_mps', 'vz_rx_mps']], [3,1])
    clk_bias = state['b_rx_m']
    clk_drift = state['b_dot_rx_mps']
    return rx_ecef, rx_v_ecef, clk_bias, clk_drift


def _find_visible_svs(gps_week, gps_tow, rx_ecef, ephem=None, sv_posvel=None, el_mask=5.):
    #TODO: Add functionality to use given satellite positions
    """Trim input ephemeris to keep only visible SVs.

    Parameters
    ----------
    gps_week : int
        Week in GPS calendar
    gps_tow : float
        GPS time of the week for simulate measurements [s]
    rx_ecef : np.ndarray
        3x1 row rx_pos ECEF position vector [m]
    ephem : gnss_lib_py.parsers.navdata.NavData
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
    #TODO: Add handling for ephem=None
    approx_posvel = find_sat(ephem, gps_tow - consts.T_TRANS, gps_week)

    # Find elevation and azimuth angles for all satellites
    approx_pos, _ = _extract_pos_vel_arr(approx_posvel)
    approx_el_az = ecef_to_el_az(np.reshape(rx_ecef, [3, 1]), approx_pos)
    # Keep attributes of only those satellites which are visible
    keep_ind = approx_el_az[0,:] > el_mask
    eph = ephem.copy(cols=np.nonzero(keep_ind))
    return eph


def _find_sv_location(gps_week, gps_tow, rx_ecef, ephem=None, sv_posvel=None):
    """Return satellite positions, difference from rx_pos position and ranges.

    Parameters
    ----------
    gps_week : int
        Week in GPS calendar
    gps_tow : float
        GPS time of the week for simulate measurements [s]
    rx_ecef : np.ndarray
        3x1 Receiver 3D ECEF position [m]
    ephem : gnss_lib_py.parsers.navdata.NavData
        DataFrame containing all satellite ephemeris parameters for gps_week and
        gps_tow, use None if not available and using precomputed satellite
        positions and velocities instead
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
    rx_ecef = np.reshape(rx_ecef, [3, 1])
    #TODO: Add handling for ephem=None
    if sv_posvel is None:
        sv_posvel = find_sat(ephem, gps_tow - consts.T_TRANS, gps_week)
        del_pos, true_range = _find_delxyz_range(sv_posvel, rx_ecef)
        t_corr = true_range/consts.C

        # Find satellite locations at (a more accurate) time of transmission
        sv_posvel = find_sat(ephem, gps_tow-t_corr, gps_week)
    del_pos, true_range = _find_delxyz_range(sv_posvel, rx_ecef)
    t_corr = true_range/consts.C
    # Corrections for the rotation of the Earth during transmission
    # sv_pos, sv_vel = _extract_pos_vel_arr(sv_posvel)
    del_x = consts.OMEGA_E_DOT*sv_posvel['x_sv_m'] * t_corr
    del_y = consts.OMEGA_E_DOT*sv_posvel['y_sv_m'] * t_corr
    sv_posvel['x_sv_m'] = sv_posvel['x_sv_m'] + del_x
    sv_posvel['y_sv_m'] = sv_posvel['y_sv_m'] + del_y

    return sv_posvel, del_pos, true_range



def _find_delxyz_range(sv_posvel, rx_ecef):
    """Return difference of satellite and rx_pos positions and range between them.

    Parameters
    ----------
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Satellite position and velocities
    rx_ecef : np.ndarray
        3x1 Receiver 3D ECEF position [m]

    Returns
    -------
    del_pos : np.ndarray
        Difference between satellite positions and receiver position
    true_range : np.ndarray
        Distance between satellite and receiver positions
    """
    rx_ecef = np.reshape(rx_ecef, [3, 1])
    satellites = len(sv_posvel)
    if np.size(rx_ecef)!=3:
        raise ValueError(f'Position not 3D, has size {np.size(rx_ecef)}')
    sv_pos, _ = _extract_pos_vel_arr(sv_posvel)
    del_pos = sv_pos - np.tile(rx_ecef, (1, satellites))
    true_range = np.linalg.norm(del_pos, axis=0)
    return del_pos, true_range


def _calculate_pseudorange_corr(gps_week, gps_tow, ephem, iono_params=None, rx_ecef=None):
    #TODO: Update units of returned values based on what the delays return
    """Incorporate corrections in measurements.

    Incorporate clock corrections (relativistic and polynomial), tropospheric
    and ionospheric atmospheric delay corrections.

    Parameters
    ----------
    gps_week : int
        GPS week for time of clock
    gps_tow : float
        Time of clock in seconds of the week
    ephem : gnss_lib_py.parsers.navdata.NavData
        Satellite ephemeris parameters for measurement SVs
    rx_ecef : np.ndarray
        3x1 array of ECEF rx_pos position [m]
    iono_params : np.ndarray
        Ionospheric atmospheric delay parameters for Klobuchar model,
        passed in 2x4 array, use None if not available
    rx_ecef : np.ndarray
        3x1 receiver position in ECEF frame of reference [m], use None
        if not available.

    Returns
    -------
    clock_corr : np.ndarray
        Satellite clock corrections [m]
    iono_delay : np.ndarray
        Estimated delay caused by the ionosphere [m]
    tropo_delay : np.ndarray
        Estimated delay caused by the troposhere [m]

    Notes
    -----
    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    """

    # Extract parameters
    # M_0  = ephem['M_0']
    # dN   = ephem['deltaN']

    # Make sure gps_tow and gps_week are arrays
    if not isinstance(gps_tow, np.ndarray):
        gps_tow = np.array(gps_tow)
    if not isinstance(gps_week, np.ndarray):
        gps_week = np.array(gps_week)

    # NOTE: Removed ionospheric delay calculation here

    # calculate clock pseudorange correction
    clock_corr = _calculate_clock_delay(gps_tow, gps_week, ephem)

    if rx_ecef is not None:
        # Calculate the tropospheric delays
        tropo_delay = _calculate_tropo_delay(gps_tow, gps_week, ephem, rx_ecef)
    else:
        warnings.warn("Receiver position not given, returning 0 "\
                    + "ionospheric delay", RuntimeWarning)
        tropo_delay = np.zeros(len(ephem))

    if iono_params is not None and rx_ecef is not None:
        iono_delay = _calculate_iono_delay(gps_tow, gps_week, ephem, iono_params, rx_ecef)
    else:
        warnings.warn("Ionospheric delay parameters or receiver position"\
                    + "not given, returning 0 ionospheric delay", \
                        RuntimeWarning)
        iono_delay = np.zeros(len(ephem))

    #TODO: Check if the corrections and delays are in meters or seconds
    return clock_corr, tropo_delay, iono_delay


def _calculate_clock_delay(gps_tow, gps_week, ephem):
    #TODO: Check if the corrections and delays are in meters or seconds
    """Calculate the modelled satellite clock delay

    Parameters
    ---------
    gps_tow : int
        GPS time of the week [s]
    gps_week : int
        GPS week for time of clock
    ephem : gnss_lib_py.parsers.navdata.NavData
        Satellite ephemeris parameters for measurement SVs

    Returns
    -------
    clock_corr : np.ndarray
        Satellite clock corrections containing all terms [m]
    corr_polynomial : np.ndarray
        Polynomial clock perturbation terms [m]
    clock_relativistic : np.ndarray
        Relativistic clock correction terms [m]

    """
    # Extract required GPS constants
    ecc        = ephem['e']     # eccentricity
    sqrt_sma = ephem['sqrtA'] # sqrt of semi-major axis

    # if np.abs(delta_t).any() > 302400:
    #     delta_t = delta_t - np.sign(delta_t)*604800

    # Compute Eccentric Anomaly
    ecc_anom = _compute_eccentric_anomaly(ephem, gps_tow, gps_week)

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
    corr_relativistic = consts.F * ecc * sqrt_sma * np.sin(ecc_anom)

    # Calculate the total clock correction including the Tgd term
    clk_corr = (corr_polynomial - ephem['TGD'] + corr_relativistic)

    return clk_corr, corr_polynomial, corr_relativistic


def _calculate_tropo_delay(gps_week, gps_tow, rx_ecef, ephem=None, sv_posvel=None):
    #TODO: Check if the corrections and delays are in meters or seconds
    """Calculate tropospheric delay

    Parameters
    ----------
    gps_week : int
        GPS week for time of clock
    gps_tow : float
        Time of clock in seconds of the week
    ephem : gnss_lib_py.parsers.navdata.NavData
        Satellite ephemeris parameters for measurement SVs
    rx_ecef : np.ndarray
        3x1 array of ECEF rx_pos position [m]
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        Precomputed positions of satellites, set to None if not available

    Returns
    -------
    tropo_delay : np.ndarray
        Tropospheric corrections to pseudorange measurements [m]

    Notes
    -----
    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    """

    # Make sure things are arrays
    if not isinstance(gps_tow, np.ndarray):
        gps_tow = np.array(gps_tow)
    if not isinstance(gps_week, np.ndarray):
        gps_week = np.array(gps_week)
    # Make sure that receiver position is 3x1
    rx_ecef = np.reshape(rx_ecef, [3,1])

    # Determine the satellite locations
    if sv_posvel is None:
        sv_posvel = find_sat(ephem, gps_tow, gps_week)
    sv_pos, _ = _extract_pos_vel_arr(sv_posvel)

    # compute elevation and azimuth
    el_az = ecef_to_el_az(rx_ecef, sv_pos)
    el_r  = np.deg2rad(el_az[0, :])

    # Calculate the WGS-84 latitude/longitude of the receiver
    rx_lla = ecef_to_geodetic(rx_ecef)
    height = rx_lla[2, :]

    # Force height to be positive
    ind = np.argwhere(height < 0).flatten()
    if len(ind) > 0:
        height[ind] = 0

    # Calculate the delay
    tropo_delay = consts.TROPO_DELAY_C1/(np.sin(el_r)+consts.TROPO_DELAY_C2) \
                     * np.exp(-height*consts.TROPO_DELAY_C3)/consts.C

    return tropo_delay


def _calculate_iono_delay(gps_tow, gps_week, ephem, iono_params, rx_ecef):
    """Calculate the ionospheric delay in pseudorange using the Klobuchar
    model Section 5.3.2_[3].

    Parameters
    ----------
    gps_week : int
        GPS week for time of clock
    gps_tow : float
        Time of clock in seconds of the week
    ephem : gnss_lib_py.parsers.navdata.NavData
        Satellite ephemeris parameters for measurement SVs
    rx_ecef : np.ndarray
        3x1 array of ECEF rx_pos position [m]
    iono_params : np.ndarray
        Ionospheric atmospheric delay parameters for Klobuchar model,
        passed in 2x4 array, use None if not available
    rx_ecef : np.ndarray
        3x1 receiver position in ECEF frame of reference [m], use None
        if not available.

    Returns
    -------
    clock_corr : np.ndarray
        Satellite clock corrections [m]
    iono_delay : np.ndarray
        Estimated delay caused by the ionosphere [m]
    tropo_delay : np.ndarray
        Estimated delay caused by the troposhere [m]

    Notes
    -----
    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    References
    ----------
    ..  [3] Misra, P. and Enge, P,
    "Global Positioning System: Signals, Measurements, and Performance."
    2nd Edition, Ganga-Jamuna Press, 2006.
    """
    #TODO: Check if the corrections and delays are in meters or seconds
    # Make sure things are arrays
    if not isinstance(gps_tow, np.ndarray):
        gps_tow = np.array(gps_tow)
    if not isinstance(gps_week, np.ndarray):
        gps_week = np.array(gps_week)

    #Reshape receiver position to 3x1
    rx_ecef = np.reshape(rx_ecef, [3,1])

    # Determine the satellite locations
    sv_posvel = find_sat(ephem, gps_tow, gps_week)
    sv_pos, _ = _extract_pos_vel_arr(sv_posvel)
    el_az = ecef_to_el_az(rx_ecef, sv_pos)
    el_r = np.deg2rad(el_az[0, :])
    az_r = np.deg2rad(el_az[1, :])

    # Calculate the WGS-84 latitude/longitude of the receiver
    wgs_llh = ecef_to_geodetic(rx_ecef)
    lat_r = np.deg2rad(wgs_llh[0, :])
    lon_r = np.deg2rad(wgs_llh[1, :])

    # Parse the ionospheric parameters
    alpha = iono_params[0,:]
    beta = iono_params[1,:]

    # Calculate the psi angle
    psi = 0.1356/(el_r+0.346) - 0.0691

    # Calculate the ionospheric geodetic latitude
    lat_i = lat_r + psi * np.cos(az_r)

    # Make sure values are in bounds
    ind = np.argwhere(np.abs(lat_i) > 1.3090)
    if len(ind) > 0:
        lat_i[ind] = 1.3090 * np.sign(lat_i[ind])
    # Calculate the ionospheric geodetic longitude
    lon_i = lon_r + psi * np.sin(az_r)/np.cos(lat_i)

    # Calculate the solar time corresponding to the gps_tow
    solar_time = 1.3751e4 * lon_i + gps_tow

    # Make sure values are in bounds
    solar_time = np.mod(solar_time,86400)

    # Calculate the geomagnetic latitude (semi-circles)
    lat_m = (lat_i + 2.02e-1 * np.cos(lon_i - 5.08))/np.pi
    # Calculate the period
    period = beta[0]+beta[1]*lat_m+beta[2]*lat_m**2+beta[3]*lat_m**3

    # Make sure values are in bounds
    ind = np.argwhere(period < 72000).flatten()
    if len(ind) > 0:
        period[ind] = 72000

    # Calculate the local time angle
    theta = 2*np.pi*(solar_time - 50400) / period

    # Calculate the amplitude term
    amp = (alpha[0]+alpha[1]*lat_m+alpha[2]*lat_m**2+alpha[3]*lat_m**3)

    # Make sure values are in bounds
    ind = np.argwhere(amp < 0).flatten()
    if len(ind) > 0:
        amp[ind] = 0

    # Calculate the slant factor
    slant_fact = 1.0 + 5.16e-1 * (1.6755-el_r)**3

    # Calculate the ionospheric delay
    iono_delay = slant_fact * 5.0e-9
    ind = np.argwhere(np.abs(theta) < np.pi/2.).flatten()
    if len(ind) > 0:
        iono_delay[ind] = slant_fact[ind]* \
            (5e-9+amp[ind]*(1-theta[ind]**2/2.+theta[ind]**4/24.))

    return iono_delay


def _compute_eccentric_anomaly(ephem, gps_tow, gps_week, tol=1e-5, max_iter=10):
    """Compute the eccentric anomaly from ephemeris parameters.
    This function extracts relevant parameters from the broadcast navigation
    ephemerides and then solves the equation `f(E) = M - E + e * sin(E) = 0`
    using the Newton-Raphson method.

    In the above equation `M` is the corrected mean anomaly, `e` is the
    orbit eccentricity and `E` is the eccentric anomaly, which is unknown.

    Parameters
    ----------
    ephem : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing ephemeris parameters of satellites
        for which states are required
    gps_tow : np.ndarray
        GPS time of the week at which positions are required [s]
    gps_week : int
        Week of GPS calendar corresponding to time of clock
    tol : float
        Tolerance for convergence of the Newton-Raphson
    max_iter : int
        Maximum number of iterations for Newton-Raphson

    Returns
    -------
    ecc_anom : np.ndarray
        Eccentric Anomaly of GNSS satellite orbits

    """
    #Extract required parameters from ephemeris and GPS constants
    delta_n   = ephem['deltaN']
    mean_anom_0  = ephem['M_0']
    sqrt_sma = ephem['sqrtA'] # sqrt of semi-major axis
    sqrt_mu_a = np.sqrt(consts.MU_EARTH) * sqrt_sma**-3 # mean angular motion
    ecc        = ephem['e']     # eccentricity
    #Times for computing positions
    gpsweek_diff = (np.mod(gps_week,1024) - np.mod(ephem['gps_week'],1024))*604800.
    delta_t = gps_tow - ephem['t_oe'] + gpsweek_diff

    # Calculate the mean anomaly with corrections
    mean_anom_corr = delta_n * delta_t
    mean_anom = mean_anom_0 + (sqrt_mu_a * delta_t) + mean_anom_corr

    # Compute Eccentric Anomaly
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
