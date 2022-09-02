"""Functions to process multi-GNSS related files.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "05 August 2022"

from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import interpolate

from gnss_lib_py.utils.sim_gnss import find_sat
from gnss_lib_py.parsers.ephemeris import EphemerisManager
import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.time_conversions import gps_millis_to_tow

def extract_sp3_func(sp3data, sidx, ipos = 10, \
                     method = 'CubicSpline', verbose = False):
    """Computing an interpolated function that represents sp3data
    around desired index and for any GNSS constellation

    Parameters
    ----------
    sp3data : Array of Sp3 classes with len == # sats
        Instance of GPS-only Sp3 class array for testing
    sidx : int
        Nearest index within the Sp3 time series centered around which the
        interpolated function needs to be computed
    ipos : int (Default: 10)
        No. of data points related to sp3data on either side of sidx value
        that will be used for computing the interpolated function
    method : string (Default: CubicSpline)
        Type of interpolation method used for interpolating sp3 data

    Returns
    -------
    func_satpos : 3-D array of scipy.interpolate.interpolate.interp1d
        Each element of 3-D array represents the interpolated function
        associated with loaded .sp3 data

    References
    ----------
    """

    func_satpos = np.empty((3,), dtype=object)
    func_satpos[:] = np.nan

    if method=='CubicSpline':
        low_i = (sidx - ipos) if (sidx - ipos) >= 0 else 0
        high_i = (sidx + ipos) if (sidx + ipos) <= len(sp3data.tym) else -1

        if verbose:
            print('Nearest sp3: ', sidx, sp3data.tym[sidx], \
                                   sp3data.xpos[sidx], sp3data.ypos[sidx], sp3data.zpos[sidx])

        func_satpos[0] = interpolate.CubicSpline(sp3data.tym[low_i:high_i], \
                                                 sp3data.xpos[low_i:high_i])
        func_satpos[1] = interpolate.CubicSpline(sp3data.tym[low_i:high_i], \
                                                 sp3data.ypos[low_i:high_i])
        func_satpos[2] = interpolate.CubicSpline(sp3data.tym[low_i:high_i], \
                                                 sp3data.zpos[low_i:high_i])

    return func_satpos

def compute_sp3_snapshot(func_satpos, cxtime, hstep = 5e-1, method='CubicSpline'):
    """Compute the satellite 3-D position and velocity via central differencing,
    given an associated 3-D array of interpolation function

    Parameters
    ----------
    func_satpos : 3-D array of scipy.interpolate.interpolate.interp1d
        Each element of 3-D array represents the interpolated function
        associated with loaded .sp3 data.
    cxtime : float
        Time at which the satellite 3-D position and velocity needs to be
        computed, given 3-D array of interpolated functions
    hstep : float (Default: 1e-5)
        Step size in seconds that will be used for computing 3-D velocity of
        any given satellite
    method : string (Default: CubicSpline)
        Type of interpolation method used for interpolating sp3 data

    Returns
    -------
    satpos_sp3 : 3-D array
        3-D satellite position in ECEF frame (Earth's rotation not included)
    satvel_sp3 : 3-D array
        3-D satellite velocity in ECEF frame (Earth's rotation not included)
    """
    if method=='CubicSpline':
        sat_x = func_satpos[0]([cxtime-0.5*hstep, cxtime, cxtime+0.5*hstep])
        sat_y = func_satpos[1]([cxtime-0.5*hstep, cxtime, cxtime+0.5*hstep])
        sat_z = func_satpos[2]([cxtime-0.5*hstep, cxtime, cxtime+0.5*hstep])

    satpos_sp3 = np.array([sat_x[1], sat_y[1], sat_z[1]])
    satvel_sp3 = np.array([ (sat_x[2]-sat_x[0]) / hstep, \
                            (sat_y[2]-sat_y[0]) / hstep, \
                            (sat_z[2]-sat_z[0]) / hstep ])

    return satpos_sp3, (satvel_sp3 * 1e3)

def extract_clk_func(clkdata, sidx, ipos = 10, \
                     method='CubicSpline', verbose = False):
    """Computing an interpolated function that represents clkdata
    around desired index and for any GNSS constellation

    Parameters
    ----------
    clkdata : Array of Clk classes with len == # sats
        Instance of GPS-only Clk class array for testing
    sidx : int
        Nearest index within the Clk time series centered around which the
        interpolated function needs to be computed
    ipos : int (Default: 10)
        No. of data points related to clkdata on either side of sidx value
        that will be used for computing the interpolated function
    method : string (Default: CubicSpline)
        Type of interpolation method used for interpolating clk data

    Returns
    -------
    func_satbias : scipy.interpolate.interpolate.interp1d
        Interpolated function associated with loaded .clk data
    """

    if method=='CubicSpline':
        low_i = (sidx - ipos) if (sidx - ipos) >= 0 else 0
        high_i = (sidx + ipos) if (sidx + ipos) <= len(clkdata.tym) else -1

        if verbose:
            print('Nearest clk: ', sidx, clkdata.tym[sidx], clkdata.clk_bias[sidx])

        func_satbias = interpolate.CubicSpline(clkdata.tym[low_i:high_i], \
                                               clkdata.clk_bias[low_i:high_i])

    return func_satbias

def compute_clk_snapshot(func_satbias, cxtime, hstep = 5e-1, method='CubicSpline'):
    """Compute the satellite clock bias and drift via central differencing,
    given an associated interpolation function

    Parameters
    ----------
    func_satbias : scipy.interpolate.interpolate.interp1d
        Interpolated function associated with loaded .clk data
    cxtime : float
        Time at which the satellite 3-D clock bias and drift needs to be
        computed, given the interpolated function
    hstep : float (Default: 1e-5)
        Step size in seconds that will be used for computing 1-D drift of
        any given satellite clock
    method : string (Default: CubicSpline)
        Type of interpolation method used for interpolating clk data

    Returns
    -------
    satbias_clk : float
        Satellite clock bias in seconds
    satdrift_clk : float
        Satellite clock drift in seconds/seconds
    """

    if method=='CubicSpline':
        sat_t = func_satbias([cxtime-0.5*hstep, cxtime, cxtime+0.5*hstep])

    satbias_clk = sat_t[1]
    satdrift_clk = (sat_t[2]-sat_t[0]) / hstep

    return satbias_clk, (satdrift_clk * 1e3)

def compute_sv_gnss_from_precise_eph(navdata, sp3_parsed_file, \
                                     clk_parsed_file, verbose = False):
    """Populate navdata class with satellite position, velocity,
    clock bias and drift, which is computed via .sp3 and .clk files

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class that depicts android derived dataset
    sp3_parsed_file : Array of gnss_lib_py.parsers.precise_ephemerides.Sp3 classes
        Instance of Sp3 class array associated with any one constellation
    clk_parsed_file : Array of gnss_lib_py.parsers.precise_ephemerides.Clk classes
        Instance of Clk class array associated with any one constellation
    verbose : bool (dafault: False)
        Flag (True/False) for whether to print intermediate steps useful
        for debugging/reviewing

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Modified instance of NavData class that computes satellite position,
        velocity, clock bias and drift using precise ephemerides
    """

    navdata_offset = 0 #1000 # Set this to zero, when android errors get fixed
    unique_gnss_id = np.unique(navdata['gnss_id'])
    if not len(unique_gnss_id)==1:
        raise RuntimeError("Input error: Multiple constellations " + \
                           "cannot be updated simultaneously")

    if not len(sp3_parsed_file) == len(clk_parsed_file):
        raise RuntimeError("Input error: Max no. of PRNs in sp3 and clk " + \
                           "parsed files do not match")

    if not sp3_parsed_file[0].const == clk_parsed_file[0].const:
        raise RuntimeError("Input error: Constellations associated with " + \
                           "sp3 and clk parsed files do not match")

    # add another if condition here that checks if navdata, sp3, clk all same
    # constellation -> after constellation flag changed in precise_ephemerides

    interp_method = 'CubicSpline' # Currently only one functionality

    # Initialize the sp3 and clk iref arrays
    sp3_iref_old = -1 * np.ones( (len(sp3_parsed_file),))
    satfunc_xyz_old = np.empty( (len(sp3_parsed_file),), dtype=object)
    satfunc_xyz_old[:] = np.nan

    clk_iref_old = -1 * np.ones( (len(clk_parsed_file),))
    satfunc_t_old = np.empty( (len(clk_parsed_file),), dtype=object)
    satfunc_t_old[:] = np.nan

    # Compute satellite information for desired time steps
    unique_timesteps = np.unique(navdata["gps_millis"])

    for t_idx, timestep in enumerate(unique_timesteps[:-1]):

        # Compute indices where gps_millis match, sort them
        # sorting is done for consistency across all satellite pos. estimation
        # algorithms as ephemerismanager inherently sorts based on prns
        idxs = np.where(navdata["gps_millis"] == timestep)[0]
        sorted_idxs = idxs[np.argsort(navdata["sv_id", idxs], axis = 0)]

        if verbose:
            print(t_idx, timestep, idxs, sorted_idxs)
            print('misc: ', navdata['gps_millis', sorted_idxs], \
                            navdata['gnss_id', sorted_idxs], \
                            navdata['sv_id', sorted_idxs], \
                            navdata['signal_type', sorted_idxs])

        # this if else condition can be removed later after navdata has been
        # fixed to represent one data point as an array as well
        if len(sorted_idxs)==1:
            visible_sats = np.array([navdata["sv_id", sorted_idxs]])
        else:
            visible_sats = navdata["sv_id", sorted_idxs]

        for sv_idx, prn in enumerate(visible_sats):

            prn = int(prn)

            # Perform nearest time step search to compute iref values for sp3 and clk
            sp3_iref = np.argmin(abs(np.array(sp3_parsed_file[prn].tym) - \
                                     (timestep - navdata_offset) ))
            clk_iref = np.argmin(abs(np.array(clk_parsed_file[prn].tym) - \
                                     (timestep - navdata_offset) ))

            if verbose:
                print('Stats: ', t_idx, timestep, prn, idxs, sorted_idxs)
                print('sp3 stats: ', sp3_iref, sp3_iref_old[prn])
                print('clk stats: ', clk_iref, clk_iref_old[prn])

            # Carry out .sp3 processing by first checking if
            # previous interpolated function holds
            if sp3_iref == sp3_iref_old[prn]:
                func_satpos = satfunc_xyz_old[prn]
            else:
                # if does not hold, recompute the interpolation function based on current iref
                if verbose:
                    print('SP3: Computing new interpolation!')
                func_satpos = extract_sp3_func(sp3_parsed_file[prn], \
                                               sp3_iref, method = interp_method)
                # Update the relevant interp function and iref values
                satfunc_xyz_old[prn] = func_satpos
                sp3_iref_old[prn] = sp3_iref

            # Compute satellite position and velocity using interpolated function
            satpos_sp3, satvel_sp3 = compute_sp3_snapshot(func_satpos, \
                                                          (timestep - navdata_offset), \
                                                          method = interp_method)

            # Adjust the satellite position based on Earth's rotation
            trans_time = navdata["raw_pr_m", sorted_idxs[sv_idx]] / consts.C
            del_x = (consts.OMEGA_E_DOT * satpos_sp3[1] * trans_time)
            del_y = (-consts.OMEGA_E_DOT * satpos_sp3[0] * trans_time)
            satpos_sp3[0] = satpos_sp3[0] + del_x
            satpos_sp3[1] = satpos_sp3[1] + del_y

            # Carry out .clk processing by first checking if previous interpolated
            # function holds
            if clk_iref == clk_iref_old[prn]:
                func_satbias = satfunc_t_old[prn]
            else:
                # if does not hold, recompute the interpolation function based on current iref
                if verbose:
                    print('CLK: Computing new interpolation!')
                func_satbias = extract_clk_func(clk_parsed_file[prn], \
                                                clk_iref, method = interp_method)
                # Update the relevant interp function and iref values
                satfunc_t_old[prn] = func_satbias
                clk_iref_old[prn] = clk_iref

            # Compute satellite clock bias and drift using interpolated function
            satbias_clk, \
            satdrift_clk = compute_clk_snapshot(func_satbias, \
                                                (timestep - navdata_offset), \
                                                method = interp_method)
            if verbose:
#                 print('after sp3:', satpos_sp3, \
#                                     satvel_sp3, \
#                                     consts.C * satbias_clk, \
#                                     consts.C * satdrift_clk)

                satpos_android = np.transpose([ navdata["x_sv_m", sorted_idxs], \
                                                navdata["y_sv_m", sorted_idxs], \
                                                navdata["z_sv_m", sorted_idxs] ])
                print( 'Android-sp3 Pos Error (m): ', \
                          np.linalg.norm(satpos_android[sv_idx] - satpos_sp3), \
                          navdata["x_sv_m", sorted_idxs[sv_idx]] - satpos_sp3[0], \
                          navdata["y_sv_m", sorted_idxs[sv_idx]] - satpos_sp3[1], \
                          navdata["z_sv_m", sorted_idxs[sv_idx]] - satpos_sp3[2] )

                satvel_android = np.transpose([ navdata["vx_sv_mps", sorted_idxs], \
                                                   navdata["vy_sv_mps", sorted_idxs], \
                                                   navdata["vz_sv_mps", sorted_idxs] ])
#                 print('android:', satpos_android[sv_idx], satvel_android[sv_idx])
                print('Android-sp3 Vel Error (m): ', \
                          np.linalg.norm(satvel_android[sv_idx] - satvel_sp3), \
                          navdata["vx_sv_mps", sorted_idxs[sv_idx]] - satvel_sp3[0], \
                          navdata["vy_sv_mps", sorted_idxs[sv_idx]] - satvel_sp3[1], \
                          navdata["vz_sv_mps", sorted_idxs[sv_idx]] - satvel_sp3[2] )

                print('Android-sp3 Clk Error (m): ', \
                          navdata["b_sv_m", sorted_idxs[sv_idx]] - consts.C * satbias_clk, \
                          navdata["b_dot_sv_mps", sorted_idxs[sv_idx]] - consts.C * satdrift_clk)

            # update x_sv_m of navdata with the estimated values from .sp3 files
            navdata['x_sv_m', sorted_idxs[sv_idx]] = np.array([satpos_sp3[0]])
            if not (navdata["x_sv_m", sorted_idxs[sv_idx]] - satpos_sp3[0] == 0.0):
                raise RuntimeError("x_sv_m of navdata not correctly updated")

            # update y_sv_m of navdata with the estimated values from .sp3 files
            navdata['y_sv_m', sorted_idxs[sv_idx]] = np.array([satpos_sp3[1]])
            if not (navdata["y_sv_m", sorted_idxs[sv_idx]] - satpos_sp3[1] == 0.0):
                raise RuntimeError("y_sv_m of navdata not correctly updated")

            # update z_sv_m of navdata with the estimated values from .sp3 files
            navdata['z_sv_m', sorted_idxs[sv_idx]] = np.array([satpos_sp3[2]])
            if not (navdata["z_sv_m", sorted_idxs[sv_idx]] - satpos_sp3[2] == 0.0):
                raise RuntimeError("z_sv_m of navdata not correctly updated")

            # update vx_sv_mps of navdata with the estimated values from .sp3 files
            navdata["vx_sv_mps", sorted_idxs[sv_idx]] = np.array([satvel_sp3[0]])
            if not (navdata["vx_sv_mps", sorted_idxs[sv_idx]] - satvel_sp3[0] == 0.0):
                raise RuntimeError("vx_sv_mps of navdata not correctly updated")

            # update vy_sv_mps of navdata with the estimated values from .sp3 files
            navdata["vy_sv_mps", sorted_idxs[sv_idx]] = np.array([satvel_sp3[1]])
            if not (navdata["vy_sv_mps", sorted_idxs[sv_idx]] - satvel_sp3[1] == 0.0):
                raise RuntimeError("vy_sv_mps of navdata not correctly updated")

            # update vz_sv_mps of navdata with the estimated values from .sp3 files
            navdata["vz_sv_mps", sorted_idxs[sv_idx]] = np.array([satvel_sp3[2]])
            if not (navdata["vz_sv_mps", sorted_idxs[sv_idx]] - satvel_sp3[2] == 0.0):
                raise RuntimeError("vz_sv_mps of navdata not correctly updated")

            # update b_sv_m of navdata with the estimated values from .clk files
            navdata["b_sv_m", sorted_idxs[sv_idx]] = np.array([consts.C * satbias_clk])
            if not (navdata["b_sv_m", sorted_idxs[sv_idx]] - consts.C * satbias_clk == 0.0):
                raise RuntimeError("b_sv_m of navdata not correctly updated")

            # update b_dot_sv_mps of navdata with the estimated values from .clk files
            navdata["b_dot_sv_mps", sorted_idxs[sv_idx]] = np.array([consts.C * satdrift_clk])
            if not (navdata["b_dot_sv_mps", sorted_idxs[sv_idx]] - \
                                            consts.C * satdrift_clk == 0.0):
                raise RuntimeError("b_dot_sv_mps of navdata not correctly updated")

            if verbose:
                print(' ')

    return navdata

def compute_sv_gps_from_brdcst_eph(navdata, verbose = False):
    """Populate navdata class comprising only GPS L1 values with
    satellite position and velocity (which is computed via .n files)

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class that depicts android derived dataset
        of only GPS L1 values
    verbose : bool (dafault: False)
        Flag (True/False) for whether to print intermediate steps useful
        for debugging/reviewing

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Modified instance of NavData class that computes satellite position,
        velocity, clock bias and drift using broadcast ephemerides
    """
    navdata_offset = 0 #1000 # Set this to zero, when android errors get fixed
    unique_gnss_id = np.unique(navdata['gnss_id'])
    if len(unique_gnss_id)==1:
        if unique_gnss_id == 'gps':
            # Need this string to create sv_id strings for ephemeris manager
            unique_gnss_id_str = 'G'
        else:
            raise RuntimeError("No non-GPS capability yet")
    else:
        raise RuntimeError("Multi-GNSS constellations cannot be updated simultaneously")

    repo = EphemerisManager()
    unique_timesteps = np.unique(navdata["gps_millis"])

    for t_idx, timestep in enumerate(unique_timesteps[:-1]):
        # Compute indices where gps_millis match, sort them
        # sorting is done for consistency across all satellite pos. estimation
        # algorithms as ephemerismanager inherently sorts based on prns
        idxs = np.where(navdata["gps_millis"] == timestep)[0]
        sorted_idxs = idxs[np.argsort(navdata["sv_id", idxs], axis = 0)]

        # compute ephem information using desired_sats, rxdatetime
        desired_sats = [unique_gnss_id_str + str(int(i)).zfill(2) \
                                           for i in navdata["sv_id", sorted_idxs]]
        rxdatetime = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc) + \
                     timedelta( seconds = (timestep - navdata_offset) * 1e-3 )
        ephem = repo.get_ephemeris(rxdatetime, satellites = desired_sats)

        if verbose:
            print(t_idx, timestep, idxs, sorted_idxs)
            print('misc: ', navdata['gps_millis', sorted_idxs], \
                            navdata['gnss_id', sorted_idxs], \
                            navdata['sv_id', sorted_idxs], \
                            navdata['signal_type', sorted_idxs], \
                            desired_sats, rxdatetime)

            satpos_android = np.transpose([ navdata["x_sv_m", sorted_idxs], \
                                            navdata["y_sv_m", sorted_idxs], \
                                            navdata["z_sv_m", sorted_idxs] ])
            satvel_android = np.transpose([ navdata["vx_sv_mps", sorted_idxs], \
                                               navdata["vy_sv_mps", sorted_idxs], \
                                               navdata["vz_sv_mps", sorted_idxs] ])
            print('android:', satpos_android, satvel_android)

        # compute satellite position and velocity based on ephem and gps_time
        # Transform satellite position to account for earth's rotation
        gps_week, gps_tow = gps_millis_to_tow(timestep - navdata_offset)
        get_sat_from_ephem = find_sat(ephem, gps_tow, gps_week)
        satpos_ephemeris = np.transpose([get_sat_from_ephem.x.values, \
                                         get_sat_from_ephem.y.values, \
                                         get_sat_from_ephem.z.values])
        satvel_ephemeris = np.transpose([get_sat_from_ephem.vx.values, \
                                         get_sat_from_ephem.vy.values, \
                                         get_sat_from_ephem.vz.values])
#         if verbose:
#             print('before ephemeris:', satpos_ephemeris, satvel_ephemeris)
        trans_time = navdata["raw_pr_m", sorted_idxs] / consts.C
        del_x = (consts.OMEGA_E_DOT * satpos_ephemeris[:,1] * trans_time)
        del_y = (-consts.OMEGA_E_DOT * satpos_ephemeris[:,0] * trans_time)
        satpos_ephemeris[:,0] = satpos_ephemeris[:,0] + del_x
        satpos_ephemeris[:,1] = satpos_ephemeris[:,1] + del_y

        if verbose:
#             print('after ephemeris:', satpos_ephemeris, satvel_ephemeris)
            print('nav-android Pos Error: ', \
                      np.linalg.norm(satpos_ephemeris - satpos_android, axis=1) )
            print('nav-android Vel Error: ', \
                      np.linalg.norm(satvel_ephemeris - satvel_android, axis=1) )

        # update x_sv_m of navdata with the estimated values from .n files
        navdata["x_sv_m", sorted_idxs] = satpos_ephemeris[:,0]
        if not max(abs(navdata["x_sv_m", sorted_idxs] - satpos_ephemeris[:,0])) == 0.0:
            raise RuntimeError("x_sv_m of navdata not correctly updated")

        # update y_sv_m of navdata with the estimated values from .n files
        navdata["y_sv_m", sorted_idxs] = satpos_ephemeris[:,1]
        if not max(abs(navdata["y_sv_m", sorted_idxs] - satpos_ephemeris[:,1])) == 0.0:
            raise RuntimeError("y_sv_m of navdata not correctly updated")

        # update z_sv_m of navdata with the estimated values from .n files
        navdata["z_sv_m", sorted_idxs] = satpos_ephemeris[:,2]
        if not max(abs(navdata["z_sv_m", sorted_idxs] - satpos_ephemeris[:,2])) == 0.0:
            raise RuntimeError("z_sv_m of navdata not correctly updated")

        # update vx_sv_mps of navdata with the estimated values from .n files
        navdata["vx_sv_mps", sorted_idxs] = satvel_ephemeris[:,0]
        if not max(abs(navdata["vx_sv_mps", sorted_idxs] - satvel_ephemeris[:,0])) == 0.0:
            raise RuntimeError("vx_sv_mps of navdata not correctly updated")

        # update vy_sv_mps of navdata with the estimated values from .n files
        navdata["vy_sv_mps", sorted_idxs] = satvel_ephemeris[:,1]
        if not max(abs(navdata["vy_sv_mps", sorted_idxs] - satvel_ephemeris[:,1])) == 0.0:
            raise RuntimeError("vy_sv_mps of navdata not correctly updated")

        # update vz_sv_mps of navdata with the estimated values from .n files
        navdata["vz_sv_mps", sorted_idxs] = satvel_ephemeris[:,2]
        if not max(abs(navdata["vz_sv_mps", sorted_idxs] - satvel_ephemeris[:,2])) == 0.0:
            raise RuntimeError("vz_sv_mps of navdata not correctly updated")

        if verbose:
            print(' ')

    return navdata
