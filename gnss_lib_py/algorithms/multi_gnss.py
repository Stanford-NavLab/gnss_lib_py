"""Functions to process multi-GNSS related files.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "05 August 2022"

import os
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import interpolate

from gnss_lib_py.utils.sim_gnss import find_sat
from gnss_lib_py.parsers.ephemeris import EphemerisManager
from gnss_lib_py.parsers.precise_ephemerides import parse_sp3, parse_clockfile
import gnss_lib_py.utils.constants as consts

def extract_sp3_func(sp3data, sidx, ipos = 10, method = 'CubicSpline'):
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

    Notes
    -----
    (1) Need to add more interpolation functions

    References
    ----------
    """

    func_satpos = np.empty((3,), dtype=object)
    func_satpos[:] = np.nan

    if method=='CubicSpline':
        low_i = (sidx - ipos) if (sidx - ipos) >= 0 else 0
        high_i = (sidx + ipos) if (sidx + ipos) <= len(sp3data.tym) else -1

        print('Nearest sp3: ', sidx, sp3data.tym[sidx], \
                               sp3data.xpos[sidx], sp3data.ypos[sidx], sp3data.zpos[sidx])

        func_satpos[0] = interpolate.CubicSpline(sp3data.tym[low_i:high_i], \
                                                 sp3data.xpos[low_i:high_i])
        func_satpos[1] = interpolate.CubicSpline(sp3data.tym[low_i:high_i], \
                                                 sp3data.ypos[low_i:high_i])
        func_satpos[2] = interpolate.CubicSpline(sp3data.tym[low_i:high_i], \
                                                 sp3data.zpos[low_i:high_i])

    return func_satpos

def compute_sp3_snapshot(func_satpos, cxtime, hstep = 1e-5, method='CubicSpline'):
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

    Notes
    -----
    (1) Need to add more interpolation functions

    References
    ----------
    """
    if method=='CubicSpline':
        sat_x = func_satpos[0]([cxtime-0.5*hstep, cxtime, cxtime+0.5*hstep])
        sat_y = func_satpos[1]([cxtime-0.5*hstep, cxtime, cxtime+0.5*hstep])
        sat_z = func_satpos[2]([cxtime-0.5*hstep, cxtime, cxtime+0.5*hstep])

    satpos_sp3 = np.array([sat_x[1], sat_y[1], sat_z[1]])
    satvel_sp3 = np.array([ (sat_x[2]-sat_x[0]) / hstep, \
                               (sat_y[2]-sat_y[0]) / hstep, \
                               (sat_z[2]-sat_z[0]) / hstep ])

    return satpos_sp3, satvel_sp3

def extract_clk_func(clkdata, sidx, ipos = 10, method='CubicSpline'):
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

    Notes
    -----
    (1) Need to add more interpolation functions
    (2) Need to add check for detecting if func_satbias is valid for
    cxtime specified

    References
    ----------
    """

    if method=='CubicSpline':
        low_i = (sidx - ipos) if (sidx - ipos) >= 0 else 0
        high_i = (sidx + ipos) if (sidx + ipos) <= len(clkdata.tym) else -1

        print('Nearest clk: ', sidx, clkdata.tym[sidx], clkdata.clk_bias[sidx])

        func_satbias = interpolate.CubicSpline(clkdata.tym[low_i:high_i], \
                                            clkdata.clk_bias[low_i:high_i])

    return func_satbias

def compute_clk_snapshot(func_satbias, cxtime, hstep = 1e-5, method='CubicSpline'):
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

    Notes
    -----
    (1) Need to add more interpolation functions
    (2) Need to add check for detecting if func_satbias is valid for
    cxtime specified

    References
    ----------
    """

    if method=='CubicSpline':
        sat_t = func_satbias([cxtime-0.5*hstep, cxtime, cxtime+0.5*hstep])

    satbias_clk = sat_t[1]
    satdrift_clk = (sat_t[2]-sat_t[0]) / hstep

    return satbias_clk, satdrift_clk

def compute_sv_sp3clk_gps_glonass(navdata, sp3_path, clk_path, multi_gnss):
    """Populate navdata class with satellite position, velocity,
    clock bias and drift, which is computed via .sp3 and .clk files

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class that depicts android derived dataset
    sp3_path : string
        Location for the unit_test sp3 measurements
    clk_path : string
        Location for the unit_test clk measurements
    multi_gnss : dict (example: {'G': (1, 'GPS_L1')} )
        Dictionary with key specifying the string specifying constellation in
        sp3 (i.e., 'G': GPS, 'R': GLONASS, 'E': Galileo). The value for each
        key depicts two values: gnss_id and signal_type defined in navdata

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Modified instance of NavData class that computes satellite position,
        velocity, clock bias and drift using precise ephemerides

    Notes
    -----
    (1) so far the functionality for gps and glonass is hard coded, can be
    automated
    (2) forgot which website to use for downloading multi-gnss data. Maybe
    TU chemnitz dataset has other constellations
    (3) Need to check if this function works for non L1/G1 values of GPS
    and GLONASS constellation, i.e., can it handle duplicate satellites
    entries of the same constellation
    (4) Need to check if this function works for navdata class with no
    satellite position, velocity, bias and drift fields
    (5) Need to check if these functions work for non-android datasets
    (6) Update multi-gnss with other constellation keys
    (7) Not sure how to address this pylint warning:
    multi_gnss.py:414:0: R0914: Too many local variables (23/15) (too-many-locals)
    multi_gnss.py:414:0: R0912: Too many branches (13/12) (too-many-branches)
    multi_gnss.py:414:0: R0915: Too many statements (56/50) (too-many-statements)
    References
    ----------
    """

#     # Initial checks for loading sp3_path
#     if not isinstance(sp3_path, str):
#         raise TypeError("sp3_path must be string")
#     if not os.path.exists(sp3_path):
#         raise OSError("file not found")

#     # Initial checks for loading clk_path
#     if not isinstance(clk_path, str):
#         raise TypeError("clk_path must be string")
#     if not os.path.exists(clk_path):
#         raise OSError("file not found")

    if not np.array_equal( np.unique(navdata["gnss_id"]), \
                           [multi_gnss[gnss][0] for gnss in multi_gnss.keys()] ):
        raise RuntimeError("Inconsistency in multi-gnss inputs")

    navdata_offset = 1 # Not sure why this offset exists
    interp_method = 'CubicSpline' # Currently only one functionality

    # Parse the sp3 and clk files for relevant constellations
    sp3_parsed_file, clk_parsed_file = {}, {}

    for gnss in multi_gnss.keys():
        sp3_parsed_file[gnss] = parse_sp3(sp3_path, constellation = gnss)
        clk_parsed_file[gnss] = parse_clockfile(clk_path, constellation = gnss)

        if not len(sp3_parsed_file[gnss]) == len(clk_parsed_file[gnss]):
            raise RuntimeError("Lengths of sp3 and clk parsed files do not match")

    # Initialize the sp3 and clk iref arrays
    sp3_iref_old, clk_iref_old = {}, {}
    satfunc_xyz_old, satfunc_t_old = {}, {}

    for gnss in multi_gnss.keys():
        sp3_iref_old[gnss] = -1 * np.ones( (len(sp3_parsed_file[gnss]),))
        satfunc_xyz_old[gnss] = np.empty( (len(sp3_parsed_file[gnss]),), dtype=object)
        satfunc_xyz_old[gnss][:] = np.nan

        clk_iref_old[gnss] = -1 * np.ones( (len(clk_parsed_file[gnss]),))
        satfunc_t_old[gnss] = np.empty( (len(clk_parsed_file[gnss]),), dtype=object)
        satfunc_t_old[gnss][:] = np.nan

    unique_timesteps = np.unique(navdata["gps_tow"])

    for t_idx, timestep in enumerate(unique_timesteps[:-1]):

        for gnss in multi_gnss.keys():

            idxs = np.where( (navdata["gps_tow",:] == timestep) & \
                             (navdata["gnss_id",:] == multi_gnss[gnss][0]) & \
                             (navdata["signal_type",:] == multi_gnss[gnss][1]) )[1]

            sorted_idxs = idxs[np.argsort(navdata["sv_id", idxs], axis = 0)]

            for sv_idx, prn in enumerate(navdata["sv_id", sorted_idxs]):

                prn = int(prn)

                # Perform nearest time step search to compute iref values for sp3 and clk
                sp3_iref = np.argmin(abs(np.array(sp3_parsed_file[gnss][prn].tym) - \
                                         (timestep-navdata_offset)))
                clk_iref = np.argmin(abs(np.array(clk_parsed_file[gnss][prn].tym) - \
                                         (timestep-navdata_offset)))

                print('Stats: ', t_idx, timestep, gnss, prn, idxs, sorted_idxs)
                print('sp3 stats: ', sp3_iref, sp3_iref_old[gnss][prn])
                print('clk stats: ', clk_iref, clk_iref_old[gnss][prn])

                # Carry out .sp3 processing by first checking if
                # previous interpolated function holds
                if sp3_iref == sp3_iref_old[gnss][prn]:
                    func_satpos = satfunc_xyz_old[gnss][prn]
                else:
                    # if does not hold, recompute the interpolation function based on current iref
                    print('SP3: Computing new interpolation!')
                    func_satpos = extract_sp3_func(sp3_parsed_file[gnss][prn], \
                                                   sp3_iref, method = interp_method)
                    # Update the relevant interp function and iref values
                    satfunc_xyz_old[gnss][prn] = func_satpos
                    sp3_iref_old[gnss][prn] = sp3_iref

                # Compute satellite position and velocity using interpolated function
                satpos_sp3, satvel_sp3 = compute_sp3_snapshot(func_satpos, \
                                                              (timestep-navdata_offset), \
                                                              method = interp_method)
                print('before sp3:', satpos_sp3, satvel_sp3)

                # Adjust the satellite position based on Earth's rotation
                trans_time = navdata["raw_pr_m", sorted_idxs[sv_idx]] / consts.C
                del_x = (consts.OMEGA_E_DOT * satpos_sp3[1] * trans_time)
                del_y = (-consts.OMEGA_E_DOT * satpos_sp3[0] * trans_time)
                satpos_sp3[0] = satpos_sp3[0] + del_x
                satpos_sp3[1] = satpos_sp3[1] + del_y

                # Carry out .clk processing by first checking if previous interpolated
                # function holds
                if clk_iref == clk_iref_old[gnss][prn]:
                    func_satbias = satfunc_t_old[gnss][prn]
                else:
                    # if does not hold, recompute the interpolation function based on current iref
                    print('CLK: Computing new interpolation!')
                    func_satbias = extract_clk_func(clk_parsed_file[gnss][prn], \
                                                 clk_iref, method = interp_method)
                    # Update the relevant interp function and iref values
                    satfunc_t_old[gnss][prn] = func_satbias
                    clk_iref_old[gnss][prn] = clk_iref

                # Compute satellite clock bias and drift using interpolated function
                satbias_clk, \
                satdrift_clk = compute_clk_snapshot(func_satbias, \
                                                    (timestep-navdata_offset), \
                                                    method = interp_method)
                print('after sp3:', satpos_sp3, \
                                    satvel_sp3, \
                                    consts.C * satbias_clk, \
                                    consts.C * satdrift_clk)

                print('Before:' )
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
                print('Android-sp3 Vel Error (m): ', \
                          np.linalg.norm(satvel_android[sv_idx] - satvel_sp3), \
                          navdata["vx_sv_mps", sorted_idxs[sv_idx]] - satvel_sp3[0], \
                          navdata["vy_sv_mps", sorted_idxs[sv_idx]] - satvel_sp3[1], \
                          navdata["vz_sv_mps", sorted_idxs[sv_idx]] - satvel_sp3[2] )
                print('Android-sp3 Clk Error (m): ', \
                          navdata["b_sv_m", sorted_idxs[sv_idx]] - consts.C * satbias_clk, \
                          navdata["b_dot_sv_mps", sorted_idxs[sv_idx]] - consts.C * satdrift_clk)
                print(' ')

                # Updating the relevant satellite fields in navdata class
                # with computed values using .sp3 and .clk files
                navdata['x_sv_m', sorted_idxs[sv_idx]] = np.array([satpos_sp3[0]])
                navdata['y_sv_m', sorted_idxs[sv_idx]] = np.array([satpos_sp3[1]])
                navdata['z_sv_m', sorted_idxs[sv_idx]] = np.array([satpos_sp3[2]])

                navdata["vx_sv_mps", sorted_idxs[sv_idx]] = np.array([satvel_sp3[0]])
                navdata["vy_sv_mps", sorted_idxs[sv_idx]] = np.array([satvel_sp3[1]])
                navdata["vz_sv_mps", sorted_idxs[sv_idx]] = np.array([satvel_sp3[2]])

                navdata["b_sv_m", sorted_idxs[sv_idx]] = np.array([consts.C * satbias_clk])
                navdata["b_dot_sv_mps", sorted_idxs[sv_idx]] = np.array([consts.C * satdrift_clk])

                if not (navdata["x_sv_m", sorted_idxs[sv_idx]][0] - satpos_sp3[0] == 0.0):
                    raise RuntimeError("x_sv_m of navdata not correctly updated")
                if not (navdata["y_sv_m", sorted_idxs[sv_idx]][0] - satpos_sp3[1] == 0.0):
                    raise RuntimeError("y_sv_m of navdata not correctly updated")
                if not (navdata["z_sv_m", sorted_idxs[sv_idx]][0] - satpos_sp3[2] == 0.0):
                    raise RuntimeError("z_sv_m of navdata not correctly updated")

                if not (navdata["vx_sv_mps", sorted_idxs[sv_idx]][0] - satvel_sp3[0] == 0.0):
                    raise RuntimeError("vx_sv_mps of navdata not correctly updated")
                if not (navdata["vy_sv_mps", sorted_idxs[sv_idx]][0] - satvel_sp3[1] == 0.0):
                    raise RuntimeError("vy_sv_mps of navdata not correctly updated")
                if not (navdata["vz_sv_mps", sorted_idxs[sv_idx]][0] - satvel_sp3[2] == 0.0):
                    raise RuntimeError("vz_sv_mps of navdata not correctly updated")

                if not (navdata["b_sv_m", sorted_idxs[sv_idx]][0] - consts.C * satbias_clk == 0.0):
                    raise RuntimeError("b_sv_m of navdata not correctly updated")
                if not (navdata["b_dot_sv_mps", sorted_idxs[sv_idx]][0] - \
                                                consts.C * satdrift_clk == 0.0):
                    raise RuntimeError("b_dot_sv_mps of navdata not correctly updated")

    return navdata

def compute_sv_eph_gps(navdata, multi_gnss):
    """Populate navdata class comprising only GPS L1 values with
    satellite position and velocity (which is computed via .n files)

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class that depicts android derived dataset
        of only GPS L1 values
    multi_gnss : tuple (example: (1, 'GPS_L1') )
        Tuple specifying gnss_id and signal_type within navdata

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Modified instance of NavData class that computes satellite position,
        velocity, clock bias and drift using broadcast ephemerides

    Notes
    -----
    (1) Considers only GPS as input, not applicable for other
    constellations
    (2) Hard coded GPS id as 1.0. Does it needs to be changed?
    (3) Confirm with ashwin about offset = 1.0 and its reasoning
    (4) Might be useful to have additional functionalities for
    navdata: sort and print functionality based on time
    (5) How to handle last outlier, when repeated satellites -> code
    breaks, i.e., ephemeris manager cannot handle duplicate satellites
    (6) Not sure how to address this pylint warning:
    multi_gnss.py:414:0: R0914: Too many local variables (23/15) (too-many-locals)
    multi_gnss.py:414:0: R0912: Too many branches (13/12) (too-many-branches)
    multi_gnss.py:414:0: R0915: Too many statements (56/50) (too-many-statements)

    References
    ----------
    """

    repo = EphemerisManager()
    navdata_offset = 1 # Not sure why this offset exists

    if all(x == navdata['gnss_id',0][0] for x in navdata['gnss_id'][0]):
        unique_gnss_id = navdata['gnss_id', 0][0]
    else:
        raise RuntimeError("No multi-GNSS capability yet")

    if unique_gnss_id == multi_gnss[0]:
        unique_gnss_id_str = 'G'
    else:
        raise RuntimeError("No non-GPS capability yet")

    if all(x == navdata['gps_week',0][0] for x in navdata['gps_week'][0]):
        unique_gps_week = navdata['gps_week', 0][0]
    else:
        raise RuntimeError("Cannot handle GPS week transition")

    unique_timesteps = np.unique(navdata["gps_tow"])

    for t_idx, timestep in enumerate(unique_timesteps[:-1]):
        idxs = np.where( (navdata["gps_tow"] == timestep) & \
                         (navdata["signal_type",:] == multi_gnss[1]) )[1]
        sorted_idxs = idxs[np.argsort(navdata["sv_id", idxs], axis = 0)]
        print(t_idx, timestep, idxs, sorted_idxs)
#         print('misc: ', navdata['gps_tow', sorted_idxs], \
#               navdata['millisSinceGpsEpoch', sorted_idxs], \
#               navdata['gps_week', sorted_idxs], \
#               navdata['gnss_id', sorted_idxs], \
#               navdata['sv_id', sorted_idxs],
#               navdata['signal_type', sorted_idxs] )

        satpos_android = np.transpose([ navdata["x_sv_m", sorted_idxs], \
                                        navdata["y_sv_m", sorted_idxs], \
                                        navdata["z_sv_m", sorted_idxs] ])
        satvel_android = np.transpose([ navdata["vx_sv_mps", sorted_idxs], \
                                           navdata["vy_sv_mps", sorted_idxs], \
                                           navdata["vz_sv_mps", sorted_idxs] ])
#         print('android:', satpos_android, satvel_android)

        desired_sats = [unique_gnss_id_str + str(int(i)).zfill(2) \
                                           for i in navdata["sv_id", sorted_idxs]]
        print(desired_sats)

        rxdatetime = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc) + \
                     timedelta( seconds = (unique_gps_week * 7 * 86400 + \
                                          (timestep-navdata_offset) ) )
#         print(rxdatetime, unique_gps_week)
        ephem = repo.get_ephemeris(rxdatetime, satellites = desired_sats)
        get_sat_from_ephem = find_sat(ephem, (timestep-navdata_offset), unique_gps_week)
        satpos_ephemeris = np.transpose([get_sat_from_ephem.x.values, \
                                         get_sat_from_ephem.y.values, \
                                         get_sat_from_ephem.z.values])
        satvel_ephemeris = np.transpose([get_sat_from_ephem.vx.values, \
                                            get_sat_from_ephem.vy.values, \
                                            get_sat_from_ephem.vz.values])
        print('before ephemeris:', satpos_ephemeris, satvel_ephemeris)
        trans_time = navdata["raw_pr_m", sorted_idxs] / consts.C
        del_x = (consts.OMEGA_E_DOT * satpos_ephemeris[:,1] * trans_time)
        del_y = (-consts.OMEGA_E_DOT * satpos_ephemeris[:,0] * trans_time)
        satpos_ephemeris[:,0] = satpos_ephemeris[:,0] + del_x
        satpos_ephemeris[:,1] = satpos_ephemeris[:,1] + del_y
        print('after ephemeris:', satpos_ephemeris, satvel_ephemeris)

        print('nav-android Pos Error: ', \
                  np.linalg.norm(satpos_ephemeris - satpos_android, axis=1) )
        print('nav-android Vel Error: ', \
                  np.linalg.norm(satvel_ephemeris - satvel_android, axis=1) )

        navdata["x_sv_m", sorted_idxs] = satpos_ephemeris[:,0] #get_sat_from_ephem.x.values
        navdata["y_sv_m", sorted_idxs] = satpos_ephemeris[:,1] #get_sat_from_ephem.y.values
        navdata["z_sv_m", sorted_idxs] = satpos_ephemeris[:,2] #get_sat_from_ephem.z.values
        navdata["vx_sv_mps", sorted_idxs] = satvel_ephemeris[:,0] #get_sat_from_ephem.vx.values
        navdata["vy_sv_mps", sorted_idxs] = satvel_ephemeris[:,1] #get_sat_from_ephem.vy.values
        navdata["vz_sv_mps", sorted_idxs] = satvel_ephemeris[:,2] #get_sat_from_ephem.vz.values

        if not max(abs(navdata["x_sv_m", sorted_idxs] - satpos_ephemeris[:,0])) == 0.0:
            raise RuntimeError("x_sv_m of navdata not correctly updated")
        if not max(abs(navdata["y_sv_m", sorted_idxs] - satpos_ephemeris[:,1])) == 0.0:
            raise RuntimeError("y_sv_m of navdata not correctly updated")
        if not max(abs(navdata["z_sv_m", sorted_idxs] - satpos_ephemeris[:,2])) == 0.0:
            raise RuntimeError("z_sv_m of navdata not correctly updated")

        if not max(abs(navdata["vx_sv_mps", sorted_idxs] - satvel_ephemeris[:,0])) == 0.0:
            raise RuntimeError("vx_sv_mps of navdata not correctly updated")
        if not max(abs(navdata["vy_sv_mps", sorted_idxs] - satvel_ephemeris[:,1])) == 0.0:
            raise RuntimeError("vy_sv_mps of navdata not correctly updated")
        if not max(abs(navdata["vz_sv_mps", sorted_idxs] - satvel_ephemeris[:,2])) == 0.0:
            raise RuntimeError("vz_sv_mps of navdata not correctly updated")

        print(' ')

    return navdata
