"""Functions to process precise ephemerides .sp3 and .clk files.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "09 June 2022"

import os

from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import interpolate

from gnss_lib_py.parsers.ephemeris import EphemerisManager
from gnss_lib_py.utils.sv_models import find_sv_states, _combine_gnss_sv_ids
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis
import gnss_lib_py.utils.constants as consts

class Sp3:
    """Class handling satellite position data from precise ephemerides

    """
    def __init__(self):
        self.xpos = []
        self.ypos = []
        self.zpos = []
        self.tym = []
        self.utc_time = []

def parse_sp3(input_path):
    """sp3 specific loading and preprocessing for any GNSS constellation

    Parameters
    ----------
    input_path : string or path-like
        Path to sp3 file

    Returns
    -------
    sp3data : dict
        Populated gnss_lib_py.parsers.precise_ephemerides.Sp3 objects
        with key of `gnss_sv_id` for each satellite.

    Notes
    -----
    The format for .sp3 files can be viewed in [1]_.

    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2015

    References
    ----------
    .. [1]  https://files.igs.org/pub/data/format/sp3d.pdf
            Accessed as of August 20, 2022
    """
    # Initial checks for loading sp3_path
    if not isinstance(input_path, (str, os.PathLike)):
        raise TypeError("input_path must be string or path-like")
    if not os.path.exists(input_path):
        raise FileNotFoundError("file not found")

    # Load in the file
    with open(input_path, 'r', encoding="utf-8") as infile:
        data = [line.strip() for line in infile]

    # Create sp3 dictionary object
    sp3data = {}

    # Loop through each line
    for dval in data:
        if len(dval) == 0:
            # No data
            continue

        if dval[0] == '*':
            # A new record
            # Get the date
            temp = dval.split()
            curr_time = datetime( int(temp[1]), int(temp[2]), \
                                  int(temp[3]), int(temp[4]), \
                                  int(temp[5]),int(float(temp[6])), \
                                  tzinfo=timezone.utc )
            gps_millis = datetime_to_gps_millis(curr_time, add_leap_secs = False)

        if 'P' in dval[0]:
            # A satellite record.  Get the satellite number, and coordinate (X,Y,Z) info
            temp = dval.split()

            gnss_sv_id = temp[0][1:]
            if gnss_sv_id not in sp3data:
                sp3data[gnss_sv_id] = Sp3()

            sp3data[gnss_sv_id].utc_time.append(curr_time)
            sp3data[gnss_sv_id].tym.append(gps_millis)
            sp3data[gnss_sv_id].xpos.append(float(temp[1])*1e3)
            sp3data[gnss_sv_id].ypos.append(float(temp[2])*1e3)
            sp3data[gnss_sv_id].zpos.append(float(temp[3])*1e3)

    return sp3data

class Clk:
    """Class handling satellite clock bias data from precise ephemerides

    """
    def __init__(self):
        self.clk_bias = []
        self.utc_time = []
        self.tym = []

def parse_clockfile(input_path):
    """Clk specific loading and preprocessing for any GNSS constellation

    Parameters
    ----------
    input_path : string or path-like
        Path to clk file

    Returns
    -------
    clkdata : dict
        Populated gnss_lib_py.parsers.precise_ephemerides.Clk objects
        with key of `gnss_sv_id` for each satellite.

    Notes
    -----
    The format for .clk files can be viewed in [2]_.

    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2015

    References
    -----
    .. [2]  https://files.igs.org/pub/data/format/rinex_clock300.txt
            Accessed as of August 24, 2022
    """

    # Initial checks for loading sp3_path
    if not os.path.exists(input_path):
        raise FileNotFoundError("file not found")

    # Create a dictionary for complete clock data
    clkdata = {}

    # Read Clock file
    with open(input_path, 'r', encoding="utf-8") as infile:
        clk = infile.readlines()

    line = 0
    while True:
        if 'OF SOLN SATS' not in clk[line]:
            del clk[line]
        else:
            line +=1
            break

    line = 0
    while True:
        if 'END OF HEADER' not in clk[line]:
            line +=1
        else:
            del clk[0:line+1]
            break

    for  clk_val in clk:

        if len(clk_val) == 0 or clk_val[0:2]!='AS':
            continue

        timelist_val = clk_val.split()

        gnss_sv_id = timelist_val[1]
        if gnss_sv_id not in clkdata:
            clkdata[gnss_sv_id] = Clk()

        curr_time = datetime(year = int(timelist_val[2]), \
                             month = int(timelist_val[3]), \
                             day = int(timelist_val[4]), \
                             hour = int(timelist_val[5]), \
                             minute = int(timelist_val[6]), \
                             second = int(float(timelist_val[7])), \
                             tzinfo=timezone.utc)
        gps_millis = datetime_to_gps_millis(curr_time, add_leap_secs = False)
        clkdata[gnss_sv_id].utc_time.append(curr_time)
        clkdata[gnss_sv_id].tym.append(gps_millis)
        clkdata[gnss_sv_id].clk_bias.append(float(timelist_val[9]))

    return clkdata

def extract_sp3(sp3data, sidx, ipos = 10, \
                     method = 'CubicSpline', verbose = False):
    """Computing interpolated function over sp3 data for any GNSS

    Parameters
    ----------
    sp3data : gnss_lib_py.parsers.precise_ephemerides.Sp3
        Instance of GPS-only Sp3 class list with len == # sats
    sidx : int
        Nearest index within sp3 time series around which interpolated
        function needs to be centered
    ipos : int
        No. of data points from sp3 data on either side of sidx
        that will be used for computing interpolated function
    method : string
        Type of interpolation method used for sp3 data (the default is
        CubicSpline, which depicts third-order polynomial)

    Returns
    -------
    func_satpos : np.ndarray
        Instance with 3-D array of scipy.interpolate.interpolate.interp1d
        that is loaded with .sp3 data
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

def extract_clk(clkdata, sidx, ipos = 10, \
                     method='CubicSpline', verbose = False):
    """Computing interpolated function over clk data for any GNSS

    Parameters
    ----------
    clkdata : gnss_lib_py.parsers.precise_ephemerides.Sp3
        Instance of GPS-only Clk class list with len == # sats
    sidx : int
        Nearest index within clk time series around which interpolated
        function needs to be centered
    ipos : int
        No. of data points from clk data on either side of sidx
        that will be used for computing interpolated function
    method : string
        Type of interpolation method used for clk data (the default is
        CubicSpline, which depicts third-order polynomial)

    Returns
    -------
    func_satbias : np.ndarray
        Instance with 1-D array of scipy.interpolate.interpolate.interp1d
        that is loaded with .clk data
    """

    if method=='CubicSpline':
        low_i = (sidx - ipos) if (sidx - ipos) >= 0 else 0
        high_i = (sidx + ipos) if (sidx + ipos) <= len(clkdata.tym) else -1

        if verbose:
            print('Nearest clk: ', sidx, clkdata.tym[sidx], clkdata.clk_bias[sidx])

        func_satbias = interpolate.CubicSpline(clkdata.tym[low_i:high_i], \
                                               clkdata.clk_bias[low_i:high_i])

    return func_satbias

def sp3_snapshot(func_satpos, cxtime, hstep = 5e-1, method='CubicSpline'):
    """Compute satellite 3-D position and velocity from sp3 interpolated function

    Parameters
    ----------
    func_satpos : np.ndarray
        Instance with 3-D array of scipy.interpolate.interpolate.interp1d
        that is loaded with .sp3 data
    cxtime : float
        Time at which the satellite 3-D position and velocity needs to be
        computed, given 3-D array of interpolated functions
    hstep : float
        Step size in milliseconds used to computing 3-D velocity of any
        given satellite using central differencing the default is 5e-1)
    method : string
        Type of interpolation method used for sp3 data (the default is
        CubicSpline, which depicts third-order polynomial)

    Returns
    -------
    satpos_sp3 : 3-D array
        Computed satellite position in ECEF frame (Earth's rotation not included)
    satvel_sp3 : 3-D array
        Computed satellite velocity in ECEF frame (Earth's rotation not included)
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

def clk_snapshot(func_satbias, cxtime, hstep = 5e-1, method='CubicSpline'):
    """Compute satellite clock bias and drift from clk interpolated function

    Parameters
    ----------
    func_satbias : scipy.interpolate._cubic.CubicSpline
        Instance with interpolated function for satellite bias from .clk data
    cxtime : float
        Time at which satellite clock bias and drift is to be computed
    hstep : float
        Step size in milliseconds used to computing clock drift using
        central differencing (the default is 5e-1)
    method : string
        Type of interpolation method used for sp3 data (the default is
        CubicSpline, which depicts third-order polynomial)

    Returns
    -------
    satbias_clk : float
        Computed satellite clock bias (in seconds)
    satdrift_clk : float
        Computed satellite clock drift (in seconds/seconds)
    """

    if method=='CubicSpline':
        sat_t = func_satbias([cxtime-0.5*hstep, cxtime, cxtime+0.5*hstep])

    satbias_clk = sat_t[1]
    satdrift_clk = (sat_t[2]-sat_t[0]) / hstep

    return satbias_clk, (satdrift_clk * 1e3)

def single_gnss_from_precise_eph(navdata, sp3_parsed_file,
                                 clk_parsed_file, inplace=False,
                                 verbose = False):
    """Compute satellite information using .sp3 and .clk for any GNSS constellation

    Either adds or replaces satellite ECEF position data and clock bias
    for any satellite that exists in the provided sp3 file

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class that depicts android derived dataset
    sp3_parsed_file : list
        Instance with list of gnss_lib_py.parsers.precise_ephemerides.Sp3
    clk_parsed_file : list
        Instance with list of gnss_lib_py.parsers.precise_ephemerides.Clk
    inplace : bool
        If true, adds satellite positions and clock bias to the input
        navdata object, otherwise returns a new NavData object with the
        satellite rows added.
    verbose : bool
        Flag (True/False) for whether to print intermediate steps useful
        for debugging/reviewing (the default is False)

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Updated NavData class with satellite information computed using
        precise ephemerides from .sp3 and .clk files
    """

    # Initialize the sp3 and clk iref arrays
    sp3_iref_old = {}
    satfunc_xyz_old = {}
    clk_iref_old = {}
    satfunc_t_old = {}


    # combine gnss_id and sv_id into gnss_sv_ids
    if inplace:
        navdata["gnss_sv_id"] = _combine_gnss_sv_ids(navdata)
    else:
        new_navdata = navdata.copy()
        new_navdata["gnss_sv_id"] = _combine_gnss_sv_ids(new_navdata)

    # add satellite indexes if not present already.
    sv_idx_keys = ['x_sv_m', 'y_sv_m', 'z_sv_m', \
                'vx_sv_mps','vy_sv_mps','vz_sv_mps', \
                'b_sv_m', 'b_dot_sv_mps']
    for sv_idx_key in sv_idx_keys:
        if sv_idx_key not in navdata.rows:
            if inplace:
                navdata[sv_idx_key] = np.nan
            else:
                new_navdata[sv_idx_key] = np.nan


    if inplace:
        iterate_navdata = navdata
    else:
        iterate_navdata = new_navdata

    for row_idx, row in enumerate(iterate_navdata):
        gnss_sv_id = str(row["gnss_sv_id"])
        # continue if no sp3 or clk data availble
        if gnss_sv_id not in sp3_parsed_file \
          or gnss_sv_id not in clk_parsed_file \
          or len(sp3_parsed_file[gnss_sv_id].tym) == 0 \
          or len(clk_parsed_file[gnss_sv_id].tym) == 0: continue

        timestep = row["gps_millis"]

        # Perform nearest time step search to compute iref values for sp3 and clk
        sp3_iref = np.argmin(abs(np.array(sp3_parsed_file[gnss_sv_id].tym) - \
                                 timestep ))
        clk_iref = np.argmin(abs(np.array(clk_parsed_file[gnss_sv_id].tym) - \
                                 timestep ))

        # Carry out .sp3 processing by first checking if
        # previous interpolated function holds
        if gnss_sv_id in sp3_iref_old and sp3_iref == sp3_iref_old[gnss_sv_id]:
            func_satpos = satfunc_xyz_old[gnss_sv_id]
        else:
            # if does not hold, recompute the interpolation function based on current iref
            if verbose:
                print('SP3: Computing new interpolation for',gnss_sv_id)
            func_satpos = extract_sp3(sp3_parsed_file[gnss_sv_id], \
                                           sp3_iref)
            # Update the relevant interp function and iref values
            satfunc_xyz_old[gnss_sv_id] = func_satpos
            sp3_iref_old[gnss_sv_id] = sp3_iref

        # Compute satellite position and velocity using interpolated function
        satpos_sp3, satvel_sp3 = sp3_snapshot(func_satpos, timestep)

        # Adjust the satellite position based on Earth's rotation
        trans_time = row["raw_pr_m"] / consts.C
        del_x = (consts.OMEGA_E_DOT * satpos_sp3[1] * trans_time)
        del_y = (-consts.OMEGA_E_DOT * satpos_sp3[0] * trans_time)
        satpos_sp3[0] = satpos_sp3[0] + del_x
        satpos_sp3[1] = satpos_sp3[1] + del_y

        # Carry out .clk processing by first checking if previous interpolated
        # function holds
        if gnss_sv_id in clk_iref_old and clk_iref == clk_iref_old[gnss_sv_id]:
            func_satbias = satfunc_t_old[gnss_sv_id]
        else:
            # if does not hold, recompute the interpolation function based on current iref
            if verbose:
                print('CLK: Computing new interpolation for',gnss_sv_id)
            func_satbias = extract_clk(clk_parsed_file[gnss_sv_id],
                                            clk_iref)
            # Update the relevant interp function and iref values
            satfunc_t_old[gnss_sv_id] = func_satbias
            clk_iref_old[gnss_sv_id] = clk_iref

        # Compute satellite clock bias and drift using interpolated function
        satbias_clk, satdrift_clk = clk_snapshot(func_satbias, timestep)

        if inplace:
            # update *_sv_m of navdata with the estimated values from .sp3 files
            navdata['x_sv_m', row_idx] = np.array([satpos_sp3[0]])
            navdata['y_sv_m', row_idx] = np.array([satpos_sp3[1]])
            navdata['z_sv_m', row_idx] = np.array([satpos_sp3[2]])

            # update v*_sv_mps of navdata with the estimated values from .sp3 files
            navdata["vx_sv_mps", row_idx] = np.array([satvel_sp3[0]])
            navdata["vy_sv_mps", row_idx] = np.array([satvel_sp3[1]])
            navdata["vz_sv_mps", row_idx] = np.array([satvel_sp3[2]])

            # update clock data of navdata with the estimated values from .clk files
            navdata["b_sv_m", row_idx] = np.array([consts.C * satbias_clk])
            navdata["b_dot_sv_mps", row_idx] = np.array([consts.C * satdrift_clk])
        else:
            # update *_sv_m of navdata with the estimated values from .sp3 files
            new_navdata['x_sv_m', row_idx] = np.array([satpos_sp3[0]])
            new_navdata['y_sv_m', row_idx] = np.array([satpos_sp3[1]])
            new_navdata['z_sv_m', row_idx] = np.array([satpos_sp3[2]])

            # update v*_sv_mps of navdata with the estimated values from .sp3 files
            new_navdata["vx_sv_mps", row_idx] = np.array([satvel_sp3[0]])
            new_navdata["vy_sv_mps", row_idx] = np.array([satvel_sp3[1]])
            new_navdata["vz_sv_mps", row_idx] = np.array([satvel_sp3[2]])

            # update clock data of navdata with the estimated values from .clk files
            new_navdata["b_sv_m", row_idx] = np.array([consts.C * satbias_clk])
            new_navdata["b_dot_sv_mps", row_idx] = np.array([consts.C * satdrift_clk])

    if inplace:
        return None
    return new_navdata

def multi_gnss_from_precise_eph(navdata, sp3_path, clk_path,
                                inplace=False, verbose = False):
    """Compute satellite information using .sp3 and .clk for multiple GNSS

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class that depicts android derived dataset
    sp3_path : path
        File path for .sp3 file to extract precise ephemerides
    clk_path : path
        File path for .clk file to extract precise ephemerides
    inplace : bool
        If true, adds satellite positions and clock bias to the input
        navdata object, otherwise returns a new NavData object with the
        satellite rows added.
    verbose : bool
        Flag for whether to print intermediate steps useful
        for debugging/reviewing (the default is False)

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Updated NavData class with satellite information computed using
        precise ephemerides from .sp3 and .clk files
    """
    sp3_parsed_gnss = parse_sp3(sp3_path)
    clk_parsed_gnss = parse_clockfile(clk_path)
    precise_navdata = single_gnss_from_precise_eph(navdata,
                                                   sp3_parsed_gnss,
                                                   clk_parsed_gnss,
                                                   inplace = inplace,
                                                   verbose = verbose)

    return precise_navdata

def sv_gps_from_brdcst_eph(navdata, verbose = False):
    """Compute satellite information using .n for any GNSS constellation

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class that depicts android derived dataset
    verbose : bool
        Flag (True/False) for whether to print intermediate steps useful
        for debugging/reviewing (the default is False)

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Updated NavData class with satellite information computed using
        broadcast ephemerides from .n files
    """
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

    for _, timestep in enumerate(unique_timesteps):
        # Compute indices where gps_millis match, sort them
        # sorting is done for consistency across all satellite pos. estimation
        # algorithms as ephemerismanager inherently sorts based on prns
        idxs = np.where(navdata["gps_millis"] == timestep)[0]
        sorted_idxs = idxs[np.argsort(navdata["sv_id", idxs], axis = 0)]

        # compute ephem information using desired_sats, rxdatetime
        desired_sats = [unique_gnss_id_str + str(int(i)).zfill(2) \
                                           for i in navdata["sv_id", sorted_idxs]]
        rxdatetime = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc) + \
                     timedelta( seconds = (timestep) * 1e-3 )
        ephem = repo.get_ephemeris(rxdatetime, satellites = desired_sats)

        # compute satellite position and velocity based on ephem and gps_time
        # Transform satellite position to account for earth's rotation
        get_sat_from_ephem = find_sv_states(timestep, ephem)
        satpos_ephemeris = np.transpose([get_sat_from_ephem["x_sv_m"], \
                                         get_sat_from_ephem["y_sv_m"], \
                                         get_sat_from_ephem["z_sv_m"]])
        satvel_ephemeris = np.transpose([get_sat_from_ephem["vx_sv_mps"], \
                                         get_sat_from_ephem["vy_sv_mps"], \
                                         get_sat_from_ephem["vz_sv_mps"]])
        trans_time = navdata["raw_pr_m", sorted_idxs] / consts.C
        del_x = (consts.OMEGA_E_DOT * satpos_ephemeris[:,1] * trans_time)
        del_y = (-consts.OMEGA_E_DOT * satpos_ephemeris[:,0] * trans_time)
        satpos_ephemeris[:,0] = satpos_ephemeris[:,0] + del_x
        satpos_ephemeris[:,1] = satpos_ephemeris[:,1] + del_y

        if verbose:
            print('after ephemeris:', satpos_ephemeris, satvel_ephemeris)
            satpos_android = np.transpose([ navdata["x_sv_m", sorted_idxs], \
                                            navdata["y_sv_m", sorted_idxs], \
                                            navdata["z_sv_m", sorted_idxs] ])
            satvel_android = np.transpose([ navdata["vx_sv_mps", sorted_idxs], \
                                               navdata["vy_sv_mps", sorted_idxs], \
                                               navdata["vz_sv_mps", sorted_idxs] ])
            print('nav-android Pos Error: ', \
                      np.linalg.norm(satpos_ephemeris - satpos_android, axis=1) )
            print('nav-android Vel Error: ', \
                      np.linalg.norm(satvel_ephemeris - satvel_android, axis=1) )

        # update *_sv_m of navdata with the estimated values from .n files
        navdata["x_sv_m", sorted_idxs] = satpos_ephemeris[:,0]
        navdata["y_sv_m", sorted_idxs] = satpos_ephemeris[:,1]
        navdata["z_sv_m", sorted_idxs] = satpos_ephemeris[:,2]

        # update v*_sv_mps of navdata with the estimated values from .n files
        navdata["vx_sv_mps", sorted_idxs] = satvel_ephemeris[:,0]
        navdata["vy_sv_mps", sorted_idxs] = satvel_ephemeris[:,1]
        navdata["vz_sv_mps", sorted_idxs] = satvel_ephemeris[:,2]

    return navdata
