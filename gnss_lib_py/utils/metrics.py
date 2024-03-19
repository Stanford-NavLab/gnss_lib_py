"""Metrics to quantify quality of state estimates or GNSS measurements.
"""

__authors__ = "Ashwin Kanhere, Daniel Neamati"
__date__ = "24 January, 2024"


import numpy as np
from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import (find_wildcard_indexes,
                                            loop_time,
                                            concat)
from gnss_lib_py.utils.coordinates import LocalCoord


def accuracy_statistics(state_estimate, ground_truth, est_type="pos",
                        statistic="mean", direction=None, percentile=None,
                        ecef_origin=None):
    """Calculate required statistic for accuracy of state estimate.

    Parameters
    ----------
    state_estimate : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing state estimates.
    ground_truth : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing ground truth corresponding to state
        estimate. Must have same time stamps as state_estimate.
    est_type : str, optional
        Type of state estimate, by default "pos". Can be "pos", "vel",
        "time" or "acc".
    statistic : str, optional
        Statistic to calculate, by default "mean". Can be "mean",
        "median", "max_min", "percentile", "quantiles", "mean_absolute" or "max_absolute".
    direction : str, optional
        Direction of the statistic. If None, is calculated in the ECEF
        frame of reference. Other options can be "ned", "enu", "3d_norm",
        "horizontal", or "along_cross_track".
    percentile : float, optional
        Percentile to calculate, if calculating percentiles. Default is
        None, in which case 95th percentile will be calculated
    """
    if est_type == "pos":
        row_wildcards = ["x_rx*_m", "y_rx*_m", "z_rx*_m"]
    elif est_type == "vel":
        row_wildcards = ["vx_rx*_mps", "vy_rx*_mps", "vz_rx*_mps"]
    elif est_type == "time":
        row_wildcards = ["b_rx*_m", "b_dot_rx*_mps"]
    elif est_type == "acc":
        row_wildcards = ["ax_rx*_mps2", "ay_rx*_mps2", "az_rx*_mps2"]
    else:
        raise ValueError(f"Invalid est_type: {est_type}")

    # Extract subsets of the state_estimate and ground_truth NavData
    se_row_dict = find_wildcard_indexes(state_estimate, row_wildcards, max_allow=1)
    gt_row_dict = find_wildcard_indexes(ground_truth, row_wildcards, max_allow=1)
    se_rows = ['gps_millis'] + [row_name[0] for row_name in se_row_dict.values()]
    gt_rows = ['gps_millis'] + [row_name[0] for row_name in gt_row_dict.values()]
    state_estimate_subset = state_estimate.copy(rows=se_rows)
    ground_truth_subset = ground_truth.copy(rows=gt_rows)

    # Calculate the error values for the given rows
    if direction is None:
        error_values = _get_vec_err(state_estimate_subset,
                                    ground_truth_subset,
                                    se_row_dict, gt_row_dict)
        error_row_dict = find_wildcard_indexes(error_values, row_wildcards)
    elif direction == "ned":
        assert est_type == "pos" or est_type == "vel" or est_type == "acc",\
            "NED errors only for position, velocities, and acc"
        error_values, error_row_dict = _get_ned_err(state_estimate_subset,
                                                    ground_truth_subset,
                                                    se_row_dict,
                                                    gt_row_dict,
                                                    ecef_origin,
                                                    est_type)
    elif direction == "enu":
        assert est_type == "pos" or est_type == "vel" or est_type == "acc",\
            "ENU errors only for position, velocities, and acc"
        error_values, error_row_dict = _get_enu_err(state_estimate_subset,
                                                    ground_truth_subset,
                                                    se_row_dict,
                                                    gt_row_dict,
                                                    ecef_origin,
                                                    est_type)
    elif direction == "3d_norm":
        assert est_type == "pos" or est_type == "vel" or est_type == "acc",\
            "3D estimate errors only for position, velocities, and acc"
        error_values = _get_single_err_sample(state_estimate_subset,
                                    ground_truth_subset,
                                    se_row_dict, gt_row_dict, err_type="3d")
        error_row_dict = {'pos_rx*_3d_m': 'pos_rx_err_3d_m'}
    elif direction == "horizontal":
        assert est_type == "pos" or est_type == "vel" or est_type == "acc",\
            "Horizontal estimate errors only for position, velocities, and acc"
        error_values = _get_horiz_err(state_estimate_subset,
                                      ground_truth_subset, est_type)
        error_row_dict = {'pos_rx*_horiz_m': 'pos_rx_err_horiz_m'}
    elif direction == "along_cross_track":
        assert est_type == "pos", \
            "Along and cross track errors only implemented for position estimates"
        raise NotImplementedError("Along/cross track errors not implemented")
    else:
        raise ValueError("Input direction of error not implemented")


    stat = {}
    # Calculate vector versions of the statistics
    for wc, error_row in error_row_dict.items():
        if statistic == "mean":
            stat_row_name = _new_row_name(wc, 'mean')
            stat[stat_row_name] = np.mean(error_values[error_row])
            stat_row_name = _new_row_name(wc, 'cov')
            stat[stat_row_name] = np.cov(error_values[error_row])
        elif statistic == "median":
            stat_row_name = _new_row_name(wc, 'median')
            stat[stat_row_name] = np.median(error_values[error_row])
        elif statistic == "max_min":
            stat_row_name = _new_row_name(wc, 'max')
            stat[stat_row_name] = np.max(error_values[error_row])
            stat_row_name = _new_row_name(wc, 'min')
            stat[stat_row_name] = np.min(error_values[error_row])
        elif statistic == "percentile":
            if percentile is None:
                percentile = 95
            stat_row_name = _new_row_name(wc, 'percentile_' + str(percentile))
            stat[stat_row_name] = np.percentile(error_values[error_row], percentile)
        elif statistic == "quantiles":
            est_quantiles = np.quantile(error_values[error_row], [0.25, 0.5, 0.75])
            for q_idx, quantile in enumerate(est_quantiles):
                stat_row_name = _new_row_name(wc, f"q{q_idx+1:01}")
                stat[stat_row_name] = quantile
        elif statistic == "mean_absolute":
            stat_row_name = _new_row_name(wc, 'mean_absolute')
            stat[stat_row_name] = np.mean(np.abs(error_values[error_row]))
        elif statistic == "max_absolute":
            stat_row_name = _new_row_name(wc, 'max_absolute')
            stat[stat_row_name] = np.max(np.abs(error_values[error_row]))
        else:
            raise ValueError(f"Input statistic {statistic} not implemented")

    return stat


def _new_row_name(wc, name):
    wc_prefix = wc.split('*')[0]
    wc_postfix = wc.split('*')[1]
    new_row_name = wc_prefix + '_' + name + wc_postfix
    return new_row_name



def _get_vec_err(state_estimate, ground_truth, se_row_dict=None,
                    gt_row_dict = None):
    """Get error samples between state estimate and ground truth.

    If the lengths of both is the same, this method assumes an ordered
    one-to-one correspondance.

    If the lenghts are not the same, this methods uses the times from
    the state estimate to find corresponding ground truth estimates and
    uses those to compute the errors.

    """
    error_values = NavData()
    if len(state_estimate) == len(ground_truth):
        # assume one-to-one correspondance
        error_values['gps_millis'] = state_estimate['gps_millis']
        for wc in se_row_dict.keys():
            err_row_name = _new_row_name(wc, 'err')
            error_values[err_row_name] = state_estimate[se_row_dict[wc]] \
                                        - ground_truth[gt_row_dict[wc]]
    else:
        # Correspondance needs to be found using time
        for milli, _, se_frame in loop_time(state_estimate, "gps_millis"):
            temp_err_frame = NavData()
            temp_err_frame['gps_millis'] = milli
            # find time index for ground_truth NavData instance
            gt_t_idx = np.argmin(np.abs(ground_truth["gps_millis"] - milli))
            # Compute error samples
            for wc in se_row_dict.keys():
                err_row_name = _new_row_name(wc, 'err')
                temp_err_frame[err_row_name] = se_frame[se_row_dict[wc]] \
                                            - ground_truth[gt_row_dict[wc], gt_t_idx]
            if len(error_values) == 0:
                error_values = temp_err_frame
            else:
                error_values = concat(error_values, temp_err_frame)
    return error_values


def _get_single_err_sample(state_estimate, ground_truth, se_row_dict,
                           gt_row_dict, err_type, est_type="pos"):
    error_values = NavData()
    se_rows = [row_name[0] for row_name in se_row_dict.values()]
    gt_rows = [row_name[0] for row_name in gt_row_dict.values()]

    if est_type =="pos":
        row_pf = err_type+"_m"
    elif est_type == "vel":
        row_pf = err_type+"_mps"
    elif est_type == "acc":
        row_pf = err_type+"_mps2"

    if len(state_estimate) == len(ground_truth):
        # assume one-to-one correspondance
        error_values['gps_millis'] = state_estimate['gps_millis']
        state_estimate_xyz = state_estimate[se_rows]
        ground_truth_xyz = ground_truth[gt_rows]
        error_values['pos_rx_err_'+row_pf] = np.linalg.norm(state_estimate_xyz
                                                    - ground_truth_xyz,
                                                    axis=0)
    else:
        # Correspondance needs to be found using time
        for milli, _, se_frame in loop_time(state_estimate, "gps_millis"):
            temp_err_frame = NavData()
            temp_err_frame['gps_millis'] = milli
            # find time index for ground_truth NavData instance
            gt_t_idx = np.argmin(np.abs(ground_truth["gps_millis"] - milli))
            # Compute error samples
            temp_err_frame['pos_rx_err_'+row_pf] = np.linalg.norm(
                                                    se_frame[se_rows] \
                                                    - ground_truth[gt_rows,
                                                                gt_t_idx])
            if len(error_values) == 0:
                error_values = temp_err_frame
            else:
                error_values = concat(error_values, temp_err_frame)
    return error_values


def _get_horiz_err(state_estimate, ground_truth, est_type = "pos"):
    if est_type == "pos":
        row_wildcards = ["x_rx*_m", "y_rx*_m"]
    elif est_type == "vel":
        row_wildcards = ["vx_rx*_mps", "vy_rx*_mps"]
    elif est_type == "acc":
        row_wildcards = ["ax_rx*_mps2", "ay_rx*_mps2"]
    else:
        raise ValueError(f"Invalid est_type: {est_type}")

    se_row_dict = find_wildcard_indexes(state_estimate, row_wildcards, max_allow=1)
    gt_row_dict = find_wildcard_indexes(ground_truth, row_wildcards, max_allow=1)
    error_values = _get_single_err_sample(state_estimate, ground_truth,
                                          se_row_dict, gt_row_dict,
                                          err_type="horiz")
    return error_values


def _get_ned_err(state_estimate, ground_truth, se_row_dict,
                 gt_row_dict, ecef_origin = None, est_type = "pos"):


    se_rows = [row_name[0] for row_name in se_row_dict.values()]
    gt_rows = [row_name[0] for row_name in gt_row_dict.values()]

    if ecef_origin is None:
        assert est_type == "pos", \
            "Need origin of NED frame of reference for vel and acc"
        ecef_origin = ground_truth[gt_rows, 0]
        ned_frame = LocalCoord.from_ecef(ecef_origin)
    else:
        ned_frame = LocalCoord.from_ecef(ecef_origin)
    if est_type == "pos":
        state_estimate_xyz = state_estimate[se_rows]
        gt_xyz = ground_truth[gt_rows]

        state_estimate_ned = ned_frame.ecef_to_ned(state_estimate_xyz)
        gt_ned = ned_frame.ecef_to_ned(gt_xyz)
        error_row_dict = {'n_rx*_m': ['n_rx_err_m'],
                          'e_rx*_m': ['e_rx_err_m'],
                          'd_rx*_m': ['d_rx_err_m']}
    else:
        state_estimate_xyz = state_estimate[se_rows]
        gt_xyz = ground_truth[gt_rows]

        state_estimate_ned = ned_frame.ecef_to_nedv(state_estimate_xyz)
        gt_ned = ned_frame.ecef_to_nedv(gt_xyz)
        if est_type == "vel":
            error_row_dict = {'vn_rx*_mps': ['vn_rx_err_mps'],
                              've_rx*_mps': ['ve_rx_err_mps'],
                              'vd_rx*_mps': ['vd_rx_err_mps']}
        if est_type == "acc":
            error_row_dict = {'an_rx*_mps2': ['an_rx_err_mps2'],
                              'ae_rx*_mps2': ['ae_rx_err_mps2'],
                              'ad_rx*_mps2': ['ad_rx_err_mps2']}

    error_values= NavData()
    error_values['gps_millis'] = state_estimate['gps_millis']
    if len(state_estimate) == len(ground_truth):
        # Assume one-to-one correspondance between state estimate and
        # ground truth
        for row_num, error_row in enumerate(error_row_dict.values()):
            error_values[error_row[0]] = state_estimate_ned[row_num, :] \
                                        - gt_ned[row_num, :]
    else:
        # Correspondance needs to be found using time
        for milli, _, _ in loop_time(state_estimate, "gps_millis"):
            temp_err_frame = NavData()
            temp_err_frame['gps_millis'] = milli
            # find time index for ground_truth NavData instance
            gt_t_idx = np.argmin(np.abs(ground_truth["gps_millis"] - milli))
            # Compute error samples
            for row_num, error_row in enumerate(error_row_dict.values()):
                temp_err_frame[error_row[0]] = state_estimate_ned[row_num, :] \
                                            - gt_ned[row_num, gt_t_idx]
            if len(error_values) == 0:
                error_values = temp_err_frame
            else:
                error_values = concat(error_values, temp_err_frame)
    return error_values, error_row_dict


def _get_enu_err(state_estimate, ground_truth, se_row_dict,
                 gt_row_dict, ecef_origin = None, est_type = "pos"):
    if est_type == "pos":
        enu_err_row_dict = {'e_rx*_m': ['e_rx_err_m'],
                          'n_rx*_m': ['n_rx_err_m'],
                          'u_rx*_m': ['u_rx_err_m']}
    if est_type == "vel":
        enu_err_row_dict = {'ve_rx*_mps': ['ve_rx_err_mps'],
                          'vn_rx*_mps': ['vn_rx_err_mps'],
                          'vu_rx*_mps': ['vu_rx_err_mps']}
    if est_type == "acc":
        enu_err_row_dict = {'ae_rx*_mps2': ['ae_rx_err_mps2'],
                          'an_rx*_mps2': ['an_rx_err_mps2'],
                          'au_rx*_mps2': ['au_rx_err_mps2']}

    ned_errors, ned_error_dict = _get_ned_err(state_estimate, ground_truth,
                                              se_row_dict,
                                              gt_row_dict, ecef_origin,
                                              est_type)
    ned_to_enu_rot_mat = np.array([[0, 1, 0],
                                   [1, 0, 0],
                                   [0, 0, -1]])
    enu_errors = NavData()
    enu_errors['gps_millis'] = ned_errors['gps_millis']
    ned_error_rows = [row_name[0] for row_name in ned_error_dict.values()]
    ned_err_array = ned_errors[ned_error_rows]
    enu_err_array = np.matmul(ned_to_enu_rot_mat, ned_err_array)
    for row_num, row in enumerate(enu_err_row_dict.values()):
        enu_errors[row[0]] = enu_err_array[row_num, :]

    return enu_errors, enu_err_row_dict
