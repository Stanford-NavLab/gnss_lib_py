"""Fault Detection and Exclusion (FDE) methods for GNSS applications.

"""

__authors__ = "D. Knowles"
__date__ = "11 Jul 2023"

import time
import numpy as np

from gnss_lib_py.navdata.operations import loop_time
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.algorithms.residuals import solve_residuals


def solve_fde(navdata, method="residual", remove_outliers=False,
              max_faults=None, threshold=None, verbose=False,
              **kwargs):
    """Detects and optionally removes GNSS measurement faults.

    Individual fault detection and exclusion (fde) methods are
    documented in their individual functions.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        NavData of GNSS measurements which must include the satellite
        positions: ``x_sv_m``, ``y_sv_m``, ``z_sv_m``, ``b_sv_m`` as
        well as the time row ``gps_millis`` and the corrected
        pseudorange ``corr_pr_m``.
    method : string
        Method for fault detection and exclusion either "residual" for
        residual-based or "edm" for Euclidean Distance Matrix-based.
    remove_outliers : bool
        If `True`, removes measurements with detected faults (fault status 1)
        and measurements with unknown fault status (fault status 2)
        from the returned NavData instance.
        If false, will detect but not exclude faults or unknowns.
    max_faults : int
        Maximum number of faults to detect and/or exclude.
    threshold : float
        Detection threshold.
    verbose : bool
        If true, prints extra debugging statements.

    Returns
    -------
    navdata : gnss_lib_py.navdata.navdata.NavData
        Result includes a new row of ``fault_<method>`` where a
        value of 1 indicates a detected fault, 0 indicates that
        no fault was detected, and 2 indicates an unknown fault status
        which is usually due to lack of necessary columns or information.

    """

    if method == "residual":
        navdata = fde_greedy_residual(navdata, max_faults=max_faults,
                                        threshold=threshold,verbose=verbose,
                                        **kwargs)
    elif method == "edm":
        navdata = fde_edm(navdata, max_faults=max_faults,
                                   threshold=threshold,verbose=verbose,
                                   **kwargs)
    else:
        raise ValueError("invalid method input for solve_fde()")

    if remove_outliers:
        navdata = navdata.where("fault_" + method, 0)

    return navdata


def fde_edm(navdata, max_faults=None, threshold=1.0, time_fde=False,
            verbose=False):
    """Euclidean distance matrix-based fault detection and exclusion.

    See [1]_ for more detailed explanation of algorithm.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        NavData of GNSS measurements which must include the satellite
        positions: ``x_sv_m``, ``y_sv_m``, ``z_sv_m``, ``b_sv_m`` as
        well as the time row ``gps_millis`` and the corrected
        pseudorange ``corr_pr_m``.
    max_faults : int
        Maximum number of faults to detect and/or exclude.
    threshold : float
        Detection threshold.
    time_fde : bool
        If true, will time the fault detection and exclusion steps and
        return that info.
    verbose : bool
        Prints extra debugging print statements if true.

    Returns
    -------
    navdata : gnss_lib_py.navdata.navdata.NavData
        Result includes a new row of ``fault_edm`` where a
        value of 1 indicates a detected fault and 0 indicates that
        no fault was detected, and 2 indicates an unknown fault status
        usually due to lack of necessary columns or information.
    timing_info : dict
        If time_fde is true, also returns a dictionary of the compute
        times in seconds for each iteration.

    References
    ----------
    ..  [1] D. Knowles and G. Gao. "Detection and Exclusion of Multiple
            Faults using Euclidean Distance Matrices." ION GNSS+ 2023.

    """

    fault_edm = []

    if threshold is None:
        threshold = 0.6

    # number of check indexes
    if max_faults is None:
        nci_nominal = 10
    else:
        nci_nominal = max_faults + 10

    if time_fde:
        compute_times = []
        navdata_timing = []

    for _, _, navdata_subset in loop_time(navdata,'gps_millis'):
        if time_fde:
            time_start = time.time()

        nsl = len(navdata_subset)
        sv_m = navdata_subset[["x_sv_m","y_sv_m","z_sv_m"]]
        corr_pr_m = navdata_subset["corr_pr_m"]

        # remove NaN indexes
        nan_idxs = sorted(list(set(np.arange(nsl)[np.isnan(sv_m).any(axis=0)]).union( \
                               set(np.arange(nsl)[np.isnan(corr_pr_m)]).union( \
                                  ))))[::-1]
        navdata_subset.remove(cols=nan_idxs,inplace=True)
        sv_m = navdata_subset[["x_sv_m","y_sv_m","z_sv_m"]]
        corr_pr_m = navdata_subset["corr_pr_m"]

        if verbose:
            print("nan_idxs:",nan_idxs)

        fault_idxs = []
        orig_idxs = np.arange(len(navdata_subset)+1) # add one for receiver index
        pre_nan_idxs = np.delete(np.arange(nsl+1), nan_idxs)

        edm = _edm_from_satellites_ranges(sv_m,corr_pr_m)

        detection_statistic = np.inf

        while detection_statistic > threshold:

            # stop removing indexes either b/c you need at least four
            # satellites or if maximum number of faults has been reached
            if len(edm) - len(fault_idxs) <= 5 \
                or (max_faults is not None and len(fault_idxs) == max_faults):
                break

            edm_detect = np.delete(np.delete(edm,fault_idxs,0),fault_idxs,1)
            detection_statistic_detect, svd_u, _, _ = _edm_detection_statistic(edm_detect)
            detection_statistic = detection_statistic_detect[0]

            if verbose:
                print("detection statistic:",detection_statistic)

            if detection_statistic < threshold:
                if verbose:
                    print("below threshold")
                break

            if verbose:
                print("before removal statistic:",detection_statistic)

            nci = min(nci_nominal,len(edm_detect))
            u3_suspects = list(np.argsort(np.abs(svd_u[:,3]))[::-1][:nci])
            u4_suspects = list(np.argsort(np.abs(svd_u[:,4]))[::-1][:nci])
            suspects = u3_suspects + u4_suspects
            counts = {i:suspects.count(i) for i in (set(suspects)-set([0]))}
            if verbose:
                print("counts:",counts)
            # suspects must be in all four singular vectors
            # also convert to the original edm indexes
            fault_suspects = [[np.delete(orig_idxs,fault_idxs)[i]]
                               for i,v in counts.items() if v == 2]

            # break out if no new suspects
            if len(fault_suspects) == 0:
                break

            if verbose:
                print("U3 abs",u3_suspects)
                print("U4 abs",u4_suspects)

            stacked_edms = [np.delete(np.delete(edm,fault_idxs+i,0),
                                                fault_idxs+i,1) \
                                                for i in fault_suspects]
            adjusted_indexes = [np.delete(orig_idxs,fault_idxs+i)
                                for i in fault_suspects]

            edms_exclude = np.stack(stacked_edms,axis=0)
            detection_statistic_exclude, _, _, _ = _edm_detection_statistic(edms_exclude)

            # also add detection with none removed
            detection_statistic_exclude += detection_statistic_detect
            fault_suspects.append([])
            adjusted_indexes.append(orig_idxs)

            if verbose:
                print("fault suspects:",fault_suspects)
                print("statistic after exclusion:",detection_statistic_exclude)

            min_idx = np.argmin(detection_statistic_exclude)
            if verbose:
                print("min index:",min_idx)
                print("best option:",fault_suspects[min_idx])

            # if nothing was removed
            if min_idx == len(fault_suspects) - 1:
                break

            fault_idxs += fault_suspects[min_idx]
            if verbose:
                print("new fault indexes:",fault_idxs)
            detection_statistic = detection_statistic_exclude[min_idx]

        # important step! remove 1 since index 0 is the receiver index
        fault_idxs = [pre_nan_idxs[i-1] for i in fault_idxs]

        fault_edm_subset = np.array([0] * nsl)
        fault_edm_subset[fault_idxs] = 1
        fault_edm_subset[nan_idxs] = 2
        fault_edm += list(fault_edm_subset)

        if time_fde:
            time_end = time.time()
            compute_times.append(time_end-time_start)
            navdata_timing += list([time_end-time_start] * nsl)


    navdata["fault_edm"] = fault_edm
    if verbose:
        print(navdata["fault_edm"])

    timing_info = {}
    if time_fde:
        navdata["compute_time_s"] = navdata_timing
        timing_info["compute_times"] = compute_times
        return navdata, timing_info
    return navdata


def fde_greedy_residual(navdata, max_faults, threshold, time_fde=False,
                        verbose=False):
    """Residual-based fault detection and exclusion.

    Implemented based on paper from Blanch et al [2]_.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        NavData of GNSS measurements which must include the satellite
        positions: ``x_sv_m``, ``y_sv_m``, ``z_sv_m``, ``b_sv_m`` as
        well as the time row ``gps_millis`` and the corrected
        pseudorange ``corr_pr_m``.
    max_faults : int
        Maximum number of faults to detect and/or exclude.
    threshold : float
        Detection threshold.
    time_fde : bool
        If true, will time the fault detection and exclusion steps and
        return that info.
    verbose : bool
        Prints extra debugging print statements if true.

    Returns
    -------
    navdata : gnss_lib_py.navdata.navdata.NavData
        Result includes a new row of ``fault_residual`` where a
        value of 1 indicates a detected fault and 0 indicates that
        no fault was detected, and 2 indicates an unknown fault status
        usually due to lack of necessary columns or information.
    timing_info : dict
        If time_fde is true, also returns a dictionary of the compute
        times in seconds for each iteration.

    References
    ----------
    .. [2] Blanch, Juan, Todd Walter, and Per Enge. "Fast multiple fault
           exclusion with a large number of measurements." Proceedings
           of the 2015 International Technical Meeting of the Institute
           of Navigation. 2015.

    """

    fault_residual = []
    if threshold is None:
        threshold = 3000


    if time_fde:
        compute_times = []
        navdata_timing = []

    for _, _, navdata_subset in loop_time(navdata,'gps_millis'):

        if time_fde:
            time_start = time.time()

        subset_length = len(navdata_subset)

        sv_m = navdata_subset[["x_sv_m","y_sv_m","z_sv_m"]]
        corr_pr_m = navdata_subset["corr_pr_m"]

        # remove NaN indexes
        nan_idxs = sorted(list(set(np.arange(subset_length)[np.isnan(sv_m).any(axis=0)]).union( \
                                  set(np.arange(subset_length)[np.isnan(corr_pr_m)]).union( \
                                  ))))[::-1]

        original_indexes = np.arange(subset_length)
        # remove NaN indexes
        navdata_subset.remove(cols=nan_idxs,inplace=True)
        original_indexes = np.delete(original_indexes, nan_idxs)

        fault_idxs = []
        if len(navdata_subset) > 5:

            # test statistic
            receiver_state = solve_wls(navdata_subset)
            solve_residuals(navdata_subset, receiver_state, inplace=True)

            chi_square = _residual_chi_square(navdata_subset, receiver_state)

            if verbose:
                print("chi squared residual test statistic:",chi_square)

            # greedy removal if chi_square above detection threshold
            while chi_square > threshold:
            # stop removing indexes either b/c you need at least four
            # satellites or if maximum number of faults has been reached
                if len(navdata_subset) < 5 or (max_faults is not None \
                                           and len(fault_idxs) >= max_faults):
                    break

                normalized_residual = _residual_exclude(navdata_subset,receiver_state)
                fault_idx = np.argsort(normalized_residual)[-1]

                navdata_subset.remove(cols=[fault_idx], inplace=True)
                fault_idxs.append(original_indexes[fault_idx])
                original_indexes = np.delete(original_indexes, fault_idx)

                # test statistic
                receiver_state = solve_wls(navdata_subset)
                solve_residuals(navdata_subset, receiver_state, inplace=True)
                chi_square = _residual_chi_square(navdata_subset, receiver_state)

                if verbose:
                    print("chi squared:",chi_square,"after removing index:",fault_idxs)

        fault_residual_subset = np.array([0] * subset_length)
        fault_residual_subset[fault_idxs] = 1
        fault_residual_subset[nan_idxs] = 2
        fault_residual += list(fault_residual_subset)

        if time_fde:
            time_end = time.time()
            compute_times.append(time_end-time_start)
            navdata_timing += list([time_end-time_start] * subset_length)

    navdata["fault_residual"] = fault_residual
    if verbose:
        print(navdata["fault_residual"])

    timing_info = {}
    if time_fde:
        navdata["compute_time_s"] = navdata_timing
        timing_info["compute_times"] = compute_times
        return navdata, timing_info
    return navdata


def evaluate_fde(navdata, method, fault_truth_row="fault_gt",
                 time_fde=False, verbose=False,
                 **kwargs):
    """Evaluate FDE methods and compute accuracy scores

    The row designated in the ``fault_truth_row`` variable must indicate
    ground truth faults according to the following convention.
    A value of 1 indicates a fault and 0 indicates no fault. 2 indicates
    an unknown fault status usually due to lack of necessary columns or
    information.

    Measurements that are returned as "unknown" (fault=2) by the fault
    detection method are excluded from all accuracy scores.

    Accuracy metrics are defined accordingly:

    True Positive (TP) : estimated = 1, truth = 1
    True Negative (TN) : estimated = 0, truth = 0
    Missed Detection (MD), False Negative : estimated = 0, truth = 1
    False Alarm (FA), False Positive : estimated = 1, truth = 0

    True Positive Rate (TPR) : TP / (TP + MD)
    True Negative Rate (TNR) : TN / (TN + FA)
    Missed Detection Rate (MDR) : MD / (MD + TP)
    False Alarm Rate (FAR) : FA / (FA + TN)

    Accuracy : (TP + TN) / (TP + TN + MD + FA)
    Balanced Accuracy (BA) : (TPR + TNR) / 2
    Precision : TP / (TP + FA)
    Recall : TPR

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        NavData of GNSS measurements which must include the satellite
        positions: ``x_sv_m``, ``y_sv_m``, ``z_sv_m``, ``b_sv_m`` as
        well as the time row ``gps_millis`` and the corrected
        pseudorange ``corr_pr_m``.
        Additionally, the ground truth fault row must exist as indicated
        by the fault the ``fault_truth_row`` variable.
    method : string
        Method for fault detection and exclusion either "residual" for
        residual-based or "edm" for Euclidean Distance Matrix-based.
    fault_truth_row : string
        Row that indicates the ground truth for the fault status. This
        row is used to provide results on how well each method performs
        at fault detection and exclusion.
    time_fde : bool
        Additional debugging info added like timing.
    verbose : bool
        Prints extra debugging print statements if true.

    Returns
    -------
    metrics : dict
        Combined metrics that were computed.
    navdata : gnss_lib_py.navdata.navdata.NavData
        Resulting NavData from ``solve_fde()``.

    """
    truth_fault_counts = []
    measurement_counts = []
    fault_percentages = []
    timesteps = 0
    for _, _, navdata_subset in loop_time(navdata,"gps_millis"):
        measurement_count = len(navdata_subset)
        truth_fault_count = len(np.argwhere(navdata_subset[fault_truth_row]==1))
        measurement_counts.append(measurement_count)
        truth_fault_counts.append(truth_fault_count)
        fault_percentages.append(truth_fault_count/measurement_count)
        timesteps += 1

    if verbose:
        print("\nDATASET METRICS")
        print("timesteps:",timesteps)
        print("\nmeasurement counts:",
              "\nmin:", int(min(measurement_counts)),
              "\nmean:", np.round(np.mean(measurement_counts),3),
              "\nmedian:", np.round(np.median(measurement_counts),3),
              "\nmax:", int(max(measurement_counts)),
              )
        print("\ntruth fault counts:",
              "\nmin:", int(min(truth_fault_counts)),
              "\nmean:", np.round(np.mean(truth_fault_counts),3),
              "\nmedian:", np.round(np.median(truth_fault_counts),3),
              "\nmax:", int(max(truth_fault_counts)),
              )
        print("\npercentage faulty per timestep as decimal:",
              "\nmin:", np.round(min(fault_percentages),3),
              "\nmean:", np.round(np.mean(fault_percentages),3),
              "\nmedian:", np.round(np.median(fault_percentages),3),
              "\nmax:", np.round(max(fault_percentages),3),
              )

    # remove_outliers must be false so that faults aren't removed
    result = solve_fde(navdata, method=method, remove_outliers=False,
                              verbose=verbose, time_fde=time_fde, **kwargs)
    if time_fde:
        navdata, timing_info = result
    else:
        navdata = result
        timing_info = {}

    estimated_faults = navdata["fault_" + method]
    truth_faults = navdata[fault_truth_row]

    total = len(estimated_faults)
    true_positive = len(np.argwhere((estimated_faults==1) & (truth_faults==1)))
    true_negative = len(np.argwhere((estimated_faults==0) & (truth_faults==0)))
    missed_detection = len(np.argwhere((estimated_faults==0) & (truth_faults==1)))
    false_alarm = len(np.argwhere((estimated_faults==1) & (truth_faults==0)))
    unknown = len(np.argwhere(estimated_faults==2))
    assert total == true_positive + false_alarm + true_negative \
                  + missed_detection + unknown

    # compute accuracy metrics
    tpr = true_positive / (true_positive + missed_detection) \
          if (true_positive + missed_detection) else 0.
    tnr = true_negative / (true_negative + false_alarm) \
          if (true_negative + false_alarm) else 0.
    mdr = missed_detection / (missed_detection + true_positive) \
          if (missed_detection + true_positive) else 0.
    far = false_alarm / (false_alarm + true_negative) \
          if (false_alarm + true_negative) else 0.

    accuracy = (true_positive + true_negative) / (true_positive \
             + true_negative + missed_detection + false_alarm)
    balanced_accuracy = (tpr + tnr) / 2.

    precision = true_positive / (true_positive + false_alarm) \
                if (true_positive + false_alarm) else 0.
    recall = tpr

    if verbose:
        print("\n")
        print(method.upper() + " FDE METRICS")

        print("total measurements:",total)
        print("true positives count, TPR:",
              true_positive, np.round(tpr,3))
        print("true negatives count, TNR:",
              true_negative, np.round(tnr,3))
        print("missed detection count, MDR:",
              missed_detection, np.round(mdr,3))
        print("false alarm count, FAR:",
              false_alarm, np.round(far,3))
        print("unknown count:",unknown)

        # accuracy metrics
        print("\nprecision:",precision)
        print("recall:",recall)
        print("accuracy:",accuracy)
        print("balanced accuracy:",balanced_accuracy)

    metrics = {}
    metrics["dataset_timesteps"] = timesteps
    metrics["measurement_counts_min"] = int(min(measurement_counts))
    metrics["measurement_counts_mean"] = np.mean(measurement_counts)
    metrics["measurement_counts_median"] = np.median(measurement_counts)
    metrics["measurement_counts_max"] = int(max(measurement_counts))
    metrics["fault_counts_min"] = int(min(truth_fault_counts))
    metrics["fault_counts_mean"] = np.mean(truth_fault_counts)
    metrics["fault_counts_median"] = np.median(truth_fault_counts)
    metrics["fault_counts_max"] = int(max(truth_fault_counts))
    metrics["faults_per_timestemp_min"] = min(fault_percentages)
    metrics["faults_per_timestemp_mean"] = np.mean(fault_percentages)
    metrics["faults_per_timestemp_median"] = np.median(fault_percentages)
    metrics["faults_per_timestemp_max"] = max(fault_percentages)
    metrics["method"] = method
    metrics["total_measurements"] = total
    metrics["true_positives_count"] = true_positive
    metrics["tpr"] = tpr
    metrics["true_negatives_count"] = true_negative
    metrics["tnr"] = tnr
    metrics["missed_detection_count"] = missed_detection
    metrics["mdr"] = mdr
    metrics["false_alarm_count"] = false_alarm
    metrics["far"] = far
    metrics["unknown_count"] = unknown
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["accuracy"] = accuracy
    metrics["balanced_accuracy"] = balanced_accuracy

    # compute timing metrics
    if "compute_times" in timing_info:
        metrics["timestep_min_ms"] = np.min(timing_info["compute_times"])*1000
        metrics["timestep_mean_ms"] = np.mean(timing_info["compute_times"])*1000
        metrics["timestep_median_ms"] = np.median(timing_info["compute_times"])*1000
        metrics["timestep_max_ms"] = np.max(timing_info["compute_times"])*1000

    return metrics, navdata


def _edm(points):
    """Creates a Euclidean distance matrix (EDM) from point locations.

    See [3]_ for more explanation.

    Parameters
    ----------
    points : np.array
        Locations of points/nodes in the graph. Numpy array of shape
        state space dimensions x number of points in graph.

    Returns
    -------
    edm : np.array
        Euclidean distance matrix as a numpy array of shape (n x n)
        where n is the number of points in the graph.
        creates edm from points

    References
    ----------
    ..  [3] I. Dokmanic, R. Parhizkar, J. Ranieri, and M. Vetterli.
        “Euclidean Distance Matrices: Essential Theory, Algorithms,
        and Applications.” 2015. arxiv.org/abs/1502.07541.

    """
    dims = points.shape[1]
    gram = (points.T).dot(points)
    edm = np.diag(gram).reshape(-1,1).dot(np.ones((1,dims))) \
        - 2.*gram + np.ones((dims,1)).dot(np.diag(gram).reshape(1,-1))
    return edm


def _edm_from_satellites_ranges(sv_pos, ranges):
    """Creates a Euclidean distance matrix (EDM) from points and ranges.

    Creates an EDM from a combination of known satellite positions as
    well as ranges from between the receiver and satellites.

    Technique introduced in detail in [4]_.

    Parameters
    ----------
    sv_pos : np.array
        known locations of satellites packed as a numpy array in the
        shape state space dimensions x number of satellites.
    ranges : np.array
        ranges between the receiver and satellites packed as a numpy
        array in the shape 1 x number of satellites

    Returns
    -------
    edm : np.array
        Euclidean distance matrix in the shape (1 + s) x (1 + s) where
        s is the number of satellites

    References
    ----------
    ..  [4] Knowles, Derek, and Grace Gao. "Euclidean distance
            matrix-based rapid fault detection and exclusion."
            NAVIGATION: Journal of the Institute of Navigation 70.1
            (2023).

    """
    num_s = sv_pos.shape[1]
    edm = np.zeros((num_s+1,num_s+1))
    edm[0,1:] = ranges**2
    edm[1:,0] = ranges**2
    edm[1:,1:] = _edm(sv_pos)

    return edm


def _edm_detection_statistic(edm):
    """Calculate the EDM FDE detection statistic [5]_.

    Parameters
    ----------
    edm : np.ndarray
        Euclidean distance matrix from GNSS measurements

    Returns
    -------
    detection_statistic : list
        EDM FDE detection statistics
    svd_u : np.ndarray
        The U matrix from SVD
    svd_s : np.nddary
        The Sigma matrix from SVD
    svd_v : np.ndarray
        The V matrix from SVD

    References
    ----------
    ..  [5] D. Knowles and G. Gao. "Detection and Exclusion of Multiple
            Faults using Euclidean Distance Matrices." ION GNSS+ 2023.

    """

    dims = edm.shape[1]

    # double center EDM to retrive the corresponding Gram matrix
    center = np.eye(dims) - (1./dims) * np.ones((dims,dims))
    gram = -0.5 * center @ edm @ center

    # calculate the singular value decomposition
    svd_u, svd_s, svd_vt = np.linalg.svd(gram,full_matrices=True)
    svd_v = svd_vt.T
    svd_s = np.atleast_2d(svd_s)

    # linear averaged and normalized by first singular value
    svd_s = np.log10(svd_s)
    detection_statistic = (np.mean(svd_s[:,3:5],axis=1)) \
                        / (svd_s[:,0])
    detection_statistic = list(detection_statistic)

    return detection_statistic, svd_u, svd_s, svd_v


def _residual_chi_square(navdata, receiver_state):
    """Chi square test for residuals.

    Implemented from Blanch et al [6]_.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        NavData of GNSS measurements which must include the satellite
        positions: ``x_sv_m``, ``y_sv_m``, ``z_sv_m`` and residuals
        ``residuals_m``.
    receiver_state : gnss_lib_py.navdata.navdata.NavData
        Reciever state that must include the receiver's state of:
        ``x_rx_wls_m``, ``y_rx_wls_m``, and ``z_rx_wls_m``.

    Returns
    -------
    chi_square : float
        Chi square test statistic.

    References
    ----------
    .. [6] Blanch, Juan, Todd Walter, and Per Enge. "Fast multiple fault
           exclusion with a large number of measurements." Proceedings
           of the 2015 International Technical Meeting of the Institute
           of Navigation. 2015.

    """

    # solve for residuals
    residuals = navdata["residuals_m"].reshape(-1,1)

    # weights
    weights = np.eye(len(navdata))

    # geometry matrix
    geo_matrix = (receiver_state[["x_rx_wls_m","y_rx_wls_m","z_rx_wls_m"]].reshape(-1,1) \
               -  navdata[["x_sv_m","y_sv_m","z_sv_m"]]).T
    geo_matrix /= np.linalg.norm(geo_matrix,axis=0)

    chi_square = residuals.T @ (weights - weights @ geo_matrix \
               @ np.linalg.pinv(geo_matrix.T @ weights @ geo_matrix) \
               @ geo_matrix.T @ weights ) @ residuals
    chi_square = chi_square.item()

    return chi_square


def _residual_exclude(navdata, receiver_state):
    """Detection statistic for Residual-based fault detection.

    Implemented from Blanch et al [7]_.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        NavData of GNSS measurements which must include the satellite
        positions: ``x_sv_m``, ``y_sv_m``, ``z_sv_m`` and residuals
        ``residuals_m``.
    receiver_state : gnss_lib_py.navdata.navdata.NavData
        Reciever state that must include the receiver's state of:
        ``x_rx_wls_m``, ``y_rx_wls_m``, and ``z_rx_wls_m``.

    Returns
    -------
    normalized_residual : np.ndarray
        Array of the normalized residual for each satellite.

    References
    ----------
    .. [7] Blanch, Juan, Todd Walter, and Per Enge. "Fast multiple fault
           exclusion with a large number of measurements." Proceedings
           of the 2015 International Technical Meeting of the Institute
           of Navigation. 2015.

    """
    # solve for residuals
    residuals = navdata["residuals_m"].reshape(-1,1)

    # weights
    weights = np.eye(len(navdata))

    # geometry matrix
    geo_matrix = (receiver_state[["x_rx_wls_m","y_rx_wls_m","z_rx_wls_m"]].reshape(-1,1) \
               -  navdata[["x_sv_m","y_sv_m","z_sv_m"]]).T
    geo_matrix /= np.linalg.norm(geo_matrix,axis=0)

    # calculate normalized residual
    x_tilde = np.linalg.pinv(geo_matrix.T @ weights @ geo_matrix) \
            @ geo_matrix.T @ weights @ residuals

    normalized_residual = np.divide(np.multiply(np.diag(weights).reshape(-1,1),
                                               (residuals - geo_matrix @ x_tilde)**2),
                                    np.diag(1 - weights @ geo_matrix @ \
                                     np.linalg.pinv(geo_matrix.T @ weights @ geo_matrix) \
                                     @ geo_matrix.T).reshape(-1,1))

    normalized_residual = normalized_residual[:,0]

    return normalized_residual
