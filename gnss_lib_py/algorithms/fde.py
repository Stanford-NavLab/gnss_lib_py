"""Fault Detection and Exclusion (FDE) methods for GNSS applications.

"""

__authors__ = "D. Knowles"
__date__ = "11 Jul 2023"

import time
import numpy as np
import matplotlib.pyplot as plt

from gnss_lib_py.algorithms.residuals import solve_residuals

def solve_fde(navdata, method="residual", remove_outliers=False,
              max_faults=None, threshold=None,verbose=False,
              **kwargs):
    """Detects and optionally removes GNSS measurement faults.

    Individual fault detection and exclusion (fde) methods are
    documented in their individual functions.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        NavData of GNSS measurements which must include the receiver's
        estimated state: x_rx*_m, y_rx*_m, z_rx*_m, b_rx*_m as well as
        the satellite states: x_sv_m, y_sv_m, z_sv_m, b_sv_m as well
        as timing "gps_millis" and the corrected pseudorange corr_pr_m.
    method : string
        Method for fault detection and exclusion either "residual" for
        residual-based, "ss" for solution separation or "edm" for
        Euclidean Distance Matrix-based.
    remove_outliers : bool
        If true, will remove detected faults from NavData instance.
        If false, will detect but not exclude faults.
    max_faults : int
        Maximum number of faults to detect and/or exclude.
    threshold : float
        Detection threshold.

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Result includes a new row of ``fault_<method>`` where a
        value of 1 indicates a detected fault and 0 indicates that
        no fault was detected, and 2 indicates an unknown fault status
        usually due to lack of necessary columns or information.

    """

    if method == "residual":
        navdata = fde_residual_old(navdata, max_faults=max_faults,
                                        threshold=threshold,verbose=verbose,
                                        **kwargs)
    elif method == "ss":
        navdata = fde_solution_separation_old(navdata, max_faults=max_faults,
                                                   threshold=threshold,verbose=verbose,
                                                   **kwargs)
    elif method == "edm":
        navdata = fde_edm(navdata, max_faults=max_faults,
                                   threshold=threshold,verbose=verbose,
                                   **kwargs)
    elif method == "edm_old":
        navdata = fde_edm_old(navdata, max_faults=max_faults,
                                   threshold=threshold,verbose=verbose,
                                   **kwargs)
    else:
        raise ValueError("invalid method input for solve_fde()")

    if remove_outliers:
        navdata = navdata.where("fault_" + method, False)

    return navdata

def fde_edm(navdata, max_faults=None, threshold=1.0, verbose=False,
            debug=False):
    """Euclidean distance matrix-based fault detection and exclusion.

    See [1]_ for more detailed explanation of algorithm.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        NavData of GNSS measurements which must include the
        the satellite states: x_sv_m, y_sv_m, z_sv_m, b_sv_m as well
        as gps_millis and the corrected pseudorange corr_pr_m.
    max_faults : int
        Maximum number of faults to detect and/or exclude.
    threshold : float
        Detection threshold.
    verbose : bool
        Prints extra debugging print statements if true.

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Result includes a new row of ``fault_edm`` where a
        value of 1 indicates a detected fault and 0 indicates that
        no fault was detected, and 2 indicates an unknown fault status
        usually due to lack of necessary columns or information.

    References
    ----------
    ..  [1] D. Knowles and G. Gao. "Detection and Exclusion of Multiple
            Faults using Euclidean Distance Matrices." ION GNSS+ 2023.

    """

    fault_edm = []
    MIN_SATELLITES = 4

    if threshold is None:
        threshold = 1E6

    # number of check indexes
    if max_faults is None:
        nci_nominal = 10
    else:
        nci_nominal = max_faults + 5

    compute_times = []

    for _, _, navdata_subset in navdata.loop_time('gps_millis'):

        if debug:
            time_start = time.time()

        nsl = len(navdata_subset)
        if verbose:
            print("gt faults:",np.argwhere(navdata_subset["fault_gt"]==1)[:,0])
        sv_m = navdata_subset[["x_sv_m","y_sv_m","z_sv_m"]]
        corr_pr_m = navdata_subset["corr_pr_m"]

        # remove NaN indexes
        nan_idxs = sorted(list(set(np.arange(nsl)[np.isnan(sv_m).any(axis=0)]).union( \
                               set(np.arange(nsl)[np.isnan(corr_pr_m)]).union( \
                                  ))))[::-1]
        fault_idxs = []
        orig_idxs = np.arange(nsl+1) # add one for receiver index

        edm = _edm_from_satellites_ranges(sv_m,corr_pr_m)

        detection_statistic = np.inf

        while detection_statistic > threshold:

            # stop removing indexes either b/c you need at least four
            # satellites or if maximum number of faults has been reached
            if len(edm) - len(fault_idxs) <= MIN_SATELLITES+1 \
                or (max_faults is not None and len(fault_idxs) == max_faults):
                break


            try:
                edm_detect = np.delete(np.delete(edm,fault_idxs,0),fault_idxs,1)
                detection_statistic_detect, svd_u, svd_s, svd_v = _edm_detection_statistic(edm_detect)
                detection_statistic = detection_statistic_detect[0]
            except Exception as exception:
                if verbose:
                    print(exception)
                print(exception)
                break

            if detection_statistic < threshold:
                if verbose:
                    print("below threshold")
                break


            original_sings = svd_s.copy()
            if verbose:
                print("before statistic:",detection_statistic)

            nci = min(nci_nominal,len(edm_detect))
            u3_suspects = list(np.argsort(np.abs(svd_u[:,3]))[::-1][:nci])
            u4_suspects = list(np.argsort(np.abs(svd_u[:,4]))[::-1][:nci])
            v3_suspects = list(np.argsort(np.abs(svd_v[:,3]))[::-1][:nci])
            v4_suspects = list(np.argsort(np.abs(svd_v[:,4]))[::-1][:nci])
            suspects = u3_suspects + u4_suspects \
                     + v3_suspects + v4_suspects
            counts = {i:suspects.count(i) for i in (set(suspects)-set([0]))}
            if verbose:
                print("counts:",counts)
            # suspects must be in all four singular vectors
            # also convert to the original edm indexes
            fault_suspects = [[np.delete(orig_idxs,fault_idxs)[i]]
                               for i,v in counts.items() if v == 4]

            # avg_u = list(np.argsort(np.mean(np.abs(svd_u)[:,3:5],axis=1))[::-1][:nci])
            # avg_v = list(np.argsort(np.mean(np.abs(svd_v)[:,3:5],axis=1))[::-1][:nci])
            # suspects = avg_u + avg_v
            # counts = {i:suspects.count(i) for i in (set(suspects)-set([0]))}
            # if verbose:
            #     print("counts:",counts)
            # # suspects must be in all four singular vectors
            # # also convert to the original edm indexes
            # fault_suspects = [[np.delete(orig_idxs,fault_idxs)[i]]
            #                    for i,v in counts.items() if v == 2]

            # print(np.mean(np.abs(svd_u)[:,3:5],axis=1))
            # print(svd_u.shape)
            # print("avg_u:",avg_u)
            # print("unitity:",list(np.argsort(np.mean(np.abs(svd_u)[:,3:5],axis=1)[::-1][:nci]))


            # break out if no new suspects
            if len(fault_suspects) == 0:
                break

            # plt.figure()
            # plt.imshow(U)
            # plt.colorbar()
            #
            # plt.figure()
            # plt.imshow(Vh)
            # plt.colorbar()

            if verbose:
                print("U3 abs",u3_suspects)
                print("U4 abs",u4_suspects)
                print("V3 abs",v3_suspects)
                print("V4 abs",v4_suspects)

            stacked_edms = [np.delete(np.delete(edm,fault_idxs+i,0),
                                                fault_idxs+i,1) \
                                                for i in fault_suspects]
            adjusted_indexes = [np.delete(orig_idxs,fault_idxs+i)
                                for i in fault_suspects]

            edms_exclude = np.stack(stacked_edms,axis=0)
            detection_statistic_exclude, _, svd_s_exclude, _ = _edm_detection_statistic(edms_exclude)

            # add also all fault suspects combined together
            if len(fault_suspects) > 1 and (max_faults is None
                or len(fault_suspects)+len(fault_idxs) <= max_faults) \
                and (len(edm) - len(fault_suspects) - len(fault_idxs)) > MIN_SATELLITES+1:
                all_faults = [i[0] for i in fault_suspects]
                edm_all_faults = np.delete(np.delete(edm,fault_idxs+all_faults,0),
                                                         fault_idxs+all_faults,1)
                detection_statistic_all, _, svd_s_all, _ = _edm_detection_statistic(edm_all_faults)

                # add to combined arrays to make argmin easy
                stacked_edms.append(edm_all_faults)
                detection_statistic_exclude += detection_statistic_all
                adjusted_indexes.append(np.delete(orig_idxs,fault_idxs+all_faults))
                fault_suspects.append(all_faults)

            # also add detection with none removed
            detection_statistic_exclude += detection_statistic_detect
            fault_suspects.append([])
            adjusted_indexes.append(orig_idxs)

            if verbose:
                print("fault suspects:",fault_suspects)
                print("statistic after exclusion:",detection_statistic_exclude)

                plt.figure()
                plt.scatter(list(range(original_sings.shape[1])),original_sings[0],label="original")
                for s_index in range(len(svd_s_exclude)):
                    plt.scatter(list(range(svd_s_exclude.shape[1])),svd_s_exclude[s_index,:],
                                label=str(fault_suspects[s_index])+" removed")
                plt.scatter(list(range(svd_s_all.shape[1])),svd_s_all[0],label="all removed")
                # plt.yscale("log")
                plt.legend()

                plt.figure()
                plt.scatter(0,detection_statistic,label="original")
                for s_index in range(len(svd_s_exclude)):
                    plt.scatter(0,detection_statistic_exclude[s_index],
                                label=str(fault_suspects[s_index])+" removed")
                plt.scatter(0,detection_statistic_all,label="all removed")
                # plt.yscale("log")
                plt.legend()

                plt.show()

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
        fault_idxs = [i-1 for i in fault_idxs]

        fault_edm_subset = np.array([0] * len(navdata_subset))
        fault_edm_subset[fault_idxs] = 1
        fault_edm_subset[nan_idxs] = 2
        fault_edm += list(fault_edm_subset)

        if debug:
            time_end = time.time()
            compute_times.append(time_end-time_start)

    navdata["fault_edm"] = fault_edm
    if verbose:
        print(navdata["fault_edm"])

    debug_info = {}
    if debug:
        debug_info["compute_times"] = compute_times
        return navdata, debug_info
    return navdata

def _edm_detection_statistic(edm):
    """Calculate the EDM FDE detection statistic.

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

    """

    dims = edm.shape[1]

    # double center EDM to retrive the corresponding Gram matrix
    center = np.eye(dims) - (1./dims) * np.ones((dims,dims))
    gram = -0.5 * center @ edm @ center

    # calculate the singular value decomposition
    svd_u, svd_s, svd_vt = np.linalg.svd(gram,full_matrices=True)
    svd_v = svd_vt.T
    svd_s = np.atleast_2d(svd_s)

    # compute the detection statistic as mean of 3rd and 4th eigs

    # in log space
    # detection_statistic = list(np.mean(svd_s[:,3:5],axis=1))

    # linear averaged
    # svd_s = np.log10(svd_s)
    # detection_statistic = (np.mean(svd_s[:,3:5],axis=1) - svd_s[:,5]) \
    #                     / (svd_s[:,2] - svd_s[:,5])
    # detection_statistic = list(detection_statistic)

    # linear averaged
    svd_s = np.log10(svd_s)
    detection_statistic = (np.mean(svd_s[:,3:5],axis=1)) \
                        / (svd_s[:,0])
    detection_statistic = list(detection_statistic)

    # log normalized
    # detection_statistic = (np.mean(svd_s[:,3:5],axis=1)) \
    #                     / (svd_s[:,0])
    # detection_statistic = list(detection_statistic)

    return detection_statistic, svd_u, svd_s, svd_v

def fde_residual_old(navdata, receiver_state, max_faults, threshold,
                 verbose=False):
    """Residual-based fault detection and exclusion.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        NavData of GNSS measurements which must include the receiver's
        estimated state: x_rx*_m, y_rx*_m, z_rx*_m, b_rx*_m as well as
        the satellite states: x_sv_m, y_sv_m, z_sv_m, b_sv_m as well
        as timing "gps_millis" and the corrected pseudorange corr_pr_m.
    receiver_state : gnss_lib_py.parsers.navdata.NavData
        Either estimated or ground truth receiver position in ECEF frame
        in meters and the estimated or ground truth receiver clock bias
        also in meters as an instance of the NavData class with the
        following rows: x_rx*_m, y_rx*_m, z_rx*_m, b_rx*_m.
    max_faults : int
        Maximum number of faults to detect and/or exclude.
    threshold : float
        Detection threshold.
    verbose : bool
        Prints extra debugging print statements if true.

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Result includes a new row of ``fault_residual`` where a
        value of 1 indicates a detected fault and 0 indicates that
        no fault was detected, and 2 indicates an unknown fault status
        usually due to lack of necessary columns or information.

    """

    fault_residual = []
    if threshold is None:
        threshold = 10

    solve_residuals(navdata, receiver_state, inplace=True)

    for _, _, navdata_subset in navdata.loop_time('gps_millis'):

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

        if len(navdata_subset) < 6:
            ri = []
        else:
            residuals = navdata_subset["residuals_m"].reshape(-1,1)

            # test statistic
            r = np.sqrt(residuals.T.dot(residuals)[0,0] \
                         / (subset_length - 4) )

            if verbose:
                print("residual test statistic:",r)

            # iterate through subsets if r is above detection threshold
            if r > threshold:
                ri = set()
                r_subsets = []
                for ss in range(len(navdata_subset)):
                    residual_subset = np.delete(residuals,ss,axis=0)
                    r_subset = np.sqrt(residual_subset.T.dot(residual_subset)[0,0] \
                                 / (len(residual_subset) - 4) )
                    if verbose:
                        r_subsets.append(r_subset)
                    # adjusted threshold metric
                    if r_subset/r < 1.:
                        ri.add(ss)

                if len(ri) == 0:
                    if verbose:
                        print("NONE fail:")
                        print("r: ",r)
                        print("ri: ",ri)
                        for rri, rrr in enumerate(residuals):
                            print(rri, rrr, r_subsets[rri]/r)
            else:
                if verbose:
                    print("threshold fail:")
                    print("r: ",r)
                    for rri, rrr in enumerate(residuals):
                        print(rri, rrr)

            ri = list(ri)

        fault_residual_subset = np.array([0] * subset_length)
        fault_residual_subset[original_indexes[ri]] = 1
        fault_residual_subset[nan_idxs] = 2
        fault_residual += list(fault_residual_subset)


    navdata["fault_residual"] = fault_residual
    if verbose:
        print(navdata["fault_residual"])

    return navdata

def fde_solution_separation_old(navdata, max_faults, threshold,
                            verbose=False):
    """Solution separation fault detection and exclusion.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        NavData of GNSS measurements which must include the receiver's
        estimated state: x_rx*_m, y_rx*_m, z_rx*_m, b_rx*_m as well as
        the satellite states: x_sv_m, y_sv_m, z_sv_m, b_sv_m as well
        as timing "gps_millis" and the corrected pseudorange corr_pr_m.
    max_faults : int
        Maximum number of faults to detect and/or exclude.
    threshold : float
        Detection threshold.
    verbose : bool
        Prints extra debugging print statements if true.

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Result includes a new row of ``fault_ss`` where a
        value of 1 indicates a detected fault and 0 indicates that
        no fault was detected, and 2 indicates an unknown fault status
        usually due to lack of necessary columns or information.

    """

    fault_ss = []
    if threshold is None:
        threshold = 100

    for _, _, navdata_subset in navdata.loop_time('gps_millis'):

        subset_length = len(navdata_subset)

        sv_m = navdata_subset[["x_sv_m","y_sv_m","z_sv_m"]]
        corr_pr_m = navdata_subset["corr_pr_m"]

        # remove NaN indexes
        nan_idxs = sorted(list(set(np.arange(subset_length)[np.isnan(sv_m).any(axis=0)]).union( \
                                  set(np.arange(subset_length)[np.isnan(corr_pr_m)]).union( \
                                  ))))[::-1]

        # fault_ss_subset = np.array([0] * len(navdata_subset))
        # fault_ss_subset[tri] = 1
        # fault_ss_subset[nan_idxs] = 2
        # fault_ss += list(fault_ss_subset)

    # navdata["fault_ss"] = fault_ss
    navdata["fault_ss"] = 0

    return navdata

def fde_edm_old(navdata, max_faults, threshold=1.0, verbose=False):
    """Euclidean distance matrix-based fault detection and exclusion.

    See [1]_ for more detailed explanation of algorithm.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        NavData of GNSS measurements which must include the receiver's
        estimated state: x_rx*_m, y_rx*_m, z_rx*_m, b_rx*_m as well as
        the satellite states: x_sv_m, y_sv_m, z_sv_m, b_sv_m as well
        as timing "gps_millis" and the corrected pseudorange corr_pr_m.
    max_faults : int
        Maximum number of faults to detect and/or exclude.
    threshold : float
        Detection threshold.
    verbose : bool
        Prints extra debugging print statements if true.

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Result includes a new row of ``fault_edm`` where a
        value of 1 indicates a detected fault and 0 indicates that
        no fault was detected, and 2 indicates an unknown fault status
        usually due to lack of necessary columns or information.

    References
    ----------
    ..  [1] D. Knowles and G. Gao. "Euclidean Distance Matrix-based
        Rapid Fault Detection and Exclusion." ION GNSS+ 2021.


    """

    fault_edm = []
    dims = 3
    if threshold is None:
        threshold = 100

    for _, _, navdata_subset in navdata.loop_time('gps_millis'):

        nsl = len(navdata_subset)

        sv_m = navdata_subset[["x_sv_m","y_sv_m","z_sv_m"]]
        corr_pr_m = navdata_subset["corr_pr_m"]

        # remove NaN indexes
        nan_idxs = sorted(list(set(np.arange(nsl)[np.isnan(sv_m).any(axis=0)]).union( \
                                  set(np.arange(nsl)[np.isnan(corr_pr_m)]).union( \
                                  ))))[::-1]

        D = _edm_from_satellites_ranges(sv_m,corr_pr_m)

        # add one to account for the receiver in D
        ri = list(np.array(nan_idxs)+1)  # index to remove
        tri = nan_idxs.copy()            # removed indexes (in transmitter frame)
        reci = 0                            # index of the receiver
        oi = np.arange(D.shape[0])                 # original indexes

        while True:

            if ri != None:
                if verbose:
                    print("removing index: ",ri)

                if isinstance(ri,np.int64):
                    # add removed index to index list passed back
                    tri.append(oi[ri]-1)
                    # keep track of original indexes (since deleting)
                    oi = np.delete(oi,ri)
                    ri = [ri]
                else:
                    for ri_val in ri:
                        oi = np.delete(oi,ri_val-1)

                for ri_val in ri:

                    # remove index from EDM
                    D = np.delete(D,ri_val,axis=0)
                    D = np.delete(D,ri_val,axis=1)

            # EDM FDE
            n = D.shape[0]  # shape of EDM
            # stop removing indexes either b/c you need at least four
            # satellites or if maximum number of faults has been reached
            if n <= 5 or (max_faults != None and len(tri) >= max_faults):
                break

            # double center EDM to retrive the corresponding Gram matrix
            J = np.eye(n) - (1./n)*np.ones((n,n))
            G = -0.5*J.dot(D).dot(J)

            try:
                # perform singular value decomposition
                U, S, Vh = np.linalg.svd(G)
            except Exception as exception:
                if verbose:
                    print(exception)
                break

            # calculate detection test statistic
            warn = S[dims]*(sum(S[dims:])/float(len(S[dims:])))/S[0]
            if verbose:
                print("\nDetection test statistic:",warn)

            if warn > threshold:
                ri = None

                u_mins = set(np.argsort(U[:,dims])[:2])
                u_maxes = set(np.argsort(U[:,dims])[-2:])
                v_mins = set(np.argsort(Vh[dims,:])[:2])
                v_maxes = set(np.argsort(Vh[dims,:])[-2:])

                def test_option(ri_option):
                    # remove option
                    D_opt = np.delete(D.copy(),ri_option,axis=0)
                    D_opt = np.delete(D_opt,ri_option,axis=1)

                    # reperform double centering to obtain Gram matrix
                    n_opt = D_opt.shape[0]
                    J_opt = np.eye(n_opt) - (1./n_opt)*np.ones((n_opt,n_opt))
                    G_opt = -0.5*J_opt.dot(D_opt).dot(J_opt)

                    # perform singular value decomposition
                    _, S_opt, _ = np.linalg.svd(G_opt)

                    # calculate detection test statistic
                    warn_opt = S_opt[dims]*(sum(S_opt[dims:])/float(len(S_opt[dims:])))/S_opt[0]

                    return warn_opt


                # get all potential options
                ri_options = u_mins | v_mins | u_maxes | v_maxes
                # remove the receiver as a potential fault
                ri_options = ri_options - set([reci])
                ri_tested = []
                ri_warns = []

                ui = -1
                while np.argsort(np.abs(U[:,dims]))[ui] in ri_options:
                    ri_option = np.argsort(np.abs(U[:,dims]))[ui]

                    # calculate test statistic after removing index
                    warn_opt = test_option(ri_option)

                    # break if test statistic decreased below threshold
                    if warn_opt < threshold:
                        ri = ri_option
                        if verbose:
                            print("chosen ri: ", ri)
                        break
                    else:
                        ri_tested.append(ri_option)
                        ri_warns.append(warn_opt)
                    ui -= 1

                # continue searching set if didn't find index
                if ri == None:
                    ri_options_left = list(ri_options - set(ri_tested))

                    for ri_option in ri_options_left:
                        warn_opt = test_option(ri_option)

                        if warn_opt < threshold:
                            ri = ri_option
                            if verbose:
                                print("chosen ri: ", ri)
                            break
                        else:
                            ri_tested.append(ri_option)
                            ri_warns.append(warn_opt)

                # if no faults decreased below threshold, then remove the
                # index corresponding to the lowest test statistic value
                if ri == None:
                    idx_best = np.argmin(np.array(ri_warns))
                    ri = ri_tested[idx_best]
                    if verbose:
                        print("chosen ri: ", ri)
            else:
                break

        fault_edm_subset = np.array([0] * len(navdata_subset))
        fault_edm_subset[tri] = 1
        fault_edm_subset[nan_idxs] = 2
        fault_edm += list(fault_edm_subset)

    navdata["fault_edm"] = fault_edm
    if verbose:
        print(navdata["fault_edm"])

    return navdata

def evaluate_fde(navdata, method, fault_truth_row="fault_gt",
                 debug=False, verbose=False,
                 **kwargs):
    """Evaluate FDE methods and compute accuracy scores

    Measurements that are returned as "unknown" (fault=2) by the fault
    detection method are excluded from all accuracy scores.

    Accuracy metrics are defined accordingly.

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
    navdata : gnss_lib_py.parsers.navdata.NavData
        NavData of GNSS measurements which must include the receiver's
        estimated state: x_rx*_m, y_rx*_m, z_rx*_m, b_rx*_m as well as
        the satellite states: x_sv_m, y_sv_m, z_sv_m, b_sv_m as well
        as timing "gps_millis" and the corrected pseudorange corr_pr_m.
    method : string
        Method for fault detection and exclusion either "residual" for
        residual-based, "ss" for solution separation or "edm" for
        Euclidean Distance Matrix-based.
    debug : bool
        Additional debugging info added like timing
    verbose : bool
        Prints extra debugging print statements if true.

    """
    truth_fault_counts = []
    measurement_counts = []
    fault_percentages = []
    timesteps = 0
    for _, _, navdata_subset in navdata.loop_time("gps_millis"):
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
                              verbose=verbose, debug=debug, **kwargs)
    if debug:
        navdata, debug_info = result
    else:
        navdata = result
        debug_info = {}

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
    try:
        tpr = true_positive / (true_positive + missed_detection)
    except ZeroDivisionError:
        tpr = 0.0
    try:
        tnr = true_negative / (true_negative + false_alarm)
    except ZeroDivisionError:
        tnr = 0.0
    try:
        mdr = missed_detection / (missed_detection + true_positive)
    except ZeroDivisionError:
        mdr = 0.0
    try:
        far = false_alarm / (false_alarm + true_negative)
    except ZeroDivisionError:
        far = 0.0

    accuracy = (true_positive + true_negative) / (true_positive \
             + true_negative + missed_detection + false_alarm)
    balanced_accuracy = (tpr + tnr) / 2.
    try:
        precision = true_positive / (true_positive + false_alarm)
    except ZeroDivisionError:
        precision = 0
    recall = tpr

    # compute timing metrics
    if "compute_times" in debug_info:
        timestep_min = np.min(debug_info["compute_times"])
        timestep_mean = np.mean(debug_info["compute_times"])
        timestep_median = np.median(debug_info["compute_times"])
        timestep_max = np.max(debug_info["compute_times"])

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
    metrics["timestep_min_ms"] = timestep_min*1000
    metrics["timestep_mean_ms"] = timestep_mean*1000
    metrics["timestep_median_ms"] = timestep_median*1000
    metrics["timestep_max_ms"] = timestep_max*1000

    return metrics


def _edm(X):
    """Creates a Euclidean distance matrix (EDM) from point locations.

    See [1]_ for more explanation.

    Parameters
    ----------
    X : np.array
        Locations of points/nodes in the graph. Numpy array of shape
        state space dimensions x number of points in graph.

    Returns
    -------
    D : np.array
        Euclidean distance matrix as a numpy array of shape (n x n)
        where n is the number of points in the graph.
        creates edm from points

    References
    ----------
    ..  [1] I. Dokmanic, R. Parhizkar, J. Ranieri, and M. Vetterli.
        “Euclidean Distance Matrices: Essential Theory, Algorithms,
        and Applications.” 2015. arxiv.org/abs/1502.07541.

    """
    n = X.shape[1]
    G = (X.T).dot(X)
    D = np.diag(G).reshape(-1,1).dot(np.ones((1,n))) \
        - 2.*G + np.ones((n,1)).dot(np.diag(G).reshape(1,-1))
    return D

def _edm_from_satellites_ranges(S,ranges):
    """Creates a Euclidean distance matrix (EDM) from points and ranges.

    Creates an EDM from a combination of known satellite positions as
    well as ranges from between the receiver and satellites.

    Parameters
    ----------
    S : np.array
        known locations of satellites packed as a numpy array in the
        shape state space dimensions x number of satellites.
    ranges : np.array
        ranges between the receiver and satellites packed as a numpy
        array in the shape 1 x number of satellites

    Returns
    -------
    D : np.array
        Euclidean distance matrix in the shape (1 + s) x (1 + s) where
        s is the number of satellites

    """
    num_s = S.shape[1]
    D = np.zeros((num_s+1,num_s+1))
    D[0,1:] = ranges**2
    D[1:,0] = ranges**2
    D[1:,1:] = _edm(S)

    return D
