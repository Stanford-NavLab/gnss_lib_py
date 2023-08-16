"""Fault Detection and Exclusion (FDE) methods for GNSS applications.

"""

__authors__ = "D. Knowles"
__date__ = "11 Jul 2023"

import numpy as np
import matplotlib.pyplot as plt

from gnss_lib_py.algorithms.residuals import solve_residuals

def solve_fde(navdata, method="residual", remove_outliers=False,
              max_faults=None, threshold=None,
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
                                        threshold=threshold,
                                        **kwargs)
    elif method == "ss":
        navdata = fde_solution_separation_old(navdata, max_faults=max_faults,
                                                   threshold=threshold,
                                                   **kwargs)
    elif method == "edm":
        navdata = fde_edm(navdata, max_faults=max_faults,
                                   threshold=threshold,
                                   **kwargs)
    elif method == "edm_old":
        navdata = fde_edm_old(navdata, max_faults=max_faults,
                                   threshold=threshold,
                                   **kwargs)
    else:
        raise ValueError("invalid method input for solve_fde()")

    if remove_outliers:
        navdata = navdata.where("fault_" + method, False)

    return navdata

def fde_edm(navdata, max_faults=1, threshold=1.0, verbose=False):
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
    dims = 3

    if threshold is None:
        threshold = 100

    data = {}
    data_45means = []

    for _, _, navdata_subset in navdata.loop_time('gps_millis'):

        nsl = len(navdata_subset)

        sv_m = navdata_subset[["x_sv_m","y_sv_m","z_sv_m"]]
        corr_pr_m = navdata_subset["corr_pr_m"]

        # remove NaN indexes
        nan_indexes = sorted(list(set(np.arange(nsl)[np.isnan(sv_m).any(axis=0)]).union( \
                                  set(np.arange(nsl)[np.isnan(corr_pr_m)]).union( \
                                  ))))[::-1]
        fault_indexes = []

        D = _edm_from_satellites_ranges(sv_m,corr_pr_m)

        while True:

            # EDM FDE
            n = D.shape[0]  # shape of EDM
            # stop removing indexes either b/c you need at least four
            # satellites or if maximum number of faults has been reached
            if n <= 5 or (max_faults != None and len(fault_indexes) >= max_faults):
                break

            # double center EDM to retrive the corresponding Gram matrix
            J = np.eye(n) - (1./n)*np.ones((n,n))
            G = -0.5*J.dot(D).dot(J)

            try:
                # perform singular value decomposition
                U, S, Vh = np.linalg.svd(G)
                # S = np.log(S)
                # S = (S-S[-1])/(S[0]-S[-1])
            except Exception as exception:
                if verbose:
                    print(exception)
                print(exception)
                stop
                break

            detection_statistic = np.mean(S[3:5])
            original_sings = S.copy()
            print("before statistic:",[detection_statistic])

            # number of check indexes
            nci = 10

            u3_suspects = list(np.argsort(np.abs(U[:,3]))[::-1][:nci])
            u4_suspects = list(np.argsort(np.abs(U[:,4]))[::-1][:nci])
            v3_suspects = list(np.argsort(np.abs(Vh[3,:]))[::-1][:nci])
            v4_suspects = list(np.argsort(np.abs(Vh[4,:]))[::-1][:nci])
            suspects = u3_suspects + u4_suspects \
                     + v3_suspects + v4_suspects

            counts = {i:suspects.count(i) for i in (set(suspects)-set([0]))}
            print("counts:",counts)
            # suspects must be in all four singular vectors
            fault_suspects = [i for i,v in counts.items() if v == 4]


            # plt.figure()
            # plt.imshow(U)
            # plt.colorbar()
            #
            # plt.figure()
            # plt.imshow(Vh)
            # plt.colorbar()

            print("U3 min",np.argmin(U[:,3]))
            print("U4 min",np.argmin(U[:,4]))
            print("U3 max",np.argmax(U[:,3]))
            print("U4 max",np.argmax(U[:,4]))
            print("V3 min",np.argmin(Vh[3,:]))
            print("V4 min",np.argmin(Vh[4,:]))
            print("V3 max",np.argmax(Vh[3,:]))
            print("V4 max",np.argmax(Vh[4,:]))

            print("U3 abs",u3_suspects)
            print("U4 abs",u4_suspects)
            print("V3 abs",v3_suspects)
            print("V4 abs",v4_suspects)
            print("fault suspects:",fault_suspects)

            def gram_removed(D,i):

                D = np.delete(np.delete(D,i,0),i,1)
                n = D.shape[0]
                # double center EDM to retrive the corresponding Gram matrix
                J = np.eye(n) - (1./n)*np.ones((n,n))
                G = -0.5*J.dot(D).dot(J)
                return G

            stacked_matrices = [gram_removed(D,i) for i in fault_suspects]
            full = np.stack(stacked_matrices,axis=0)
            print(full.shape)
            U, S, Vh = np.linalg.svd(full,full_matrices=True)
            print("shapes:")
            print(U.shape)
            # print(U[:,:,0])
            print(S.shape)
            # print(S[:,0])
            print(Vh.shape)
            new_detection_statistic = np.mean(S[:,3:5],axis=1)
            print("after exclusion:",new_detection_statistic)

            _, bonus_sings, _ = np.linalg.svd(gram_removed(D,fault_suspects))

            # hi

            plt.figure()
            for s_index in range(len(S)):
                plt.scatter(list(range(S.shape[1])),S[s_index,:],
                            label=str(fault_suspects[s_index])+" removed")
            plt.scatter(list(range(len(original_sings))),original_sings,label="original")
            plt.scatter(list(range(len(bonus_sings))),bonus_sings,label="all")
            plt.yscale("log")
            plt.legend()
            plt.show()


            # plt.scatter(range(len(S)),S)
            # plt.yscale("log")
            # plt.show()
            data_45means.append(np.mean(S[3:5]))

            break


        # while True:
        #
        #     if ri != None:
        #         if verbose:
        #             print("removing index: ",ri)
        #
        #         if isinstance(ri,np.int64):
        #             # add removed index to index list passed back
        #             tri.append(oi[ri]-1)
        #             # keep track of original indexes (since deleting)
        #             oi = np.delete(oi,ri)
        #             ri = [ri]
        #         else:
        #             for ri_val in ri:
        #                 oi = np.delete(oi,ri_val-1)
        #
        #         for ri_val in ri:
        #
        #             # remove index from EDM
        #             D = np.delete(D,ri_val,axis=0)
        #             D = np.delete(D,ri_val,axis=1)
        #
        #     # EDM FDE
        #     n = D.shape[0]  # shape of EDM
        #     # stop removing indexes either b/c you need at least four
        #     # satellites or if maximum number of faults has been reached
        #     if n <= 5 or (max_faults != None and len(tri) >= max_faults):
        #         break
        #
        #     # double center EDM to retrive the corresponding Gram matrix
        #     J = np.eye(n) - (1./n)*np.ones((n,n))
        #     G = -0.5*J.dot(D).dot(J)
        #
        #     try:
        #         # perform singular value decomposition
        #         U, S, Vh = np.linalg.svd(G)
        #     except Exception as exception:
        #         if verbose:
        #             print(exception)
        #         break
        #
        #     # calculate detection test statistic
        #     warn = S[dims]*(sum(S[dims:])/float(len(S[dims:])))/S[0]
        #     if verbose:
        #         print("\nDetection test statistic:",warn)
        #
        #     if warn > threshold:
        #         ri = None
        #
        #         u_mins = set(np.argsort(U[:,dims])[:2])
        #         u_maxes = set(np.argsort(U[:,dims])[-2:])
        #         v_mins = set(np.argsort(Vh[dims,:])[:2])
        #         v_maxes = set(np.argsort(Vh[dims,:])[-2:])
        #
        #         def test_option(ri_option):
        #             # remove option
        #             D_opt = np.delete(D.copy(),ri_option,axis=0)
        #             D_opt = np.delete(D_opt,ri_option,axis=1)
        #
        #             # reperform double centering to obtain Gram matrix
        #             n_opt = D_opt.shape[0]
        #             J_opt = np.eye(n_opt) - (1./n_opt)*np.ones((n_opt,n_opt))
        #             G_opt = -0.5*J_opt.dot(D_opt).dot(J_opt)
        #
        #             # perform singular value decomposition
        #             _, S_opt, _ = np.linalg.svd(G_opt)
        #
        #             # calculate detection test statistic
        #             warn_opt = S_opt[dims]*(sum(S_opt[dims:])/float(len(S_opt[dims:])))/S_opt[0]
        #
        #             return warn_opt
        #
        #
        #         # get all potential options
        #         ri_options = u_mins | v_mins | u_maxes | v_maxes
        #         # remove the receiver as a potential fault
        #         ri_options = ri_options - set([reci])
        #         ri_tested = []
        #         ri_warns = []
        #
        #         ui = -1
        #         while np.argsort(np.abs(U[:,dims]))[ui] in ri_options:
        #             ri_option = np.argsort(np.abs(U[:,dims]))[ui]
        #
        #             # calculate test statistic after removing index
        #             warn_opt = test_option(ri_option)
        #
        #             # break if test statistic decreased below threshold
        #             if warn_opt < threshold:
        #                 ri = ri_option
        #                 if verbose:
        #                     print("chosen ri: ", ri)
        #                 break
        #             else:
        #                 ri_tested.append(ri_option)
        #                 ri_warns.append(warn_opt)
        #             ui -= 1
        #
        #         # continue searching set if didn't find index
        #         if ri == None:
        #             ri_options_left = list(ri_options - set(ri_tested))
        #
        #             for ri_option in ri_options_left:
        #                 warn_opt = test_option(ri_option)
        #
        #                 if warn_opt < threshold:
        #                     ri = ri_option
        #                     if verbose:
        #                         print("chosen ri: ", ri)
        #                     break
        #                 else:
        #                     ri_tested.append(ri_option)
        #                     ri_warns.append(warn_opt)
        #
        #         # if no faults decreased below threshold, then remove the
        #         # index corresponding to the lowest test statistic value
        #         if ri == None:
        #             idx_best = np.argmin(np.array(ri_warns))
        #             ri = ri_tested[idx_best]
        #             if verbose:
        #                 print("chosen ri: ", ri)
        #     else:
        #         break

        fault_edm_subset = np.array([0] * len(navdata_subset))
        fault_edm_subset[fault_indexes] = 1
        fault_edm_subset[nan_indexes] = 2
        fault_edm += list(fault_edm_subset)

    data["data_45means"] = data_45means

    navdata["fault_edm"] = fault_edm
    if verbose:
        print(navdata["fault_edm"])

    return navdata, data


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
        nan_indexes = sorted(list(set(np.arange(subset_length)[np.isnan(sv_m).any(axis=0)]).union( \
                                  set(np.arange(subset_length)[np.isnan(corr_pr_m)]).union( \
                                  ))))[::-1]

        original_indexes = np.arange(subset_length)
        # remove NaN indexes
        navdata_subset.remove(cols=nan_indexes,inplace=True)
        original_indexes = np.delete(original_indexes, nan_indexes)

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
        fault_residual_subset[nan_indexes] = 2
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
        nan_indexes = sorted(list(set(np.arange(subset_length)[np.isnan(sv_m).any(axis=0)]).union( \
                                  set(np.arange(subset_length)[np.isnan(corr_pr_m)]).union( \
                                  ))))[::-1]

        # fault_ss_subset = np.array([0] * len(navdata_subset))
        # fault_ss_subset[tri] = 1
        # fault_ss_subset[nan_indexes] = 2
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
        nan_indexes = sorted(list(set(np.arange(nsl)[np.isnan(sv_m).any(axis=0)]).union( \
                                  set(np.arange(nsl)[np.isnan(corr_pr_m)]).union( \
                                  ))))[::-1]

        D = _edm_from_satellites_ranges(sv_m,corr_pr_m)

        # add one to account for the receiver in D
        ri = list(np.array(nan_indexes)+1)  # index to remove
        tri = nan_indexes.copy()            # removed indexes (in transmitter frame)
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
        fault_edm_subset[nan_indexes] = 2
        fault_edm += list(fault_edm_subset)

    navdata["fault_edm"] = fault_edm
    if verbose:
        print(navdata["fault_edm"])

    return navdata

def evaluate_fde(navdata, method, fault_truth_row, verbose=True,
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
    solve_fde(navdata, method=method, remove_outliers=False, **kwargs)

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
