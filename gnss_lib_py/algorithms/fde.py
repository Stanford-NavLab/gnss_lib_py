"""Fault Detection and Exclusion (FDE) methods for GNSS applications.

"""

__authors__ = "D. Knowles"
__date__ = "11 Jul 2023"

import numpy as np

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
        navdata = fde_residual(navdata, max_faults=max_faults,
                                        threshold=threshold,
                                        **kwargs)
    elif method == "ss":
        navdata = fde_solution_separation(navdata, max_faults=max_faults,
                                                   threshold=threshold,
                                                   **kwargs)
    elif method == "edm":
        navdata = fde_edm(navdata, max_faults=max_faults,
                                   threshold=threshold,
                                   **kwargs)
    else:
        raise ValueError("invalid method input for solve_fde()")

    if remove_outliers:
        navdata = navdata.where("fault_" + method, False)

    return navdata


def fde_residual(navdata, receiver_state, max_faults, threshold,
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

def fde_solution_separation(navdata, max_faults, threshold,
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

def fde_edm(navdata, max_faults, threshold=1.0, verbose=False):
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
