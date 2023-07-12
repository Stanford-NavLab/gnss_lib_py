"""Fault Detection and Exclusion (FDE) methods for GNSS applications.

"""

__authors__ = "D. Knowles"
__date__ = "11 Jul 2023"

import numpy as np

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
        Result includes a new row of ``fault_<method>`` which has a
        boolean 0/1 value where 1 indicates a fault and 0 indicates that
        no fault was detected.

    """

    if method == "residual":
        navdata = fde_residual(navdata, max_faults, threshold,
                               **kwargs)
    elif method == "ss":
        navdata = fde_solution_separation(navdata, max_faults, threshold,
                                          **kwargs)
    elif method == "edm":
        navdata = fde_edm(navdata, max_faults, threshold,
                          **kwargs)
    else:
        raise ValueError("invalid method input for solve_fde()")

    if remove_outliers:
        navdata = navdata.where("fault_" + method, False)

    return navdata


def fde_residual(navdata, max_faults, threshold):
    """Residual-based fault detection and exclusion.

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

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Result includes a new row of ``fault_residual`` which has a
        boolean 0/1 value where 1 indicates a fault and 0 indicates that
        no fault was detected.

    """

    navdata["fault_residual"] = 0

    return navdata

def fde_solution_separation(navdata, max_faults, threshold):
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

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Result includes a new row of ``fault_ss`` which has a
        boolean 0/1 value where 1 indicates a fault and 0 indicates that
        no fault was detected.

    """

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
        Result includes a new row of ``fault_edm`` which has a
        boolean 0/1 value where 1 indicates a fault and 0 indicates that
        no fault was detected.

    References
    ----------
    ..  [1] D. Knowles and G. Gao. "Euclidean Distance Matrix-based
        Rapid Fault Detection and Exclusion." ION GNSS+ 2021.


    """

    fault_edm = []
    dims = 3
    if threshold is None:
        threshold = 1000

    for timestep, _, navdata_subset in navdata.loop_time('gps_millis'):

        # navdata_subset = navdata_subset.where("signal_type",("l5","e5a"),"neq")
        # print()

        nsl = len(navdata_subset)
        if nsl < 5:
            break

        sv_m = navdata_subset[["x_sv_m","y_sv_m","z_sv_m"]]
        corr_pr_m = navdata_subset["corr_pr_m"]

        # remove NaN indexes
        nan_indexes = sorted(list(set(np.arange(nsl)[np.isnan(sv_m).any(axis=0)]).union( \
                                  set(np.arange(nsl)[np.isnan(corr_pr_m)]).union( \
                                  set(navdata_subset.argwhere("signal_type",
                                  ("l5","e5a","nan")))))))[::-1]

        D = _edm_from_satellites_ranges(sv_m,corr_pr_m)

        # add one to account for the receiver in D
        ri = list(np.array(nan_indexes)+1)  # index to remove
        tri = nan_indexes                   # removed indexes (in transmitter frame)
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
        fault_edm += list(fault_edm_subset)

    navdata["fault_edm"] = fault_edm

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
