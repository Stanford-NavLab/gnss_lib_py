"""Functions useful to perform dgnss operations.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "20 Jul 2022"

from datetime import datetime, timedelta, timezone
import numpy as np

import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.sim_gnss import find_sat
from gnss_lib_py.parsers.ephemeris import EphemerisManager

def compute_snapshot_dgpscorr(ephem, gps_week, raw_pr_m, gps_tow, x_gt_m, y_gt_m, z_gt_m):
    """Compute pseudorange corrections based on differential GPS concept via use of
    the presurveyed ground truth position of publicly-available, post-processed
    base station data.

    Parameters
    ----------
    ephem : pd.DataFrame
        DataFrame containing ephemeris parameters of satellites for which states
        are required
    gpsweek : int
        Week of GPS calendar corresponding to time of clock
    gps_tow : ndarray
        GPS time of the week at which diferential GPS corrections are to be
        calculated for desired satellites [s]
    x_gt_m : ndarray
        X-ECEF ground truth positions for a static base station [m]
    y_gt_m : ndarray
        Y-ECEF ground truth positions for a static base station [m]
    z_gt_m : ndarray
        Z-ECEF ground truth positions for a static base station [m]

    Returns
    -------
    delta_pr_m : ndarray
        Differential GPS corrections to be added for desired satellites

    Notes
    -----
    TOCLARIFY:
    (1) Not sure if it is odd to have arrays of gps_tow, x_gt_m, y_gt_m,
    z_gt_m which are basically identical arrays. But the reason I still have
    them like arrays is because they are independently extracted from rinex2
    NavData object
    Added assertions to check that they are indeed equal
    (2) Parts of this function can be replaced with find_sv_location of
    sim_gnss: which is why not writing unit tests for this, in case it is
    a repeat of what is already written
    (3) R0913: Too many arguments (7/5) (too-many-arguments)

    TODO:
    (1) corner cases not handled, like gps_week is expected to be a single
    value, cannot be applicable for dataset in transition across dates
    (2) assumption that gps_week of derived and base station does not
    change over entire dataset

    """
    if not all(x == gps_tow[0] for x in gps_tow):
        raise RuntimeError("Elements in GPS time of week are not equal")
    if not all(x == x_gt_m[0] for x in x_gt_m):
        raise RuntimeError("Elements in x_gt_m are not equal")
    if not all(x == y_gt_m[0] for x in y_gt_m):
        raise RuntimeError("Elements in y_gt_m are not equal")
    if not all(x == z_gt_m[0] for x in z_gt_m):
        raise RuntimeError("Elements in z_gt_m are not equal")
    if not len(ephem) == len(gps_tow) == len(x_gt_m) \
                      == len(y_gt_m) == len(z_gt_m) :
        raise RuntimeError("Lengths of input arrays do not match")

    trans_time = raw_pr_m / consts.C
    get_sat_from_ephem = find_sat(ephem, gps_tow - trans_time, gps_week)
    sat_xyz = np.transpose([get_sat_from_ephem.x.values, \
                            get_sat_from_ephem.y.values, \
                            get_sat_from_ephem.z.values])
#     print(satXYZ_ephemeris)
    del_x = (consts.OMEGA_E_DOT * sat_xyz[:,1] * trans_time)
    del_y = (-consts.OMEGA_E_DOT * sat_xyz[:,0] * trans_time)
    sat_xyz[:,0] = sat_xyz[:,0] + del_x
    sat_xyz[:,1] = sat_xyz[:,1] + del_y
    print(sat_xyz)

    base_gtruth = np.vstack(( x_gt_m, y_gt_m, z_gt_m ))
    base_gtruth = np.transpose(base_gtruth)
    exp_obs_pseudo = np.linalg.norm(base_gtruth - sat_xyz, axis=1)
    delta_pr_m = exp_obs_pseudo - raw_pr_m

    return delta_pr_m

def compute_all_dgpscorr(derived, base):
    """Compute pseudorange corrections for derived object across all time stamps
    using differential GPS concept, i.e., via base object

    Parameters
    ----------
    derived : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class that depicts android derived dataset
    base : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class that depicts base station dataset

    Returns
    -------
    derived : gnss_lib_py.parsers.navdata.NavData
        Modified instance of the NavData class that computes corrected
        pseudoranges using differential GPS

    Notes
    -----
    TOCLARIFY:
    (1) unique_gnss_id_str I just hard-coded it as 'G' is there a way to
    automate it based on current gnss_lib_py structure.
    (2) Check if derived object should by-default initialize with
    corr_pr_m or it should be nan.
    (3) R0914: Too many local variables (21/15) (too-many-locals)

    TODO:
    (2) The main bottleneck for this being differential GPS and not
    differential GNSS is because satellite ephemeris can only be extracted
    for GPS and not other constellation (in a non-android derived way).
    Once SP3 files are merged to main, I can update these codes to allow
    L1 multi-constellation.

    """
    if all(x == derived['gnss_id',0][0] for x in derived['gnss_id'][0]):
        unique_gnss_id = derived['gnss_id', 0][0]
    else:
        raise RuntimeError("No multi-GNSS capability yet")

    if unique_gnss_id == 1.0:
        unique_gnss_id_str = 'G'
    else:
        raise RuntimeError("No non-GPS capability yet")

    if all(x == derived['gps_week',0][0] for x in derived['gps_week'][0]):
        unique_gps_week = derived['gps_week', 0][0]
    else:
        raise RuntimeError("Cannot handle GPS week transition")

    print(unique_gnss_id, unique_gps_week)

    repo = EphemerisManager()
    unique_timesteps = np.unique(derived["gps_tow"])

    for t_idx, timestep in enumerate(unique_timesteps):
        derived_idxs = np.where(derived["gps_tow"] == timestep)[1]

        min_idx = np.argmin(abs(base['gps_tow'][0] - timestep))
        base_idxs = np.where( (base['gps_tow'] == base['gps_tow', min_idx][0]) & \
                                          (base['gnss_id'] == unique_gnss_id) )[1]
        print(t_idx, timestep, derived_idxs, min_idx, base_idxs)
        print(derived['sv_id', derived_idxs], base['sv_id', base_idxs])

        # base_remove_idxs = np.where(np.isnan(base['raw_pr_m', base_idxs]))
        # base_avail_svs = base['sv_id', base_idxs]
        # if np.size(base_remove_idxs)>0:
        #     base_avail_svs = np.delete(base_avail_svs, base_remove_idxs)

        common_svs, \
        derived_local_idxs, \
        base_local_idxs = np.intersect1d(derived['sv_id', derived_idxs], \
                                                     base['sv_id', base_idxs], \
                                                     return_indices=True)
        print(common_svs, derived_local_idxs, base_local_idxs)

        derived_global_idxs = derived_idxs[derived_local_idxs]
        base_global_idxs = base_idxs[base_local_idxs]
        print(derived['raw_pr_m', derived_global_idxs], \
              base['raw_pr_m', base_global_idxs])

        rxdatetime = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc) + \
                     timedelta(seconds=(unique_gps_week * 7 * 86400 + timestep))
        desired_sats = [unique_gnss_id_str + str(int(i)).zfill(2) for i in common_svs]
        print('rxdatetime and desired sats: ', rxdatetime, desired_sats)
        ephem = repo.get_ephemeris(rxdatetime, satellites=desired_sats)
        print('checking for base station consistency!')
        print(base['gnss_id', base_global_idxs], \
              base['gps_tow', base_global_idxs], \
              base['sv_id', base_global_idxs], \
              base['raw_pr_m', base_global_idxs])
        delta_pr_m = compute_snapshot_dgpscorr(ephem, unique_gps_week, \
                                               base['raw_pr_m', base_global_idxs], \
                                               base['gps_tow', base_global_idxs], \
                                               base['x_gt_m', base_global_idxs], \
                                               base['y_gt_m', base_global_idxs], \
                                               base['z_gt_m', base_global_idxs])
        print('delta_pr_m: ', delta_pr_m)
        print('before: ', delta_pr_m +\
              derived['raw_pr_m', derived_global_idxs] - \
              derived['corr_pr_m', derived_global_idxs])

        derived['corr_pr_m', derived_global_idxs] = delta_pr_m + \
                                                    derived['raw_pr_m', derived_global_idxs]
        print('after: ', delta_pr_m + \
              derived['raw_pr_m', derived_global_idxs] - \
              derived['corr_pr_m', derived_global_idxs])


        print(' ')

    return derived
