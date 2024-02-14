"""Classes for GNSS-based Kalman Filter implementations

"""

__authors__ = "Ashwin Kanhere, Derek Knowles"
__date__ = "25 Jan 2020"

import warnings

import numpy as np

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import loop_time
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.utils.coordinates import ecef_to_geodetic
from gnss_lib_py.utils.filters import BaseExtendedKalmanFilter

def solve_gnss_ekf(measurements, init_dict = None,
                   params_dict = None, delta_t_decimals=-2):
    """Runs a GNSS Extended Kalman Filter across each timestep.

    Runs an Extended Kalman Filter across each timestep and adds a new
    row for the receiver's position and clock bias.

    Parameters
    ----------
    measurements : gnss_lib_py.navdata.navdata.NavData
        Instance of the NavData class
    init_dict : dict
        Initialization dict with initial states and covariances.
    params_dict : dict
        Dictionary of parameters for GNSS EKF.

    Returns
    -------
    state_estimate : gnss_lib_py.navdata.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """

    # check that all necessary rows exist
    measurements.in_rows(["gps_millis","corr_pr_m",
                          "x_sv_m","y_sv_m","z_sv_m",
                          ])

    if init_dict is None:
        init_dict = {}

    if "state_0" not in init_dict:
        pos_0 = None
        for _, _, measurement_subset in loop_time(measurements,"gps_millis",
                                        delta_t_decimals=delta_t_decimals):
            pos_0 = solve_wls(measurement_subset)
            if pos_0 is not None:
                break

        state_0 = np.zeros((7,1))
        if pos_0 is not None:
            state_0[:3,0] = pos_0[["x_rx_wls_m","y_rx_wls_m","z_rx_wls_m"]]
            state_0[6,0] = pos_0[["b_rx_wls_m"]]

        init_dict["state_0"] = state_0

    if "sigma_0" not in init_dict:
        sigma_0 = np.eye(init_dict["state_0"].size)
        init_dict["sigma_0"] = sigma_0

    if "Q" not in init_dict:
        process_noise = np.eye(init_dict["state_0"].size)
        init_dict["Q"] = process_noise

    if "R" not in init_dict:
        measurement_noise = np.eye(1) # gets overwritten
        init_dict["R"] = measurement_noise

    # initialize parameter dictionary
    if params_dict is None:
        params_dict = {}

    if "motion_type" not in params_dict:
        params_dict["motion_type"] = "constant_velocity"

    if "measure_type" not in params_dict:
        params_dict["measure_type"] = "pseudorange"

    # create initialization parameters.
    gnss_ekf = GNSSEKF(init_dict, params_dict)

    states = []

    for timestamp, delta_t, measurement_subset in loop_time(measurements,"gps_millis"):
        pos_sv_m = measurement_subset[["x_sv_m","y_sv_m","z_sv_m"]].T
        pos_sv_m = np.atleast_2d(pos_sv_m)

        corr_pr_m = measurement_subset["corr_pr_m"].reshape(-1,1)

        # remove NaN indexes
        not_nan_indexes = ~np.isnan(pos_sv_m).any(axis=1) & ~np.isnan(corr_pr_m).any(axis=1)
        pos_sv_m = pos_sv_m[not_nan_indexes]
        corr_pr_m = corr_pr_m[not_nan_indexes]

        # prediction step
        predict_dict = {"delta_t" : delta_t}
        gnss_ekf.predict(predict_dict=predict_dict)

        # update step
        update_dict = {"pos_sv_m" : pos_sv_m.T}
        update_dict["measurement_noise"] = np.eye(pos_sv_m.shape[0])
        gnss_ekf.update(corr_pr_m, update_dict=update_dict)

        states.append([timestamp] + np.squeeze(gnss_ekf.state).tolist())

    states = np.array(states)

    if states.size == 0:
        warnings.warn("No valid state estimate computed in solve_gnss_ekf, "\
                    + "returning None.", RuntimeWarning)
        return None

    state_estimate = NavData()
    state_estimate["gps_millis"] = states[:,0]
    state_estimate["x_rx_ekf_m"] = states[:,1]
    state_estimate["y_rx_ekf_m"] = states[:,2]
    state_estimate["z_rx_ekf_m"] = states[:,3]
    state_estimate["vx_rx_ekf_mps"] = states[:,4]
    state_estimate["vy_rx_ekf_mps"] = states[:,5]
    state_estimate["vz_rx_ekf_mps"] = states[:,6]
    state_estimate["b_rx_ekf_m"] = states[:,7]

    lat,lon,alt = ecef_to_geodetic(state_estimate[["x_rx_ekf_m",
                                                   "y_rx_ekf_m",
                                                   "z_rx_ekf_m"]].reshape(3,-1))
    state_estimate["lat_rx_ekf_deg"] = lat
    state_estimate["lon_rx_ekf_deg"] = lon
    state_estimate["alt_rx_ekf_m"] = alt

    return state_estimate

class GNSSEKF(BaseExtendedKalmanFilter):
    """GNSS-only EKF implementation.

    States: 3D position, 3D velocity and clock bias (in m).
    The state vector is :math:`\\bar{x} = [x, y, z, v_x, v_y, v_y, b]^T`

    Attributes
    ----------
    params_dict : dict
        Dictionary of parameters that may include the following.
    delta_t : float
        Time between prediction instances
    motion_type : string
        Type of motion (``stationary`` or ``constant_velocity``)
    measure_type : string
        NavData types (pseudorange)
    """
    def __init__(self, init_dict, params_dict):
        super().__init__(init_dict, params_dict)

        self.delta_t = params_dict.get('dt',1.0)
        self.motion_type = params_dict.get('motion_type','stationary')
        self.measure_type = params_dict.get('measure_type','pseudorange')

    def dyn_model(self, u, predict_dict=None):
        """Nonlinear dynamics

        Parameters
        ----------
        u : np.ndarray
            Control signal, not used for propagation
        predict_dict : dict
            Additional prediction parameters, including ``delta_t``
            updates.

        Returns
        -------
        new_x : np.ndarray
            Propagated state
        """
        if predict_dict is None: #pragma: no cover
            predict_dict = {}

        A = self.linearize_dynamics(predict_dict)
        new_x = A @ self.state
        return new_x

    def measure_model(self, update_dict):
        """Measurement model

        Pseudorange model adds true range and clock bias estimate:
        :math:`\\rho = \\sqrt{(x-x_{sv})^2 + (y-y_{sv})^2 + (z-z_{sv})^2} + b`.
        See [1]_ for more details and models.

        ``pos_sv_m`` must be a key in update_dict and must be an array
        of shape [3 x N] with rows of x_sv_m, y_sv_m, and z_sv_m in that
        order.

        Parameters
        ----------
        update_dict : dict
            Update dictionary containing satellite positions with key
            ``pos_sv_m``.

        Returns
        -------
        z : np.ndarray
            Expected measurement, depending on type (pseudorange)
        References
        ----------
        .. [1] Morton, Y. Jade, Frank van Diggelen, James J. Spilker Jr,
            Bradford W. Parkinson, Sherman Lo, and Grace Gao, eds.
            Position, navigation, and timing technologies in the 21st century:
            integrated satellite navigation, sensor systems, and civil
            applications. John Wiley & Sons, 2021.
        """
        if self.measure_type=='pseudorange':
            pos_sv_m = update_dict['pos_sv_m']
            pseudo = np.sqrt((self.state[0] - pos_sv_m[0, :])**2
                           + (self.state[1] - pos_sv_m[1, :])**2
                           + (self.state[2] - pos_sv_m[2, :])**2) \
                           + self.state[6]
            z = np.reshape(pseudo, [-1, 1])
        else: #pragma: no cover
            raise NotImplementedError
        return z

    def linearize_dynamics(self, predict_dict=None):
        """Linearization of dynamics model

        Parameters
        ----------
        predict_dict : dict
            Additional predict parameters, not used in current implementation

        Returns
        -------
        A : np.ndarray
            Linear dynamics model depending on motion_type
        predict_dict : dict
            Dictionary of prediction parameters.
        """

        if predict_dict is None: # pragma: no cover
            predict_dict = {}

        # uses delta_t from predict_dict if exists, otherwise delta_t
        # from the class initialization.
        delta_t = predict_dict.get('delta_t', self.delta_t)

        if self.motion_type == 'stationary':
            A = np.eye(7)
        elif self.motion_type == 'constant_velocity':
            A = np.eye(7)
            A[:3, -4:-1] = delta_t*np.eye(3)
        else: # pragma: no cover
            raise NotImplementedError
        return A

    def linearize_measurements(self, update_dict):
        """Linearization of measurement model

        Parameters
        ----------
        update_dict : dict
            Update dictionary containing satellite positions with key
            ``pos_sv_m``.

        Returns
        -------
        H : np.ndarray
            Jacobian of measurement model, of dimension
            #measurements x #states
        """
        if self.measure_type == 'pseudorange':
            pos_sv_m = update_dict['pos_sv_m']
            m = np.shape(pos_sv_m)[1]
            H = np.zeros([m, self.state_dim])
            pseudo_expect = self.measure_model(update_dict)
            rx_pos = np.reshape(self.state[:3], [-1, 1])
            H[:, :3] = (rx_pos - pos_sv_m).T/pseudo_expect
            H[:, 6] = 1
        else: # pragma: no cover
            raise NotImplementedError
        return H
