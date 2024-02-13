"""Classes for External Filter implementations

"""

__authors__ = "Shubh Gupta"
__date__ = "25 Jan 2020"

import warnings

import numpy as np

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import loop_time
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.utils.coordinates import ecef_to_geodetic
from abc import ABC, abstractmethod
from gnss_lib_py.utils.filters import _check_col_vect, _check_square_mat

def solve_gnss_pf(measurements, init_dict = None,
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
        Dictionary of parameters for GNSS pf.

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
        process_noise = 0.0*np.eye(init_dict["state_0"].size)
        process_noise[:3, :3] = 100*np.eye(3)
        init_dict["Q"] = process_noise

    if "R" not in init_dict:
        measurement_noise = np.eye(1) # gets overwritten
        init_dict["R"] = measurement_noise

    # initialize parameter dictionary
    if params_dict is None:
        params_dict = {}

    if "motion_type" not in params_dict:
        params_dict["motion_type"] = "stationary"

    if "measure_type" not in params_dict:
        params_dict["measure_type"] = "pseudorange"

    if "num_particles" not in params_dict:
        params_dict["num_particles"] = 10000

    # create initialization parameters.
    gnss_pf = GNSSPF(init_dict, params_dict)

    states = []

    for timestamp, delta_t, measurement_subset in loop_time(measurements,"gps_millis"):
        pos_sv_m = measurement_subset[["x_sv_m","y_sv_m","z_sv_m"]].T
        pos_sv_m = np.atleast_2d(pos_sv_m)

        corr_pr_m = measurement_subset["corr_pr_m"].reshape(-1,1)

        # remove NaN indexes
        # not_nan_indexes = ~np.isnan(pos_sv_m).any(axis=1)
        not_nan_indexes = ~np.isnan(pos_sv_m).any(axis=1) & ~np.isnan(corr_pr_m).any(axis=1)
        pos_sv_m = pos_sv_m[not_nan_indexes]
        corr_pr_m = corr_pr_m[not_nan_indexes]

        # prediction step
        predict_dict = {"delta_t" : delta_t}
        gnss_pf.predict(predict_dict=predict_dict)

        # update step
        update_dict = {"pos_sv_m" : pos_sv_m.T}
        update_dict["measurement_noise"] = np.eye(pos_sv_m.shape[0])
        gnss_pf.update(corr_pr_m, update_dict=update_dict)

        states.append([timestamp] + np.squeeze(gnss_pf.state).tolist())

    states = np.array(states)

    if states.size == 0:
        warnings.warn("No valid state estimate computed in solve_gnss_pf, "\
                    + "returning None.", RuntimeWarning)
        return None

    state_estimate = NavData()
    state_estimate["gps_millis"] = states[:,0]
    state_estimate["x_rx_pf_m"] = states[:,1]
    state_estimate["y_rx_pf_m"] = states[:,2]
    state_estimate["z_rx_pf_m"] = states[:,3]
    state_estimate["vx_rx_pf_mps"] = states[:,4]
    state_estimate["vy_rx_pf_mps"] = states[:,5]
    state_estimate["vz_rx_pf_mps"] = states[:,6]
    state_estimate["b_rx_pf_m"] = states[:,7]

    lat,lon,alt = ecef_to_geodetic(state_estimate[["x_rx_pf_m",
                                                   "y_rx_pf_m",
                                                   "z_rx_pf_m"]].reshape(3,-1))
    state_estimate["lat_rx_pf_deg"] = lat
    state_estimate["lon_rx_pf_deg"] = lon
    state_estimate["alt_rx_pf_m"] = alt

    return state_estimate

class GNSSPF(ABC):
    """GNSS-only PF implementation.

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
        super().__init__()

        self.delta_t = params_dict.get('dt',1.0)
        self.motion_type = params_dict.get('motion_type','stationary')
        self.measure_type = params_dict.get('measure_type','pseudorange')
        self.num_particles = params_dict.get('num_particles',1000) # N
        self.particles_dim = init_dict['state_0'].size # d
        self.state = init_dict['state_0']
        self.sigma = init_dict['sigma_0']
        # Store cholesky decomposition of Q and R
        self.Q = init_dict['Q']
        self.R = init_dict['R']
        self.Q_cholesky = self.stable_cholesky(self.Q)

        # State shaped d x N distributed by sigma around state_0
        self.init_state(init_dict['state_0'])

    def init_state(self, state_0):
        """Initialize the state of the filter"""
        sigma_cholesky = self.stable_cholesky(self.sigma)
        self.particles = state_0 + sigma_cholesky @ np.random.randn(self.particles_dim, self.num_particles)
        self.log_wt = np.zeros(self.num_particles)

    def stable_cholesky(self, Q):
        n = Q.shape[0]
        # Initialize the Q_cholesky matrix with zeros
        Q_cholesky = np.zeros_like(Q)

        # Find the rank of the non-zero leading square submatrix
        # This is effectively the size of the submatrix to decompose
        rank = 0
        for i in range(n):
            if not np.isclose(Q[i, i], 0, atol=1e-8):
                rank += 1
            else:
                break  # Stop at the first zero diagonal entry

        # Perform Cholesky decomposition on the non-zero submatrix if rank > 0
        if rank > 0:
            Q_nonzero_submatrix = Q[:rank, :rank]
            # Directly assign the decomposed submatrix to the corresponding part of Q_cholesky
            Q_cholesky[:rank, :rank] = np.linalg.cholesky(Q_nonzero_submatrix)
        
        return Q_cholesky


    def predict(self, u=None, predict_dict=None):
        """Predict the state of the filter given the control input

        Parameters
        ----------
        u : np.ndarray
            The control signal given to the actual system, dimension state_dim x D
        predict_dict : dict
            Additional parameters needed to implement predict step
        """
        if u is None:
            u = np.zeros((self.particles_dim,1))
        if predict_dict is None:
            predict_dict = {}
        assert _check_col_vect(u, np.size(u)), "Control input is not a column vector"
        self.particles = self.dyn_model(u, predict_dict)  # Can pass parameters via predict_dict
        self.state, self.sigma = self.empirical_covariance()
        assert self.particles.shape==(self.particles_dim, self.num_particles), "Incorrect state shape after prediction"
        assert _check_square_mat(self.sigma, self.particles_dim), "Incorrect covariance shape after prediction"

    def update(self, z, update_dict=None):
        """Update the state of the filter given a noisy measurement of the state

        Parameters
        ----------
        z : np.ndarray
            Noisy measurement of state, dimension M x 1
        update_dict : dict
            Additional parameters needed to implement update step
        """
        if update_dict is None:
            update_dict = {}

        # uses process_noise from update_dict if exists, otherwise use
        # process_noise from the class initialization.
        measurement_noise = update_dict.get('measurement_noise', self.R)
        R_cholesky = np.linalg.cholesky(measurement_noise)
        assert _check_col_vect(z, np.size(z)), "Measurements are not a column vector"
        if self.measure_type == 'pseudorange':
            pos_sv_m = update_dict['pos_sv_m']
            pseudo_expect = self.measure_model(update_dict) # M x N
            residual = np.linalg.solve(R_cholesky, (z - pseudo_expect)) # M x N
        else: # pragma: no cover
            raise NotImplementedError
        
        # Updating log weights
        self.log_wt = self.log_wt - np.sum(np.square(residual), axis=0)
        self.log_wt = self.log_wt - np.max(self.log_wt)
        
        # # Check effective size and resample particles
        # effective_sample_size = self.calc_effective_sample_size()
        # if effective_sample_size < 0.5:
        #     self.resample_particles()

        # Resample every time
        self.resample_particles()

        # Update covariance
        self.state, self.sigma = self.empirical_covariance()
        assert self.particles.shape==(self.particles_dim, self.num_particles), "Incorrect state shape after update"
        assert _check_square_mat(self.sigma, self.particles_dim), "Incorrect covariance shape after update"

    def calc_effective_sample_size(self):
        """Calculate the effective particle size of the filter

        Returns
        -------
        effective_sample_size : float
            Effective particle size of the filter
        """
        weights = np.exp(self.log_wt - np.max(self.log_wt))
        effective_sample_size = 1.0/np.sum(weights**2)
        # print('effective_sample_size ', effective_sample_size)
        return effective_sample_size

    def resample_particles(self):
        """
        Resamples particles based on their log weights.

        Parameters:
            particles (ndarray): Array of particles of shape (N, D) where N is the number of particles and D is the dimensionality of each particle.
            log_weights (ndarray): Array of log weights of shape (N,).

        Returns:
            resampled_particles (ndarray): Resampled particles.
        """
        # Calculate weights from log weights
        weights = np.exp(self.log_wt - np.max(self.log_wt))  # Subtracting max for numerical stability

        # Resample indices
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, replace=True, p=weights/np.sum(weights))

        # Resample particles
        self.particles = self.particles[:, indices]
        self.log_wt = np.zeros(self.num_particles)


    def dyn_model(self, u, predict_dict=None):
        """Nonlinear dynamics (stochastic)

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

        # uses delta_t from predict_dict if exists, otherwise delta_t
        # from the class initialization.
        delta_t = predict_dict.get('delta_t', self.delta_t)

        if self.motion_type == 'stationary':
            A = np.eye(7) # Assumes d is 7
        elif self.motion_type == 'constant_velocity':
            A = np.eye(7)
            A[:3, -4:-1] = delta_t*np.eye(3)
        else: # pragma: no cover
            raise NotImplementedError
        
        # Create new_x by matrix multiplying A and state and adding a random noise according to Q
        new_x = A @ self.particles + self.Q_cholesky @ np.random.randn(self.particles_dim, self.num_particles)

        # d X N
        return new_x

    def measure_model(self, update_dict):
        """Measurement model (vectorized)

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
            pos_sv_m = update_dict['pos_sv_m'] # 3 x M
            
            pseudo = np.sqrt((self.particles[0, :, None] - pos_sv_m[0, :])**2
                           + (self.particles[1, :, None] - pos_sv_m[1, :])**2
                           + (self.particles[2, :, None] - pos_sv_m[2, :])**2) \
                           + self.particles[6, :, None]     # N x M
            z = pseudo.T    # M x N
        else: #pragma: no cover
            raise NotImplementedError
        return z

    def empirical_covariance(self):
        """Empirical covariance of the filter, computed from state samples

        Returns
        -------
        C : np.ndarray
        """
        # Convert log weights to weights
        weights = np.exp(self.log_wt - np.max(self.log_wt))  # Subtracting max for numerical stability

        # Compute weighted mean
        weighted_mean = np.average(self.particles, axis=1, weights=weights).reshape(-1, 1)

        # Compute weighted covariance
        weighted_diff = self.particles - weighted_mean
        cov = np.dot(weighted_diff, (weighted_diff * weights[None, :]).T) / weights.sum()

        return weighted_mean, cov
