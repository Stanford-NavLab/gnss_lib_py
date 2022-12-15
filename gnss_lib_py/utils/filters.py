"""Parent classes for Kalman filter algorithms.

"""

__authors__ = "Ashwin Kanhere, Shivam Soni"
__date__ = "20 January 2020"

from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import sqrtm


class BaseFilter(ABC):
    """Class with general filter implementation framework

    Attributes
    ----------
    state_0 : np.ndarray
        Initial state estimate
    sigma_0 : np.ndarray
        Current uncertainty estimated for state estimate (2D covariance)
    """

    def __init__(self, state_0, sigma_0):
        self.state_dim = state_0.size
        assert _check_col_vect(state_0, self.state_dim), "Incorrect initial state shape"
        assert _check_square_mat(sigma_0, self.state_dim), "Incorrect initial cov shape"
        self.state = state_0
        self.sigma = sigma_0

    @abstractmethod
    def predict(self):
        """Predict the state of the filter based on some dynamics model
        """
        pass

    @abstractmethod
    def update(self):
        """Update the state of the filter based on some measurement model
        """
        pass


class BaseExtendedKalmanFilter(BaseFilter):
    """Class with general extended Kalman filter implementation

    Attributes
    ----------
    Q : np.ndarray
        Process noise covariance, tunable parameter
    R : np.ndarray
        Measurement noise covariance, tunable parameter
    params_dict : dict
        Dictionary of additional parameters required, implementation dependent
    """

    def __init__(self, init_dict, params_dict):
        super().__init__(init_dict['state_0'], init_dict['sigma_0'])
        assert _check_square_mat(init_dict['Q'], self.state_dim)
        self.Q = init_dict['Q']
        self.R = init_dict['R']
        self.params_dict = params_dict

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
            u = np.zeros((self.state_dim,1))
        if predict_dict is None:
            predict_dict = {}
        assert _check_col_vect(u, np.size(u)), "Control input is not a column vector"
        self.state = self.dyn_model(u, predict_dict)  # Can pass parameters via predict_dict
        A = self.linearize_dynamics(predict_dict)
        self.sigma = A @ self.sigma @ A.T + self.Q
        assert _check_col_vect(self.state, self.state_dim), "Incorrect state shape after prediction"
        assert _check_square_mat(self.sigma, self.state_dim), "Incorrect covariance shape after prediction"

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

        assert _check_col_vect(z, np.size(z)), "Measurements are not a column vector"
        H = self.linearize_measurements(update_dict)  # Can pass arguments via update_dict
        S = H @ self.sigma @ H.T + measurement_noise
        K = self.sigma @ H.T @ np.linalg.inv(S)
        z_expect = self.measure_model(update_dict)  # Can pass arguments via update_dict
        assert _check_col_vect(z_expect, np.size(z)), "Expected measurements are not a column vector"
        # Updating state
        self.state = self.state + K @ (z - z_expect)
        # Update covariance
        self.sigma = (np.eye(self.state_dim) - K @ H) @ self.sigma
        assert _check_col_vect(self.state, self.state_dim), "Incorrect state shape after update"
        assert _check_square_mat(self.sigma, self.state_dim), "Incorrect covariance shape after update"

    @abstractmethod
    def linearize_dynamics(self, predict_dict=None):
        """Linearization of system dynamics, should return A matrix
        """
        raise NotImplementedError

    @abstractmethod
    def linearize_measurements(self, update_dict=None):
        """Linearization of measurement model, should return H matrix
        """
        raise NotImplementedError

    @abstractmethod
    def measure_model(self, update_dict=None):
        """Non-linear measurement model
        """
        raise NotImplementedError

    @abstractmethod
    def dyn_model(self, u, predict_dict=None):
        """Non-linear dynamics model
        """
        raise NotImplementedError


class BaseKalmanFilter(BaseExtendedKalmanFilter):
    """General Kalman Filter implementation. Implementated as special
    case of BaseExtendedKalmanFilter with linear dynamics and measurement
    model
    """

    def dyn_model(self, u, predict_dict=None):
        """Linear dynamics model

        Parameters
        ----------
        u : np.ndarray
            Control input to system
        predict_dict : dict
            Additional parameters that might be requried for prediction

        Returns
        -------
        new_x : State after propagation
        """
        if predict_dict is None:
            predict_dict = {}
        A = self.linearize_dynamics(predict_dict)
        B = self.get_B(predict_dict)
        new_x = A @ self.state + B @ u
        return new_x

    def measure_model(self, update_dict=None):
        """Linear measurement model

        Parameters
        ----------
        update_dict : dict
            Additional parameters that might be requried for update

        Returns
        -------
        z_expect : Measurement expected for current state
        """
        if update_dict is None:
            update_dict = {}
        H = self.linearize_measurements(update_dict)
        z_expect = H @ self.state
        return z_expect

    @abstractmethod
    def get_B(self, predict_dict=None):
        """Map from control to state, should return B matrix
        """
        raise NotImplementedError


class BaseUnscentedKalmanFilter(BaseFilter):
    """General Unscented Kalman Filter implementation.
    Class with general Unscented Kalman filter implementation

    Attributes
    ----------
    Q : np.ndarray
        Process noise covariance, tunable parameter
    R : np.ndarray
        Measurement noise covariance, tunable parameter
    params_dict : dict
        Dictionary of additional parameters required, implementation dependent
    """

    def __init__(self, init_dict, params_dict):
        super().__init__(init_dict['state_0'], init_dict['sigma_0'])
        assert _check_square_mat(init_dict['Q'], self.state_dim)
        self.Q = init_dict['Q']
        self.R = init_dict['R']
        if 'lam' in init_dict:
            self.lam = init_dict['lam']
        else:
            self.lam = 2

        if 'N_sig' in init_dict:
            self.N_sig = init_dict['N_sig']
        else:
            self.N_sig = int(2 * self.state_dim + 1)
        self.params_dict = params_dict

    def predict(self, u, predict_dict=None):
        """Predict the state of the filter given the control input

        Parameters
        ----------
        u : np.ndarray
            The control signal given to the actual system, dimension state_dim x D
        predict_dict : dict
            Additional parameters needed to implement predict step
        """

        if predict_dict is None:
            predict_dict = {}

        N = self.state_dim
        N_sig = self.N_sig
        x_t_tm = np.zeros((N, N_sig))

        # Compute U-Transform:
        x_tm_tm, W = self.U_transform()

        for ind in range(N_sig):
            # Todo: Change x_tm_tm[:, ind] Shape
            x_sigma = np.expand_dims(x_tm_tm[:, ind], axis=1)
            x_t_tm[:, [ind]] = self.dyn_model(x_sigma, u, predict_dict)

        # Compute Inverse U-Transform:
        mu_t_tm, S_t_tm = self.inv_U_transform(W, x_t_tm)
        S_t_tm = S_t_tm + self.Q
        self.state = mu_t_tm
        self.sigma = S_t_tm
        assert _check_col_vect(self.state, self.state_dim), "Incorrect state shape after prediction"
        assert _check_square_mat(self.sigma, self.state_dim), "Incorrect covariance shape after prediction"

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

        assert _check_col_vect(z, np.size(z)), "Measurements are not a column vector"
        N = self.state_dim
        N_sig = self.N_sig

        y_t_tm = np.zeros((np.shape(self.R)[0], N_sig))
        S_xy_t_tm = np.zeros((N, np.shape(z)[0]))

        x_t_tm, W = self.U_transform()

        for ind in range(N_sig):
            y_t_tm[:, [ind]] = self.measure_model(x_t_tm[:, [ind]])

        y_hat_t_tm, S_y_t_tm = self.inv_U_transform(W, y_t_tm)
        S_y_t_tm = S_y_t_tm + self.R

        for ind in range(N_sig):
            S_xy_t_tm = S_xy_t_tm + W[ind] * np.outer((x_t_tm[:, [ind]] - self.state),  (y_t_tm[:, [ind]] - y_hat_t_tm))

        meas_res = z - y_hat_t_tm
        self.state = self.state + S_xy_t_tm @ np.linalg.inv(S_y_t_tm) @ meas_res
        self.sigma = self.sigma - S_xy_t_tm @ np.linalg.inv(S_y_t_tm) @ S_xy_t_tm.T

        assert _check_col_vect(self.state, self.state_dim), "Incorrect state shape after update"
        assert _check_square_mat(self.sigma, self.state_dim), "Incorrect covariance shape after update"

    def U_transform(self):
        """
        Sigma Point Transform
        """
        N = self.state_dim
        N_sig = self.N_sig
        X = np.zeros([N, N_sig])
        W = np.zeros([N_sig, 1])
        delta = sqrtm((self.lam + N) * self.sigma)
        X[:, 0] = np.squeeze(self.state)

        for ind in range(N):
            X[:, ind + 1] = np.squeeze(self.state) + delta[:, ind]
            X[:, ind + 1 + N] = np.squeeze(self.state) - delta[:, ind]

        W[0] = self.lam / (self.lam + N)
        W[1:] = (1 / (2 * (self.lam + N))) * np.ones([2 * N, 1])

        return X, W

    def inv_U_transform(self, W, x_t_tm):
        """
        Inverse Sigma Point Transform
        """

        N = x_t_tm.shape[0]
        N_sig = self.N_sig
        mu = np.sum(np.multiply(np.transpose(W), x_t_tm), axis=1)
        S = np.zeros([N, N])
        x_hat = x_t_tm - np.expand_dims(mu, axis=1)
        for ind in range(N_sig):
            S = S + W[ind] * np.outer(x_hat[:, [ind]], x_hat[:, [ind]])

        return np.expand_dims(mu, axis=1), S

    @abstractmethod
    def measure_model(self, x, update_dict=None):
        """Non-linear measurement model
        """
        raise NotImplementedError

    @abstractmethod
    def dyn_model(self, x, u, predict_dict=None):
        """Non-linear dynamics model
        """
        raise NotImplementedError

def _check_col_vect(vect, dim):
    """Boolean for whether input vector is column shaped or not

    Parameters
    ----------
    vect : np.ndarray
        Input vector
    dim : int
        Number of row elements in column vector
    """
    return np.shape(vect)[0] == dim and np.shape(vect)[1] == 1

def _check_square_mat(mat, dim):
    """Boolean for whether input matrices are square or not

    Parameters
    ----------
    vect : np.ndarray
        Input matrix
    dim : int
        Number of elements for row and column = N for N x N
    """

    return np.shape(mat)[0] == dim and np.shape(mat)[1] == dim
