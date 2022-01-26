"""Parent classes for Kalman filter algorithms

"""

__authors__ = "Ashwin Kanhere, Shivam Soni"
__date__ = "20 Januray 2020"

import numpy as np
from abc import ABC, abstractmethod

from gnss_lib_py.utils.matrices import check_col_vect, check_square_mat


class BaseFilter(ABC):
    """Class with general filter implementation framework

    Attributes
    ----------
    x_dim : int
        Dimension of the state vector being estimated
    x : np.ndarray
        Current state estimate
    P : np.ndarray
        Current uncertainty estimated for state estimate (2D covariance)
    """
    def __init__(self, x_dim, x0, P0):
        self.x_dim = x_dim
        assert check_col_vect(x0, self.x_dim), "Incorrect initial state shape"
        assert check_square_mat(P0, self.x_dim), "Incorrect initial cov shape"
        self.x = x0
        self.P = P0

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
    params_dict : Dict
        Dictionary of additional parameters required, implementation dependent
    """
    def __init__(self, init_dict, params_dict):
        super().__init__(init_dict['x_dim'], init_dict['x0'], init_dict['P0'])
        assert check_square_mat(init_dict['Q'], self.x_dim)
        self.Q = init_dict['Q']
        self.R = init_dict['R']
        self.params_dict = params_dict

    def predict(self, u, predict_dict=None):
        """Predict the state of the filter given the control input

        Parameters
        ----------
        u : np.ndarray
            The control signal given to the actual system, dimension x_dim x D
        predict_dict : Dict
            Additional parameters needed to implement predict step
        """
        self.x = self.dyn_model(u, predict_dict) # Can pass parameters via predict_dict
        A = self.linearize_dynamics(predict_dict)
        self.P = A @ self.P @ A.T + self.Q
        assert check_col_vect(self.x, self.x_dim), "Incorrect state shape after prediction"
        assert check_square_mat(self.P, self.x_dim), "Incorrect covariance shape after prediction"

    def update(self, z, update_dict=None):         
        """Update the state of the filter given a noisy measurement of the state

        Parameters
        ----------
        z : np.ndarray
            Noisy measurement of state, dimension M x 1
        update_dict : Dict
            Additional parameters needed to implement update step
        """
        assert check_col_vect(z, np.size(z)), "Measurements are not a column vector"
        H = self.linearize_measurements(update_dict) # Can pass arguments via update_dict      
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        z_expect = self.measure_model(update_dict) # Can pass arguments via update_dict
        #Updating state
        self.x = self.x + K @ (z - z_expect)
        # Update covariance
        self.P = (np.eye(self.x_dim) - K @ H) @ self.P
        assert check_col_vect(self.x, self.x_dim), "Incorrect state shape after update"
        assert check_square_mat(self.P, self.x_dim), "Incorrect covariance shape after update"

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
        predict_dict : Dict
            Additional parameters that might be requried for prediction

        Returns
        -------
        new_x : State after propagation
        """
        A = self.linearize_dynamics(predict_dict)
        B = self.get_B(predict_dict)
        new_x = A @ self.x + B @ u
        return new_x

    def measure_model(self, update_dict=None):
        """Linear measurment model

        Parameters
        ----------
        update_dict : Dict
            Additional parameters that might be requried for update

        Returns
        -------
        z_expect : Measurement expected for current state
        """
        H = self.linearize_measurements(update_dict)
        z_expect = H @ self.x
        return z_expect
    
    @abstractmethod
    def get_B(self, predict_dict=None):
        """Map from control to state, should return B matrix
        """
        raise NotImplementedError