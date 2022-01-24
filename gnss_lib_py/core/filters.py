"""Parent classes for Kalman filter algorithms
"""

__authors__ = "Ashwin Kanhere, Shivam Soni"
__date__ = "20 Januray 2020"

import numpy as np
from abc import abstractmethod

from gnss_lib_py.utils.matrices import check_col_vect, check_square_mat


class BaseFilter():

    def __init__(self, x_dim, x0, P0):
        self.x_dim = x_dim
        assert check_col_vect(x0, self.x_dim), "Incorrect initial state shape"
        assert check_square_mat(P0, self.x_dim), "Incorrect initial cov shape"
        self.x = x0
        self.P = P0

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def update(self):
        pass


class BaseKalmanFilter(BaseFilter):

    def __init__(self, init_dict, params_dict):
        super.__init__(init_dict['x_dim'], init_dict['x0'], init_dict['P0'])
        assert check_square_mat(init_dict['Q'], self.x_dim)
        self.Q = init_dict['Q']
        self.R = init_dict['R_scale']
        self.params_dict = params_dict

    def predict(self, u, predict_dict=None):
        A = self.get_A(predict_dict)
        B = self.get_B(predict_dict)
        self.x = A @ self.x + B @ u
        self.P = A @ self.P @ A.T + self.Q
        assert check_col_vect(self.x, self.x_dim), "Incorrect state shape after prediction"
        assert check_square_mat(self.P, self.x_dim), "Incorrect covariance shape after prediction"

    def update(self, z, update_dict=None):
        assert check_col_vect(z, np.size(z)), "Measurements are not a column vector"
        H = self.get_H(update_dict) # Can pass arguments via update_dict      
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        z_expect = H @ self.x # Can pass arguments via update_dict
        #Updating state
        self.x = self.x + K @ (z - z_expect)
        # Update covariance
        self.P = (np.eye(self.x_dim) - K @ H) @ self.P
        assert check_col_vect(self.x, self.x_dim), "Incorrect state shape after update"
        assert check_square_mat(self.P, self.x_dim), "Incorrect covariance shape after update"

    def get_A(predict_dict=None):
        pass

    def get_B(predict_dict=None):
        pass
    
    def get_H(update_dict=None):
        pass



class BaseExtendedKalmanFilter(BaseFilter):

    def __init__(self, init_dict, params_dict):
        super.__init__(init_dict['x_dim'], init_dict['x0'], init_dict['P0'])
        assert check_square_mat(init_dict['Q'], self.x_dim)
        self.Q = init_dict['Q']
        self.R = init_dict['R_scale']
        self.params_dict = params_dict

    def predict(self, u, predict_dict=None):
        self.x = self.dyn_func(u, predict_dict) # Can pass parameters via predict_dict
        A = self.linearize_dynamics(predict_dict)
        self.P = A @ self.P @ A.T + self.Q
        assert check_col_vect(self.x, self.x_dim), "Incorrect state shape after prediction"
        assert check_square_mat(self.P, self.x_dim), "Incorrect covariance shape after prediction"

    def update(self, z, update_dict=None): 
        assert check_col_vect(z, np.size(z)), "Measurements are not a column vector"
        H = self.linearize_measurements(update_dict) # Can pass arguments via update_dict      
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        z_expect = self.expect_measure(update_dict) # Can pass arguments via update_dict
        #Updating state
        self.x = self.x + K @ (z - z_expect)
        # Update covariance
        self.P = (np.eye(self.x_dim) - K @ H) @ self.P
        assert check_col_vect(self.x, self.x_dim), "Incorrect state shape after update"
        assert check_square_mat(self.P, self.x_dim), "Incorrect covariance shape after update"

    @abstractmethod
    def linearize_dynamics(self, predict_dict=None):
        # This function should return A
        pass

    @abstractmethod
    def linearize_measurements(self, update_dict=None):
        # This function should return H such that z = H x
        pass

    @abstractmethod
    def meas_model(self, update_dict=None):
        pass

    @abstractmethod
    def dyn_model(u, predict_dict=None):
        # This function should perform non-linear propagation of the state
        pass
