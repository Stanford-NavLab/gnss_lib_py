"""Tests for a general EKF
"""

__authors__ = "Ashwin Kanhere, Shivam Soni"
__date__ = "24 Januray 2020"

import numpy as np
import pandas as pd
import math
import pytest
from numpy.random import default_rng

from gnss_lib_py.core.filters import BaseKalmanFilter


class MSD_EKF(BaseKalmanFilter):
    #TODO: Define the state space, measurment space etc. here
    def __init__(self, init_dict, params_dict):
        self.rng = default_rng()
        self.k = params_dict['k']
        self.m = params_dict['m']
        self.b = params_dict['b']
        super().__init__(init_dict, params_dict)
    
    def linearize_dynamics(self, predict_dict=None):
        A = np.array([[0, 1], [-self.k / self.m, -self.b / self.m]])
        return A

    def linearize_measurements(self, update_dict=None):
        H = np.array([[1, 0]])
        return H

    def get_B(self, predict_dict=None):
        B = np.zeros([2,1])
        return B

    # def measure_model(self, update_dict=None):
    #     meas = self.x[0]
    #     return meas

    # def dyn_model(self, u, predict_dict=None):
    #     A = self.linearize_dynamics()
    #     B = np.zeros([2,1])
    #     # prop_noise = self.rng.multivariate_normal(np.zeros(self.x_dim), self.Q, size=1)
    #     new_state = A @ self.x + B @ u # + prop_noise.T
    #     return new_state



@pytest.fixture(name="times")
def msd_times():
    t_end = 20  # Seconds
    time_step = 0.01
    t_vals = np.linspace(0, t_end, int(1 / time_step) + 1)
    return t_vals


@pytest.fixture(name="x_exact")
def msd_exact_sol(times):
    zeta = 0.5
    omega_n = 2
    omega_d = omega_n * np.sqrt(1 - zeta ** 2)
    y0 = 1
    yd0 = 1
    C1 = y0
    C2 = (yd0 + zeta * omega_n * y0) / omega_d
    C = np.sqrt(C1 ** 2 + C2 ** 2)
    phi = np.arctan2(C1, C2)
    exp_term = np.exp(-zeta * omega_n * times)
    sin_term = np.sin(omega_d * times + phi)
    exact_sol = exp_term * C * sin_term
    return exact_sol


@pytest.fixture(name="init_dict")
def msd_ekf_params():
    ekf_init_dict = {'x_dim': 2,
                     'x0': np.array([[1], [1]]),
                     'P0': np.eye(2)}
    return ekf_init_dict


@pytest.fixture(name="msd_params")
def msd_params():
    params = {'m': 1,
            'b' : 2,
            'k' : 4}


def msd_filter_sol(times, x_exact, init_dict, q, r, filter_type):
    if filter_type=='ekf':
        init_dict['Q'] = q*np.eye(init_dict['x_dim'])
        init_dict['R'] = r*np.eye(1)
        params_dict = {'m': 1,
                    'b' : 2,
                    'k' : 4}
        msd_filter = MSD_EKF(init_dict, params_dict)
    else:
        raise NotImplementedError
    t_len = np.size(times)
    u = np.array([[0]])
    run_rng = default_rng()

    x_filter = np.empty([0, 1])
    P_pre = np.empty([0, 2, 2])
    P_post = np.empty([0, 2, 2])
    
    for t_idx in range(t_len):
        msd_filter.predict(u)
        P_pre_temp = np.reshape(msd_filter.P, [1, msd_filter.x_dim, msd_filter.x_dim])
        P_pre = np.concatenate([P_pre, P_pre_temp], axis=0)
        z = np.reshape(x_exact[t_idx] + run_rng.normal(0, r, size=1), [-1, 1])
        msd_filter.update(z)
        # Save positions and covariance matrices
        x_filter = np.append(x_filter, msd_filter.x[0])
        P_post_temp = np.reshape(msd_filter.P, [1, msd_filter.x_dim, msd_filter.x_dim])
        P_post = np.concatenate([P_post, P_post_temp], axis=0)
    return x_filter, P_pre, P_post


@pytest.mark.parametrize("filter_type", ['ekf'])
def test_exact_sol(times, x_exact, init_dict, filter_type):
    #TODO: Check why increasing the q value below from 0.001 to 0.01 makes the test pass
    x_filter, _, _ = msd_filter_sol(times, x_exact, init_dict, 0.01, 0.0001, filter_type)
    #TODO: Why are different indices being compared here? Add findings as a note
    print('x_exact shape', x_exact.shape)
    print('x_filter shape', x_filter.shape)
    np.testing.assert_array_almost_equal(x_exact, x_filter, decimal=2)


@pytest.mark.parametrize("filter_type", ['ekf'])
@pytest.mark.parametrize('q, r',
                        [(0.01, 0.00001),
                        (0.00001, 0.01),
                        (0.01, 0.01)])
def test_filter_cov_tests(times, x_exact, init_dict, q, r, filter_type):
    _, P_pre, P_post = msd_filter_sol(times, x_exact, init_dict, q, r, filter_type)
    # Test that all P matrices are positive semidefinite
    for idx in range(np.shape(P_pre)[0]):
        # Cholesky factorization will fail if matrix is not PSD
        np.linalg.cholesky(P_pre[idx, :, :])
        np.linalg.cholesky(P_post[idx, :, :])
    # Covariance should decrease after EKF update
    np.testing.assert_array_less(P_post[:, 0, 0], P_pre[:, 0, 0])
    np.testing.assert_array_less(P_post[:, 1, 1], P_pre[:, 1, 1])
    # Covariance should increase after EKF prediction
    np.testing.assert_array_less(P_post[:-1, 0, 0], P_pre[1:, 0, 0])
    np.testing.assert_array_less(P_post[:-1, 1, 1], P_pre[1:, 1, 1])
