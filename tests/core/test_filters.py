"""Tests for a general KF

"""

__authors__ = "Ashwin Kanhere, Shivam Soni"
__date__ = "24 Januray 2020"

import numpy as np
import pytest
from numpy.random import default_rng

from gnss_lib_py.core.filters import BaseKalmanFilter
from gnss_lib_py.core.filters import BaseUnscentedKalmanFilter


class MSD_EKF(BaseKalmanFilter):
    """Mass spring damper system for testing EKF implementation

    Attributes
    ----------
    k : float
        Spring constant of system
    m : float
        Mass of system
    b : float
        Damping coeeficient of system
    """

    # TODO: Define the state space, measurment space etc. here
    def __init__(self, init_dict, params_dict):
        self.k = params_dict['k']
        self.m = params_dict['m']
        self.b = params_dict['b']
        super().__init__(init_dict, params_dict)

    def linearize_dynamics(self, predict_dict=None):
        """Linearization of dynamics model

        Parameters
        ----------
        predict_dict : Dict
            Additional predict parameters, not used in current implementation

        Returns
        -------
        A : np.ndarray
            Linear dynamics model for MSD
        """
        A = np.array([[0, 1], [-self.k / self.m, -self.b / self.m]])
        return A

    def linearize_measurements(self, update_dict=None):
        """Linearization of measurement model

        Parameters
        ----------
        update_dict : Dict
            Additional update parameters, not used in current implementation

        Returns
        -------
        H : np.ndarray
            Jacobian of measurement model, dimension 2 x 1
        """
        H = np.array([[1, 0]])
        return H

    def get_B(self, predict_dict=None):
        """Map from control space to state space . No control signals in
        this example

        Parameters
        ----------
        predict_dict : Dict
            Additional predict parameters, not used in current implementation

        Returns
        -------
        B : np.ndarray
            2 x 1 array of all zero elements
        """
        B = np.zeros([2, 1])
        return B


class MSD_UKF(BaseUnscentedKalmanFilter):
    """Mass spring damper system for testing EKF implementation

    Attributes
    ----------
    k : float
        Spring constant of system
    m : float
        Mass of system
    b : float
        Damping coefficient of system
    """

    # TODO: Define the state space, measurment space etc. here
    def __init__(self, init_dict, params_dict):
        self.k = params_dict['k']
        self.m = params_dict['m']
        self.b = params_dict['b']
        super().__init__(init_dict, params_dict)

    def dyn_model(self, x, u, predict_dict=None):
        """Full dynamics model

        Parameters
        ----------
        x : UKF belief state under unscented transformation
        u : Input vector
        predict_dict : Dict
            Additional update parameters, not used in current implementation

        Returns
        -------
        x_new : np.ndarray
            predicted state, dimension 2 x 1
        """

        A = np.array([[0, 1], [-self.k / self.m, -self.b / self.m]])
        B = np.zeros([2, 1])
        x_new = A @ x + B @ u
        return x_new

    def linearize_measurements(self, update_dict=None):
        """Linearization of measurement model

        Parameters
        ----------
        update_dict : Dict
            Additional update parameters, not used in current implementation

        Returns
        -------
        H : np.ndarray
            Jacobian of measurement model, dimension 2 x 1
        """
        H = np.array([[1, 0]])
        return H

    def measure_model(self, x, update_dict=None):
        """Full measurement model

        Parameters
        ----------
        update_dict : Dict
            Additional update parameters, not used in current implementation
        x: np.ndarray
            State for measurement model

        Returns
        -------
        y : np.ndarray
            measurement, dimension 2 x 1
        """
        H = self.linearize_measurements()
        z_expect = H @ x
        return z_expect


@pytest.fixture(name="times")
def msd_times():
    """Return time vector for MSD system  evolution

    Returns
    -------
    msd_times : np.ndarray
        Vector of linearly spaced time values from 0 to 20 seconds
    """
    t_end = 20  # Seconds
    time_step = 0.01
    t_vals = np.linspace(0, t_end, int(1 / time_step) + 1)
    return t_vals


@pytest.fixture(name="x_exact")
def msd_exact_sol(times):
    """Compute exact solution for mass-spring-damper position evolution

    Parameters
    ----------
    times : np.ndarray
        Vector of linearly spaced time values for MSD

    Returns
    -------
    exact_sol : np.ndarray
        Vector of exact position solutions for MSD system
    """
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
    """Return dictionary of initial parameters for MSD KF implementation

    Returns
    -------
    ekf_init_dict : Dict
        Dictionary with KF state dimension, initial state and covariance
    """
    ekf_init_dict = {'x_dim': 2,
                     'x0': np.array([[1], [1]]),
                     'P0': np.eye(2)}
    return ekf_init_dict


@pytest.fixture(name="init_dict")
def msd_ukf_params():
    """Return dictionary of initial parameters for MSD UKF implementation

    Returns
    -------
    ekf_init_dict : Dict
        Dictionary with UKF state dimension, initial state and covariance
    """
    ukf_init_dict = {'x_dim': 2,
                     'x0': np.array([[1], [1]]),
                     'P0': np.eye(2)}
    return ukf_init_dict


@pytest.fixture(name="params_dict")
def msd_params():
    """Return dictionary of additional parameters for MSD KF implementation

    Returns
    -------
    params : Dict
        Dictionary with MSD mass, spring constant and damping coefficient
    """
    params = {'m': 1,
              'b': 2,
              'k': 4}
    return params


def msd_filter_sol(times, x_exact, init_dict, params_dict, q, r, filter_type):
    """Run filter for all time steps

    Parameters
    ----------
    times : np.ndarray
        Vector containing time instances for exact state evolution
    x_exact : np.ndarray
        Vector containing exact state positions
    init_dict : Dict
        Dictionary of filter initialization parameters
    params_dict : Dict
        Dictionary of MSD parameters
    q : float
        Process noise covariance values on diagonal term
    r : float
        Measurement noise covariance values on diagonal term
    filter_type : string
        Type of filter to test, currently ekf

    Returns
    -------
    x_filter : np.ndarray
        Position estimated by filter, dimension T x 1
    P_pre : np.ndarray
        State covariance after prediction step, dimension T x 2 x 2
    P_post : np.ndarray
        State covariance after update step, dimension T x 2 x 2
    """
    if filter_type == 'ekf':
        init_dict['Q'] = q * np.eye(init_dict['x_dim'])
        init_dict['R'] = r * np.eye(1)
        msd_filter = MSD_EKF(init_dict, params_dict)
    elif filter_type == 'ukf':
        init_dict['Q'] = q * np.eye(init_dict['x_dim'])
        init_dict['R'] = r * np.eye(1)
        msd_filter = MSD_UKF(init_dict, params_dict)
        # TODO: elif for 'ukf' - Done
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


@pytest.mark.parametrize("filter_type", ['ekf', 'ukf'])
def test_exact_sol(times, x_exact, init_dict, params_dict, filter_type):
    """Compare solution from KF to exact solution of the system

    Parameters
    ----------
    times : np.ndarray
        Vector containing time instances for exact state evolution
    x_exact : np.ndarray
        Vector containing exact state positions
    init_dict : Dict
        Dictionary of filter initialization parameters
    params_dict : Dict
        Dictionary of MSD parameters
    filter_type : string
        Type of filter to test, currently ekf
    """
    # TODO: Check why increasing the q value below from 0.001 to 0.01 makes the test pass
    x_filter, _, _ = msd_filter_sol(times, x_exact, init_dict, params_dict, 0.01, 0.0001, filter_type)
    # TODO: Why are different indices being compared here? Add findings as a note
    # Note: Due to the setup, at time-step k=2, the estimation is for k=1.
    # print('x_exact shape', x_exact.shape)
    # print('x_filter shape', x_filter.shape)
    np.testing.assert_array_almost_equal(x_exact, x_filter, decimal=2)


@pytest.mark.parametrize("filter_type", ['ekf', 'ukf'])
@pytest.mark.parametrize('q, r',
                         [(0.01, 0.00001),
                          (0.00001, 0.01),
                          (0.01, 0.01)])
def test_filter_cov_tests(times, x_exact, init_dict, params_dict, q, r, filter_type):
    """
    Test that covariance is PSD, decreases after update and increases
    after prediction step

    Parameters
    ----------
    times : np.ndarray
        Vector containing time instances for exact state evolution
    x_exact : np.ndarray
        Vector containing exact state positions
    init_dict : Dict
        Dictionary of EKF initialization parameters
    params_dict : Dict
        Dictionary of MSD parameters
    q : float
        Process noise covariance values on diagonal term
    r : float
        Measurement noise covariance values on diagonal term
    filter_type : string
        Type of filter to test, currently ekf
    """
    _, P_pre, P_post = msd_filter_sol(times, x_exact, init_dict, params_dict, q, r, filter_type)
    # Test that all P matrices are positive semidefinite
    for idx in range(np.shape(P_pre)[0]):
        # Cholesky factorization will fail if matrix is not PD
        np.linalg.cholesky(P_pre[idx, :, :])
        np.linalg.cholesky(P_post[idx, :, :])
    # Covariance should decrease after EKF update
    np.testing.assert_array_less(P_post[:, 0, 0], P_pre[:, 0, 0])
    np.testing.assert_array_less(P_post[:, 1, 1], P_pre[:, 1, 1])
    # Covariance should increase after EKF prediction
    np.testing.assert_array_less(P_post[:-1, 0, 0], P_pre[1:, 0, 0])
    np.testing.assert_array_less(P_post[:-1, 1, 1], P_pre[1:, 1, 1])
