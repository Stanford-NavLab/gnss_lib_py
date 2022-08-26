"""Test for GNSS filtering algorithms.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "25 January 2022"

import pytest
import numpy as np
from numpy.random import default_rng

from gnss_lib_py.algorithms.gnss_filters import GNSSEKF


@pytest.fixture(name='init_dict')
def gnss_init_params():
    """Testing parameters for GNSS-only EKF

    Returns
    -------
    init_dict : Dict
        Dictionary with KF initialization parameters
    """
    x0 = np.zeros([7, 1])
    x0[0] = -2700628.97971166
    x0[1] = -4292443.61165747
    x0[2] =  3855152.80233124
    x_dim = 7
    Q = 5*np.eye(x_dim)
    R = 5*np.eye(7) # Test has 7 measurements
    init_dict = {'x_dim': x_dim,
                'x0': x0,
                'P0': 5*np.eye(x_dim),
                'Q': Q,
                'R': R}
    return init_dict


@pytest.fixture(name="params_dict")
def gnss_run_params():
    """Run time parameters for GNSS-only EKF test

    Returns
    -------
    params_dict : Dict
        Dicitonary with satellite positions, dt and measure type
    """
    sv_pos = np.array([
        [16033696.6255441, -19379870.1543683, -8529912.10997747],
        [-2904524.74773657, -26399200.3920454, -253278.373568479],
        [-3692615.01847773, -15682408.1669833, 21115038.8039792],
        [21478993.5690158, -2493694.84909697, 15422467.9135949],
        [-13910706.2648025, -17206261.2592405, 14692005.3082363],
        [13057746.5436005, -11000091.2964189, 20344813.6378203],
        [8021833.71066202, -20554715.8959834, 14784181.8724606]])
    sv_pos = sv_pos.T
    dt = 0.1
    params_dict = {'sv_pos': sv_pos,
                'dt': dt,
                'measure_type': 'pseudorange'}
    return params_dict

@pytest.mark.parametrize('motion_type', ['stationary',
                                        'constant_velocity'])
def test_stationary_filter(init_dict, params_dict, motion_type):
    """Test if solution of EKF with small measurement noise is submeter
    close to truth

    Parameters
    ----------
    init_dict : Dict
        Dictionary of initialization parameters
    params_dict : Dict
        Dictionary of run-time parameters for GNSS-only EKF
    motion_type : string
        Stationary or constant velocity
    """
    # Run stationary filter for 10 timesteps and verify that obtained position is near original position
    run_rng = default_rng()
    params_dict['motion_type'] = motion_type
    update_dict = {'sv_pos': params_dict['sv_pos']}
    x_dim = init_dict['x_dim']
    gnss_ekf = GNSSEKF(init_dict, params_dict)
    t_total = 2
    x = np.reshape(init_dict['x0'][:3], [-1, 1])
    true_range = np.linalg.norm(params_dict['sv_pos'] - x, axis=0)
    true_range = np.reshape(true_range, [-1, 1])
    for _ in range(int(t_total/params_dict['dt'])):
        gnss_ekf.predict(np.zeros([x_dim, 1]))
        # meas_noise = run_rng.multivariate_normal(np.zeros(x_dim), init_dict['R'])
        meas_noise = np.zeros(7)
        z = true_range + np.reshape(meas_noise, [-1, 1])
        gnss_ekf.update(z, update_dict)
    pos_ekf = gnss_ekf.x[:3]
    np.testing.assert_allclose(pos_ekf, init_dict['x0'][:3], atol=0.1)
