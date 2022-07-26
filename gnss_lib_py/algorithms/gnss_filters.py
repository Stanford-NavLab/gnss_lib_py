"""Classes for GNSS-based Kalman Filter implementations

"""

__authors__ = "Ashwin Kanhere"
__date__ = "25 Januray 2020"

import numpy as np

from gnss_lib_py.utils.filters import BaseExtendedKalmanFilter

class GNSSEKF(BaseExtendedKalmanFilter):
    """GNSS-only EKF implementation.

    States: 3D position, 3D velocity and clock bias (in m).
    The state vector is :math:`\\bar{x} = [x, y, z, v_x, v_y, v_y, b]^T`

    Attributes
    ----------
    dt : float
        Time between prediction instances
    motion_type : string
        Type of motion (stationary or constant velocity)
    measure_type : string
        NavData types (pseudoranges)
    """
    def __init__(self, init_dict, params_dict):
        super().__init__(init_dict, params_dict)
        self.dt = params_dict['dt']
        try:
            self.motion_type = params_dict['motion_type']
        except KeyError:
            self.motion_type = 'stationary'
        try:
            self.measure_type = params_dict['measure_type']
        except KeyError:
            self.measure_type = 'pseudoranges'

    def dyn_model(self, u, predict_dict=None):
        """Non linear dynamics

        Parameters
        ----------
        u : np.ndarray
            Control signal, not used for propagation
        predict_dict : Dict
            Additional prediction parameters, not used currently

        Returns
        -------
        new_x : np.ndarray
            Propagated state
        """
        A = self.linearize_dynamics()
        new_x = A @ self.x
        return new_x

    def measure_model(self, update_dict):
        """Measurement model

        Pseudorange model adds true range and clock bias estimate:
        :math:`\\rho = \\sqrt{(x-x_{sv})^2 + (y-y_{sv})^2 + (z-z_{sv})^2} + b`.
        See [1]_ for more details and models.

        Parameters
        ----------
        update_dict : Dict
            Update dictionary containing satellite positions with key 'sv_pos'

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
            sv_pos = update_dict['sv_pos']
            pseudo = np.sqrt((self.x[0] - sv_pos[0, :])**2
                            + (self.x[1] - sv_pos[1, :])**2
                            + (self.x[2] - sv_pos[2, :])**2) \
                            + self.x[6]
            z = np.reshape(pseudo, [-1, 1])
        else:
            raise NotImplementedError
        return z

    def linearize_dynamics(self, predict_dict=None):
        """Linearization of dynamics model

        Parameters
        ----------
        predict_dict : Dict
            Additional predict parameters, not used in current implementation

        Returns
        -------
        A : np.ndarray
            Linear dynamics model depending on motion_type
        """
        if self.motion_type == 'stationary':
            A = np.eye(7)
        elif self.motion_type == 'constant_velocity':
            A = np.eye(7)
            A[:3, -4:-1] = self.dt*np.eye(3)
        else:
            raise NotImplementedError
        return A

    def linearize_measurements(self, update_dict):
        """Linearization of measurement model

        Parameters
        ----------
        update_dict : Dict
            Update dictionary containing satellite positions with key 'sv_pos'

        Returns
        -------
        H : np.ndarray
            Jacobian of measurement model, dimension M x N
        """
        if self.measure_type == 'pseudorange':
            sv_pos = update_dict['sv_pos']
            m = np.shape(sv_pos)[1]
            H = np.zeros([m, self.x_dim])
            pseudo_expect = self.measure_model(update_dict)
            rx_pos = np.reshape(self.x[:3], [-1, 1])
            H[:, :3] = (rx_pos - sv_pos).T/pseudo_expect
            H[:, 6] = 1
        else:
            raise NotImplementedError
        return H
