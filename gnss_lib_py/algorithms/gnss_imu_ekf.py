
"""Classes for GNSS_IMU-based Extended Kalman Filter implementations

"""

__authors__ = "Mahesh Saboo"
__date__ = "1 May 2022"


import numpy as np
from gnss_lib_py.core.filters import BaseExtendedKalmanFilter


class GNSS_IMU_EKF(BaseExtendedKalmanFilter):
    """GNSS_IMU EKF implementation.

    States: 2D position, 2D velocity, heading angle and clock bias (in m).
    The state vector is :math:`\\bar{x} = [x,y,v_x,v_y,psi,b]^T`
    
    Attributes
    ----------
    dt : float
        Time between prediction instances
    z_pos : float
        constant z position for 2D case
    motion_type : string
        Type of motion (constant velocity)
    measure_type : string
        Measurement types (pseudoranges)
    """
    def __init__(self, init_dict, params_dict):
        super().__init__(init_dict, params_dict)
        self.dt = params_dict['dt']
        self.z_pos = init_dict['z_pos']
        self.params_dict = params_dict
        self.init_dict = init_dict
        try:
            self.motion_type = params_dict['motion_type']
        except KeyError:
            self.motion_type = 'constant_velocity'
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
        #u = [ax,ay,wz]
        imu_val = u
        new_x = self.x
        new_x[0] += new_x[2]*self.dt
        new_x[1] += new_x[3]*self.dt
        new_x[2] += imu_val[0]*np.cos(new_x[4]) - imu_val[1]*np.sin(new_x[4])
        new_x[3] += imu_val[0]*np.sin(new_x[4]) + imu_val[1]*np.cos(new_x[4])
        new_x[4] += imu_val[2]*self.dt
        return new_x

    def measure_model(self, update_dict):
        """Measurement model

        Pseudorange model adds true range and clock bias estimate:
        :math:`\\rho = \\sqrt{(x-x_{sat})^2 + (y-y_{sat})^2 + (z-z_{sat})^2} + b`.
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
                            + (self.z_pos - sv_pos[2, :])**2) \
                            + self.x[5]
            z = np.reshape(pseudo, [-1, 1])
        else:
            raise NotImplementedError
        return z

    def linearize_dynamics(self, u,predict_dict=None):
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
        if self.motion_type == 'constant_velocity':
            #u = [ax,ay,wz]
            imu_val = u
            M = -imu_val[0]*np.sin(self.x[4]) - imu_val[1]*np.cos(self.x[4])
            N = imu_val[0]*np.cos(self.x[4]) - imu_val[1]*np.sin(self.x[4])
            A = np.array([[1,0,self.dt,0,0,0],
                         [0,1,0,self.dt,0,0],
                         [0,0,1,0,M,0],
                         [0,0,0,1,N,0],
                         [0,0,0,0,1,0],
                         [0,0,0,0,0,1]],dtype=float)
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
            rx_pos = np.reshape(self.x[:2], [-1, 1])
            H[:, :2] = (rx_pos - sv_pos[:2,:]).T/pseudo_expect
            H[:, 5] = 1
        else:
            raise NotImplementedError
        return H