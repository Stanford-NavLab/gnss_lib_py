########################################################################
# Author(s):    Shubh Gupta
# Date:         19 July 2021
# Desc:         Point solution methods using GNSS measurements
########################################################################

import numpy as np
from utils.constants import GPSConsts


def solve_pos(prange, X, Y, Z, B, e=1e-3):
    # TODO: Modify code to perform WLS if weights are given
    """Find user position, clock bias using WLS or NR methods

    Find user position, clock bias by solving the weighted least squares (WLS) problem.
    If no weights are given, the Newton Raphson (NR) position, clock bias solution is used instead. 

    Parameters
    ----------
    prange : ndarray
        Measured pseudoranges, dimension m
    X : ndaarray
        Satellite ECEF x positions, dimension m
    Y : ndaarray
        Satellite ECEF y positions, dimension m
    Z : ndaarray
        Satellite ECEF z positions, dimension m
    B : ndaarray
        Satellite onboard clock bias, dimension m
    e : float
        Termination threshold for LS solver

    Returns
    -------
    x_fix : ndarray
        Solved 3D position and clock bias estimates

    """
    gpsconsts = GPSConsts()
    if len(prange)<4:
        return np.empty(4)
    x, y, z, cdt = 100., 100., 100., 0.

    x0 = np.array([x, y, z, cdt])
    x_fix, res_err = newton_raphson(_f, _df, x0, e=e)
    x_fix[-1] = x_fix[-1]*1e6/gpsconsts.C
    
    def _f(vars):
        """Difference between expected and received pseudoranges

        Parameters
        ----------
        vars : list
            List of estimates for position and time

        Returns
        -------
        delta_prange: ndarray
            Float difference between expected and measured pseudoranges

        """
        x, y, z, cdt = list(vars)
        tilde_prange = np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
        _prange = tilde_prange + cdt - B
        delta_prange = prange-_prange
        return delta_prange

    def _df(vars):
        """Jacobian of expected pseudorange 

        Parameters
        ----------
        vars : list
            List of estimates for position and time

        Returns
        -------
        derivatives: ndarray
            mx4 Jacobian matrix of expected pseudorange
        """
        x, y, z, cdt = list(vars)
        tilde_prange = np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
        _prange = tilde_prange + cdt - B
        derivatives = np.zeros((len(prange), 4))
        derivatives[:, 0] = -(x - X)/tilde_prange
        derivatives[:, 1] = -(y - Y)/tilde_prange
        derivatives[:, 2] = -(z - Z)/tilde_prange
        derivatives[:, 3] = -1
        return derivatives

    return x_fix


def newton_raphson(f, df, x0, e=1e-3, lam=1.):
    """Newton Raphson method to find zero of function. 

    Parameters
    ----------
    f : method
        Function whose zero is required
    df : method
        Function that outputs derivative of f.
    x0: ndarray
        Initial guess of solution
    e: float
        Maximum difference between consecutive guesses for termination
    lam: float
        Scaling factor for step taken at each iteration

    Returns
    -------
    x0 : ndarray
        Solution for zero of function 
    f_norm : float
        Norm of function magnitude at solution point

    """
    delta_x = np.ones_like(x0)
    while np.sum(np.abs(delta_x))>e:
        delta_x = lam*(np.linalg.pinv(df(x0)) @ f(x0))
        x0 = x0 - delta_x
    f_norm = np.linalg.norm(f(x0))
    return x0, f_norm