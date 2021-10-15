########################################################################
# Author(s):    Shubh Gupta, Bradley Collicott
# Date:         19 July 2021
# Desc:         Point solution methods using GNSS measurements
########################################################################

import os
import sys
# append <path>/gnss_libpy/gnss_libpy/ to path
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
import numpy as np

from core.constants import GPSConsts


def solvepos(
    prange_measured:np.ndarray, 
    x_sv:np.ndarray, 
    y_sv:np.ndarray, 
    z_sv:np.ndarray, 
    bias_clock_sv:np.ndarray, 
    e:float=1e-3
):
    # TODO: Modify code to perform WLS if weights are given
    # TODO: Change inputs to either DataFrame or Matrix
    """Find user position, clock bias using WLS or NR methods

    Find user position, clock bias by solving the weighted least squares
    (WLS) problem for n satellites. If no weights are given, the Newton
    Raphson (NR) position, clock bias solution is used instead.

    Parameters
    ----------
    prange_measured : ndarray
        Measured pseudoranges, dimension n
    x_sv : ndaarray
        Satellite ECEF x positions, dimension n, units [m]
    y_sv : ndaarray
        Satellite ECEF y positions, dimension n, units [m]
    z_sv : ndaarray
        Satellite ECEF z positions, dimension n, units [m]
    bias_clock_sv : ndaarray
        Range bias due to satellite clock offset (c*dt), dimension n, units [m]
    e : float
        Termination threshold for LS solver, units [~]

    Returns
    -------
    X_fix : ndarray
        Solved 3D position and clock bias estimates, dimension 1-by-4, units [m] and [ms]
    """

    def _f(vars):
        """Difference between expected and received pseudoranges

        Parameters
        ----------
        vars : list
            List of estimates for position and time
        x_fix : float
            User ECEF x position estimate, scalar, units [m]
        y_fix : float
            User ECEF y position estimate, scalar, units [m]
        z_fix : float
            User ECEF z position estimate, scalar, units [m]
        bias_clock_user : float
            Range bias due to user clock offset (c*dt), scalar, units [m]

        Returns
        -------
        deltaprange: ndarray
            Float difference between expected and measured pseudoranges
        """
        x_fix, y_fix, z_fix, bias_clock_user = list(vars)
        range_geometric = np.sqrt((x_fix - x_sv)**2 + (y_fix - y_sv)**2 + (z_fix - z_sv)**2)
        prange_expected = range_geometric + bias_clock_user - bias_clock_sv
        deltaprange = prange_measured - prange_expected
        return deltaprange

    def _df(vars):
        """Jacobian of expected pseudorange

        Parameters
        ----------
        vars : list
            List of estimates for position and time
        x_fix : float
            User ECEF x position estimate, scalar, units [m]
        y_fix : float
            User ECEF y position estimate, scalar, units [m]
        z_fix : float
            User ECEF z position estimate, scalar, units [m]
        bias_clock_user : float
            Range bias due to user clock offset (c*dt), scalar, units [m]

        Returns
        -------
        derivatives: ndarray
            Jacobian matrix of expected pseudorange, dimension n-by-4
        """
        x_fix, y_fix, z_fix, bias_clock_user = list(vars)
        range_geometric = np.sqrt((x_fix - x_sv)**2 + (y_fix - y_sv)**2 + (z_fix - z_sv)**2)
        derivatives = np.zeros((len(prange_measured), 4))
        derivatives[:, 0] = -(x_fix - x_sv)/range_geometric
        derivatives[:, 1] = -(y_fix - y_sv)/range_geometric
        derivatives[:, 2] = -(z_fix - z_sv)/range_geometric
        derivatives[:, 3] = -1
        return derivatives

    gpsconsts = GPSConsts()
    if len(prange_measured)<4:
        return np.empty(4)
    # Inital guess for position estimate and clock offset bias
    x_fix, y_fix, z_fix, bias_clock_user = 100., 100., 100., 0.

    x0 = np.array([x_fix, y_fix, z_fix, bias_clock_user])
    X_fix, res_err = newtonraphson(_f, _df, x0, e=e)
    # Return user clock bias in units [ms]
    X_fix[-1] = X_fix[-1]*1e6/gpsconsts.C

    return X_fix

def newtonraphson(f, df, x0, e=1e-3, lam=1., max_count = 20):
    """Newton Raphson method to find zero of function.

    Parameters
    ----------
    f : method
        Function whose zero is required.
    df : method
        Function that outputs derivative of f.
    x0: ndarray
        Initial guess of solution.
    e: float
        Maximum difference between consecutive guesses for termination.
    lam: float
        Scaling factor for step taken at each iteration.
    max_count : int
        Maximum number of iterations to perform before raising an error.

    Returns
    -------
    x0 : ndarray
        Solution for zero of function.
    f_norm : float
        Norm of function magnitude at solution point.
    """
    delta_x = np.ones_like(x0)
    count = 0
    while np.sum(np.abs(delta_x)) > e:
        delta_x = lam*(np.linalg.pinv(df(x0)) @ f(x0))
        x0 = x0 - delta_x
        count += 1
        if count >= max_count:
            raise RuntimeError("Newton Raphson did not converge.")
    f_norm = np.linalg.norm(f(x0))
    return x0, f_norm
