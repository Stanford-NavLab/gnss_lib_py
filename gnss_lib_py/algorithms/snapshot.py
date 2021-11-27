"""Point solution methods using GNSS measurements.

This module contains point solution methods for estimating position
at a single GNSS measurement epoch. Position is solved using Newton-Raphson
or Weighted Least Squares algorithms.

Notes
-----
    Weighted Least Squares solver is not yet implemented. There is not an input
    field for specifying weighting matrix.

"""

__authors__ = "Shubh Gupta, Bradley Collicott"
__date__ = "19 July 2021"

import numpy as np

def solvepos(
    prange_measured:np.ndarray,
    x_sv:np.ndarray,
    y_sv:np.ndarray,
    z_sv:np.ndarray,
    b_clk_sv:np.ndarray,
    tol:float=1e-3
):
    # TODO: Modify code to perform WLS if weights are given
    # TODO: Change inputs to either DataFrame or Matrix
    """Find user position, clock bias using WLS or NR methods.

    Find user position, clock bias by solving the weighted least squares
    (WLS) problem for n satellites. If no weights are given, the Newton
    Raphson (NR) position, clock bias solution is used instead.

    Parameters
    ----------
    prange_measured : ndarray
        Measured pseudoranges, dimension n
    x_sv : np.ndarray
        Satellite x positions, dimension n, units [m]
    y_sv : np.ndarray
        Satellite y positions, dimension n, units [m]
    z_sv : np.ndarray
        Satellite z positions, dimension n, units [m]
    b_clk_sv : np.ndarray
        Range bias due to satellite clock offset (c*dt), dimension n, units [m]
    tol : float
        Termination threshold for LS solver, units [~]

    Returns
    -------
    user_fix : np.ndarray
        Solved 3D position and clock bias estimate, dimension 4-by-1,
        units [m]
    """

    def _compute_prange_residual(user_fix):
        """Compute the difference between expected and received pseudoranges.

        Parameters
        ----------
        user_fix : list
            List of estimates for position and time
        x_fix : float
            User x position estimate, scalar, units [m]
        y_fix : float
            User y position estimate, scalar, units [m]
        z_fix : float
            User z position estimate, scalar, units [m]
        b_clk_u : float
            Range bias due to user clock offset (c*dt), scalar, units [m]

        Returns
        -------
        prange_residual: np.ndarray
            Float difference between expected and measured pseudoranges

        """
        x_fix, y_fix, z_fix, b_clk_u = list(user_fix)
        range_geometric = np.sqrt((x_fix - x_sv)**2
                                + (y_fix - y_sv)**2
                                + (z_fix - z_sv)**2)

        prange_expected = range_geometric + b_clk_u - b_clk_sv
        prange_residual = prange_measured - prange_expected
        return prange_residual

    def _compute_prange_partials(user_fix):
        # TODO: Vectorize jacobian calculation
        """Compute the Jacobian of expected pseudorange with respect to the
        user states.

        Parameters
        ----------
        user_fix : list
            List of estimates for position and time
        x_fix : float
            User x position estimate, scalar, units [m]
        y_fix : float
            User y position estimate, scalar, units [m]
        z_fix : float
            User z position estimate, scalar, units [m]
        b_clk_u : float
            Range bias due to user clock offset (c*dt), scalar, units [m]

        Returns
        -------
        derivatives: np.ndarray
            Jacobian matrix of expected pseudorange, dimension n-by-4

        """
        x_fix, y_fix, z_fix, _ = list(user_fix)
        range_geometric = np.sqrt((x_fix - x_sv)**2
                                + (y_fix - y_sv)**2
                                + (z_fix - z_sv)**2)

        derivatives = np.zeros((len(prange_measured), 4))
        derivatives[:, 0] = -(x_fix - x_sv)/range_geometric
        derivatives[:, 1] = -(y_fix - y_sv)/range_geometric
        derivatives[:, 2] = -(z_fix - z_sv)/range_geometric
        derivatives[:, 3] = -1
        return derivatives

    if len(prange_measured)<4:
        return np.empty(4)
    # Inital guess for position estimate and clock offset bias
    x_fix, y_fix, z_fix, b_clk_u = 0., 0., 0., 0.
    user_fix = np.array([x_fix, y_fix, z_fix, b_clk_u])

    user_fix, _ = newton_raphson(
        _compute_prange_residual,
        _compute_prange_partials,
        user_fix,
        tol=tol
    )

    return user_fix.reshape([-1,1])

def newton_raphson(f_x, df_dx, x_0, tol = 1e-3, lam = 1., max_count = 20):
    """Newton-Raphson method to find zero of function.

    Parameters
    ----------
    f_x : method
        Function whose zero is required.
    df_dx : method
        Function that outputs derivative of f_x.
    x_0: np.ndarray
        Initial guess of solution.
    tol: float
        Maximum difference between consecutive guesses for termination.
    lam: float
        Scaling factor for step taken at each iteration.
    max_count : int
        Maximum number of iterations to perform before raising an error.

    Returns
    -------
    x0 : np.ndarray
        Solution for zero of function.
    f_norm : float
        Norm of function magnitude at solution point.

    """
    delta_x = np.ones_like(x_0)
    count = 0
    while np.sum(np.abs(delta_x)) > tol:
        delta_x = lam*(np.linalg.pinv(df_dx(x_0)) @ f_x(x_0))
        x_0 = x_0 - delta_x
        count += 1
        if count >= max_count:
            raise RuntimeError("Newton Raphson did not converge.")
    f_norm = np.linalg.norm(f_x(x_0))
    return x_0, f_norm
