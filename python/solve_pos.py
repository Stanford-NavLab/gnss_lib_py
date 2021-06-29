import numpy as np
from . import constants

def newton_raphson(f, df, x0, e=1e-3, lam=1.):
    delta_x = np.ones_like(x0)
    while np.sum(np.abs(delta_x))>e:
        delta_x = lam*(np.linalg.pinv(df(x0)) @ f(x0))
        x0 = x0 - delta_x
    return x0, np.linalg.norm(f(x0))

def _solve_pos(prange, X, Y, Z, B, e=1e-3):
  if len(prange)<4:
    return np.empty(4)
  x, y, z, cdt = 100., 100., 100., 0.
  
  def f(vars):
    x, y, z, cdt = list(vars)
    tilde_prange = np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
    _prange = tilde_prange + cdt - B
    delta_prange = prange-_prange
    return delta_prange

  def df(vars):
    x, y, z, cdt = list(vars)
    tilde_prange = np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
    _prange = tilde_prange + cdt - B
    delta_prange = prange-_prange
    derivatives = np.zeros((len(prange), 4))
    derivatives[:, 0] = -(x - X)/tilde_prange
    derivatives[:, 1] = -(y - Y)/tilde_prange
    derivatives[:, 2] = -(z - Z)/tilde_prange
    derivatives[:, 3] = -1
    return derivatives
  
  x0 = np.array([x, y, z, cdt])
  x_fix, res_err = newton_raphson(f, df, x0, e=e)
  x_fix[-1] = x_fix[-1]*1e6/constants.LIGHTSPEED

  return x_fix