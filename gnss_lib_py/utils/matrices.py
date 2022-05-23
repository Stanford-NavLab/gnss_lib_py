"""Matrix size, condition check wrappers
"""

__authors__ = "Ashwin Kanhere"
__date__ = "19 June, 2020"


import numpy as np


def check_col_vect(vect, dim):
    """Boolean for whether input vector is column shaped or not

    Parameters
    ----------
    vect : np.ndarray
        Input vector
    dim : int
        Number of row elements in column vector
    """
    check = False
    if np.shape(vect)[0] == dim and np.shape(vect)[1] == 1:
        check = True
    return check
    

def check_square_mat(mat, dim):
    """Boolean for whether input matrices are square or not

    Parameters
    ----------
    vect : np.ndarray
        Input matrix
    dim : int
        Number of elements for row and column = N for N x N
    """
    check = False
    if np.shape(mat)[0] == dim and np.shape(mat)[1] == dim:
        check = True
    return check