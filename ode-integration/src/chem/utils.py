import numpy as np
from typing import Callable, Tuple


def convert_force_fxn_into_deriv_fxn(force_function: Callable) -> Callable:
    """
    Converts a force function into a derivative function.  Note that this
    very deep in functional programming

    Parameters
    ----------
    force_function : Callable
        A function that calculates the force acting on the system given the current position and velocity.

    Returns
    -------
    Callable
        A function that calculates the derivative of the system given the current position and velocity.

    Notes
    -----
    We are explicitly assuming the mass is 1.
    """

    # Functional programming tricks!  We define a function inside a function,
    # which we then return.
    def deriv_function(x: np.ndarray, v: np.ndarray) -> np.ndarray:
        force = force_function(x)
        deriv = np.stack((v, force), axis=-2)
        return deriv

    return deriv_function
