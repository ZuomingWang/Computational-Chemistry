"""
Using Vectors and Matrices
"""

import numpy as np
from typing import Tuple, Union


def normalize_wavefunction(psi: np.ndarray, dx: float) -> np.ndarray:
    """
    Normalizes a wavefunction by dividing by the square root of the integral of the square of the wavefunction.

    Args:
        psi (np.ndarray): The wavefunction to normalize.
        dx (float): The spacing between points in the grid.

    Returns:
        np.ndarray: The normalized wavefunction.
    """
    integral = np.sum(np.abs(psi) ** 2) * dx
    norm = np.sqrt(integral)
    return psi / norm


def evaluate_top_eigenvector_with_power_iteration(
    A: np.ndarray,
    num_iter: int,
) -> Tuple[np.ndarray, Union[float, np.complex128]]:
    """
    Power iteration is an algorithm to find the top eigenvector of a matrix.
    (i.e. the eigenvector corresponding to the largest eigenvalue)
    Despite its simplicity, it is a surprisingly powerful algorithm (in fact,
    it is the basis for the PageRank algorithm used by Google).

    The algorithm is straightforward.  One starts with a random vector
    with ||v|| = 1.  Then,
    1. Multiply the matrix by the vector.
    2. Normalize the resulting vector.

    Repeating this process converges to the top eigenvector of the matrix.
    The eigenvalue can then be approximated by the Rayleigh quotient:
    v^dagger A v.

    Args:
        A (np.ndarray): The matrix for which to find the top eigenvector.
        num_iter (int): The number of iterations to perform

    Returns:
        np.ndarray: The top eigenvector of the matrix,
        float: The eigenvalue corresponding to the top eigenvector.

    Notes:
        A more efficient implementation would check for convergence and stop
        if the vector has not changed significantly between iterations.
        If you are interested in this, you can try implementing it as an extension.
    """
    n = A.shape[0]
    if np.iscomplexobj(A):
        v = np.random.randn(n) + 1j * np.random.randn(n)
    else:
        v = np.random.randn(n)
    v /= np.linalg.norm(v)
    for _ in range(num_iter):
        v = A @ v
        v /= np.linalg.norm(v)
    eigenvalue = np.vdot(v, A @ v)
    return v, eigenvalue


def convert_to_fractional_coordinates(
    lattice_vectors: np.ndarray, cartesian_coordinates: np.ndarray
) -> np.ndarray:
    """
    In crystallography, fractional coordinates are used to describe the position of atoms
    within a crystal unit cell.  Given a set of lattice vectors that define a unit cell,
    an atom's fractional coordinates represent how far along each lattice vector the atom
    is.  For example, an atom at the center of a unit cell has fractional coordinates
    of (0.5, 0.5, 0.5), and atoms located at fractional coordinates (0, 0, 0) or (1, 1, 1)
    are located at a corner of the unit cell.

    Here, you will write a code that converts Cartesian coordinates to fractional coordinates.

    Args:
        lattice_vectors (np.ndarray): A 3x3 matrix where each row is a lattice vector.
        cartesian_coordinates (np.ndarray): An Nx3 matrix where each row is a Cartesian coordinate.

    Returns:
        np.ndarray: An Nx3 matrix where each row is the corresponding fractional coordinate.
    """
    inv_lattice = np.linalg.inv(lattice_vectors)
    return cartesian_coordinates @ inv_lattice


def _gaussian_matrix(bandwidth: float, grid_points: np.ndarray) -> np.ndarray:
    """
    Constructs an num_points x num_points matrix such that entry (i, j) evaluates the
    Gaussian smoother

    K(x, x') = exp(-||x - x'||^2 / (2 * bandwidth^2))

    Args:
        bandwidth (float): The bandwidth of the Gaussian.
        grid_points (np.ndarray): The grid points at which to evaluate the
            Gaussian.  Explicitly assumed to be a single column vector
            of positions.

    Returns:
        np.ndarray: The Gaussian matrix, where entry (i, j) is the
        Gaussian evaluated at grid_points[i] and grid_points[j].

    Notes:
        The name for this function starts with a single underscore.  This is a common
        convention in Python to indicate that the function is a helper function.  It is
        not expected to be called by the user directly, but is primarily for internal use.
    """
    x = grid_points.reshape(-1, 1)
    diffs = x - x.T
    squared_dists = diffs ** 2
    K = np.exp(-squared_dists / (2 * (bandwidth ** 2)))
    return K


def apply_gaussian_smoothing(
    y: np.ndarray, x_min: np.ndarray, x_max: np.ndarray, bandwidth=1.0
) -> np.ndarray:
    """
    Applies a Gaussian smoothing to a signal to remove high-frequency noise.
    Mathematically, the smoothed signal is given by

    s(x) = (K y) / (K 1)

    where K is the matrix of evaluated Gaussian values, y is the original signal,
    and 1 is the vector of ones.
    Args:
        y (np.ndarray): The original signal, assumed to be taken at evenly spaced points
            between x_min and x_max.
        x_min (np.ndarray): The minimum value of the x-axis.
        x_max (np.ndarray): The maximum value of the x-axis.
        bandwidth (float): The bandwidth of the Gaussian.

    Returns:
        np.ndarray: The baseline drift-corrected signal.

    """
    n = len(y)
    grid = np.linspace(x_min, x_max, n)
    K = _gaussian_matrix(bandwidth, grid)
    K_y = K @ y
    K_ones = K @ np.ones(n)
    smoothed = K_y / K_ones
    return smoothed