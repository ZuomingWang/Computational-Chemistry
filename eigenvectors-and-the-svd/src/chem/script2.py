import numpy as np


def apply_low_rank_svd_filter(A: np.ndarray, rank: int) -> np.ndarray:
    """
    Apply a low rank SVD filter to the input matrix A. This function should
    return the filtered matrix.

    Specifically, this function should perform the following steps:
    1. Compute the SVD of the input matrix A.
    2. Set all but the first `rank` singular values to zero.
    3. Reconstruct and return the matrix using the modified singular values.

    Args:
        A (np.ndarray) : The input matrix to filter.
        rank (int) : The rank of the filter, i.e., the number of singular values to keep.

    Returns:
        np.ndarray : The filtered matrix.
    """
    # Compute SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Keep only the top `rank` singular values
    S[rank:] = 0
    
    # Reconstruct the filtered matrix
    return U @ np.diag(S) @ Vt


def evaluate_1d_plane_wave_hamiltonian(
    grid_points: np.ndarray,
    potential_energy: np.ndarray,
    k_max: int,
) -> np.ndarray:
    """
    Evaluate elements of the Hamiltonian matrix in the basis of plane-waves.
    The plane waves are defined as
        psi_k(x) = exp(i 2 \pi k x / L) / sqrt(L),
    where X is the grid point,  and L is the length of the domain (the maximum
    grid point minus the minimum grid point).

    We use reduced units, with hbar = 1 and m = 1.  Consequently, the Hamiltonian
    operator is given by:
        H = -1/2 d^2/dx^2 + V(x),
    where V(x) is the potential energy at each grid point.

    Your job is to evaluate the Hamiltonian matrix in the basis of plane waves:
        <psi_k | H | psi_j> = \int dx psi_k^*(x) H psi_j(x).

    Args:
        grid_points (np.ndarray) : An array of shape (n,) containing the grid points.
            We assume that the grid points are evenly spaced.
        potential_energy (np.ndarray) : An array of shape (n,) containing the potential
            energy at each grid point.
        k_max (int) : The maximum value of k to include in the basis of plane waves:
            the basis will include all integer k from -k_max to k_max.

    Returns:
        np.ndarray : An (2k+1) x (2k+1) matrix representing the Hamiltonian matrix,
            evaluated at from -k_max to k_max in the basis of plane waves.
    """
    n = len(grid_points)
    L = grid_points[-1] - grid_points[0]  # Domain length
    k_vals = np.arange(-k_max, k_max + 1)
    num_k = len(k_vals)
    
    H = np.zeros((num_k, num_k), dtype=np.complex128)
    dx = grid_points[1] - grid_points[0]  # Assume uniform spacing
    
    
    # Kinetic energy term
    for i, k_i in enumerate(k_vals):
        for j, k_j in enumerate(k_vals):
            if k_i == k_j:
                H[i, j] = (2 * np.pi * k_i / L) ** 2 / 2
    
    # Potential energy term
    for i, k_i in enumerate(k_vals):
        for j, k_j in enumerate(k_vals):
            sum_integral = np.sum(
                potential_energy * np.exp(-1j * 2 * np.pi * (k_i - k_j) * grid_points / L)
            )
            H[i, j] += sum_integral * dx / L
    
    return H


def propagate_wavefunctions(
    coeffs_0: np.ndarray,
    potential: np.ndarray,
    grid_points: np.ndarray,
    k_max: int,
    time_points: np.ndarray,
) -> np.ndarray:
    """
    Given an initial wavefunction and a Hamiltonian matrix,
    propagate the wavefunctions forward in time using the eigendecomposition
    of the Hamiltonian.

    Args:
        coeffs_0 (np.ndarray) : An array of shape (2k+1,) containing the coefficients
            of the wavefunction in the plane wave basis.
        potential (np.ndarray) : An array of shape (n,) containing the potential
            energy at each grid point.
        grid_points (np.ndarray) : An array of shape (n,) containing the grid points.
        k_max (int) : The maximum value of k to include in the basis of plane waves.
        time_points (np.ndarray) : An array of shape (m,) containing the times at which
            to evaluate the wavefunctions.

    Returns:
        np.ndarray : An array of shape (2k+1, m) containing the coefficients of the
            wavefunctions at each time point.  The i-th column of the output array
            should contain the coefficients for the i-th time point.
    """
    # Compute the Hamiltonian matrix
    H = evaluate_1d_plane_wave_hamiltonian(grid_points, potential, k_max)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    # Transform initial coefficients to eigenbasis
    c = eigenvectors.T.conj() @ coeffs_0
    
    # Time evolution
    wavefunctions = np.zeros((len(coeffs_0), len(time_points)), dtype=np.complex128)
    for i, t in enumerate(time_points):
        wavefunctions[:, i] = eigenvectors @ (c * np.exp(-1j * eigenvalues * t))
    
    return wavefunctions
