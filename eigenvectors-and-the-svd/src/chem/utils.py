import numpy as np


def expand_wavefunction_from_coefficients(
    coeffs: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """
    Given a set of coefficients for a wavefunction in the plane wave basis,
    expand the wavefunction in real space.

    Args:
        coeffs (np.ndarray) : An array of shape (2k+1,) containing the coefficients
            of the wavefunction in the plane wave basis.
        grid (np.ndarray) : An array of shape (n,) containing the grid points.

    Returns:
        np.ndarray : An array of shape (n,) containing the wavefunction in real space.
    """
    L = grid[-1] - grid[0]
    kmax = (len(coeffs) - 1) // 2
    k_values = np.arange(-kmax, kmax + 1)

    psi_x = np.zeros_like(grid, dtype=np.complex128)
    for coeff_k, k in zip(coeffs, k_values):
        psi_x += coeff_k * np.exp(1j * 2 * np.pi * k * grid / L) / np.sqrt(L)
    return psi_x
