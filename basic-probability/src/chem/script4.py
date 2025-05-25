import numpy as np


def sample_exponential_using_inverse_transform(uniform: float, lam: float) -> float:
    """
    Samples an exponential random variable with rate parameter lambda
    using the inverse transform method.

    An exponential random variable has the probability density function:
    f(x) = lambda * exp(-lambda * x) for x >= 0, and f(x) = 0 for x < 0.

    Parameters
    ----------
    uniform : float
        A random number sampled from a uniform distribution between 0 and 1.
    lam : float
        The rate parameter of the exponential distribution.

    Returns
    -------
    float
        The sampled exponential random variable.
    """
        # Using the inverse transform: x = -ln(1-u)/lam.
    return -np.log(1.0 - uniform) / lam


def calculate_free_energy_difference(
    prob_mass_fxn: np.ndarray, state_a_indices: np.ndarray, state_b_indices: np.ndarray
) -> float:
    """
    Calculates the free energy difference between two states, A and B, given the
    probability mass function and the indices of the states in the probability mass
    function for the two states.

    Parameters
    ----------
    prob_mass_fxn : np.ndarray
        The probability mass function of the system.
    state_a_indices : np.ndarray
        The indices of the states in state A.
    state_b_indices : np.ndarray
        The indices of the states in state B.

    Returns
    -------
    float
        The free energy difference between states A and B, in units of kT.
    """
    # Sum probabilities over the provided indices
    p_a = np.sum(prob_mass_fxn[state_a_indices])
    p_b = np.sum(prob_mass_fxn[state_b_indices])
    
    if p_a <= 0 or p_b <= 0:
        raise ValueError("The sum of probabilities for each state must be positive.")
    
    # Compute the free energy difference ΔF = -ln(P_B / P_A)
    delta_F = -np.log(p_b / p_a)
    return delta_F


def evaluate_prob_density_fxn_of_rv_sum(
    prob_mass_fxn_a: np.ndarray, prob_mass_fxn_b: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """
    Evaluates the probability density function of the sum of two random variables a and b.
    The probability density function of the sum is given by the convolution of the
    probability density functions of the individual random variables.

    Parameters
    ----------
    prob_density_fxn_a: np.ndarray
        The probability density function of random variable A
        evaluated at the gridpoints in `grid`.
    prob_density_fxn_b: np.ndarray
        The probability density function of random variable B
        evaluated at the gridpoints in `grid`.
    grid: np.ndarray
        The gridpoints at which the probability density functions are evaluated,
        in increasing order.  Assumed to be evenly spaced, of size $M$

    Returns
    -------
    np.ndarray
        The probability mass function of the sum of the two random variables,
        evaluate at all 2 M - 1 possible points, from grid[0] + grid[0] to grid[-1] + grid[-1].
    """
    ### This is a perfectly fine solution, and how you would do it in practice.
    # grid_spacing = grid[1] - grid[0]
    # prob_density_fxn_sum = np.convolve(prob_mass_fxn_a, prob_mass_fxn_b) * grid_spacing
    # return prob_density_fxn_sum
    M = len(grid)
    grid_spacing = grid[1] - grid[0]
    out_length = 2 * M - 1
    # Initialize the convolution result array with zeros.
    conv_result = np.zeros(out_length)
    
    # Manually compute the convolution
    for i in range(M):
        for j in range(M):
            conv_result[i + j] += prob_mass_fxn_a[i] * prob_mass_fxn_b[j]
    
    # Multiply by grid_spacing to account for the discretization of the integration
    conv_result *= grid_spacing
    return conv_result

