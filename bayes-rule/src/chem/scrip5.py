import numpy as np


def bayesian_particle_identification(
    signal: np.ndarray,
    positions: np.ndarray,
    prior: np.ndarray,
    particle_width: float,
    noise: float,
) -> np.ndarray:
    """
    Calculates the log posterior probability for the location of a particle,
    and its width, in a 1D signal.  The prior is modeled as a Bell curve with
    possible origin positions given in the numpy array `positions`.

    In the absence of noise, our particle would emit a signal that is a Gaussian

    .. math::

        f(x) = \exp\left(-\frac{(x - x_0)^2}{2 w^2}\right)

    where $x_0$ is the position of the particle and $w$ is the width of the particle.

    Given a value of the position, the likelihood of the signal
    is then given by

    .. math::

        L(x) = \prod_i \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(f(x_i) - s_i)^2}{2 sigma^2}\right)

    where $s_i$ is the value of the signal at x_i, $f(x)$ is the model of the
    particle given above, and $\sigma$ is the standard deviation of the
    experimental noise.

    Parameters
    ----------
    signal : np.ndarray
        The 1D signal that the particle emits.
    positions : np.ndarray
        A discretization of space, given by a 1D array.  These are the
        possible positions of the particle, as well as the points at
        which the signal was measured.  As usual, assumed to be evenly
        spaced.
    prior : np.ndarray
        The prior probability of the particle being at each position and
        width.  This is a 1D array with shape (len(positions))
    widths : float
        The width of the particle.
    noise : float
        The standard deviation of the noise in the signal.

    Returns
    -------
    np.ndarray
        The log posterior probability of the particle being at each position.
        Note that the log posterior is not normalized, so you will need to
    """
    # Ensure that the input arrays have correct dimensions
    if len(signal) != len(positions):
        raise ValueError("Signal and positions arrays must have the same length.")
    if len(prior) != len(positions):
        raise ValueError("Prior array must have the same length as positions.")
    
    # Compute the expected signal for each possible particle position
    likelihoods = np.zeros_like(positions, dtype=np.float64)
    
    for i, x0 in enumerate(positions):
        model_signal = np.exp(-((positions - x0) ** 2) / (2 * particle_width ** 2))
        log_likelihood = -np.sum(((model_signal - signal) ** 2) / (2 * noise ** 2))
        likelihoods[i] = log_likelihood
    
    # Compute the log posterior
    log_posterior = likelihoods + np.log(prior)
    
    # Normalize the posterior
    posterior = np.exp(log_posterior)
    posterior /= np.sum(posterior) * (positions[1] - positions[0])
    log_posterior = np.log(np.clip(posterior, a_min=1e-10, a_max=None))  # Avoid log(0)
    
    return log_posterior
