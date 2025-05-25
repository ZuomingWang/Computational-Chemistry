import numpy as np
from typing import Tuple
from tqdm import tqdm


def gibbs_sample_ising_model(
    initial_spins: np.ndarray,
    J: float = 1.0,
    kT: float = 1.0,
    N_steps: int = 100,
) -> np.ndarray:
    """
    Perform Gibbs sampling on an Ising model.

    Parameters
    ----------
    initial_spins : np.ndarray
        The initial spins of the Ising model.
    J : float
        The interaction constant.
    N_steps : int
        The number of steps to perform.  In each step we update each spin once
        in a random order.

    Returns
    -------
    np.ndarray
        The final spins of the Ising model.
    """
    spins = initial_spins.copy()
    N = spins.shape[0]

    list_of_all_index_pairs = [(i, j) for i in range(N) for j in range(N)]

    samples = [initial_spins]
    for _ in tqdm(range(N_steps)):
        # Shuffle index pair list
        np.random.shuffle(list_of_all_index_pairs)
        for k in list_of_all_index_pairs:
            p = _evaluate_gibbs_sampling_probability(k, spins, J, kT)
            spins[k[0], k[1]] = 2 * (np.random.rand() < p) - 1
        samples.append(spins.copy())
    return np.array(samples)


def _evaluate_gibbs_sampling_probability(
    k: Tuple[int, int], spins: np.ndarray, J: float = 1.0, kT: float = 1.0
) -> float:
    """
    Evaluates the probability of sampling a value of +1 for a given site in the Ising model.

    Parameters
    ----------
    k : Tuple[int, int]
        The index of the site.
    spins : np.ndarray
        The spins of the Ising model.
    J : float
        The interaction constant.
    kT : float
        The product of the Boltzmann constant and the temperature.

    Returns
    -------
    float
        The probability of sampling a value of +1 for the site.
    """
    i, j = k
    N = spins.shape[0]
    
    # Sum the spins of the four nearest neighbors with periodic boundaries
    sum_neighbors = (
        spins[(i - 1) % N, j] +  # Up
        spins[(i + 1) % N, j] +  # Down
        spins[i, (j - 1) % N] +  # Left
        spins[i, (j + 1) % N]    # Right
    )
    
    # Compute the probability
    p = 1 / (1 + np.exp(-2 * J / kT * sum_neighbors))
    return p
    raise NotImplementedError("You have to implement this function!")


def total_ising_model_energy(
    spins: np.ndarray,
    J: float = 1.0,
) -> float:
    r"""
    Calculate the energy of an Ising model.

    The energy of an Ising model is given by

    .. math::

        E = -\sum_{i, j \in NN} J_{ij} s_i s_j

    where $s_i$ is the spin of site $i$ and $J_{ij}$ is the interaction
    between sites $i$ and $j$. The sum is over pairs of nearest neighbors
    $i$ and $j$.  On a square lattice (as in this case), the nearest
    neighbors are the four sites to the left, right, above, and below
    site $k$.  The lattice is assumed to be periodic, so that the
    a spin on the left edge has a nearest neighbor on the right edge,
    a spin on the top edge has a nearest neighbor on the bottom edge, etc.


    Parameters
    ----------
    spins : np.ndarray
        The spins of the Ising model.
    J : float
        The interaction constant.

    Returns
    -------
    float
        The energy of the Ising model.

    Notes
    -----
    Be sure not to double-count the energy of each pair of spins!
    """
    # Horizontal interactions (right neighbors)
    horizontal = np.sum(spins * np.roll(spins, -1, axis=1))
    # Vertical interactions (down neighbors)
    vertical = np.sum(spins * np.roll(spins, -1, axis=0))
    # Total energy
    energy = -J * (horizontal + vertical)
    return energy
    raise NotImplementedError("You have to implement this function!")
