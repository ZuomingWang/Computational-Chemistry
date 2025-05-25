import numpy as np
from typing import Callable, Tuple
from abc import ABC, abstractmethod
from chem.velocity_verlet import velocity_verlet


class MetropolisMonteCarlo:
    def __init__(self, energy_function):
        """
        Initialize the Metropolis Monte Carlo algorithm.

        Parameters
        ----------
        energy_function : Callable
            A function that calculates the energy of a given state.
            (or equivalently, the negative log probability density for the state.)
        """
        self.energy_function = energy_function

    def sample_trajectory(
        self, x_0: np.ndarray, n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a trajectory using the Metropolis Monte Carlo algorithm.

        Parameters
        ----------
        x_0 : np.ndarray
            The initial state of the system.

        Returns
        -------
        np.ndarray
            The trajectory caused by running the Metropolis Monte Carlo algorithm.
        np.ndarray
            The energies of the states in the trajectory.
        """
        x = x_0.copy()
        E = self.energy_function(x)
        traj = [x_0.copy()]
        energies = [E]

        for _ in range(n_steps):
            random_uniform = np.random.uniform()
            x, E = self._metropolis_monte_carlo_update(x, E, random_uniform)
            traj.append(x.copy())
            energies.append(E)
        return np.array(traj), np.array(energies)

    def _metropolis_monte_carlo_update(
        self,
        x: np.ndarray,
        E_current: float,
        random_uniform: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Perform a single Metropolis update step.

        Parameters
        ----------
        x : np.ndarray
            The current state of the system.
        proposal_function : Callable
            A function that generates a proposed new state given the current state.
            Assumed to be symmetric, i.e., the probability of proposing a new state
            i from state j is the same as proposing state j from state i.
        energy_function : Callable
            A function that calculates the energy of a given state.
            We assume that kT=1, so that the energy is equivalent to the negative log
            probability density for the state.
        random_uniform : float
            A random number drawn uniformly distributed between 0 and 1.

        Returns
        -------
        np.ndarray
            The updated state of the system.
        float
            The energy of the updated state.

        Notes
        -----
        will need to use self.proposal_function and self.energy_function
        """
        x_new = self.proposal_function(x)
        E_new = self.energy_function(x_new)
        delta_E = E_new - E_current
        if random_uniform < np.exp(-delta_E):
            return (x_new, E_new)
        else:
            return x, E_current

    @abstractmethod
    def proposal_function(self, x_t: np.ndarray) -> np.ndarray:
        """
        Generate a proposed new state given the current state.
        Assumed to be symmetric, i.e., the probability of proposing a new state
        i from state j is the same as proposing state j from state i.

        Parameters
        ----------
        x_t : np.ndarray
            The current state of the system.

        Returns
        -------
        np.ndarray
            The proposed new state.
        """
        pass


class GaussianMetropolisMC(MetropolisMonteCarlo):
    def __init__(self, energy_function: Callable, sigma: float):
        """
        Initialize the Gaussian Proposal Metropolis algorithm.

        Parameters
        ----------
        energy_function : Callable
            A function that calculates the energy of a given state.
            (or equivalently, the negative log probability density for the state.)
        sigma : float
            The standard deviation of the Gaussian noise used for proposals.
        """
        super().__init__(energy_function)
        self.sigma = sigma

    def proposal_function(self, x: np.ndarray) -> np.ndarray:
        """
        Generate a proposed new state by adding Gaussian noise to the current state.

        Parameters
        ----------
        x : np.ndarray
            The current state of the system.

        Returns
        -------
        np.ndarray
            The proposed new state.
        """
        return x + np.random.normal(0, self.sigma, size=x.shape)


class HamiltonianMC(MetropolisMonteCarlo):
    def __init__(
        self,
        energy_function: Callable,
        force_function: Callable,
        dt: float,
        n_steps_proposal: int,
    ):
        """
        Initialize the Hamiltonian Monte Carlo algorithm.

        Parameters
        ----------
        force_function : Callable
            A function that calculates the force acting on the system given the current position and velocity.
        dt : float
            The time step for the integration.
        n_steps_proposal : int
            The number of steps to run per proposal
        """
        super().__init__(energy_function)
        self.force_function = force_function
        self.dt = dt
        self.n_steps_proposal = n_steps_proposal

    def proposal_function(self, x: np.ndarray) -> np.ndarray:
        """
        Generate a proposed new state using Hamiltonian Monte Carlo.

        Parameters
        ----------
        x : np.ndarray
            The current state of the system.
        dt : float
            The time step for the integration.
        force_function : Callable
            A function that calculates the force acting on the system given the current position and velocity.

        Returns
        -------
        np.ndarray
            The proposed new state.
        """
        v = np.random.normal(size=x.shape)  # Boltzmann distribution with mass and kT =1
        x_0 = np.concatenate([x, v])
        x_new = velocity_verlet(
            x_0, n_steps=1, dt=self.dt, force_function=self.force_function
        )
        return x_new[-1][:len(x)]

    # def advanced_sample(self, x_init: np.ndarray) -> np.ndarray:
    #     self._metropolis_monte_carlo_update()
    def advanced_sample(self, x_init: np.ndarray) -> Tuple[np.ndarray, float]:
        E_current = self.energy_function(x_init)
        random_uniform = np.random.uniform()
        return self._metropolis_monte_carlo_update(x_init, E_current, random_uniform)
