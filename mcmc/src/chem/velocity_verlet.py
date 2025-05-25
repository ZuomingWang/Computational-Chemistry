import numpy as np
from typing import Callable, Tuple
from scipy.optimize import fsolve
from abc import ABC, abstractmethod


def velocity_verlet(
    x_0: np.ndarray,
    n_steps: int,
    dt: float,
    force_function: Callable,
) -> np.ndarray:
    """
    Performs an update of velocity verlet for n steps.

    Velocity Verlet is a simple algorithm that is closely related to the
    leapfrog algorithm that we discussed in class:  The updated proceeds in the following steps:

    1. Calculate the acceleration at the current position and velocity.
        a(t) = f(q_t, v_t)
    2. Integrate the velocity to t = t + dt/2 using Forward Euler
        v(t + dt/2) = v(t) + a(t) * dt / 2
    3. Calculate the new position at t = t + dt using the velocity at t + dt/2 (midpoint update)
        q(t + dt) = q(t) + v(t + dt/2) * dt
    4. Calculate the new acceleration at the new position at t = t + dt
        a(t + dt) = f(q(t + dt), v(t + dt/2))
    5. Integrate the velocity to t = t + dt using Backward Euler
        v(t + dt) = v(t + dt/2) + a(t + dt) * dt / 2

    This process is then repeated to integrate the trajectory.

    Parameters
    ----------
    x_0 : np.ndarray
        The state of the system at time t = 0.  This should be a length 2N
        array where $N$ is the number of position coordinates.  The first N
        elements are all of the position coordinates, and the second N elements
        are all of the velocity coordinates.
    n_steps : int
        The number of steps to run the algorithm for.
    dt : float
        The time step for the integration.
    force_function : Callable
        A function that calculates the force acting on the system given the
        current position and velocity.  The force is assumed to be a function
        of only the position , i.e. f(q).

    Returns
    -------
    np.ndarray
        The resulting trajectory of the system after n_steps.  Should be a
        length 2N array of dimension (N_steps+1) x 2N, where N_steps is the
        number of steps taken.  (It is N_steps+1 because we include the initial
        state).  The first N columns are the position coordinates, and the
        second N columns are the velocity coordinates.


    Notes
    -----
    - We are explicitly assuming the mass is 1.
    - You can re-use the force evaluation from step 4 to evaluate step 1 in the
      next iteration.
    - A quick calculation should convince you that this is mathematically
      equivalent to the leapfrog algorithm.
    """
    # Split the initial state into position and velocity
    N = len(x_0) // 2
    q = x_0[:N].copy()
    v = x_0[N:].copy()

    a = force_function(q)  # Initial acceleration

    trajectory = np.zeros((n_steps + 1, 2 * N))
    trajectory[0, :N] = q
    trajectory[0, N:] = v 


    for step in range(1, n_steps + 1):
        # Step 2: Velocity half-step
        v_half = v + 0.5 * dt * a

        # Step 3: Position full-step
        q = q + dt * v_half

        # Step 4: Acceleration at new position
        a_new = force_function(q)

        # Step 5: Velocity full-step
        v = v_half + 0.5 * dt * a_new

        a = a_new  # Save for next loop

        trajectory[step, :N] = q
        trajectory[step, N:] = v

    return trajectory