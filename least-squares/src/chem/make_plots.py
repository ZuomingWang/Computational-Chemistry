import numpy as np
import matplotlib.pyplot as plt
from chem import script3
from typing import Callable


def double_well_potential(x: np.ndarray) -> np.ndarray:
    """
    Evaluates the double well potential at the given points.

    Args:
        x (np.ndarray): The points at which to evaluate the potential.

    Returns:
        np.ndarray: The potential values.
    """
    return (x - 2) ** 2 * (x + 2) ** 2 + x


def perform_force_field_regression():
    """
    Here, we attempt to regress a force field from noisy measurements of a double well potential.
    We will use a polynomial basis to represent the force field, and use Tikhonov regularization
    to stabilize the regression.
    """
    x = np.linspace(-3, 3, 200)
    U = double_well_potential(x)

    # Randomly sample 10 points: these are our ``measurements''
    np.random.seed(0)
    sampled_indices = np.random.choice(len(x), 20, replace=False)
    sampled_x = x[sampled_indices]
    sampled_U = U[sampled_indices] + np.random.normal(0, 6.0, len(sampled_x))

    # Evaluate the first 20 polynomial basis functions
    basis_functions_at_samples = np.array([sampled_x**i for i in range(10)]).T
    basis_functions_at_all_points = np.array([x**i for i in range(10)]).T

    values_of_alpha = np.logspace(-3, 1, 9)

    regularized_solutions = []
    residuals = []
    for alpha in values_of_alpha:
        coeffs = script3.tikhonov_lst_sq(basis_functions_at_samples, sampled_U, alpha)
        U_hat = basis_functions_at_all_points @ coeffs

        residual = np.sum((sampled_U - basis_functions_at_samples @ coeffs) ** 2)
        regularized_solutions.append(U_hat)
        residuals.append(residual)

    # Plot the results  One big plot on the left, and a 3 x 3 grid of small plots on the right
    fig = plt.figure(figsize=(12, 6))
    ax_big = plt.subplot2grid((3, 6), (0, 0), rowspan=3, colspan=3)
    ax_big.plot(values_of_alpha, residuals, marker="o", color="k")
    ax_big.set_xscale("log")
    # ax_big.set_yscale("log")
    ax_big.set_xlabel("alpha")
    ax_big.set_ylabel("Residual")
    ax_big.set_title("Residual vs. alpha")

    ax_small = [plt.subplot2grid((3, 6), (i // 3, 3 + i % 3)) for i in range(9)]
    for i, ax in enumerate(ax_small):
        ax.plot(x, U, label="True")
        ax.plot(x, regularized_solutions[i], label="Regressed")
        ax.plot(sampled_x, sampled_U, ".", label="Sampled")
        ax.set_title(r"$\alpha$ = %.2f" % values_of_alpha[i])
        ax.set_ylim(-10, 20)
        # ax.legend()

    ax_small[0].legend(fontsize=5)
    plt.tight_layout()
    plt.show()


def build_1d_transition_matrix(
    U_fxn: Callable, grid: np.ndarray, kT: float
) -> np.ndarray:
    """
    Build a 1D transition matrix from a potential energy function.

    Args:
        U_fxn (Callable): The potential energy function.
        grid (np.ndarray): The grid of points in the 1D space.
        kT (float): The thermal energy.

    Returns:
        np.ndarray: The transition matrix.

    Notes: Here we are using a paradigm known as ``function programming''.
    We are passing a function as an argument to another function.
    This is a powerful paradigm that allows for more modular and
    reusable code.
    """
    n = len(grid)
    T = np.zeros((n, n))

    U_values = U_fxn(grid)
    dU_forward = U_values[1:] - U_values[:-1]
    dU_backward = U_values[:-1] - U_values[1:]

    # Hop forward / backward with a probability proportional to the Boltzmann factor
    T_matrix = np.eye(n)
    T_matrix += np.diag(np.exp(-dU_forward / kT), 1)
    T_matrix += np.diag(np.exp(-dU_backward / kT), -1)

    # Normalize the rows
    T_matrix /= T_matrix.sum(axis=1, keepdims=True)
    return T_matrix


def plot_double_well_committors():

    grid = np.linspace(-3, 3, 200)

    state_a = np.where(grid < -2)[0]
    state_b = np.where(grid > 2)[0]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    axes[0].plot(grid, double_well_potential(grid), color="k")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("U(x)")
    axes[0].set_ylim(None, 20)
    axes[0].set_title("Double well potential")

    for i, kT in enumerate([1, 10, 100]):
        transition_matrix = build_1d_transition_matrix(double_well_potential, grid, kT)
        committor = script3.calculate_committor(transition_matrix, state_a, state_b)
        axes[i + 1].plot(grid, committor, label=f"kT = {kT}")
        axes[i + 1].set_xlabel("x")
        axes[i + 1].set_ylabel("q(x)")
        axes[i + 1].set_title(f"kT = {kT}")
        # axes[i+1].legend()

    plt.tight_layout()
    plt.savefig("double_well_committors.png")
    plt.show()


def __main__():
    plot_double_well_committors()
    perform_force_field_regression()


if __name__ == "__main__":
    __main__()
