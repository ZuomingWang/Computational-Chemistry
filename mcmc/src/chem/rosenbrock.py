import numpy as np
import matplotlib.pyplot as plt


def rosenbrock_function(x: np.ndarray) -> float:
    """
    Evaluates the Rosenbrock function, defined as

    .. math::
        f(x_1, x_2) =  (1 - x_0)^2 + 100 (x_1 - x_0^2)^2

    Parameters
    ----------
    x : np.ndarray
        The input vector: a point in 2D space.

    Returns
    -------
    float
        The value of the Rosenbrock function at the input point.
    """
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2 / 2.0


def rosenbrock_force(x: np.ndarray) -> np.ndarray:
    """
    Returns the force from a particle with the Rosenbrock function
    as its potential energy (recall the force is the negative gradient of the potential energy).

    The Rosenbrock potential energy is defined as above.

    Parameters
    ----------
    x : np.ndarray
        The input vector: a point in 2D space.

    Returns
    -------
    np.ndarray
        The force vector at the input point.
    """
    return -0.5 * np.array(
        [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)]
    )


def plot_rosenbrock_in_2D(ax):
    X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 100), np.linspace(-1, 4, 100))
    Z = rosenbrock_function(np.array([X, Y]))

    ax.contour(X, Y, Z, levels=np.arange(0, 100, 2), cmap="viridis")

    # Heatmap
    ax.imshow(
        Z,
        extent=(-2.5, 2.5, -1, 4),
        origin="lower",
        cmap="viridis",
        alpha=0.5,
        vmin=0,
        vmax=100,
    )

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1, 4)

    return ax
