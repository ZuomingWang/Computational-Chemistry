import matplotlib.pyplot as plt
import numpy as np
import vector_matrix_utils as vm


def _return_Fe_Sb4_O_12_coordinates():
    """
    Returns data for the FeSb4O12 structure,
    made into a helper function for readability.
    """
    cartesian_coordinates = np.array(
        [
            [0.01769674, 1.91924638, 5.72068035],
            [5.12805664, 3.07549455, 0.20025973],
            [2.59680857, 3.95538202, 4.12822048],
            [0.03824663, 0.0731397, 3.78827363],
            [2.56355624, 3.93029255, 0.50601907],
            [3.69563694, 3.25955815, 4.59656579],
            [4.41735961, 2.7107689, 2.02378929],
            [3.51603823, 5.90550249, 7.7452789],
            [4.23701279, 5.7000416, 4.96400074],
            [0.93410937, 3.75726729, 1.1360034],
            [3.30735598, 4.31810911, 2.32421634],
            [1.84413982, 3.53183292, 6.10597634],
            [4.20469946, 4.04331036, 7.36843095],
            [0.89243174, 2.12063078, 3.83343547],



            [1.60020855, 1.90892166, 0.73624676],
            [0.55996, 0.60976724, 5.79697979],
            [1.72187752, 4.90065308, 3.9966277],
        ]
    )

    lattice_vectors = np.array(
        [
            [5.15940000e00, 3.04326565e00, 6.73997199e-03],
            [0.00000000e00, 4.55921844e00, 7.37660517e-01],
            [0.00000000e00, 0.00000000e00, 7.53598006e00],
        ]
    )
    return lattice_vectors, cartesian_coordinates


def plot_Fe_Sb4_O_12():
    """
    Plot the FeSb4O12 structure in both Cartesian and fractional coordinates.
    """
    lattice_vectors, cartesian_coordinates = _return_Fe_Sb4_O_12_coordinates()

    predicted_fractional_coordinates = vm.convert_to_fractional_coordinates(
        lattice_vectors, cartesian_coordinates
    )

    # Side-by-side 3d scatter plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})

    ax[0].scatter(
        cartesian_coordinates[:, 0],
        cartesian_coordinates[:, 1],
        cartesian_coordinates[:, 2],
    )
    ax[1].scatter(
        predicted_fractional_coordinates[:, 0],
        predicted_fractional_coordinates[:, 1],
        predicted_fractional_coordinates[:, 2],
    )

    ax[0].set_title("Cartesian coordinates")
    ax[1].set_title("Fractional coordinates")

    for a in ax:
        a.set_xlabel("x")
        a.set_ylabel("y")
        a.set_zlabel("z")
        a.set_aspect("equal")

    plt.show()


def plot_smoothed_sine_wave():
    """
    Plot a noisy sine wave and the result of applying a Gaussian smoother.
    """
    x = np.linspace(-8, 8, 1000)
    y = np.sin(np.pi * x / 2.0)

    np.random.seed(0)
    y += 0.1 * np.random.randn(len(y))

    smoothed_signal = vm.apply_gaussian_smoothing(y, -10, 10, 0.40)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(x, y, label="Noisy signal")
    plt.plot(x, smoothed_signal, label="Smoothed signal")
    plt.legend()

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()


def __main__():
    plot_Fe_Sb4_O_12()
    plot_smoothed_sine_wave()
    return


if __name__ == "__main__":
    __main__()
    