import numpy as np
import matplotlib.pyplot as plt
from chem import script2
from chem.utils import expand_wavefunction_from_coefficients
from skimage import data


def _load_noisy_blobs_image(noise_level: float = 2.0) -> np.ndarray:
    """
    Load the blobs image from the skimage.data module,
    with added Gaussian noise.

    Returns:
        np.ndarray: The blobs image.
    """
    original_image = data.binary_blobs().astype(float) * 2 - 1
    noise = np.random.randint(0, 2, original_image.shape)
    noise = noise * 2 - 1
    blobs_image = original_image + noise * 1.5
    blobs_image = np.clip(blobs_image, -1, 1)
    return original_image, blobs_image


def plot_low_rank_filtered_image():
    """
    Plot the original blobs image, the noisy blobs image, and the filtered blobs image
    for a low rank SVD filter with ranks 1, 10, 20, and 50.
    """
    original_image, noisy_blobs_image = _load_noisy_blobs_image()
    ranks = [1, 5, 10, 20]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axf = axes.flatten()

    # Top left is the original image
    axes[0, 0].imshow(original_image, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Top center is the noisy image
    axes[0, 1].imshow(noisy_blobs_image, cmap="gray")
    axes[0, 1].set_title("Noisy Image")
    axes[0, 1].axis("off")

    for i, rank in enumerate(ranks):
        filtered_image = script2.apply_low_rank_svd_filter(noisy_blobs_image, rank)
        axf[i + 2].imshow(filtered_image, cmap="gray")
        axf[i + 2].set_title(f"Rank {rank}")
        axf[i + 2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_wavefunction_propagation():
    """
    Plots the propagation of a wavefunction in a double well potential.
    """
    # Construct the potential energy surface
    grid = np.linspace(0, 1, 1000)
    potential_energy = np.cos(4.0 * np.pi * grid) + 0.6 * np.cos(2.0 * np.pi * grid)
    potential_energy *= 100.0

    kmax = 20
    k_values = np.arange(-kmax, kmax + 1)

    # Construct the initial wavefunction coefficients
    coeffs_0 = np.exp(k_values**2 / -8.0)
    coeffs_0 = coeffs_0 * np.exp(
        -1j * np.pi * k_values / 2.0
    )  # Translate to first well
    coeffs_0 = coeffs_0 / np.linalg.norm(coeffs_0)  # Normalize the coefficients

    # Propagate the wavefunction in time
    timepoints = np.linspace(0, 1.0, 16)
    coeffs_t = script2.propagate_wavefunctions(
        coeffs_0, potential_energy, grid, kmax, timepoints
    )

    # Plot the wavefunction at each time point
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), sharex=True, sharey=True)
    axf = axes.flatten()

    for i, ax in enumerate(axf):
        wavefunction_t = expand_wavefunction_from_coefficients(coeffs_t[:, i], grid)
        ax.plot(grid, np.abs(wavefunction_t) ** 2, label="$|psi(x)|^2$")
        nax = ax.twinx()
        nax.plot(grid, potential_energy, label="V(x)", linestyle="--", c="k")
        ax.set_title(f"t = {timepoints[i]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$|\psi(x)|^2$")
    plt.tight_layout(pad=1.0)
    plt.show()


def __main__():
    plot_low_rank_filtered_image()
    plot_wavefunction_propagation()
    return


if __name__ == "__main__":
    __main__()
