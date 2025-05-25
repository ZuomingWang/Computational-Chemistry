import chem.script6 as script6
import numpy as np
import argparse
import matplotlib.pyplot as plt


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the trajectory of an Ising model."
    )
    parser.add_argument(
        "--N",
        type=int,
        default=10,
        help="The size of the Ising model.",
    )
    parser.add_argument(
        "--J",
        type=float,
        default=1.0,
        help="The interaction constant.",
    )
    parser.add_argument(
        "--kT",
        type=float,
        default=1.0,
        help="The product of the Boltzmann constant and the temperature.",
    )
    parser.add_argument(
        "--N_steps",
        type=int,
        default=1000,
        help="The number of steps to perform.",
    )

    parser.add_argument(
        "--plot_every",
        type=int,
        default=1,
        help="Plot every this many steps.",
    )

    return parser.parse_args()


def plot_trajectory(trajectory: np.ndarray, plot_every: int = 1):
    trajectory = trajectory[::plot_every]
    N = trajectory.shape[1]
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes.imshow(trajectory[0], cmap="bwr", vmin=-1, vmax=1)
    axes.set_xticks([])
    axes.set_yticks([])

    # Update the plot for each step using a slider
    def update_plot(i):
        axes.imshow(trajectory[i], cmap="bwr", vmin=-1, vmax=1)
        axes.set_title(f"Step {i}")

    # Create the slider
    from matplotlib.widgets import Slider

    slider_axes = plt.axes([0.1, 0.01, 0.8, 0.03])
    slider = Slider(
        slider_axes, "Step", 0, trajectory.shape[0] - 1, valinit=0, valstep=1
    )
    slider.on_changed(update_plot)

    plt.show()


def main(args):
    # Set the random seed
    np.random.seed(0)

    # Unpack the arguments
    N = args.N
    J = args.J
    kT = args.kT
    N_steps = args.N_steps

    # Generate the initial spins
    # initial_spins = 2 * np.random.binomial(1, 0.75, size=(N, N)) - 1
    initial_spins = 2 * np.random.binomial(1, 0.5, size=(N, N)) - 1

    # Perform Gibbs sampling
    trajectory = script6.gibbs_sample_ising_model(initial_spins, J, kT, N_steps)

    plot_trajectory(trajectory, args.plot_every)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
