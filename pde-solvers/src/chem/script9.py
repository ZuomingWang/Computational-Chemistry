import numpy as np


class ReactionConvectionDiffusion:
    def __init__(
        self,
        x_0: float,
        x_N: float,
        reaction_rates: np.ndarray,
        D: float,
        v: float,
        dt: float,
    ):
        r"""
        Initialize a Solver for the reaction diffusion equation.
        The rection diffusion equation is given by:

        du / dt  = D (d^2 u / dx^2) + v (du / dx)  + k(x) u(x)

        where u is the concentration of the specie, D is the diffusion coefficient, and k(x) is the reaction rate.
        at the point x.

        We are going to use periodic boundary conditions: to calculate the derivative at the first point,
        we will use the last point, i.e.

        f'(x_0) \approx (f(x_1) - f(x_{N})) / (2 * dx)

        and for the second derivative,

        f''(x_0) \approx (f(x_1) - 2 * f(x_0) + f(x_{N})) / (dx^2)

        where N is the total number of points.

        Parameters
        ----------
        x_0 : float
            The leftmost point of the grid.
        x_N : float
            The rightmost point of the grid.
        reaction_rates : np.ndarray`
            The reaction rates at each point in the grid (k(x) in the equation above.).
            The shape of the array should be (N_points,).  where N_points is the number of points in the grid.
        D : float
            The diffusion coefficient.
        v : float
            The convection velocity.
        dt : float
            The time step to use in the integration.
        """
        self.reaction_rates = reaction_rates
        self.N_points = reaction_rates.shape[0]
        self.D = D
        self.v = v

        # Set up the grid
        self.x = np.linspace(x_0, x_N, self.N_points)
        self.dx = self.x[1] - self.x[0]  # Space step
        self.dt = dt  # Time step

        self.second_derivative_matrix = _create_second_derivative_matrix(
            self.N_points, self.dx
        )
        self.first_derivative_matrix = _create_first_derivative_matrix(
            self.N_points, self.dx
        )

        self.operator_matrix = _create_operator_matrix(
            reaction_rates,
            D,
            v,
            self.first_derivative_matrix,
            self.second_derivative_matrix,
        )

        self.crank_nicolson_update_matrix = _create_crank_nicolson_update_matrix(
            self.operator_matrix, dt
        )
        self.implicit_update_matrix = _create_implicit_update_matrix(
            self.operator_matrix, dt
        )
        self.explicit_update_matrix = _create_explicit_update_matrix(
            self.operator_matrix, dt
        )

    def integrate_pde(
        self, u_0: np.ndarray, number_of_steps: int, integrator="crank_nicolson"
    ) -> np.ndarray:
        r"""
        Integrate the PDE forwards in time.

        Parameters
        ----------
        u_0 : np.ndarray
            The initial condition for the system.
        number_of_steps : int
            The number of steps to integrate.
        integrator : str`
            The integrator to use. Can be "crank_nicolson", "implicit", or "explicit".

        Returns
        -------
        np.ndarray
            The solution to the PDE at each time step.
        """
        # Initialize the solution array
        u = np.zeros((number_of_steps, self.N_points))
        u[0, :] = u_0

        # Check the integrator
        if integrator == "implicit":
            update_matrix = self.implicit_update_matrix
        elif integrator == "explicit":
            update_matrix = self.explicit_update_matrix
        elif integrator == "crank_nicolson":
            update_matrix = self.crank_nicolson_update_matrix
        else:
            raise ValueError(
                f"Integrator {integrator} not recognized. Use 'crank_nicolson', 'implicit', or 'explicit'."
            )

        # Integrate the PDE
        for i in range(1, number_of_steps):
            u[i, :] = update_matrix @ u[i - 1, :]

        return u


# Storing these functions outside of the class allows for easier testing
# and reusability in other parts of the code.


def _create_first_derivative_matrix(N_points: int, dx: float) -> np.ndarray:
    r"""
    Create a matrix that approximates the first derivative of a function using
    a central difference:

    .. math::
        f'(x_0) \approx (f(x_{i+1}) - f(x_{i-1})) / (2 * dx)

    where N is the total number of points.
    At the boundaries, we use periodic boundary conditions.  We loop around the grid
    using the first point to calculate the derivative at the last point,
    i.e. f'(x_N) \approx (f(x_0) - f(x_{N-1})) / (2 * dx)
    and f'(x_0) \approx (f(x_1) - f(x_{N})) / (2 * dx)

    Parameters
    ----------
    N_points : int
        The number of points in the grid.
    dx : float
        The space step.

    Returns
    -------
    np.ndarray
        The first derivative matrix.
    """
    matrix = np.zeros((N_points, N_points))

    for i in range(N_points):
        matrix[i, (i - 1) % N_points] = -0.5 / dx
        matrix[i, (i + 1) % N_points] = 0.5 / dx

    return matrix


def _create_second_derivative_matrix(N_points: int, dx: float) -> np.ndarray:
    r"""
    Create a matrix that approximates the second derivative of a function using
    a central difference:

    .. math::
        f''(x_0) \approx (f(x_{i+1}) - 2 * f(x_i) + f(x_{i-1})) / (dx^2)

    where N is the total number of points.  Like with the first derivative, we use periodic
    boundary conditions.

    Parameters
    ----------
    N_points : int
        The number of points in the grid.
    dx : float
        The space step.

    Returns
    -------
    np.ndarray
        The second derivative matrix.
    """
    matrix = np.zeros((N_points, N_points))

    for i in range(N_points):
        matrix[i, i] = -2.0 / dx**2
        matrix[i, (i - 1) % N_points] = 1.0 / dx**2
        matrix[i, (i + 1) % N_points] = 1.0 / dx**2

    return matrix


def _create_operator_matrix(
    reaction_rates: np.ndarray,
    D: float,
    v: float,
    first_derivative_matrix: np.ndarray,
    second_derivative_matrix: np.ndarray,
) -> np.ndarray:
    """
    Create the the right hand side matrix that approximates the operator

    D * (d^2 u / dx^2) + k(x) u(x) + v du / dx

    Parameters
    ----------
    reaction_rates : np.ndarray
        The reaction rates at each point in the grid (k(x) in the equation above.).
        The shape of the array should be (N_points,).  where N_points is the number of points in the grid.
    D : float
        The diffusion coefficient.
    v : float
        The convection velocity.
    first_derivative_matrix : np.ndarray
        The first derivative matrix.
    second_derivative_matrix : np.ndarray
        The second derivative matrix.

    Returns
    -------
    np.ndarray
        The operator matrix.
    """
    N_points = reaction_rates.shape[0]
    reaction_matrix = np.diag(reaction_rates)
    return D * second_derivative_matrix + v * first_derivative_matrix + reaction_matrix


def _create_explicit_update_matrix(
    operator_matrix: np.ndarray, dt: float
) -> np.ndarray:
    """
    Create a matrix that applies an explicit update to the system.

    Parameters
    ----------
    operator_matrix : np.ndarray
        The operator matrix.
    dt : float
        The time step.

    Returns
    -------
    np.ndarray
        The explicit update matrix.  The explicit method output should be
        given by applying this matrix to the solution at the previous time step.
    """
    return np.eye(operator_matrix.shape[0]) + dt * operator_matrix


def _create_crank_nicolson_update_matrix(
    operator_matrix: np.ndarray, dt: float
) -> np.ndarray:
    """
    Create a matrix that applies a Crank-Nicolson update to the system.

    Parameters
    ----------
    operator_matrix : np.ndarray
        The operator matrix.
    dt : float
        The time step.

    Returns
    -------
    np.ndarray
        The Crank-Nicolson update matrix.  The Crank-Nicolson method output should be
        given by applying this matrix to the solution at the previous time step.
    """
    from numpy.linalg import inv
    I = np.eye(operator_matrix.shape[0])
    A = I - 0.5 * dt * operator_matrix
    B = I + 0.5 * dt * operator_matrix
    return inv(A) @ B


def _create_implicit_update_matrix(
    operator_matrix: np.ndarray, dt: float
) -> np.ndarray:
    """
    Create a matrix that applies an implicit update to the system.

    Parameters
    ----------
    operator_matrix : np.ndarray
        The operator matrix.
    dt : float
        The time step.

    Returns
    -------
    np.ndarray
        The implicit update matrix.  The implicit method output should be
        given by applying this matrix to the solution at the previous time step.
    """
    from numpy.linalg import inv
    I = np.eye(operator_matrix.shape[0])
    return inv(I - dt * operator_matrix)
