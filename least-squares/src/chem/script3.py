import numpy as np


def tikhonov_lst_sq(A: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """
    Perform a least squares regression using Tikhonov regularization.
    In Tikhonov regularization, the solution is given by:
        x = U (S^T S + alpha I)^-1 S^T b
    where U, S, and V are the singular value decomposition of A.
    and alpha is the regularization parameter that controls the tradeoff between
    the fit to the data and the size of the coefficients.

    Args:
        A (np.ndarray): The matrix A of the linear system to solve.
        b (np.ndarray): The vector b of the linear system to solve.
        alpha (float): The regularization parameter.

    Returns:
        np.ndarray: The solution to the linear system.
    """
    # Compute the singular value decomposition
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Calculate U^T b
    UTb = np.dot(U.T, b)
    
    # Create a diagonal matrix of 1/(s_i^2 + alpha)
    S_squared_inv = np.zeros_like(S)
    for i in range(len(S)):
        S_squared_inv[i] = S[i] / (S[i]**2 + alpha)
    
    # Calculate S_inv * U^T * b
    S_inv_UTb = S_squared_inv * UTb[:len(S)]
    
    # Calculate V^T * S_inv * U^T * b
    x = np.dot(Vt.T, S_inv_UTb)
    
    return x


def calculate_committor(
    transition_matrix: np.ndarray, state_A: np.ndarray, state_B: np.ndarray
) -> np.ndarray:
    """
    Calculates the committor function for a Markov state model:
    the probability that if the system is in state i, it will reach a state in set B
    before reaching a state in set A.

    The committor function obeys the following equation:
        q_i = 1 if i is in set B
        q_i = 0 if i is in set A
        q_i = sum_j T_ij q_j for all other i

    Args:
        transition_matrix (np.ndarray): The transition matrix of the Markov state model.
            Of shape (n, n), where n is the number of states.
        state_A (np.ndarray): The indices of the states in set A.
            Of variable length
        state_B (np.ndarray): The indices of the states in set B.
            Of variable length

    Returns:
        np.ndarray: The committor function.
    """
    n = transition_matrix.shape[0]
    q = np.zeros(n)
    
    # Set boundary conditions
    q[state_B] = 1.0
    
    # Identify the states that are neither in A nor in B
    intermediate_states = np.ones(n, dtype=bool)
    intermediate_states[state_A] = False
    intermediate_states[state_B] = False
    
    # Create a linear system to solve for the intermediate states
    # We need to solve (I - T) q = 0 for the intermediate states
    # with the boundary conditions already set
    I = np.eye(n)
    linear_system = I - transition_matrix
    
    # For boundary states (A and B), set the corresponding row in the linear system
    # to ensure q_i = 0 for i in A and q_i = 1 for i in B
    linear_system[state_A] = 0
    linear_system[state_B] = 0
    
    for i in state_A:
        linear_system[i, i] = 1  # Set diagonal to 1
    
    for i in state_B:
        linear_system[i, i] = 1  # Set diagonal to 1
    
    # Set up the right-hand side
    rhs = np.zeros(n)
    rhs[state_B] = 1.0
    
    # Solve the linear system
    q = np.linalg.solve(linear_system, rhs)
    
    return q


def regress_first_order_coefficients(
    concentrations: np.ndarray, time_points: np.ndarray
) -> np.ndarray:
    """
    Given a set of concentrations observed at different time points, estimate
    the first order coefficients of the concentrations by:
        1. Estimating the derivative of the concentrations with respect to time
           using a forward dinite difference. (dC/dt = (C(t+dt) - C(t)) / dt).
        2. Performing a least squares regression to estimate the first order
           coefficients of the concentrations using the derivative estimates.

    Args:
        concentrations (np.ndarray) : An array of shape (m, n) containing the
            concentrations of n species at m time points.
        time_points (np.ndarray) : An array of shape (m,) containing the times
            at which the concentrations were observed.

    Returns:
        np.ndarray : An array of shape (n, n) containing the estimated first
            order coefficients of the concentrations.  The i-th column of the
            output array should contain the coefficients for the i-th species.
            The coefficients should be ordered in the same way as the input
            concentrations.
    """
    m, n = concentrations.shape
    
    # Step 1: Calculate the derivatives using forward finite difference
    # We can only compute m-1 derivatives
    dc_dt = np.zeros((m-1, n))
    
    for i in range(m-1):
        dt = time_points[i+1] - time_points[i]
        dc_dt[i, :] = (concentrations[i+1, :] - concentrations[i, :]) / dt
    
    # Step 2: Perform least squares regression to estimate coefficients
    # For each species (column), we need to solve:
    # dc_i/dt = sum_j k_ij * c_j
    # where k_ij are the first-order coefficients
    
    # Use the concentrations at the corresponding time points for regression
    C = concentrations[:-1, :]  # Use concentrations at time points where we calculated derivatives
    
    # Initialize the coefficients matrix
    coefficients = np.zeros((n, n))
    
    # For each species, perform regression to estimate its coefficients
    for i in range(n):
        # Solve for the i-th column of coefficients
        # dc_i/dt = C * k_i where k_i is the i-th column of coefficients
        coefficients[:, i] = np.linalg.lstsq(C, dc_dt[:, i], rcond=None)[0]
    
    return coefficients
