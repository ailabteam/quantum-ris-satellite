# classical_solvers.py
import numpy as np
import cvxpy as cp
import time

def solve_random(params):
    """Returns a random RIS configuration."""
    return np.random.choice([-1, 1], size=params["N"])

def solve_benchmark_user0(h_sr, H_ru, params):
    """Optimizes phases for user 0 only."""
    cascaded_k0 = H_ru[0, :] * h_sr.T.flatten()
    return np.sign(np.cos(np.angle(cascaded_k0)))

def solve_sdr(Q_matrix, params):
    """
    Solves the RIS phase optimization problem using Semidefinite Relaxation.
    """
    N = params["N"]
    num_randomizations = params["sdr_num_randomizations"]
    
    # 1. Define and solve the SDP problem
    V = cp.Variable((N, N), hermitian=True)
    constraints = [cp.diag(V) == 1, V >> 0]
    objective = cp.Maximize(cp.real(cp.trace(Q_matrix @ V)))
    problem = cp.Problem(objective, constraints)
    
    print("Solving SDP with CVXPY...")
    start_time = time.time()
    # Use a solver that is typically available with CVXPY's standard installation
    problem.solve(solver=cp.SCS, verbose=False)
    end_time = time.time()
    print(f"SDP solved in {end_time - start_time:.2f} seconds.")

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("WARNING: SDR solver failed. Returning a random solution.")
        return solve_random(params)

    V_star = V.value
    
    # 2. Perform Gaussian Randomization to extract a 1-bit solution
    best_v = None
    max_objective_val = -np.inf
    
    try:
        # Add a small identity matrix to ensure the matrix is positive definite for Cholesky
        L = np.linalg.cholesky(V_star + 1e-6 * np.eye(N))
    except np.linalg.LinAlgError:
        print("Cholesky failed. Using eigenvalue decomposition as fallback.")
        eigvals, eigvecs = np.linalg.eigh(V_star)
        eigvals[eigvals < 0] = 0
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    print(f"Performing {num_randomizations} Gaussian randomizations...")
    for _ in range(num_randomizations):
        # Generate a random complex vector
        r = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1)) / np.sqrt(2)
        # Create a candidate vector and quantize to 1-bit
        v_candidate = np.sign(np.real(L.T.conj() @ r)).flatten()
        
        # Check the objective value for this candidate
        current_val = np.real(v_candidate.T @ Q_matrix @ v_candidate)
        
        if current_val > max_objective_val:
            max_objective_val = current_val
            best_v = v_candidate
            
    return best_v
