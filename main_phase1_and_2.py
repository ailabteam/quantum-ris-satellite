import numpy as np

# -------------------------------------------------------------------
# PHASE 1: SYSTEM MODEL AND CHANNEL SIMULATION
# -------------------------------------------------------------------

def setup_parameters_workable():
    """A workable parameter setup based on a LEO satellite scenario."""
    params = {
        # System Layout
        "N": 256, # Number of RIS elements
        "K": 4,   # Number of ground users
        "sat_pos": np.array([0, 0, 1000e3]), # LEO satellite at 1000 km altitude
        "ris_pos": np.array([0, 0, 50]),
        "user_area_center": np.array([0, 100, 0]),
        "user_area_radius": 50,

        # RF and Channel Parameters
        "fc": 4e9, # Carrier frequency (4 GHz)
        "c": 3e8,
        "lambda_c": 3e8 / 4e9,

        # Antenna Gains (dBi)
        "gain_sat_dBi": 40,
        "gain_ris_ele_dBi": 5,
        "gain_user_dBi": 10,

        # Power and Noise
        "Pt_dBm": 40, # Satellite Tx Power = 10W
        "P_sat": 10**((40 - 30) / 10),
        "noise_figure_dB": 5,
        "noise_power_dbm": -174 + 10 * np.log10(10e6) + 5, # BW = 10MHz
        "sigma_sq": 10**(((-174 + 10 * np.log10(10e6) + 5) - 30) / 10),
        
        # Rician Fading Factors
        "k_sr": 10,
        "k_ru": 3
    }
    # Convert gains from dBi to linear scale
    for key in ["gain_sat_dBi", "gain_ris_ele_dBi", "gain_user_dBi"]:
        linear_key = key.replace("_dBi", "").replace("gain_", "G_").upper()
        params[linear_key] = 10**(params[key] / 10)
    return params

def create_rician_channel(dim1, dim2, distance, k_factor, lambda_c, G_tx, G_rx):
    """Creates a Rician fading channel matrix of size (dim1, dim2)."""
    if distance == 0: distance = 1e-6 # Avoid division by zero
    path_loss_val = (lambda_c / (4 * np.pi * distance))**2
    
    total_gain = np.sqrt(path_loss_val * G_tx * G_rx)
    
    # LoS component with phase determined by distance
    h_los = np.exp(-1j * 2 * np.pi * distance / lambda_c)
    
    # NLoS component (random)
    h_nlos = (np.random.randn(dim1, dim2) + 1j * np.random.randn(dim1, dim2)) / np.sqrt(2)
    
    channel = total_gain * (
        np.sqrt(k_factor / (k_factor + 1)) * h_los +
        np.sqrt(1 / (k_factor + 1)) * h_nlos
    )
    return channel

def generate_system_channels(params):
    """Generates all channel matrices for the system."""
    N, K = params["N"], params["K"]
    
    # Generate user positions
    user_positions = np.zeros((K, 3))
    for k in range(K):
        r = np.sqrt(np.random.rand()) * params["user_area_radius"]
        theta = 2 * np.pi * np.random.rand()
        user_positions[k, 0] = params["user_area_center"][0] + r * np.cos(theta)
        user_positions[k, 1] = params["user_area_center"][1] + r * np.sin(theta)
        user_positions[k, 2] = 1.5
        
    dist_sr = np.linalg.norm(params["sat_pos"] - params["ris_pos"])
    dist_ru = np.array([np.linalg.norm(params["ris_pos"] - user_pos) for user_pos in user_positions])
    
    # S-R Channel (vector of size N x 1)
    h_sr = np.zeros((N, 1), dtype=np.complex128)
    for n in range(N):
        h_sr[n] = create_rician_channel(1, 1, dist_sr, params["k_sr"], params["lambda_c"], params["G_SAT"], params["G_RIS_ELE"])

    # R-U Channel (matrix of size K x N)
    H_ru = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        for n in range(N):
            H_ru[k, n] = create_rician_channel(1, 1, dist_ru[k], params["k_ru"], params["lambda_c"], params["G_RIS_ELE"], params["G_USER"])
            
    return h_sr, H_ru

def calculate_sum_snr(v, h_sr, H_ru, P_k, sigma_sq):
    """Calculates the sum-SNR for a given RIS configuration vector v."""
    N, K = h_sr.shape[0], H_ru.shape[0]
    Phi = np.diag(v)
    total_snr = 0
    for k in range(K):
        h_ru_k = H_ru[k, :].reshape(1, N)
        effective_channel = h_ru_k @ Phi @ h_sr
        channel_gain = np.abs(effective_channel)**2
        snr_k = (P_k * channel_gain) / sigma_sq
        total_snr += snr_k
    return total_snr.item()


# -------------------------------------------------------------------
# PHASE 2: PROBLEM MAPPING TO ISING HAMILTONIAN
# -------------------------------------------------------------------

def create_ising_hamiltonian_coeffs(h_sr, H_ru, P_k, sigma_sq):
    """
    Calculates the Ising Hamiltonian coefficients (J matrix and h vector).
    Goal: MAXIMIZE sum-SNR <=> MINIMIZE H = -sum_SNR.
    Ising Model: H = sum_{i<j} J_ij s_i s_j + sum_i h_i s_i
    """
    N, K = h_sr.shape[0], H_ru.shape[0]
    
    # 1. Calculate the Q matrix from the quadratic form of sum-SNR
    # sum-SNR = v^T * Q * v
    Q = np.zeros((N, N), dtype=np.complex128)
    
    # Pre-calculate cascaded channels c_kn = h_ru,k,n * h_sr,n
    C = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        C[k, :] = H_ru[k, :] * h_sr.T.flatten()
        
    # Build Q matrix where Q_nm = sum_k (P_k/sigma^2) * c_kn * c_km*
    for n in range(N):
        for m in range(N):
            q_nm = np.sum((P_k / sigma_sq) * C[:, n] * np.conj(C[:, m]))
            Q[n, m] = q_nm
    
    # 2. Extract Ising parameters J and h from H = -v^T Q v
    # H = const - sum_{n<m} 2*Re(Q_nm) v_n v_m
    J = np.zeros((N, N))
    h = np.zeros(N) # No linear terms in this formulation
    
    for n in range(N):
        for m in range(n + 1, N):
            J[n, m] = -2 * np.real(Q[n, m])

    # The diagonal terms of Q contribute to a constant energy offset
    cost_offset = -np.sum(np.real(np.diag(Q)))
            
    return J, h, cost_offset

# -------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# -------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- PHASE 1: Generate System and Channels ---
    print("--- Phase 1: Workable LEO Scenario Setup ---")
    
    params = setup_parameters_workable()
    N_val, K_val = params["N"], params["K"]
    print(f"Parameters: N={N_val}, K={K_val}, Sat Altitude={params['sat_pos'][2]/1e3}km")
    
    np.random.seed(42) # For reproducible results
    h_sr, H_ru = generate_system_channels(params)
    print("Channel matrices generated successfully.")

    # --- Verification Step ---
    print("\n--- Verifying Model with Near-Optimal Configuration ---")
    # Find a near-optimal 1-bit configuration for a single user (k=0)
    k_test = 0
    h_ru_k_test = H_ru[k_test, :]
    cascaded_vector = h_ru_k_test * h_sr.T.flatten()
    v_optimal_1bit = np.sign(np.cos(np.angle(cascaded_vector)))

    P_k = params["P_sat"] / K_val

    # Calculate sum-SNR for all users with this configuration
    sum_snr_val = calculate_sum_snr(v_optimal_1bit, h_sr, H_ru, P_k, params["sigma_sq"])
    
    print(f"Resulting Sum-SNR (all users): {sum_snr_val:.4f}")
    if sum_snr_val > 0:
        print(f"Resulting Sum-SNR in dB: {10 * np.log10(sum_snr_val):.2f} dB")

    # Calculate SNR for the target user only
    Phi = np.diag(v_optimal_1bit)
    h_ru_k0 = H_ru[0, :].reshape(1, N_val)
    effective_channel_k0 = h_ru_k0 @ Phi @ h_sr
    channel_gain_k0 = np.abs(effective_channel_k0)**2
    snr_k0 = (P_k * channel_gain_k0) / params["sigma_sq"]
    
    print(f"\nSNR for the target user (k=0) only: {snr_k0.item():.4f}")
    if snr_k0 > 0:
        print(f"SNR for the target user (k=0) in dB: {10 * np.log10(snr_k0.item()):.2f} dB")

    # --- PHASE 2: Get Ising Hamiltonian Coefficients ---
    print("\n\n--- Phase 2: Mapping to Ising Model ---")
    
    J, h, offset = create_ising_hamiltonian_coeffs(h_sr, H_ru, P_k, params["sigma_sq"])

    print(f"Shape of J matrix (couplings): {J.shape}")
    print(f"Shape of h vector (fields): {h.shape}")
    print("\nIsing J matrix (first 5x5 elements):")
    print(J[:5, :5])
    print(f"\nIsing h vector (first 5 elements): {h[:5]}")
    print(f"\nConstant energy offset: {offset:.4f}")
