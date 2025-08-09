import numpy as np

# -------------------------------------------------------------------
# Phase 1.1: System Parameters & Channel Model
# -------------------------------------------------------------------

def setup_parameters():
    """Defines all simulation parameters in a dictionary."""
    params = {
        # RIS
        "N": 16,  # Number of RIS elements

        # Users
        "K": 4,  # Number of ground users

        # Satellite
        "sat_pos": np.array([0, 0, 35786e3]),  # GEO satellite position (meters)

        # RIS
        "ris_pos": np.array([0, 100, 50]),     # RIS position (e.g., on a building)

        # Users' Area
        "user_area_center": np.array([10, 200, 0]), # Center of the user area
        "user_area_radius": 50,                    # Radius of the user area (meters)

        # Channel Parameters
        "fc": 4e9,  # Carrier frequency (4 GHz)
        "c": 3e8,   # Speed of light (m/s)
        "lambda_c": 3e8 / 4e9, # Wavelength

        "k_sr": 10,   # Rician factor for Satellite-RIS link (high LoS)
        "k_ru": 3,    # Rician factor for RIS-User link (moderate LoS)

        # Power & Noise
        "Pt_dBm": 30, # Satellite transmit power in dBm
        "P_sat": 10**((30 - 30) / 10), # Transmit power in Watts (1W)
        "noise_figure_dB": 5, # Noise figure at the user receiver in dB
        "noise_power_dbm": -174 + 10 * np.log10(10e6) + 5, # -174 dBm/Hz, 10MHz bandwidth, 5dB NF
        "sigma_sq": 10**(((-174 + 10 * np.log10(10e6) + 5) - 30) / 10) # Noise power in Watts
    }
    return params

def get_path_loss(d, lambda_c):
    """Calculates free-space path loss."""
    return (lambda_c / (4 * np.pi * d))**2

def create_rician_channel(dim1, dim2, distance, k_factor, lambda_c):
    """
    Creates a Rician fading channel.
    dim1, dim2: dimensions of the channel matrix (e.g., N, 1)
    """
    path_loss = get_path_loss(distance, lambda_c)

    # Line-of-Sight (LoS) component (deterministic, based on geometry)
    # For simplicity, we assume a simple plane wave model for LoS part.
    h_los = np.ones((dim1, dim2), dtype=np.complex128)

    # Non-Line-of-Sight (NLoS) component (random)
    h_nlos = (np.random.randn(dim1, dim2) + 1j * np.random.randn(dim1, dim2)) / np.sqrt(2)

    channel = np.sqrt(path_loss) * (
        np.sqrt(k_factor / (k_factor + 1)) * h_los +
        np.sqrt(1 / (k_factor + 1)) * h_nlos
    )
    return channel

def generate_system_channels(params):
    """Generates all channels for the system."""
    # Unpack parameters
    N = params["N"]
    K = params["K"]

    # 1. Generate user positions randomly in a circular area
    user_positions = np.zeros((K, 3))
    for k in range(K):
        r = np.sqrt(np.random.rand()) * params["user_area_radius"]
        theta = 2 * np.pi * np.random.rand()
        user_positions[k, 0] = params["user_area_center"][0] + r * np.cos(theta)
        user_positions[k, 1] = params["user_area_center"][1] + r * np.sin(theta)
        user_positions[k, 2] = 1.5 # User height

    # 2. Calculate distances
    dist_sr = np.linalg.norm(params["sat_pos"] - params["ris_pos"])
    dist_ru = np.array([np.linalg.norm(params["ris_pos"] - user_pos) for user_pos in user_positions])

    # 3. Generate Satellite-RIS channel (h_sr)
    h_sr = create_rician_channel(N, 1, dist_sr, params["k_sr"], params["lambda_c"])

    # 4. Generate RIS-Users channels (H_ru)
    H_ru = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        # Note: h_ru_k is a row vector in our formulation h_ru,k * Phi * h_sr
        # So we create a (1, N) channel and place it in the matrix
        h_ru_k = create_rician_channel(1, N, dist_ru[k], params["k_ru"], params["lambda_c"])
        H_ru[k, :] = h_ru_k

    return h_sr, H_ru, user_positions

# -------------------------------------------------------------------
# Phase 1.2: Objective Function Calculation
# -------------------------------------------------------------------

def calculate_sum_snr(v, h_sr, H_ru, P_k, sigma_sq):
    """
    Calculates the sum of SNRs for all users.
    Args:
        v (np.array): A 1D array of size N with elements in {-1, 1}, representing phase shifts.
        h_sr (np.array): Satellite-RIS channel (N x 1).
        H_ru (np.array): RIS-Users channel matrix (K x N).
        P_k (float): Transmit power allocated per user (assuming equal power for now).
        sigma_sq (float): Noise power.
    Returns:
        float: The sum of SNRs.
    """
    N = h_sr.shape[0]
    K = H_ru.shape[0]

    # Create the diagonal phase shift matrix Phi
    Phi = np.diag(v) # For v_n in {-1, 1}, e^(j*theta_n) is just v_n

    total_snr = 0
    for k in range(K):
        h_ru_k = H_ru[k, :].reshape(1, N) # Get channel for user k
        
        # Effective channel for user k: h_ru,k * Phi * h_sr
        effective_channel = h_ru_k @ Phi @ h_sr
        
        # Channel gain
        channel_gain = np.abs(effective_channel)**2
        
        # SNR for user k
        snr_k = (P_k * channel_gain) / sigma_sq
        total_snr += snr_k
        
    return total_snr.item() # .item() to get a single float value


# -------------------------------------------------------------------
# Main execution block to test the setup
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Phase 1: Simulation Setup ---")
    
    # 1. Load parameters
    params = setup_parameters()
    print(f"Parameters: N={params['N']}, K={params['K']}, Power={params['Pt_dBm']}dBm")
    
    # 2. Generate a single instance of the channels
    np.random.seed(42) # for reproducibility
    h_sr, H_ru, user_pos = generate_system_channels(params)
    print(f"Shape of h_sr: {h_sr.shape}")
    print(f"Shape of H_ru: {H_ru.shape}")

    # 3. Test the objective function with a random phase configuration
    # For 1-bit RIS, v_n is either 1 or -1
    v_random = np.random.choice([-1, 1], size=params["N"])
    
    # Assume equal power allocation for now
    P_k = params["P_sat"] / params["K"]

    sum_snr_random = calculate_sum_snr(v_random, h_sr, H_ru, P_k, params["sigma_sq"])
    
    print("\n--- Testing with a random RIS configuration ---")
    print(f"Random RIS vector v: {v_random}")
    print(f"Resulting Sum-SNR: {sum_snr_random}")
    print(f"Resulting Sum-SNR in dB: {10 * np.log10(sum_snr_random)}")
