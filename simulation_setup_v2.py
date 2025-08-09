import numpy as np

# -------------------------------------------------------------------
# Phase 1.1: System Parameters & Channel Model (Version 2)
# -------------------------------------------------------------------

def setup_parameters_v2():
    """Defines all simulation parameters, including antenna gains."""
    params = {
        # System Layout
        "N": 16,
        "K": 4,
        "sat_pos": np.array([0, 0, 35786e3]),
        "ris_pos": np.array([0, 100, 50]),
        "user_area_center": np.array([10, 200, 0]),
        "user_area_radius": 50,

        # Channel & RF Parameters
        "fc": 4e9,
        "c": 3e8,
        "lambda_c": 3e8 / 4e9,
        "k_sr": 10,
        "k_ru": 3,

        # ------ NEW: Antenna Gains ------
        "gain_sat_dBi": 40,   # Satellite antenna gain in dBi
        "gain_ris_ele_dBi": 3, # Gain of a single RIS element in dBi
        "gain_user_dBi": 0,    # User antenna gain (omni-directional) in dBi

        # Power & Noise
        "Pt_dBm": 30,
        "P_sat": 10**((30 - 30) / 10),
        "noise_figure_dB": 5,
        "noise_power_dbm": -174 + 10 * np.log10(10e6) + 5,
        "sigma_sq": 10**(((-174 + 10 * np.log10(10e6) + 5) - 30) / 10)
    }
    # Convert gains from dBi to linear scale
    params["G_sat"] = 10**(params["gain_sat_dBi"] / 10)
    params["G_ris_ele"] = 10**(params["gain_ris_ele_dBi"] / 10)
    params["G_user"] = 10**(params["gain_user_dBi"] / 10)
    return params

def get_path_loss(d, lambda_c):
    """Calculates free-space path loss (as a linear value < 1)."""
    return (lambda_c / (4 * np.pi * d))**2

def create_rician_channel_v2(dim1, dim2, distance, k_factor, lambda_c, G_tx, G_rx):
    """
    Creates a Rician fading channel, now including antenna gains.
    G_tx: Transmit antenna gain (linear)
    G_rx: Receive antenna gain (linear)
    """
    path_loss_val = get_path_loss(distance, lambda_c)

    # Total channel gain now includes path loss and antenna gains
    total_gain = np.sqrt(path_loss_val * G_tx * G_rx)

    h_los = np.ones((dim1, dim2), dtype=np.complex128)
    h_nlos = (np.random.randn(dim1, dim2) + 1j * np.random.randn(dim1, dim2)) / np.sqrt(2)

    channel = total_gain * (
        np.sqrt(k_factor / (k_factor + 1)) * h_los +
        np.sqrt(1 / (k_factor + 1)) * h_nlos
    )
    return channel

def generate_system_channels_v2(params):
    """Generates all channels for the system using the updated functions."""
    N, K = params["N"], params["K"]

    user_positions = np.zeros((K, 3))
    for k in range(K):
        r = np.sqrt(np.random.rand()) * params["user_area_radius"]
        theta = 2 * np.pi * np.random.rand()
        user_positions[k, 0] = params["user_area_center"][0] + r * np.cos(theta)
        user_positions[k, 1] = params["user_area_center"][1] + r * np.sin(theta)
        user_positions[k, 2] = 1.5

    dist_sr = np.linalg.norm(params["sat_pos"] - params["ris_pos"])
    dist_ru = np.array([np.linalg.norm(params["ris_pos"] - user_pos) for user_pos in user_positions])

    # S-R Channel: Tx is Sat, Rx is RIS element
    h_sr = create_rician_channel_v2(N, 1, dist_sr, params["k_sr"], params["lambda_c"], params["G_sat"], params["G_ris_ele"])

    # R-U Channels: Tx is RIS element, Rx is User
    H_ru = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        h_ru_k = create_rician_channel_v2(1, N, dist_ru[k], params["k_ru"], params["lambda_c"], params["G_ris_ele"], params["G_user"])
        H_ru[k, :] = h_ru_k

    return h_sr, H_ru, user_positions

# (The function calculate_sum_snr remains the same)
def calculate_sum_snr(v, h_sr, H_ru, P_k, sigma_sq):
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
# Main execution block
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Phase 1 (V2): Simulation Setup with Antenna Gains ---")
    
    params = setup_parameters_v2()
    print(f"Parameters: N={params['N']}, K={params['K']}, Sat Gain={params['gain_sat_dBi']}dBi, RIS Ele Gain={params['gain_ris_ele_dBi']}dBi")
    
    np.random.seed(42)
    h_sr, H_ru, user_pos = generate_system_channels_v2(params)
    
    v_random = np.random.choice([-1, 1], size=params["N"])
    P_k = params["P_sat"] / params["K"]

    sum_snr_random = calculate_sum_snr(v_random, h_sr, H_ru, P_k, params["sigma_sq"])
    
    print("\n--- Testing with a random RIS configuration ---")
    print(f"Resulting Sum-SNR: {sum_snr_random}")
    if sum_snr_random > 0:
        print(f"Resulting Sum-SNR in dB: {10 * np.log10(sum_snr_random)}")
    else:
        print("Sum-SNR is not positive, cannot compute dB.")
