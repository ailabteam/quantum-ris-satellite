import numpy as np

# A workable parameter setup based on LEO satellite
def setup_parameters_workable():
    params = {
        "N": 256,
        "K": 4,
        "sat_pos": np.array([0, 0, 1000e3]),
        "ris_pos": np.array([0, 0, 50]),
        "user_area_center": np.array([0, 100, 0]),
        "user_area_radius": 50,
        "fc": 4e9,
        "c": 3e8,
        "lambda_c": 3e8 / 4e9,
        "gain_sat_dBi": 40,
        "gain_ris_ele_dBi": 5,
        "gain_user_dBi": 10,
        "Pt_dBm": 40,
        "P_sat": 10**((40 - 30) / 10),
        "noise_figure_dB": 5,
        "noise_power_dbm": -174 + 10 * np.log10(10e6) + 5,
        "sigma_sq": 10**(((-174 + 10 * np.log10(10e6) + 5) - 30) / 10),
        "k_sr": 10,
        "k_ru": 3
    }
    # Convert gains from dBi to linear scale
    for key in ["gain_sat_dBi", "gain_ris_ele_dBi", "gain_user_dBi"]:
        linear_key = key.replace("_dBi", "")
        # Creates keys: G_SAT, G_RIS_ELE, G_USER
        params[linear_key.upper().replace("GAIN_", "G_")] = 10**(params[key] / 10)
    return params

def create_rician_channel_v2(dim1, dim2, distance, k_factor, lambda_c, G_tx, G_rx):
    path_loss_val = (lambda_c / (4 * np.pi * distance))**2
    if distance == 0: path_loss_val = 1.0 # Avoid division by zero
    
    total_gain = np.sqrt(path_loss_val * G_tx * G_rx)
    # Using a deterministic but complex LoS component for better modeling
    h_los = np.exp(-1j * 2 * np.pi * distance / lambda_c)
    h_nlos = (np.random.randn(dim1, dim2) + 1j * np.random.randn(dim1, dim2)) / np.sqrt(2)
    channel = total_gain * (np.sqrt(k_factor / (k_factor + 1)) * h_los + np.sqrt(1 / (k_factor + 1)) * h_nlos)
    return channel

def generate_system_channels_v2(params):
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
    
    h_sr = np.zeros((N, 1), dtype=np.complex128)
    for n in range(N):
        # FIXED: Using uppercase keys G_SAT, G_RIS_ELE
        h_sr[n] = create_rician_channel_v2(1, 1, dist_sr, params["k_sr"], params["lambda_c"], params["G_SAT"], params["G_RIS_ELE"])

    H_ru = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        for n in range(N):
            # FIXED: Using uppercase keys G_RIS_ELE, G_USER
            H_ru[k, n] = create_rician_channel_v2(1, 1, dist_ru[k], params["k_ru"], params["lambda_c"], params["G_RIS_ELE"], params["G_USER"])
            
    return h_sr, H_ru

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
    
# Main execution
if __name__ == "__main__":
    print("--- Phase 1 (V4.1): Workable LEO Scenario (Fixed) ---")
    
    params = setup_parameters_workable()
    print(f"Parameters: N={params['N']}, K={params['K']}, Sat Altitude={params['sat_pos'][2]/1e3}km")
    
    np.random.seed(42)
    h_sr, H_ru = generate_system_channels_v2(params)
    
    # Test with OPTIMAL phases for one user to see the potential
    k_test = 0
    h_ru_k_test = H_ru[k_test, :]
    cascaded_vector = h_ru_k_test * h_sr.T.flatten()
    
    # Quantize to 1-bit (-1, 1)
    # v_n = 1 if phase is in [-pi/2, pi/2], -1 otherwise
    v_optimal_1bit = np.sign(np.cos(np.angle(cascaded_vector)))
    # For a perfect match, v must be complex. For 1-bit, this is the best we can do.

    P_k = params["P_sat"] / params["K"]

    sum_snr_optimal = calculate_sum_snr(v_optimal_1bit, h_sr, H_ru, P_k, params["sigma_sq"])
    
    print("\n--- Testing with nearly-optimal 1-bit RIS configuration ---")
    print(f"Resulting Sum-SNR: {sum_snr_optimal}")
    if sum_snr_optimal > 0:
        print(f"Resulting Sum-SNR in dB: {10 * np.log10(sum_snr_optimal)}")
    else:
        print("Sum-SNR is not positive.")
