import numpy as np

# A workable parameter setup based on LEO satellite
def setup_parameters_workable():
    params = {
        "N": 256, # A more realistic number of RIS elements
        "K": 4,   # Back to multi-user case
        "sat_pos": np.array([0, 0, 1000e3]), # LEO satellite at 1000 km altitude
        "ris_pos": np.array([0, 0, 50]),
        "user_area_center": np.array([0, 100, 0]),
        "user_area_radius": 50,
        "fc": 4e9,
        "c": 3e8,
        "lambda_c": 3e8 / 4e9,

        # Gains - A more optimistic but realistic scenario
        "gain_sat_dBi": 40,   # Satellite antenna gain
        "gain_ris_ele_dBi": 5,    # RIS element gain
        "gain_user_dBi": 10,  # User antenna with some gain

        # Power & Noise
        "Pt_dBm": 40, # Satellite Tx Power = 10W
        "P_sat": 10**((40 - 30) / 10),
        "noise_figure_dB": 5,
        "noise_power_dbm": -174 + 10 * np.log10(10e6) + 5,
        "sigma_sq": 10**(((-174 + 10 * np.log10(10e6) + 5) - 30) / 10),
        
        # Channel Fading parameters (back to Rician)
        "k_sr": 10,
        "k_ru": 3
    }
    # Convert gains from dBi to linear scale
    for key in ["gain_sat_dBi", "gain_ris_ele_dBi", "gain_user_dBi"]:
        linear_key = key.replace("_dBi", "")
        params[linear_key.upper()] = 10**(params[key] / 10)
    return params

# Use the V2 functions which are more general
def create_rician_channel_v2(dim1, dim2, distance, k_factor, lambda_c, G_tx, G_rx):
    path_loss_val = (lambda_c / (4 * np.pi * distance))**2
    total_gain = np.sqrt(path_loss_val * G_tx * G_rx)
    h_los = np.exp(1j * 2 * np.pi * np.random.rand()) # Add a random phase to LoS
    h_nlos = (np.random.randn(dim1, dim2) + 1j * np.random.randn(dim1, dim2)) / np.sqrt(2)
    channel = total_gain * (np.sqrt(k_factor / (k_factor + 1)) * h_los + np.sqrt(1 / (k_factor + 1)) * h_nlos)
    return channel

def generate_system_channels_v2(params):
    N, K = params["N"], params["K"]
    user_positions = np.zeros((K, 3))
    # ... (user generation code from v2)
    for k in range(K):
        r = np.sqrt(np.random.rand()) * params["user_area_radius"]
        theta = 2 * np.pi * np.random.rand()
        user_positions[k, 0] = params["user_area_center"][0] + r * np.cos(theta)
        user_positions[k, 1] = params["user_area_center"][1] + r * np.sin(theta)
        user_positions[k, 2] = 1.5
        
    dist_sr = np.linalg.norm(params["sat_pos"] - params["ris_pos"])
    dist_ru = np.array([np.linalg.norm(params["ris_pos"] - user_pos) for user_pos in user_positions])
    
    # We now model the channel vectors more carefully
    # h_sr is a vector where each element is the channel to one RIS element
    h_sr = np.zeros((N, 1), dtype=np.complex128)
    for n in range(N):
        # Assume all elements are close enough to have same distance
        h_sr[n] = create_rician_channel_v2(1, 1, dist_sr, params["k_sr"], params["lambda_c"], params["G_SAT"], params["G_RIS_ELE"])

    H_ru = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        for n in range(N):
            H_ru[k, n] = create_rician_channel_v2(1, 1, dist_ru[k], params["k_ru"], params["lambda_c"], params["G_RIS_ELE"], params["G_USER"])
            
    return h_sr, H_ru

# Using the original sum-snr function
def calculate_sum_snr(v, h_sr, H_ru, P_k, sigma_sq):
    # ... (same as before)
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
    print("--- Phase 1 (V4): Workable LEO Scenario ---")
    
    params = setup_parameters_workable()
    print(f"Parameters: N={params['N']}, K={params['K']}, Sat Altitude={params['sat_pos'][2]/1e3}km")
    
    np.random.seed(42)
    h_sr, H_ru = generate_system_channels_v2(params)
    
    # Test with OPTIMAL phases for this channel realization
    # v_n should cancel the phase of h_ru,k,n * h_sr,n
    # Let's check for user 1.
    k_test = 0
    h_ru_k_test = H_ru[k_test, :]
    cascaded_vector = h_ru_k_test * h_sr.T.flatten() # Element-wise product
    
    # Optimal phase for each element is the conjugate of the cascaded channel's phase
    optimal_phases = np.exp(-1j * np.angle(cascaded_vector))
    # For 1-bit, we quantize this
    v_optimal_1bit = np.sign(np.cos(np.angle(cascaded_vector)))

    P_k = params["P_SAT"] / params["K"]

    sum_snr_optimal = calculate_sum_snr(v_optimal_1bit, h_sr, H_ru, P_k, params["sigma_sq"])
    
    print("\n--- Testing with nearly-optimal 1-bit RIS configuration ---")
    print(f"Resulting Sum-SNR: {sum_snr_optimal}")
    print(f"Resulting Sum-SNR in dB: {10 * np.log10(sum_snr_optimal)}")
