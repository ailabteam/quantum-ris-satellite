import numpy as np

# A simplified parameter setup for debugging
def setup_parameters_debug():
    params = {
        "N": 16,
        "K": 1, # FOCUS ON A SINGLE USER FIRST
        "sat_pos": np.array([0, 0, 35786e3]),
        "ris_pos": np.array([0, 0, 50]), # RIS directly below satellite for simplicity
        "user_pos": np.array([0, 100, 1.5]), # A single user at a fixed position
        "fc": 4e9,
        "c": 3e8,
        "lambda_c": 3e8 / 4e9,
        "gain_sat_dBi": 50, # Increased satellite gain
        "gain_ris_ele_dBi": 5, # Increased RIS element gain
        "gain_user_dBi": 0,
        "Pt_dBm": 30,
        "P_sat": 1,
        "noise_power_dbm": -99, # Simplified noise: -174dBm/Hz + 10log10(10e6) + 5dB NF
        "sigma_sq": 10**((-99 - 30) / 10)
    }
    params["G_sat"] = 10**(params["gain_sat_dBi"] / 10)
    params["G_ris_ele"] = 10**(params["gain_ris_ele_dBi"] / 10)
    params["G_user"] = 10**(params["gain_user_dBi"] / 10)
    return params

def get_path_loss(d, lambda_c):
    return (lambda_c / (4 * np.pi * d))**2

# Simplified LoS-only channel model
def create_los_channel(num_elements, distance, lambda_c, G_tx, G_rx):
    """Creates a pure Line-of-Sight channel vector."""
    path_loss_val = get_path_loss(distance, lambda_c)
    total_gain = np.sqrt(path_loss_val * G_tx * G_rx)
    
    # In LoS, the phase is determined by distance. Assume all elements
    # receive the same phase from the far-away satellite.
    # We can model this as a simple real-valued gain.
    # The random phase component can be added later.
    return total_gain * np.ones((num_elements, 1), dtype=np.complex128)

def generate_system_channels_debug(params):
    N, K = params["N"], params["K"]
    
    dist_sr = np.linalg.norm(params["sat_pos"] - params["ris_pos"])
    dist_ru = np.linalg.norm(params["ris_pos"] - params["user_pos"])

    # S-R Channel: Pure LoS
    h_sr = create_los_channel(N, dist_sr, params["lambda_c"], params["G_sat"], params["G_ris_ele"])

    # R-U Channel: Pure LoS
    # Here, h_ru is a (1, N) row vector for a single user
    h_ru = create_los_channel(N, dist_ru, params["lambda_c"], params["G_ris_ele"], params["G_user"]).T
    
    # We now return a single user channel for simplicity
    return h_sr, h_ru

# Simplified SNR calculation for a single user
def calculate_snr_single_user(v, h_sr, h_ru, P_sat, sigma_sq):
    Phi = np.diag(v)
    effective_channel = h_ru @ Phi @ h_sr
    channel_gain = np.abs(effective_channel)**2
    snr = (P_sat * channel_gain) / sigma_sq
    return snr.item()

# -------------------------------------------------------------------
# Main execution block for debugging
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Phase 1 (V3): Debugging with Simplified LoS Model ---")
    
    params = setup_parameters_debug()
    print(f"Parameters: N={params['N']}, K={params['K']}, Sat Gain={params['gain_sat_dBi']}dBi")
    
    h_sr, h_ru = generate_system_channels_debug(params)

    # Let's inspect the channel magnitudes
    print(f"\nMagnitude of h_sr elements: {np.abs(h_sr[0, 0])}")
    print(f"Magnitude of h_ru elements: {np.abs(h_ru[0, 0])}")

    # **THE IDEAL CASE**: All phases are perfectly aligned
    # This is the best possible performance we can get.
    # We align the RIS phases to cancel the phase of the cascaded channel.
    # For our simple LoS model, all elements are in-phase, so v_n = 1 is optimal.
    v_optimal = np.ones(params["N"]) 
    
    # In the ideal case, the gains add up coherently.
    # The total channel gain should be roughly (N * |h_ru_n| * |h_sr_n|)^2
    
    snr_optimal = calculate_snr_single_user(v_optimal, h_sr, h_ru, params["P_sat"], params["sigma_sq"])

    print("\n--- Testing with the OPTIMAL RIS configuration (perfect alignment) ---")
    print(f"Resulting SNR: {snr_optimal}")
    if snr_optimal > 0:
        print(f"Resulting SNR in dB: {10 * np.log10(snr_optimal)}")
    else:
        print("SNR is not positive.")
