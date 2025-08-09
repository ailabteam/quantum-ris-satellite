# config.py
import numpy as np

def get_params():
    """Returns a dictionary of all simulation parameters."""
    params = {
        # --- System Layout ---
        "N": 256,
        "K": 4,
        "sat_pos": np.array([0, 0, 1000e3]),
        "ris_pos": np.array([0, 0, 50]),
        "user_area_center": np.array([0, 100, 0]),
        "user_area_radius": 50,
        
        # --- RF and Channel ---
        "fc": 4e9,
        "c": 3e8,
        "lambda_c": 3e8 / 4e9,
        "k_sr": 10,
        "k_ru": 3,
        
        # --- Antenna Gains (dBi) ---
        "gain_sat_dBi": 40,
        "gain_ris_ele_dBi": 5,
        "gain_user_dBi": 10,

        # --- Power and Noise ---
        "Pt_dBm": 40,
        "P_sat": 10**((40 - 30) / 10),
        "noise_figure_dB": 5,
        "sigma_sq": 10**(((-174 + 10 * np.log10(10e6) + 5) - 30) / 10), # BW = 10MHz

        # --- QAOA Parameters ---
        "qaoa_p_layers": 3,
        "qaoa_n_steps": 80,
        "qaoa_lr": 0.05,
        "qaoa_sim_qubits": 10, # Number of qubits to simulate for QAOA

        # --- SDR Parameters ---
        "sdr_num_randomizations": 100
    }

    # Convert gains from dBi to linear scale
    for key in ["gain_sat_dBi", "gain_ris_ele_dBi", "gain_user_dBi"]:
        linear_key = key.replace("_dBi", "").replace("gain_", "G_").upper()
        params[linear_key] = 10**(params[key] / 10)
        
    return params
