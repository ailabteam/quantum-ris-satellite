# system_model.py
import numpy as np

def create_rician_channel(dim1, dim2, distance, k_factor, lambda_c, G_tx, G_rx):
    if distance < 1e-6: distance = 1e-6
    path_loss_val = (lambda_c / (4 * np.pi * distance))**2
    total_gain = np.sqrt(path_loss_val * G_tx * G_rx)
    h_los = np.exp(-1j * 2 * np.pi * distance / lambda_c)
    h_nlos = (np.random.randn(dim1, dim2) + 1j * np.random.randn(dim1, dim2)) / np.sqrt(2)
    channel = total_gain * (np.sqrt(k_factor / (k_factor + 1)) * h_los + np.sqrt(1 / (k_factor + 1)) * h_nlos)
    return channel

def generate_system_channels(params):
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
        h_sr[n] = create_rician_channel(1, 1, dist_sr, params["k_sr"], params["lambda_c"], params["G_SAT"], params["G_RIS_ELE"])
    H_ru = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        for n in range(N):
            H_ru[k, n] = create_rician_channel(1, 1, dist_ru[k], params["k_ru"], params["lambda_c"], params["G_RIS_ELE"], params["G_USER"])
    return h_sr, H_ru

def get_Q_matrix(h_sr, H_ru, params):
    """Calculates the Q matrix for the quadratic objective function."""
    N, K = params["N"], params["K"]
    P_k = params["P_sat"] / K
    
    Q = np.zeros((N, N), dtype=np.complex128)
    C = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        C[k, :] = H_ru[k, :] * h_sr.T.flatten()
    
    for n in range(N):
        for m in range(N):
            q_nm = np.sum((P_k / params["sigma_sq"]) * C[:, n] * np.conj(C[:, m]))
            Q[n, m] = q_nm
    return Q

def calculate_sum_snr_from_Q(v, Q):
    """Calculates sum-SNR directly from the Q matrix and a vector v."""
    # sum-SNR = v^H Q v
    # For real v={-1,1}, this is v^T Q v
    return np.real(v.T @ Q @ v)
