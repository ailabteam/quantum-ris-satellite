import numpy as np
import pennylane as qml
import torch 
import time
import matplotlib
# Use a non-interactive backend for shell environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# PHASE 1: SYSTEM MODEL AND CHANNEL SIMULATION
# -------------------------------------------------------------------

def setup_parameters_workable():
    """A workable parameter setup based on a LEO satellite scenario."""
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
        "noise_power_dbm": -174 + 10 * np.log10(10e6),
        "sigma_sq": 10**(((-174 + 10 * np.log10(10e6) + 5) - 30) / 10),
        "k_sr": 10,
        "k_ru": 3
    }
    for key in ["gain_sat_dBi", "gain_ris_ele_dBi", "gain_user_dBi"]:
        linear_key = key.replace("_dBi", "").replace("gain_", "G_").upper()
        params[linear_key] = 10**(params[key] / 10)
    return params

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

def create_ising_hamiltonian_coeffs(h_sr, H_ru, P_k, sigma_sq):
    N, K = h_sr.shape[0], H_ru.shape[0]
    Q = np.zeros((N, N), dtype=np.complex128)
    C = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        C[k, :] = H_ru[k, :] * h_sr.T.flatten()
    for n in range(N):
        for m in range(N):
            q_nm = np.sum((P_k / sigma_sq) * C[:, n] * np.conj(C[:, m]))
            Q[n, m] = q_nm
    J = np.zeros((N, N))
    h = np.zeros(N)
    for n in range(N):
        for m in range(n + 1, N):
            J[n, m] = -2 * np.real(Q[n, m])
    return J, h

# -------------------------------------------------------------------
# PHASE 3: QAOA SOLVER (FIXED AGAIN)
# -------------------------------------------------------------------

def run_qaoa_optimization(N, J, h, p_layers, n_steps=100, lr=0.1):
    if N > 12:
        N_qaoa = 10
        print(f"\nWARNING: System N={N} is too large for classical simulation.")
        print(f"--> Reducing to N_qaoa={N_qaoa} for QAOA demonstration.")
        J_qaoa = J[:N_qaoa, :N_qaoa]
        h_qaoa = h[:N_qaoa]
    else:
        N_qaoa = N
        J_qaoa = J
        h_qaoa = h

    print(f"\nSetting up QAOA with p={p_layers} layers for {N_qaoa} qubits...")
    dev = qml.device("default.qubit", wires=N_qaoa)
    
    cost_coeffs = [J_qaoa[i, j] for i in range(N_qaoa) for j in range(i + 1, N_qaoa) if J_qaoa[i, j] != 0]
    cost_obs = [qml.PauliZ(i) @ qml.PauliZ(j) for i in range(N_qaoa) for j in range(i + 1, N_qaoa) if J_qaoa[i, j] != 0]
    cost_h = qml.Hamiltonian(cost_coeffs, cost_obs)
    
    # --- FINAL FIX: Manually construct the mixer Hamiltonian ---
    mixer_coeffs = [1.0] * N_qaoa
    mixer_obs = [qml.PauliX(i) for i in range(N_qaoa)]
    mixer_h = qml.Hamiltonian(mixer_coeffs, mixer_obs)
    
    def qaoa_circuit_manual(params, **kwargs):
        gammas, betas = params[0], params[1]
        for i in range(N_qaoa):
            qml.Hadamard(wires=i)
        for p in range(p_layers):
            qml.qaoa.cost_layer(gammas[p], cost_h)
            qml.qaoa.mixer_layer(betas[p], mixer_h)

    @qml.qnode(dev, interface="torch")
    def cost_function(params):
        qaoa_circuit_manual(params)
        return qml.expval(cost_h)

    params = (2 * np.pi * np.random.rand(p_layers), 2 * np.pi * np.random.rand(p_layers))
    params_torch = [torch.tensor(p, requires_grad=True) for p in params]
    optimizer = torch.optim.Adam(params_torch, lr=lr)
    
    print("Starting QAOA optimization...")
    cost_history = []
    start_time = time.time()
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = cost_function(params_torch)
        loss.backward()
        optimizer.step()
        cost_history.append(loss.item())
        if (step + 1) % 10 == 0:
            print(f"Step {step+1:3d}: Cost = {loss.item():.8f}")
    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.2f} seconds.")

    @qml.qnode(dev)
    def probability_circuit(params):
        qaoa_circuit_manual(params)
        return qml.probs(wires=range(N_qaoa))

    final_params = (params_torch[0].detach().numpy(), params_torch[1].detach().numpy())
    probs = probability_circuit(final_params)
    most_likely_outcome = np.argmax(probs)
    binary_string = format(most_likely_outcome, f'0{N_qaoa}b')
    v_qaoa_small = np.array([1 if bit == '0' else -1 for bit in binary_string])
    
    v_qaoa_full = np.ones(N)
    v_qaoa_full[:N_qaoa] = v_qaoa_small
    return v_qaoa_full, cost_history, N_qaoa

# -------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# -------------------------------------------------------------------
if __name__ == "__main__":
    
    print("--- Setting up System and Mapping to Ising Model ---")
    params = setup_parameters_workable()
    N_val, K_val = params["N"], params["K"]
    print(f"Parameters: N={N_val}, K={K_val}, Sat Altitude={params['sat_pos'][2]/1e3}km")
    np.random.seed(42)
    h_sr, H_ru = generate_system_channels(params)
    P_k = params["P_sat"] / K_val
    J, h = create_ising_hamiltonian_coeffs(h_sr, H_ru, P_k, params["sigma_sq"])
    print("System setup and Ising mapping complete.")

    p_layers = 3
    v_qaoa, history, N_qaoa = run_qaoa_optimization(N_val, J, h, p_layers, n_steps=80, lr=0.05)
    
    print("\n\n--- FINAL PERFORMANCE COMPARISON ---")
    v_random = np.random.choice([-1, 1], size=N_val)
    cascaded_k0 = H_ru[0, :] * h_sr.T.flatten()
    v_benchmark = np.sign(np.cos(np.angle(cascaded_k0)))
    
    snr_random = calculate_sum_snr(v_random, h_sr, H_ru, P_k, params["sigma_sq"])
    snr_benchmark = calculate_sum_snr(v_benchmark, h_sr, H_ru, P_k, params["sigma_sq"])
    snr_qaoa = calculate_sum_snr(v_qaoa, h_sr, H_ru, P_k, params["sigma_sq"])
    
    db_random = 10*np.log10(snr_random) if snr_random > 0 else -np.inf
    db_benchmark = 10*np.log10(snr_benchmark) if snr_benchmark > 0 else -np.inf
    db_qaoa = 10*np.log10(snr_qaoa) if snr_qaoa > 0 else -np.inf
    
    print("\nPerformance (Sum-SNR in dB):")
    print("--------------------------------")
    print(f"1. Random Phases:                    {db_random:.2f} dB")
    print(f"2. Benchmark (Optimized for User 0): {db_benchmark:.2f} dB")
    print(f"3. QAOA (Optimized for {N_qaoa}/{N_val} elements): {db_qaoa:.2f} dB")
    print("--------------------------------")
    
    print("\n--- Saving plots to files ---")
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(history, color='navy', linewidth=2)
    ax1.set_xlabel("Optimization Steps", fontsize=12)
    ax1.set_ylabel("Cost (Expected Energy)", fontsize=12)
    ax1.set_title(f"QAOA Cost Function Convergence (p={p_layers}, N={N_qaoa})", fontsize=14)
    fig1.savefig('qaoa_convergence.png', dpi=600, bbox_inches='tight')
    print("Convergence plot saved to 'qaoa_convergence.png'")

    methods = [f'Random', f'Benchmark\n(User 0)', f'QAOA\n(N={N_qaoa})']
    snr_values_db = [db_random, db_benchmark, db_qaoa]
    colors = ['#808080', '#ff7f0e', '#1f77b4']

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    bars = ax2.bar(methods, snr_values_db, color=colors, width=0.6)
    ax2.set_ylabel('Sum-SNR (dB)', fontsize=12)
    ax2.set_title('Performance Comparison of RIS Configurations', fontsize=14)
    min_val = min(snr_values_db)
    max_val = max(snr_values_db)
    ax2.set_ylim(bottom=min_val - abs(min_val*0.1) - 2, top=max_val + abs(max_val*0.1) + 2)
    
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom')

    fig2.savefig('performance_comparison.png', dpi=600, bbox_inches='tight')
    print("Performance comparison plot saved to 'performance_comparison.png'")
