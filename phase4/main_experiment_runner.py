# main_experiment_runner.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import our own modules
import config
import system_model as sm
import classical_solvers as cs
import quantum_solver as qs

def run_single_experiment():
    """Runs a single simulation and compares all solvers."""
    
    # 1. Setup
    params = config.get_params()
    np.random.seed(42)
    print("--- Setting up System Channels ---")
    h_sr, H_ru = sm.generate_system_channels(params)
    Q = sm.get_Q_matrix(h_sr, H_ru, params)
    print("System setup complete.")

    # 2. Solve with all methods
    print("\n--- Solving with Classical Methods ---")
    v_random = cs.solve_random(params)
    v_benchmark = cs.solve_benchmark_user0(h_sr, H_ru, params)
    v_sdr = cs.solve_sdr(Q, params)
    
    print("\n--- Solving with Quantum Method ---")
    v_qaoa = qs.run_qaoa_optimization(Q, params)
    
    # 3. Calculate and print results
    snr_random = sm.calculate_sum_snr_from_Q(v_random, Q)
    snr_benchmark = sm.calculate_sum_snr_from_Q(v_benchmark, Q)
    snr_sdr = sm.calculate_sum_snr_from_Q(v_sdr, Q)
    snr_qaoa = sm.calculate_sum_snr_from_Q(v_qaoa, Q)
    
    db_random = 10*np.log10(snr_random) if snr_random > 0 else -np.inf
    db_benchmark = 10*np.log10(snr_benchmark) if snr_benchmark > 0 else -np.inf
    db_sdr = 10*np.log10(snr_sdr) if snr_sdr > 0 else -np.inf
    db_qaoa = 10*np.log10(snr_qaoa) if snr_qaoa > 0 else -np.inf

    print("\n\n--- FINAL PERFORMANCE COMPARISON ---")
    print("--------------------------------")
    print(f"1. Random:         {db_random:.2f} dB")
    print(f"2. Benchmark (U0): {db_benchmark:.2f} dB")
    print(f"3. SDR:            {db_sdr:.2f} dB")
    print(f"4. QAOA (N={params['qaoa_sim_qubits']}): {db_qaoa:.2f} dB")
    print("--------------------------------")

    # 4. Plotting
    methods = ['Random', 'Benchmark\n(User 0)', 'SDR', f'QAOA\n(N={params["qaoa_sim_qubits"]})']
    snr_values_db = [db_random, db_benchmark, db_sdr, db_qaoa]
    colors = ['#808080', '#ff7f0e', '#2ca02c', '#1f77b4'] # Gray, Orange, Green, Blue

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(methods, snr_values_db, color=colors, width=0.6)
    ax.set_ylabel('Sum-SNR (dB)', fontsize=12)
    ax.set_title('Performance Comparison of RIS Configurations', fontsize=14)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom')

    fig.savefig('results/single_run_comparison.png', dpi=600, bbox_inches='tight')
    print("\nComparison plot saved to 'results/single_run_comparison.png'")

if __name__ == "__main__":
    # Create a directory for results if it doesn't exist
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
        
    run_single_experiment()
