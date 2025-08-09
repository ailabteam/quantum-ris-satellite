# run_vs_N.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# Import our own modules
import config
import system_model as sm
import classical_solvers as cs
import quantum_solver as qs

def run_experiment_vs_N(N_list):
    """
    Runs the simulation for a list of N values and saves the results.
    """
    results = []

    for N_val in N_list:
        print(f"\n{'='*20} RUNNING FOR N = {N_val} {'='*20}")
        
        # 1. Setup with the current N
        params = config.get_params()
        params['N'] = N_val
        np.random.seed(42) # Use the same seed for fair comparison
        
        print("--- Setting up System Channels ---")
        h_sr, H_ru = sm.generate_system_channels(params)
        Q = sm.get_Q_matrix(h_sr, H_ru, params)
        print("System setup complete.")

        # 2. Run all solvers
        # --- Classical Solvers ---
        print("\n--- Solving with Classical Methods ---")
        v_random = cs.solve_random(params)
        v_benchmark = cs.solve_benchmark_user0(h_sr, H_ru, params)
        v_sdr = cs.solve_sdr(Q, params)
        
        # --- Quantum Solver ---
        print("\n--- Solving with Quantum Method ---")
        # NOTE: QAOA is always run on a small subset, but the problem space (Q) is for the full N
        v_qaoa = qs.run_qaoa_optimization(Q, params)

        # 3. Calculate metrics
        snr_random = sm.calculate_sum_snr_from_Q(v_random, Q)
        snr_benchmark = sm.calculate_sum_snr_from_Q(v_benchmark, Q)
        snr_sdr = sm.calculate_sum_snr_from_Q(v_sdr, Q)
        snr_qaoa = sm.calculate_sum_snr_from_Q(v_qaoa, Q)

        # 4. Store results
        current_run = {
            'N': N_val,
            'SNR_Random': snr_random,
            'SNR_Benchmark': snr_benchmark,
            'SNR_SDR': snr_sdr,
            'SNR_QAOA': snr_qaoa
        }
        results.append(current_run)
        print(f"\n--- Results for N = {N_val} ---")
        print(f"SDR: {10*np.log10(snr_sdr):.2f} dB, QAOA: {10*np.log10(snr_qaoa):.2f} dB")

    # 5. Save to CSV
    df = pd.DataFrame(results)
    df.to_csv('results/results_vs_N.csv', index=False)
    print("\nFull results saved to 'results/results_vs_N.csv'")
    return df

def plot_results_vs_N(df):
    """Plots the results from the dataframe."""
    print("\nPlotting results...")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Convert to dB for plotting
    for col in df.columns:
        if 'SNR' in col:
            df[f'dB_{col}'] = 10 * np.log10(df[col].clip(lower=1e-15))

    ax.plot(df['N'], df['dB_SNR_Random'], 'o--', color='gray', label='Random')
    ax.plot(df['N'], df['dB_SNR_Benchmark'], 's--', color='orange', label='Benchmark (User 0)')
    ax.plot(df['N'], df['dB_SNR_SDR'], '^-', color='green', label='SDR')
    ax.plot(df['N'], df['dB_SNR_QAOA'], 'd-', color='blue', label=f'QAOA (N={config.get_params()["qaoa_sim_qubits"]})')
    
    ax.set_xlabel('Number of RIS Elements (N)', fontsize=12)
    ax.set_ylabel('Sum-SNR (dB)', fontsize=12)
    ax.set_title('Performance vs. Number of RIS Elements', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle=':')
    
    fig.savefig('results/plot_vs_N.png', dpi=600, bbox_inches='tight')
    print("Plot saved to 'results/plot_vs_N.png'")

if __name__ == "__main__":
    # Ensure results directory exists
    import os
    if not os.path.exists('results'):
        os.makedirs('results')

    # Define the list of N values to test
    N_list_to_run = [64, 128, 256] 
    
    start_time_total = time.time()
    results_df = run_experiment_vs_N(N_list_to_run)
    plot_results_vs_N(results_df)
    end_time_total = time.time()
    
    print(f"\nTotal experiment time: {(end_time_total - start_time_total)/60:.2f} minutes.")
