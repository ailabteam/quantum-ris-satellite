# plot_generator.py
import pandas as pd
import numpy as np
import matplotlib
# Use a non-interactive backend suitable for shell environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_performance_vs_N(df, save_path):
    """
    Generates and saves a line plot of Sum-SNR (dB) vs. N.
    This will be Figure 3 in the paper.
    """
    print("Generating plot: Performance vs. N...")
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(df['N'], df['dB_SNR_Random'], 'o--', color='gray', label='Random', linewidth=1.5, markersize=7)
    ax.plot(df['N'], df['dB_SNR_Benchmark'], 's--', color='#ff7f0e', label='Benchmark (User 0)', linewidth=1.5, markersize=7)
    ax.plot(df['N'], df['dB_SNR_SDR'], '^-', color='#2ca02c', label='SDR', linewidth=2, markersize=8)
    ax.plot(df['N'], df['dB_SNR_QAOA'], 'd-', color='#1f77b4', label=f'Proposed QAOA', linewidth=2.5, markersize=8)
    
    ax.set_xlabel('Number of RIS Elements (N)', fontsize=14)
    ax.set_ylabel('Sum-SNR (dB)', fontsize=14)
    ax.set_title('Performance vs. Number of RIS Elements', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle=':')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xticks(df['N'])

    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Plot saved to '{save_path}'")

def plot_gain_comparison_at_N256(df, save_path):
    """
    Generates and saves a bar chart showing the performance GAIN over the Random baseline.
    This will be Figure 4 in the paper.
    """
    print("\nGenerating plot: Performance Gain Comparison at N=256...")
    
    data_n256 = df[df['N'] == 256].iloc[0]
    
    # Calculate gain over the random baseline
    random_baseline_db = data_n256['dB_SNR_Random']
    gain_benchmark = data_n256['dB_SNR_Benchmark'] - random_baseline_db
    gain_sdr = data_n256['dB_SNR_SDR'] - random_baseline_db
    gain_qaoa = data_n256['dB_SNR_QAOA'] - random_baseline_db

    methods = ['Benchmark\n(User 0)', 'SDR', 'Proposed\nQAOA']
    gains_db = [gain_benchmark, gain_sdr, gain_qaoa]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 7))
    
    bars = ax.bar(methods, gains_db, color=colors, width=0.5, zorder=3)
    
    ax.set_ylabel('Performance Gain over Random Baseline (dB)', fontsize=14)
    ax.set_title('Optimization Gain at N = 256', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12, rotation=0)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle=':', zorder=1)
    
    ax.bar_label(bars, fmt='%.2f dB', fontsize=12, fontweight='bold', padding=5)

    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Plot saved to '{save_path}'")


if __name__ == "__main__":
    input_csv_path = 'results/results_vs_N.csv'
    output_dir = 'results'
    
    if not os.path.exists(input_csv_path):
        print(f"Error: Input file not found at '{input_csv_path}'")
        print("Please run 'run_vs_N.py' first to generate the data.")
    else:
        df = pd.read_csv(input_csv_path)

        for col in df.columns:
            if 'SNR' in col:
                df[f'dB_{col}'] = 10 * np.log10(df[col].clip(lower=1e-20))
        
        # Figure 3 in the paper
        plot_performance_vs_N(df, os.path.join(output_dir, 'fig_performance_vs_N.png'))
        
        # Figure 4 in the paper
        plot_gain_comparison_at_N256(df, os.path.join(output_dir, 'fig_gain_at_N256.png'))
        
        print("\nAll plots for the paper have been generated successfully.")
