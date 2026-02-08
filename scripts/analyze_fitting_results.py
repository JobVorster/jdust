import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ifu_analysis.jdspextract import load_spectra,merge_subcubes,read_aperture_ini
import os

source_name = 'L1448MM1'
aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt'%(source_name)
input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/cavity_modelling/'  

if os.path.isfile(aperture_filename):
    aper_names,aper_sizes,coord_list = read_aperture_ini(aperture_filename)

for aperture in aper_names:
    if aperture != 'B1':
        continue
    # Read the CSV file
    df = pd.read_csv(input_foldername + 'fitting_results_L1448MM1_%s.csv'%(aperture))

    # Get last 10000 iterations
    df_plot = df.tail(10000).copy()
    df_plot['iteration'] = df_plot['ID'] - df_plot['ID'].min()

    # Define parameters to plot (excluding ID and chi2_red for now)
    params = ['temp', 'scaling', 'temp2', 'scaling2', 'surface_density', 'Jv_Scale',
              'olmg50-0.1um', 'olmg50-1.5um', 'olmg50-6um', 
              'pyrmg70-0.1um', 'pyrmg70-1.5um', 'pyrmg70-6um']

    # Create figure with subplots
    fig, axes = plt.subplots(5, 3, figsize=(15, 12))
    axes = axes.flatten()

    # Plot each parameter
    for i, param in enumerate(params):
        ax = axes[i]
        ax.plot(df_plot['iteration'], df_plot[param], linewidth=0.5, alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(param)
        ax.set_title(param)
        ax.grid(True, alpha=0.3)
        
        # Use log scale for very small values
        if param in ['scaling', 'scaling2', 'Jv_Scale'] or 'um' in param:
            ax.set_yscale('log')

    # Plot chi2_red in the last panel
    axes[-1].plot(df_plot['iteration'], df_plot['chi2_red'], linewidth=0.5, color='red')
    axes[-1].set_xlabel('Iteration')
    axes[-1].set_ylabel('chi2_red')
    axes[-1].set_title('Reduced Chi-squared')
    axes[-1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(input_foldername + 'parameter_evolution_%s.png'%(aperture), dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics for last 1000 iterations
    print("\nSummary statistics (last 1000 iterations):")
    print("=" * 80)
    df_summary = df.tail(100)

    for param in ['temp', 'temp2', 'surface_density'] + [col for col in df.columns if 'um' in col]:
        mean_val = df_summary[param].mean()
        std_val = df_summary[param].std()
        print(f"{param:20s}: {mean_val:.6e} ± {std_val:.6e}")

    print(f"\nchi2_red: {df_summary['chi2_red'].mean():.2f} ± {df_summary['chi2_red'].std():.2f}")