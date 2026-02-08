#!/usr/bin/env env python
"""
Script to create corner plots for MCMC fitting results.
Visualizes posterior distributions and correlations between parameters.
Handles parameters with no dynamic range (fixed values).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import corner
from ifu_analysis.jdspextract import load_spectra, merge_subcubes, read_aperture_ini
import os


def make_corner_plots(input_foldername, aperture, n_iterations=10000, save_plots=True):
    """
    Create corner plots for MCMC fitting results.
    
    Parameters:
    -----------
    input_foldername : str
        Path to folder containing the CSV file
    aperture : str
        Aperture identifier for the filename
    n_iterations : int
        Number of last iterations to plot (default: 10000)
    save_plots : bool
        Whether to save the plots to files (default: True)
    """
    
    # Read the data
    print(f"Reading data from: {input_foldername}fitting_results_L1448MM1_{aperture}.csv")
    df = pd.read_csv(input_foldername + f'fitting_results_L1448MM1_{aperture}.csv')
    
    # Get last n_iterations
    df_plot = df.tail(n_iterations).copy()
    df_plot['iteration'] = df_plot['ID'] - df_plot['ID'].min()
    
    print(f"Loaded {len(df)} total iterations, using last {len(df_plot)} for plotting")
    
    # Define parameters to plot
    params = ['temp', 'scaling', 'temp2', 'scaling2', 'surface_density', 'Jv_Scale',
              'olmg50-0.1um', 'olmg50-1.5um', 'olmg50-6um', 
              'pyrmg70-0.1um', 'pyrmg70-1.5um', 'pyrmg70-6um']
    
    # Create labels for better readability
    labels = ['Temp', 'Scaling', 'Temp 2', 'Scaling 2', 'Surface Density', r'$J_v$ Scale',
              'Olivine 0.1μm', 'Olivine 1.5μm', 'Olivine 6.0μm',
              'Pyroxene 0.1μm', 'Pyroxene 1.5μm', 'Pyroxene 6.0μm']
    
    # Check for parameters with no dynamic range and filter them out
    params_to_plot = []
    labels_to_plot = []
    
    print("\nChecking parameter ranges:")
    print("-" * 60)
    for param, label in zip(params, labels):
        param_range = np.ptp(df_plot[param]*1.2)  # peak-to-peak (max - min)
        param_std = np.std(df_plot[param])
        param_min = np.min(df_plot[param])
        param_max = np.max(df_plot[param])
        
        if param_range == 0 or param_std == 0:
            print(f"{label:20s}: FIXED at {df_plot[param].iloc[0]:.6e}")
        else:
            print(f"{label:20s}: [{param_min:.6e}, {param_max:.6e}] (std={param_std:.6e})")
            params_to_plot.append(param)
            labels_to_plot.append(label)
    print("-" * 60)
    
    if len(params_to_plot) == 0:
        print("ERROR: No parameters with dynamic range found!")
        return None, df_plot
    
    # Extract only the varying parameters
    data = df_plot[params_to_plot].values
    
    print(f"\nPlotting {len(params_to_plot)} out of {len(params)} parameters")
    
    # Create the corner plot
    print("Creating corner plot...")
    fig = corner.corner(
        data,
        labels=labels_to_plot,
        quantiles=[0.16, 0.5, 0.84],  # 1-sigma and median
        show_titles=True,
        title_fmt='.3e',  # Scientific notation for better readability
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 10},
        smooth=1.0,
        color='steelblue',
        fill_contours=True,
        levels=(0.68, 0.95),  # 1-sigma and 2-sigma contours
        plot_datapoints=True,
        plot_density=True,
        hist_kwargs={'density': True}
    )
    
    # Add a title to the figure
    fig.suptitle(f'L1448MM1 {aperture} - Last {n_iterations} iterations', 
                 fontsize=14, y=1.0)
    
    plt.tight_layout()
    
    # Save the plot
    if save_plots:
        output_filename = f'corner_plot_L1448MM1_{aperture}.png'
        plt.savefig(input_foldername + output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved corner plot to: {input_foldername}{output_filename}")
        
        # Also save as PDF for publication quality
        output_filename_pdf = f'corner_plot_L1448MM1_{aperture}.pdf'
        plt.savefig(input_foldername + output_filename_pdf, bbox_inches='tight')
        print(f"Saved corner plot to: {input_foldername}{output_filename_pdf}")
    
    plt.show()
    
    # Print some statistics
    print("\n" + "="*70)
    print("Parameter Statistics (median ± std):")
    print("="*70)
    for param, label in zip(params, labels):
        median = np.median(df_plot[param])
        std = np.std(df_plot[param])
        param_range = np.ptp(df_plot[param])
        
        if param_range == 0:
            print(f"{label:20s}: {median:12.6e} (FIXED)")
        else:
            p16 = np.percentile(df_plot[param], 16)
            p84 = np.percentile(df_plot[param], 84)
            print(f"{label:20s}: {median:12.6e} +{p84-median:12.6e} -{median-p16:12.6e}")
    print("="*70)
    
    return fig, df_plot


def make_subset_corner_plots(input_foldername, aperture, n_iterations=10000):
    """
    Create separate corner plots for different parameter groups.
    Useful when you have too many parameters for a single plot.
    """
    
    # Read the data
    df = pd.read_csv(input_foldername + f'fitting_results_L1448MM1_{aperture}.csv')
    df_plot = df.tail(n_iterations).copy()
    
    # Group 1: Temperature and scaling parameters
    params_thermal = ['temp', 'scaling', 'temp2', 'scaling2', 'surface_density', 'Jv_Scale']
    labels_thermal = ['Temp', 'Scaling', 'Temp 2', 'Scaling 2', 'Surface Density', 
                      r'$J_v$ T', r'$J_v$ Scale']
    
    # Group 2: Dust grain parameters - Olivine
    params_olivine = ['olmg50-0.1um', 'olmg50-1.5um', 'olmg50-6um']
    labels_olivine = ['Olivine 0.1μm', 'Olivine 1.5μm', 'Olivine 6μm']
    
    # Group 3: Dust grain parameters - Pyroxene
    params_pyroxene = ['pyrmg70-0.1um', 'pyrmg70-1.5um', 'pyrmg70-6um']
    labels_pyroxene = ['Pyroxene 0.1μm', 'Pyroxene 1.5μm', 'Pyroxene 6μm']
    
    # Create corner plot for thermal parameters
    print("Creating corner plot for thermal parameters...")
    
    # Filter out fixed parameters
    thermal_varying = []
    thermal_labels_varying = []
    for param, label in zip(params_thermal, labels_thermal):
        if np.ptp(df_plot[param]) > 0:
            thermal_varying.append(param)
            thermal_labels_varying.append(label)
        else:
            print(f"  Skipping fixed parameter: {param} = {df_plot[param].iloc[0]:.6e}")
    
    if len(thermal_varying) > 0:
        fig1 = corner.corner(
            df_plot[thermal_varying].values,
            labels=thermal_labels_varying,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.3e',
            color='steelblue'
        )
        fig1.suptitle(f'Thermal Parameters - {aperture}', fontsize=14)
        plt.savefig(input_foldername + f'corner_plot_thermal_{aperture}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved thermal corner plot")
    else:
        print("  No varying thermal parameters to plot!")
        fig1 = None
    
    # Create corner plot for olivine parameters
    print("Creating corner plot for olivine parameters...")
    
    olivine_varying = []
    olivine_labels_varying = []
    for param, label in zip(params_olivine, labels_olivine):
        if np.ptp(df_plot[param]) > 0:
            olivine_varying.append(param)
            olivine_labels_varying.append(label)
        else:
            print(f"  Skipping fixed parameter: {param} = {df_plot[param].iloc[0]:.6e}")
    
    if len(olivine_varying) > 0:
        fig2 = corner.corner(
            df_plot[olivine_varying].values,
            labels=olivine_labels_varying,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.3e',
            color='forestgreen'
        )
        fig2.suptitle(f'Olivine Grain Sizes - {aperture}', fontsize=14)
        plt.savefig(input_foldername + f'corner_plot_olivine_{aperture}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved olivine corner plot")
    else:
        print("  No varying olivine parameters to plot!")
        fig2 = None
    
    # Create corner plot for pyroxene parameters
    print("Creating corner plot for pyroxene parameters...")
    
    pyroxene_varying = []
    pyroxene_labels_varying = []
    for param, label in zip(params_pyroxene, labels_pyroxene):
        if np.ptp(df_plot[param]) > 0:
            pyroxene_varying.append(param)
            pyroxene_labels_varying.append(label)
        else:
            print(f"  Skipping fixed parameter: {param} = {df_plot[param].iloc[0]:.6e}")
    
    if len(pyroxene_varying) > 0:
        fig3 = corner.corner(
            df_plot[pyroxene_varying].values,
            labels=pyroxene_labels_varying,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.3e',
            color='coral'
        )
        fig3.suptitle(f'Pyroxene Grain Sizes - {aperture}', fontsize=14)
        plt.savefig(input_foldername + f'corner_plot_pyroxene_{aperture}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved pyroxene corner plot")
    else:
        print("  No varying pyroxene parameters to plot!")
        fig3 = None
    
    plt.show()
    
    print("\nFinished creating subset corner plots!")
    
    return fig1, fig2, fig3


# Example usage
if __name__ == "__main__":
    # Set your parameters here
    input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/cavity_modelling/'  
    source_name = 'L1448MM1'
    aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt' % (source_name)
    
    if os.path.isfile(aperture_filename):
        aper_names, aper_sizes, coord_list = read_aperture_ini(aperture_filename)
    
    for aperture in aper_names:
        if aperture != 'B1':
            continue
        print("\n" + "="*70)
        print(f"Processing aperture: {aperture}")
        print("="*70)
        
        # Option 1: Create a single comprehensive corner plot
        fig, df_plot = make_corner_plots(input_foldername, aperture, n_iterations=10000)
        
        # Option 2: Create separate corner plots for different parameter groups
        # Uncomment the lines below if you prefer separate plots
        print("\nCreating subset plots...")
        fig1, fig2, fig3 = make_subset_corner_plots(input_foldername, aperture, n_iterations=10000)