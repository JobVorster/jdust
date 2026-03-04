# PSF Subtraction Systematics Figure 1
# Rows 1 and 2: PSF scaling and uncertainties vs wavelength
#
# Author: Job Vorster — jobvorster8@gmail.com
# Version: 2026-03-03

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ifu_analysis.jdpsfsub import get_options_csv_fn, get_pixel_scale

# =============================================================================
# USER PARAMETERS
# =============================================================================

source_name        = 'L1448-mm'
subband_arr        = ['1A','1B','1C','2A','2B','2C','3A','3B','3C','4A','4B','4C']
fn_band_arr        = ['ch1-short','ch1-medium','ch1-long',
                      'ch2-short','ch2-medium','ch2-long',
                      'ch3-short','ch3-medium','ch3-long',
                      'ch4-short','ch4-medium','ch4-long']

options_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/PSFSub_Docs/L1448MM1/FIT_OPTIONS/'
figure_savepath    = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/publication_figures/psf_systematics_fig1_%s.png' % source_name

# SNR threshold below which Gaussian fit is unreliable (saturated channels)
SNR_SAT = 2.5

# =============================================================================
# END USER PARAMETERS
# =============================================================================

# Colors
C_DATA    = 'steelblue'
C_SMOOTH  = 'tomato'
C_FIT     = '#4477AA'
C_SMOOTH2 = '#EE6677'

# =============================================================================
# STEP 1: Load PSF options CSVs
# =============================================================================

def load_subband_data(subband_arr, fn_band_arr, options_foldername, snr_sat):
    '''
    Load PSF options CSVs for all subbands.
    Masks saturated channels and computes uncertainty columns.

    Parameters
    ----------
    subband_arr : list of str
        Subband labels e.g. ['1A', '1B', ...].
    fn_band_arr : list of str
        Band filenames e.g. ['ch1-short', ...].
    options_foldername : str
        Path to folder containing options CSV files.
    snr_sat : float
        SNR threshold below which channels are masked.

    Returns
    -------
    subband_data : list of dict
        Each dict has keys 'subband' and 'df'.
    '''
    subband_data = []

    for subband, fn_band in zip(subband_arr, fn_band_arr):
        csv_fn = get_options_csv_fn(options_foldername, source_name, fn_band)
        try:
            df = pd.read_csv(csv_fn)
        except FileNotFoundError:
            print('Warning: %s not found, skipping subband %s' % (csv_fn, subband))
            continue

        sat_mask = (df['fit_scaling'] / df['dfit_scaling']) < snr_sat
        df.loc[sat_mask, 'dfit_scaling'] = np.nan
        df.loc[sat_mask, 'fit_scaling']  = np.nan

        df['dtotal_scaling'] = np.sqrt(df['dfit_scaling']**2 + df['dsmooth_scaling']**2)
        df['dfit_scaling_frac']    = df['dfit_scaling']    / df['smooth_scaling']
        df['dsmooth_scaling_frac'] = df['dsmooth_scaling'] / df['smooth_scaling']
        df['dtotal_scaling_frac']  = df['dtotal_scaling']  / df['smooth_scaling']

        subband_data.append({'subband': subband, 'df': df})

    print('Loaded %d subbands successfully.' % len(subband_data))
    return subband_data


# =============================================================================
# STEP 2: Plot
# =============================================================================

def plot_psf_systematics(subband_data, figure_savepath):
    '''
    Plot PSF scaling (row 1) and uncertainties (row 2).

    Parameters
    ----------
    subband_data : list of dict
        Output of load_subband_data.
    figure_savepath : str
        Path to save the figure.
    '''
    panel_height = 3.46 * 0.5
    fig, axes = plt.subplots(2, 1, figsize=(3.46, panel_height * 2))
    fig.subplots_adjust(hspace=0.08)

    ax_scl     = axes[0]
    ax_unc_abs = axes[1]

    FONTSIZE_LABEL  = 7
    FONTSIZE_LEGEND = 6
    FONTSIZE_TICK   = 6

    # --- Row 1: I_PSF scaling ---
    for i, sd in enumerate(subband_data):
        df   = sd['df']
        mask = df['mask'].astype(bool)

        label_data   = r'$I_{\rm PSF}$'          if i == 0 else None
        label_sigma  = r'$\sigma_{\rm PSF}$'     if i == 0 else None
        label_smooth = r'Smoothed $I_{\rm PSF}$' if i == 0 else None

        ax_scl.plot(df['um'], df['fit_scaling'],
                    color=C_DATA, alpha=0.2, lw=0.5, label=label_data)
        ax_scl.plot(df['um'], df['dtotal_scaling'],
                    color='grey', alpha=1.0, lw=0.5, label=label_sigma)
        ax_scl.errorbar(df['um'][mask], df['fit_scaling'][mask],
                        yerr=df['dfit_scaling'][mask],
                        fmt='o', ms=1, color=C_DATA, alpha=0.6, zorder=2)
        ax_scl.plot(df['um'], df['smooth_scaling'],
                    color=C_SMOOTH, lw=1.0, zorder=4, label=label_smooth)

    ax_scl.set_yscale('log')
    ax_scl.set_ylabel(r'$I_{\rm PSF}$ (MJy sr$^{-1}$)', fontsize=FONTSIZE_LABEL)
    ax_scl.tick_params(labelbottom=False, labelsize=FONTSIZE_TICK)
    ax_scl.set_xlim(4.7, 27.5)
    ax_scl.set_ylim(1e-1, 1.2e5)
    ax_scl.legend(fontsize=FONTSIZE_LEGEND, loc='upper left', ncol=3, frameon=False)

    # --- Row 2: Uncertainties ---
    for sd in subband_data:
        df = sd['df']
        ax_unc_abs.plot(df['um'], df['dfit_scaling'],    color=C_FIT,     lw=0.5, alpha=1.0)
        ax_unc_abs.plot(df['um'], df['dsmooth_scaling'], color=C_SMOOTH2, lw=0.5, alpha=1.0)

    ax_unc_abs.set_yscale('log')
    ax_unc_abs.set_ylabel(r'$\sigma_{I_{\rm PSF}}$ (MJy sr$^{-1}$)', fontsize=FONTSIZE_LABEL)
    ax_unc_abs.set_xlabel(r'Wavelength ($\mu$m)', fontsize=FONTSIZE_LABEL)
    ax_unc_abs.tick_params(labelsize=FONTSIZE_TICK)
    ax_unc_abs.set_xlim(4.7, 27.5)

    legend_r2 = [
        Line2D([0], [0], color=C_FIT,     lw=0.5, label=r'$\sigma_{\rm fit}$'),
        Line2D([0], [0], color=C_SMOOTH2, lw=0.5, label=r'$\sigma_{\rm smooth}$'),
    ]
    ax_unc_abs.legend(handles=legend_r2, fontsize=FONTSIZE_LEGEND, loc='upper right', ncol=2, frameon=False)

    plt.savefig(figure_savepath, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()
    print('Saved figure to %s' % figure_savepath)


# =============================================================================
# MAIN
# =============================================================================

subband_data = load_subband_data(subband_arr, fn_band_arr, options_foldername, SNR_SAT)
plot_psf_systematics(subband_data, figure_savepath)