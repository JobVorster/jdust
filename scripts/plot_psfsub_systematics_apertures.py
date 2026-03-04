# PSF Subtraction Systematics Figure 2
# One panel per aperture showing nominal spectrum with PSF uncertainty envelope
#
# Author: Job Vorster - jobvorster8@gmail.com
# Version: 2026-03-03

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ifu_analysis.jdspextract import load_spectra, merge_subcubes

# =============================================================================
# USER PARAMETERS
# =============================================================================

source_name   = 'L1448-mm'
READ_STITCHED = True

# List of apertures - one panel per aperture
aperture_list = ['A1', 'B0', 'B1', 'B2', 'C0', 'C1', 'C2']

# Nominal spectrum - stitched CSV or unstitched .spectra
if READ_STITCHED:
    spectra_nom_folder   = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/L1448MM1_STITCHED_PSFSUB/'
    spectra_nom_template = 'L1448MM1_aper{aper}_stitched.csv'
else:
    spectra_nom_folder   = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/L1448MM1_Extracted/PSFSUB/'
    spectra_nom_template = 'L1448MM1_aper{aper}_unstitched.spectra'

# Plus/minus - always unstitched .spectra
spectra_plus_folder    = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/L1448MM1_Extracted/PSFSUB_PLUS/'
spectra_minus_folder   = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/L1448MM1_Extracted/PSFSUB_MINUS/'
spectra_psf_template   = 'L1448MM1_aper{aper}_unstitched.spectra'

figure_savepath = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/publication_figures/psf_systematics_fig2_%s.png' % source_name

# =============================================================================
# END USER PARAMETERS
# =============================================================================

C_DATA = 'steelblue'


# =============================================================================
# STEP 1: Helper functions
# =============================================================================

def extract_stitched(filename):
    '''
    Load a stitched spectrum CSV and return wavelength, flux, and uncertainty.

    Parameters
    ----------
    filename : str
        Full path to the stitched CSV file (output of interactive_stitcher.py).
        Expected columns: um, flux_stitched, unc_stitched.

    Returns
    -------
    um : array
        Wavelength array (um).
    flux : array
        Stitched flux (Jy).
    flux_unc : array
        Stitched flux uncertainty (Jy).
    '''
    df = pd.read_csv(filename, comment='#')
    return df['um_stitched'].values, df['flux_stitched'].values, df['unc_stitched'].values


def load_aperture_spectra(aperture, spectra_nom_folder, spectra_nom_template,
                          spectra_plus_folder, spectra_minus_folder,
                          spectra_psf_template, read_stitched):
    '''
    Load nominal, plus, and minus spectra for a single aperture.

    Parameters
    ----------
    aperture : str
        Aperture name e.g. 'C1'.
    spectra_nom_folder : str
        Folder containing nominal spectra.
    spectra_nom_template : str
        Filename template for nominal spectrum with {aper} placeholder.
    spectra_plus_folder : str
        Folder containing plus-perturbed spectra.
    spectra_minus_folder : str
        Folder containing minus-perturbed spectra.
    spectra_psf_template : str
        Filename template for plus/minus spectra with {aper} placeholder.
        Always unstitched .spectra format.
    read_stitched : bool
        If True, load nominal from stitched CSV via extract_stitched.

    Returns
    -------
    um : array
    flux_nom : array
    unc_nom : array
    flux_plus : array
    flux_minus : array
    '''
    psf_fn    = spectra_psf_template.format(aper=aperture)
    res_plus  = merge_subcubes(load_spectra(spectra_plus_folder  + psf_fn))
    res_minus = merge_subcubes(load_spectra(spectra_minus_folder + psf_fn))

    nom_fn = spectra_nom_folder + spectra_nom_template.format(aper=aperture)
    if read_stitched:
        um, flux_nom, unc_nom = extract_stitched(nom_fn)
    else:
        res_nom = merge_subcubes(load_spectra(nom_fn))
        um, flux_nom, unc_nom = res_nom['um'], res_nom['flux'], res_nom['flux_unc']

    return um, flux_nom, unc_nom, res_plus['flux'], res_minus['flux']


# =============================================================================
# STEP 2: Plot
# =============================================================================

def plot_aperture_uncertainties(aperture_list, spectra_nom_folder, spectra_nom_template,
                                spectra_plus_folder, spectra_minus_folder,
                                spectra_psf_template, read_stitched, figure_savepath):
    '''
    Plot one panel per aperture showing nominal spectrum and PSF uncertainty.

    Detected channels (flux > unc_nom): flux in C_DATA, sigma_total in grey thin.
    Undetected channels (flux <= unc_nom): no flux plotted, sigma_total in black thick.

    Parameters
    ----------
    aperture_list : list of str
        Aperture names to plot.
    spectra_nom_folder : str
        Folder containing nominal spectra.
    spectra_nom_template : str
        Filename template for nominal spectrum.
    spectra_plus_folder : str
        Folder containing plus-perturbed spectra.
    spectra_minus_folder : str
        Folder containing minus-perturbed spectra.
    spectra_psf_template : str
        Filename template for plus/minus spectra.
    read_stitched : bool
        If True, load nominal from stitched CSV.
    figure_savepath : str
        Path to save the figure.
    '''
    n_aper = len(aperture_list)
    panel_height = 3.46 * 0.3
    fig, axes = plt.subplots(n_aper, 1, figsize=(2*3.46, panel_height * n_aper), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    FONTSIZE_LABEL  = 7
    FONTSIZE_LEGEND = 6
    FONTSIZE_TICK   = 6

    if n_aper == 1:
        axes = [axes]

    for ax, aperture in zip(axes, aperture_list):
        try:
            um, flux_nom, unc_nom, flux_plus, flux_minus = load_aperture_spectra(
                aperture, spectra_nom_folder, spectra_nom_template,
                spectra_plus_folder, spectra_minus_folder,
                spectra_psf_template, read_stitched
            )
        except Exception as e:
            print('Warning: could not load aperture %s - %s' % (aperture, e))
            ax.text(0.5, 0.5, 'Could not load %s' % aperture,
                    transform=ax.transAxes, ha='center', va='center', color='red',
                    fontsize=FONTSIZE_LABEL)
            ax.set_ylabel(r'$F_\nu$ (Jy)', fontsize=FONTSIZE_LABEL)
            continue

        sigma_psf   = np.abs(flux_plus - flux_minus) / 2.0
        sigma_total = np.sqrt(unc_nom**2 + sigma_psf**2)

        detected   = flux_nom > unc_nom
        undetected = ~detected

        # Flux - only where detected
        flux_plot = flux_nom.copy().astype(float)
        flux_plot[undetected] = np.nan

        # sigma_total - grey thin where detected, black thick where undetected
        sigma_det   = sigma_total.copy().astype(float)
        sigma_undet = sigma_total.copy().astype(float)
        sigma_det[undetected] = np.nan
        sigma_undet[detected] = np.nan

        ax.plot(um, flux_plot,
                color=C_DATA, lw=0.3, alpha=0.9, zorder=4, label=aperture)
        ax.plot(um, sigma_det,
                color='grey', lw=0.5, alpha=0.8, zorder=3, label=r'$\sigma_{\rm total}$')
        ax.plot(um, sigma_undet,
                color='black', lw=1.0, alpha=1.0, zorder=3)

        ax.set_ylabel(r'$F_\nu$ (Jy)', fontsize=FONTSIZE_LABEL)
        ax.set_yscale('log')
        ax.set_xlim(4.7, 27.5)
        ax.set_ylim(bottom=5e-6)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.legend(fontsize=FONTSIZE_LEGEND, loc='upper left', ncol=2, frameon=False)

    axes[-1].set_xlabel(r'Wavelength ($\mu$m)', fontsize=FONTSIZE_LABEL)
    axes[-1].tick_params(labelsize=FONTSIZE_TICK)

    plt.savefig(figure_savepath, bbox_inches='tight', dpi=350)
    plt.show()
    plt.close()
    print('Saved figure to %s' % figure_savepath)


# =============================================================================
# MAIN
# =============================================================================

plot_aperture_uncertainties(
    aperture_list,
    spectra_nom_folder,
    spectra_nom_template,
    spectra_plus_folder,
    spectra_minus_folder,
    spectra_psf_template,
    READ_STITCHED,
    figure_savepath,
)