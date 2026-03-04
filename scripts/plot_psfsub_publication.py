# PSF Subtraction Systematics Figure
# Produces a three-row figure showing:
#   Row 1: I_PSF scaling vs wavelength (log scale)
#   Row 2: Fractional and absolute PSF scaling uncertainties (twin axes)
#   Row 3: Nominal PSF-subtracted spectrum with plus/minus perturbed spectra

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ifu_analysis.jdpsfsub import get_options_csv_fn, get_pixel_scale
from ifu_analysis.jdspextract import load_spectra, merge_subcubes

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
figure_savepath    = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/publication_figures/psf_systematics_%s.png' % source_name

# SNR threshold below which Gaussian fit is unreliable (saturated channels)
SNR_SAT = 2.5

# Paths to extracted spectra (nominal, plus, minus) — one aperture at a time
aperture_name      = 'C1'
spectra_nom_path   = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/L1448MM1_Extracted/PSFSUB/L1448MM1_aper%s_unstitched.spectra'%(aperture_name)
spectra_plus_path  = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/L1448MM1_Extracted/PSFSUB_PLUS/L1448MM1_aper%s_unstitched.spectra'%(aperture_name)
spectra_minus_path = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/L1448MM1_Extracted/PSFSUB_MINUS/L1448MM1_aper%s_unstitched.spectra'%(aperture_name)


# =============================================================================
# END USER PARAMETERS
# =============================================================================

# Colors
C_DATA    = 'steelblue'
C_SMOOTH  = 'tomato'
C_FILL    = 'tomato'
C_FIT     = '#4477AA'   # blue
C_SMOOTH2 = '#EE6677'   # red
C_TOTAL   = '#228833'   # green

############################################################
# STEP 1: I/O — PSF options CSVs
############################################################

subband_data = []

for subband, fn_band in zip(subband_arr, fn_band_arr):
	csv_fn = get_options_csv_fn(options_foldername, source_name, fn_band)

	try:
		df = pd.read_csv(csv_fn)
	except FileNotFoundError:
		print(f'Warning: {csv_fn} not found, skipping subband {subband}')
		continue

	pix_scale = get_pixel_scale(subband)

	# Mask saturated channels — Gaussian fit unreliable where SNR < SNR_SAT
	sat_mask = (df['fit_scaling'] / df['dfit_scaling']) < SNR_SAT
	df.loc[sat_mask, 'dfit_scaling'] = np.nan
	df.loc[sat_mask, 'fit_scaling']  = np.nan

	# Total scaling uncertainty (absolute)
	df['dtotal_scaling'] = np.sqrt(df['dfit_scaling']**2 + df['dsmooth_scaling']**2)

	# Fractional scaling uncertainties (for log scale)
	df['dfit_scaling_frac']    = df['dfit_scaling']    / df['smooth_scaling']
	df['dsmooth_scaling_frac'] = df['dsmooth_scaling'] / df['smooth_scaling']
	df['dtotal_scaling_frac']  = df['dtotal_scaling']  / df['smooth_scaling']

	subband_data.append({'subband': subband, 'df': df})

res_nom   = merge_subcubes(load_spectra(spectra_nom_path))
res_plus  = merge_subcubes(load_spectra(spectra_plus_path))
res_minus = merge_subcubes(load_spectra(spectra_minus_path))

um_nom    = res_nom['um']
flux_nom  = res_nom['flux']
unc_nom = res_nom['flux_unc']
flux_plus = res_plus['flux']
flux_minus= res_minus['flux']

print(f'Loaded {len(subband_data)} subbands successfully')

############################################################
# FIGURE SETUP
############################################################

fig, axes = plt.subplots(3, 1, figsize=(12, 12))
fig.subplots_adjust(hspace=0.08)

ax_scl  = axes[0]
ax_unc_abs  = axes[1]
ax_spec = axes[2]

############################################################
# STEP 2: ROW 1 — I_PSF
############################################################

for i,sd in enumerate(subband_data):
	df   = sd['df']
	mask = df['mask'].astype(bool)

	if i == 0:
		ax_scl.plot(df['um'], df['fit_scaling'],
					color=C_DATA, alpha=0.2, lw=0.8,label=r'$I_{\rm PSF}$')
		ax_scl.plot(df['um'], df['dtotal_scaling'],
					color='grey', alpha=1, lw=1,label=r'$\sigma_{\rm PSF}$')
		ax_scl.errorbar(df['um'][mask], df['fit_scaling'][mask],
						yerr=df['dfit_scaling'][mask],
						fmt='o', ms=2, color=C_DATA, alpha=0.6, zorder=2)
		ax_scl.plot(df['um'], df['smooth_scaling'],
					color=C_SMOOTH, lw=1.5, zorder=4,label=r'Smoothed $I_{\rm PSF}$')
	else:
		ax_scl.plot(df['um'], df['fit_scaling'],
					color=C_DATA, alpha=0.2, lw=0.8)
		ax_scl.plot(df['um'], df['dtotal_scaling'],
					color='grey', alpha=1, lw=1)
		ax_scl.errorbar(df['um'][mask], df['fit_scaling'][mask],
						yerr=df['dfit_scaling'][mask],
						fmt='o', ms=2, color=C_DATA, alpha=0.6, zorder=2)
		ax_scl.plot(df['um'], df['smooth_scaling'],
					color=C_SMOOTH, lw=1.5, zorder=4)

ax_scl.set_yscale('log')
ax_scl.set_ylabel(r'$I_{\rm PSF}$ (MJy sr$^{-1}$)', fontsize=12)
ax_scl.tick_params(labelbottom=False)
ax_scl.set_xlim(4.7, 27.5)
ax_scl.set_ylim(1e-1,1.2e5)

ax_scl.legend(fontsize=10, loc='upper left')

############################################################
# STEP 3: ROW 2 — Uncertainties (fractional left, absolute right)
############################################################

for sd in subband_data:
	df = sd['df']

	ax_unc_abs.plot(df['um'], df['dfit_scaling'],    color=C_FIT,     lw=1.0, alpha=1)
	ax_unc_abs.plot(df['um'], df['dsmooth_scaling'], color=C_SMOOTH2, lw=1.0, alpha=1)

ax_unc_abs.set_yscale('log')
ax_unc_abs.set_ylabel(r'$\sigma_{I_{\rm PSF}}$ (MJy sr$^{-1}$)', fontsize=12)
ax_unc_abs.tick_params(labelbottom=False)
ax_unc_abs.set_xlim(4.7, 27.5)
#ax_unc_abs.set_ylim(1e-1,1.2e5)

legend_r2 = [
	Line2D([0], [0], color=C_FIT,     lw=1.0, label=r'$\sigma_{\rm fit}$'),
	Line2D([0], [0], color=C_SMOOTH2, lw=1.0, label=r'$\sigma_{\rm smooth}$'),
]
ax_unc_abs.legend(handles=legend_r2, fontsize=10, loc='upper right')

############################################################
# STEP 4: ROW 3 — Nominal spectrum with plus/minus perturbed spectra
############################################################

ax_spec.plot(um_nom, flux_nom,
			 color=C_DATA, lw=1.0, alpha=0.9, zorder=4, label=aperture_name)
sigma_data = unc_nom
sigma_psf = np.abs(res_plus['flux'] - res_minus['flux'])

sigma_total = (sigma_data**2 + sigma_psf**2)**0.5


ax_spec.plot(um_nom, unc_nom,
			 color='grey', lw=0.8, alpha=0.6, zorder=3, label=r'$\sigma_{\rm pipe}$')
ax_spec.plot(um_nom, sigma_total,
			 color='black', lw=1, alpha=1, zorder=3, label=r'$\sigma_{\rm PSF}$')

ax_spec.fill_between(um_nom,unc_nom,sigma_total,color=C_FILL, alpha=0.4,label=r'$\sigma_{\rm PSF} > \sigma_{\rm pipe}$')

ax_spec.set_xlabel(r'Wavelength ($\mu$m)', fontsize=12)
ax_spec.set_ylabel(r'$F_\nu$ (Jy)', fontsize=12)
ax_spec.set_xlim(4.7, 27.5)
ax_spec.set_yscale('log')
ax_spec.legend(fontsize=10, loc='upper left')

############################################################
# STEP 5: SAVING
############################################################
plt.savefig(figure_savepath,bbox_inches='tight',dpi=150)
plt.show()
plt.close()


print(f'Saved figure to {figure_savepath}')