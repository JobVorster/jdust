# PSF Subtraction Systematics Figure
# Produces a three-row figure showing:
#   Row 1: I_PSF scaling vs wavelength (log scale)
#   Row 2: Fractional and absolute PSF scaling uncertainties (twin axes)
#   Row 3: PSF-subtracted spectrum with uncertainty envelope (placeholder)

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

options_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/PSFSub_Docs/L1448MM1/'
figure_savepath    = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/publication_figures/psf_systematics_%s.png' % source_name

# SNR threshold below which Gaussian fit is unreliable (saturated channels)
SNR_SAT = 2.5

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
# STEP 1: I/O
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

print(f'Loaded {len(subband_data)} subbands successfully')

############################################################
# FIGURE SETUP
############################################################

fig, axes = plt.subplots(3, 1, figsize=(12, 12))
fig.subplots_adjust(hspace=0.08)

ax_scl  = axes[0]
ax_unc  = axes[1]
ax_spec = axes[2]

ax_unc_abs = ax_unc.twinx()

############################################################
# STEP 2: ROW 1 — I_PSF
############################################################

for sd in subband_data:
	df   = sd['df']
	mask = df['mask'].astype(bool)

	ax_scl.plot(df['um'], df['fit_scaling'],
				color=C_DATA, alpha=0.2, lw=0.8)
	ax_scl.errorbar(df['um'][mask], df['fit_scaling'][mask],
					yerr=df['dfit_scaling'][mask],
					fmt='o', ms=2, color=C_DATA, alpha=0.6, zorder=2)
	ax_scl.plot(df['um'], df['smooth_scaling'],
				color=C_SMOOTH, lw=1.5, zorder=4)
	ax_scl.fill_between(df['um'],
						df['smooth_scaling'] - df['dtotal_scaling'],
						df['smooth_scaling'] + df['dtotal_scaling'],
						color=C_FILL, alpha=0.4,zorder=3)

ax_scl.set_yscale('log')
ax_scl.set_ylabel(r'$I_{\rm PSF}$ (MJy sr$^{-1}$)', fontsize=12)
ax_scl.tick_params(labelbottom=False)
ax_scl.set_xlim(4.7, 27.5)

legend_r1 = [
	Line2D([0], [0], color=C_DATA,   lw=1.0, alpha=0.6, label='data'),
	Line2D([0], [0], color=C_SMOOTH, lw=1.5,             label='smooth baseline'),
	plt.Rectangle((0,0), 1, 1, fc=C_FILL, alpha=0.2,    label=r'$\sigma_{\rm total}$'),
]
ax_scl.legend(handles=legend_r1, fontsize=10, loc='upper left')

############################################################
# STEP 3: ROW 2 — Uncertainties (fractional left, absolute right)
############################################################

for sd in subband_data:
	df = sd['df']

	# Fractional — left axis
	ax_unc.plot(df['um'], df['dfit_scaling_frac'],    color=C_FIT,     lw=1.0, alpha=0.8)
	ax_unc.plot(df['um'], df['dsmooth_scaling_frac'], color=C_SMOOTH2, lw=1.0, alpha=0.8)
	ax_unc.plot(df['um'], df['dtotal_scaling_frac'],  color=C_TOTAL,   lw=1.5, alpha=1.0, zorder=4)

	# Absolute — right axis
	ax_unc_abs.plot(df['um'], df['dfit_scaling'],    color=C_FIT,     lw=1.0, alpha=0.4, ls='--')
	ax_unc_abs.plot(df['um'], df['dsmooth_scaling'], color=C_SMOOTH2, lw=1.0, alpha=0.4, ls='--')
	ax_unc_abs.plot(df['um'], df['dtotal_scaling'],  color=C_TOTAL,   lw=1.5, alpha=0.6, ls='--', zorder=4)

ax_unc.set_yscale('log')
ax_unc_abs.set_yscale('log')
ax_unc.set_ylabel(r'$\sigma_{I_{\rm PSF}} / I_{\rm PSF}$', fontsize=12)
ax_unc_abs.set_ylabel(r'$\sigma_{I_{\rm PSF}}$ (MJy sr$^{-1}$)', fontsize=12)
ax_unc.set_xlabel(r'Wavelength ($\mu$m)', fontsize=12)
ax_unc.tick_params(labelbottom=True)
ax_unc.set_xlim(4.7, 27.5)

legend_r2 = [
	Line2D([0], [0], color=C_FIT,     lw=1.0, label=r'$\sigma_{\rm fit}$'),
	Line2D([0], [0], color=C_SMOOTH2, lw=1.0, label=r'$\sigma_{\rm smooth}$'),
	Line2D([0], [0], color=C_TOTAL,   lw=1.5, label=r'$\sigma_{\rm total}$'),
	Line2D([0], [0], color='grey',    lw=1.0, ls='-',  label='fractional (left)'),
	Line2D([0], [0], color='grey',    lw=1.0, ls='--', label='absolute (right)'),
]
ax_unc.legend(handles=legend_r2, fontsize=10, loc='upper right')

############################################################
# STEP 4: ROW 3 PLACEHOLDER
############################################################

ax_spec.text(0.5, 0.5,
			 'PSF subtraction uncertainty spectrum — pending perturbed wrapper runs',
			 ha='center', va='center', fontsize=12, color='grey',
			 transform=ax_spec.transAxes)
ax_spec.set_xlabel(r'Wavelength ($\mu$m)', fontsize=12)
ax_spec.set_ylabel(r'$F_\nu$ (MJy sr$^{-1}$)', fontsize=12)
ax_spec.set_xlim(4.7, 27.5)
ax_spec.tick_params(labelbottom=True)

############################################################
# STEP 5: SAVING
############################################################

plt.savefig(figure_savepath, bbox_inches='tight', dpi=150)
plt.close()
print(f'Saved figure to {figure_savepath}')