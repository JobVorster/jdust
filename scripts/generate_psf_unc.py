# Perturbed PSF Subtraction Wrapper
# Produces PSF-subtracted cubes perturbed by the combined fitting and smoothing uncertainties.
# Requires the _options.csv files produced by the original PSF subtraction wrapper.

# Imports
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from ifu_analysis.jdpsfsub import generate_single_miri_mrs_psf, get_aper_mask, get_options_csv_fn
from ifu_analysis.jdutils import unpack_hdu

# =============================================================================
# USER PARAMETERS — edit this block only
# =============================================================================

source_name       = 'L1448-mm'                  # Source name, used for filenames
subband_arr = ['1A','1B','1C','2A','2B','2C','3A','3B','3C','4A','4B','4C']

fn_band_arr = ['ch1-short','ch1-medium','ch1-long','ch2-short','ch2-medium','ch2-long','ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long']

input_foldername  = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/'
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/PSFSub_Docs/L1448MM1/'

aper_coords       = ['03h25m38.8898s', '+30d44m05.612s']  # RA, Dec of source
mask_psfsub       = True    # Mask the PSF region in the output cubes
bfe_factor        = 0.35    # Brighter-fatter effect factor, same as original wrapper
base_factor       = 1       # Base PSF size factor, same as original wrapper

# =============================================================================
# END USER PARAMETERS
# =============================================================================

fn_arr          = [input_foldername + '%s_%s_s3d_LSRcorr_stripecorr.fits' % (source_name, x) for x in fn_band_arr]
options_csv_arr = [get_options_csv_fn(output_foldername, source_name, x) for x in fn_band_arr]


for filename, options_csv, subband in zip(fn_arr, options_csv_arr, subband_arr):

	print('Doing perturbed PSF subtraction for subband %s' % subband)

	# Step 2: Read options CSV and compute perturbed arrays
	df = pd.read_csv(options_csv)

	um             = df['um'].values
	smooth_scaling = df['smooth_scaling'].values
	smooth_xoffset = df['smooth_xoffset'].values
	smooth_yoffset = df['smooth_yoffset'].values

	dfit_scaling    = df['dfit_scaling'].values
	dfit_xoffset    = df['dfit_xoffset'].values
	dfit_yoffset    = df['dfit_yoffset'].values

	dsmooth_scaling = df['dsmooth_scaling'].values
	dsmooth_xoffset = df['dsmooth_xoffset'].values
	dsmooth_yoffset = df['dsmooth_yoffset'].values

	# Combine uncertainties in quadrature
	dtotal_scaling = np.sqrt(dfit_scaling**2 + dsmooth_scaling**2)
	dtotal_xoffset = np.sqrt(dfit_xoffset**2 + dsmooth_xoffset**2)
	dtotal_yoffset = np.sqrt(dfit_yoffset**2 + dsmooth_yoffset**2)

	# Perturbed arrays
	scaling_plus  = smooth_scaling + dtotal_scaling
	scaling_minus = smooth_scaling - dtotal_scaling
	xoffset_plus  = smooth_xoffset + dtotal_xoffset
	xoffset_minus = smooth_xoffset - dtotal_xoffset
	yoffset_plus  = smooth_yoffset + dtotal_yoffset
	yoffset_minus = smooth_yoffset - dtotal_yoffset

	# Step 3: Loop and subtraction
	data_cube, unc_cube, dq_cube, hdr, um_cube, shp = unpack_hdu(filename)
	wcs = WCS(hdr).dropaxis(2)

	psfsub_plus  = data_cube.copy()
	psfsub_minus = data_cube.copy()

	for channel in range(len(um)):
		print('Perturbed PSF subtraction: %d of %d' % (channel, len(um)))

		chan_map = data_cube[channel]

		# Plus perturbation
		psf_plus, _ = generate_single_miri_mrs_psf(subband, channel,
			x_offset_arcsec=xoffset_plus[channel],
			y_offset_arcsec=yoffset_plus[channel],
			shp=np.shape(chan_map))
		psf_plus /= np.nanmax(psf_plus)
		psfsub_plus[channel] = chan_map - psf_plus * scaling_plus[channel]

		# Minus perturbation
		psf_minus, _ = generate_single_miri_mrs_psf(subband, channel,
			x_offset_arcsec=xoffset_minus[channel],
			y_offset_arcsec=yoffset_minus[channel],
			shp=np.shape(chan_map))
		psf_minus /= np.nanmax(psf_minus)
		psfsub_minus[channel] = chan_map - psf_minus * scaling_minus[channel]

		if mask_psfsub:
			aper_mask_bfe = get_aper_mask(um_cube[channel], aper_coords, base_factor + bfe_factor, wcs, shp)
			psfsub_plus[channel][aper_mask_bfe == 1]  = np.nan
			psfsub_minus[channel][aper_mask_bfe == 1] = np.nan

	# Step 4: Saving
	fn_base = output_foldername + source_name + '_%s_s3d_LSRcorr_stripecorr' % subband

	hdu_plus  = fits.open(filename).copy()
	hdu_minus = fits.open(filename).copy()

	hdu_plus[1].data  = psfsub_plus
	hdu_minus[1].data = psfsub_minus

	hdu_plus.writeto(fn_base  + '_psfsub_plus.fits',  overwrite=True)
	hdu_minus.writeto(fn_base + '_psfsub_minus.fits', overwrite=True)

	print('Saved perturbed PSF subtracted cubes for subband %s' % subband)