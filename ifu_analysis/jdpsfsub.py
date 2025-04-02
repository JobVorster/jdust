#PSF Modelling and Subtraction

#PSF functions were inspired by scripts from @author: Łukasz Tychoniec tychoniec@strw.leidenuniv.nl

from astropy.modeling import models, fitting
from ifu_analysis.jdutils import is_nan_map
import numpy as np
import stpsf
import matplotlib.pyplot as plt
from photutils.centroids import centroid_2dg

def Gauss2D_fit(data):
	'''
	Fits a 2D gaussian to a 2D map.

	Parameters
	----------

	data : 2D array
		A map.

	Returns
	-------

	w : astropy model
		The fitted model.

	model_data : 2D array
		The gaussian fit in the same shape as the input data.

	Notes
	-----

	Thanks to AGN Gazer on stackoverflow the outline of this function.
	'''

	inds_finite = np.isfinite(data)

	fit_w = fitting.LevMarLSQFitter()

	y0, x0 = np.unravel_index(np.nanargmax(data), data.shape)

	sigma = 5
	amp = np.nanmax(data)
	w = models.Gaussian2D(amp, x0, y0, sigma, sigma)
	yi, xi = np.indices(data.shape)
	g = fit_w(w, xi[inds_finite], yi[inds_finite], data[inds_finite])
	model_data = g(xi, yi)
	return w, model_data

def generate_single_miri_mrs_psf(subband,spectral_channel,filename=None,
	verbose=False,fov_pixels = 101,x_offset_arcsec = None, y_offset_arcsec = None,shp = None):
	'''
	Generates a SINGLE MIRI MRS PSF if the subband (e.g. 4A), and channel (zero indexed, e.g. 200) is specified.
	The implementation is not the most straightforward (one could use calc_psf straighaway instead of calc_datacube).
	If the filename is specified, it also saves the PSF HDU.

	Parameters
	----------

	subband : string
		MIRI MRS Subband (e.g. 3A, 2A)

	spectral_channel : integer
		Channel of the respective subband (e.g. 300).

	filename : string
		Filename to save to.

	verbose : boolean
		Whether to give verbose information. 

	fov_pixels : integer
		Size of PSF model.
	
	x_offset_arcsec : float
		Offset for the PSF model (in arcsec).

	y_offset_arcsec : float
		Offset for the PSF model (in arcsec).
	
	shp : 2x1 tuple
		Shape of a cropped PSF.

	Returns
	-------

	PSFCube_Init : HDU
		The DET_DIST subHDU is the PSF data.

	pix_scale : float
		Pixelscale for this subband and channel.
	'''
	miri = stpsf.MIRI()
	miri.mode = 'IFU'
	miri.band = subband
	waves = miri.get_IFU_wavelengths()
	if (subband == '3A') and (spectral_channel > 767):
		spectral_channel = 767
	waves = [waves[spectral_channel]]

	jitter_sigma = 0.007

	miri.options['jitter'] = 'gaussian'   # jitter model name or None
	miri.options['jitter_sigma'] = jitter_sigma # in arcsec per axis, default 0.007

	if x_offset_arcsec:
		miri.options['source_offset_x'] = x_offset_arcsec  # in units of arcseconds
	if y_offset_arcsec:
		miri.options['source_offset_y'] = y_offset_arcsec

	PSFCube_Init = miri.calc_datacube(waves,fov_pixels=fov_pixels)


	PSF_map = PSFCube_Init['DET_DIST'].data[0]
	if shp:
		PSF_map = PSF_map[0:shp[0],0:shp[1]]

	#From the documentation, pixelscale. 
	#'Ch1': (0.177, 0.196),
	#'Ch2': (0.280, 0.196),
	#'Ch3': (0.390, 0.245),
	#'Ch4': (0.656, 0.273),

	pix_scale = float(miri.pixelscale)
	if verbose:
		print("Band is", miri.band)
		print(f"Pixelscale for band {miri.band} is {miri.pixelscale} arcsec/pix")
		print ("Num of channels in the model", len(waves))

	#Free up memory, otherwise it crashes.
	del PSFCube_Init
	del miri

	return PSF_map, pix_scale

def get_offsets(channel_map,unc_map,subband,spectral_channel,SNR_percentile=98.5):
	'''
	Get the offsets between the peak of a channel map and a PSF map.

	Parameters
	----------

	channel_map : 2D array
		Channel map.

	unc_map : 2D array
		Uncertainties on the channel map.

	subband : string
		MIRI MRS Subband.

	spectral_channel : integer
		Spectral channel on which to do the fit.

	SNR_percentile : float
		Signal-to-noise percentile to calculate the fitting mask.

	Returns
	-------

	x_offset_arcsec : float
		Offset between the peak in the channel map and the psf in arcsec.

	y_offset_arcsec : float
		Offset between the peak in the channel map and the psf in arcsec.
	'''
	
	SNR_arr = np.array(channel_map/unc_map).flatten()
	SNR_threshold = np.nanpercentile(SNR_arr,[SNR_percentile])[0]
	mask = channel_map/unc_map <SNR_threshold
	x_sci, y_sci = centroid_2dg(channel_map,unc_map,mask)
	#psf_map, pix_scale = generate_single_miri_mrs_psf(subband,spectral_channel)
	pix_scale = get_pixel_scale(subband)
	x_psf, y_psf = [50,50] #This should be fov_pixels //2
	x_offset_arcsec,y_offset_arcsec = (x_sci-x_psf)*pix_scale, (y_sci-y_psf)*pix_scale

	return x_offset_arcsec, y_offset_arcsec

def subtract_psf_cube(data_cube,unc_cube,subband,SNR_percentile):
	'''
	Point Spread Function subtraction for an entire MIRI MRS cube.

	The results is very sensitive to the SNR_percentile across different subbands.

	Parameters
	----------

	data_cube : 3D array
		Intensity cube in MJy sr-1

	unc_cube : 3D array
		Uncertainties in the data_cube.

	subband : string
		MIRI MRS Subband, e.g. 3A, 4C.

	SNR_percentile : float
		Percentile of SNR map to consider for cutoff for the fit. 
		This should be as close to 100 as possible to exclude extended emission, but should allow enough pixels for a good fit.

	Returns
	-------

	psfsub_cube : 3D array
		PSF subtracted cube in MJy sr-1

	x_offset_arr : 1D array
		Offsets in arcsec used for the subtraction.

	t_offset_arr : 1D array
		Offsets in arcsec used for the subtraction.

	scaling_arr : 1D array
		Scaling factors of the PSF used for the subtraction.
	'''

	psfsub_cube = np.zeros(np.shape(data_cube))
	x_offset_arr = []
	y_offset_arr = []
	scaling_arr = []
	for channel, (chan_map, unc_map) in enumerate(zip(data_cube,unc_cube)):	
		print('Channel %d of %d'%(channel,len(data_cube)))
		if not is_nan_map(chan_map):
			x_offset_arcsec, y_offset_arcsec = get_offsets(chan_map,unc_map,subband,channel,SNR_percentile)
			psf_woffset, pix_scale = generate_single_miri_mrs_psf(subband,channel,
				x_offset_arcsec = x_offset_arcsec,y_offset_arcsec = y_offset_arcsec,shp=np.shape(chan_map))
			psf_woffset /= np.nanmax(psf_woffset)
			w, model_data = Gauss2D_fit(chan_map)
			scaling = w.amplitude
			psf_map = psf_woffset*scaling
			

			psfsub_cube[channel] = chan_map - psf_map
			if (0):	
				plt.close()
				plt.figure(figsize=(16,6))
				plt.subplot(131)
				plt.imshow(chan_map)
				plt.colorbar(location='top',fraction=0.046)
				plt.subplot(132)
				plt.imshow(psf_map)
				plt.colorbar(location='top',fraction=0.046)
				plt.subplot(133)
				plt.imshow(psfsub_cube[channel],cmap='coolwarm',vmin=-0.05*scaling,vmax=0.05*scaling)
				plt.colorbar(location='top',fraction=0.046)
				plt.show()

			x_offset_arr.append(x_offset_arcsec)
			y_offset_arr.append(y_offset_arcsec)
			scaling_arr.append(scaling.value)

	return psfsub_cube,x_offset_arr,y_offset_arr,scaling_arr

def get_pixel_scale(subband):
	'''
	Get the pixel scale for a MIRI MRS subband.

	Parameters
	----------

	subband : string
		MIRI MRS Subband, e.g. 3A, 4C.

	Returns
	-------

	pix_scale : float
		Pixel scale in arcsec.
	'''
	miri = stpsf.MIRI()
	miri.mode = 'IFU'
	miri.band = subband
	pix_scale = float(miri.pixelscale)
	return pix_scale