#PSF Modelling and Subtraction


#PSF functions were inspired by scripts from @author: Łukasz Tychoniec tychoniec@strw.leidenuniv.nl
#

from astropy.modeling import models, fitting
from ifu_analysis.jdutils import is_nan_map,get_JWST_PSF,define_circular_aperture,unpack_hdu
import numpy as np
import stpsf
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils.centroids import centroid_2dg
import pandas as pd
from matplotlib.colors import LogNorm
from photutils.background import MedianBackground,Background2D
from pybaselines import Baseline
from scipy.interpolate import CubicSpline

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

def get_offsets(channel_map,unc_map,subband,spectral_channel,mask_method,mask_par):
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

	mask_method : string
		Method to use to mask channel maps before fitting the centroid.
		Two method are currently supported:

			SNR
			---

			This method takes a SNR percentage, and cuts all pixels below that percentage. It is easy to use, but can be unreliable if there is bright extended emission.

			APERTURE
			--------

			This method places an aperture of 2*FWHM at the coordinates specified, and uses that as a mask.
	
	mask_par : float for mask_method=='SNR' and CircularAperture for mask_method=='APERTURE'.
		Object used to mask the channel map before centroiding.

	Returns
	-------

	x_offset_arcsec : float
		Offset between the peak in the channel map and the psf in arcsec.

	y_offset_arcsec : float
		Offset between the peak in the channel map and the psf in arcsec.
	'''
	
	if mask_method == 'SNR':
		SNR_percentile = mask_par
		SNR_arr = np.array(channel_map/unc_map).flatten()
		SNR_threshold = np.nanpercentile(SNR_arr,[SNR_percentile])[0]
		mask = channel_map/unc_map <SNR_threshold

	elif mask_method == 'APERTURE':
		shp = np.shape(channel_map)
		aper = mask_par

		#This line creates a 2D boolean mask from the aperture.
		#Method='center' makes sure the mask is just 0s and 1s, with no float values in between.
		#To_image lets the mask have the same shape as the relevant image.
		#dtype=int turns the 2D array of floats to an integer array, which acts as a boolean array.
		#The last np.mod(Array+1,2) is to invert the array (1-->0, 0-->1).
		mask = np.mod(np.array(aper.to_mask(method='center').to_image(shp),dtype=int)+1,2)
	
	else:
		raise ValueError('Please specify a masking method for the centroiding. Centroid calculations on the whole map will be inaccurate.')

	x_sci, y_sci = centroid_2dg(channel_map,unc_map,mask)
	#psf_map, pix_scale = generate_single_miri_mrs_psf(subband,spectral_channel)
	pix_scale = get_pixel_scale(subband)
	x_psf, y_psf = [50,50] #This should be fov_pixels //2
	x_offset_arcsec,y_offset_arcsec = (x_sci-x_psf)*pix_scale, (y_sci-y_psf)*pix_scale

	return x_offset_arcsec, y_offset_arcsec


def get_cube_offsets_scaling(um,data_cube,unc_cube,subband,mask_method,base_factor,aper_coords,wcs,saveto_psf_gen):
	x_offset_arr = []
	y_offset_arr = []
	scaling_arr = []

	shp = np.shape(data_cube[0])
	for channel, (chan_map,unc_map) in enumerate(zip(data_cube,unc_cube)):
		print('Getting PSF Parameters: %d of %d'%(channel,len(data_cube)))
		if not is_nan_map(chan_map):
			if mask_method == 'APERTURE':
				if aper_coords == None:
					raise ValueError('Please specify an aperture coordinate for aperture masking.')
				elif wcs == None:
					raise ValueError('Please specify a wcs for aperture masking.')
				else:
					fwhm = base_factor*get_JWST_PSF(um[channel])
					RA, Dec = aper_coords
					aper = define_circular_aperture(RA,Dec,fwhm)
					mask_par = aper.to_pixel(wcs)
					aper_mask = np.mod(np.array(mask_par.to_mask(method='center').to_image(shp),dtype=int),2)
			else:
				print('No masking method other than APERTURE is supported.')
			

			x_offset_arcsec, y_offset_arcsec = get_offsets(chan_map,unc_map,subband,channel,mask_method,mask_par)
			w, model_data = Gauss2D_fit(chan_map)
			scaling = w.amplitude.value
			x_offset_arr.append(x_offset_arcsec)
			y_offset_arr.append(y_offset_arcsec)
			scaling_arr.append(scaling)
		else:
			x_offset_arr.append(np.nan)
			y_offset_arr.append(np.nan)
			scaling_arr.append(np.nan)

	x_offset_arr = np.array(x_offset_arr)
	y_offset_arr = np.array(y_offset_arr)
	scaling_arr = np.array(scaling_arr)


	#Line flagging.
	#Do continuum classification on scaling array.
	um_inds = np.where(um < 27)
	lam,scale,num_std = 1e4,5,3
	baseline_fitter = Baseline(um[um_inds], check_finite=True)
	scaling_baseline, scaling_params = baseline_fitter.fabc(scaling_arr[um_inds], lam=lam,scale=scale,num_std=num_std)
	mask = scaling_params['mask']

	mask_inds = np.where(mask == 1)
	#Interpolation on the scaling, x-offset, y-offset.
	baseline_fitter = Baseline(um[um_inds][mask_inds], check_finite=True)
	scaling_baseline, scaling_params = baseline_fitter.fabc(scaling_arr[um_inds][mask_inds], lam = lam,scale=scale,num_std=num_std)
	xoffset_baseline, xoffset_params = baseline_fitter.fabc(x_offset_arr[um_inds][mask_inds], lam = lam,scale=scale,num_std=num_std)
	yoffset_baseline, yoffset_params = baseline_fitter.fabc(y_offset_arr[um_inds][mask_inds], lam = lam,scale=scale,num_std=num_std)

	#Now the offset arrays become the baselines.
	#We have to interpolate for the masked wavelengths.
	cs_scaling = CubicSpline(um[um_inds][mask_inds],scaling_baseline,extrapolate=False)
	cs_xoffset = CubicSpline(um[um_inds][mask_inds],xoffset_baseline,extrapolate=False)
	cs_yoffset = CubicSpline(um[um_inds][mask_inds],yoffset_baseline,extrapolate=False)

	if (1):
		pix_scale = get_pixel_scale(subband)
		plt.figure(figsize = (16,9))
		plt.subplot(131)
		plt.plot(um,scaling_arr,alpha=0.3,label='data')
		plt.scatter(um[um_inds][mask],scaling_arr[um_inds][mask],label='lines masked')
		plt.plot(um[um_inds][mask],scaling_baseline,color='red',label='baseline')
		plt.plot(um,cs_scaling(um),color='green',label='interpolated baseline')
		plt.legend()
		plt.title('scaling')
		plt.subplot(132)
		plt.plot(um,x_offset_arr/pix_scale,alpha=0.3,label='data')
		plt.scatter(um[um_inds][mask],x_offset_arr[um_inds][mask]/pix_scale,label='lines masked')
		plt.plot(um[um_inds][mask],xoffset_baseline/pix_scale,color='red',label='baseline')
		plt.plot(um,cs_xoffset(um)/pix_scale,color='green',label='interpolated baseline')
		plt.title('x-offset')
		plt.legend()
		plt.subplot(133)
		plt.plot(um,y_offset_arr/pix_scale,alpha=0.3,label='data')
		plt.scatter(um[um_inds][mask],y_offset_arr[um_inds][mask]/pix_scale,label='lines masked')
		plt.plot(um[um_inds][mask],yoffset_baseline/pix_scale,color='red',label='baseline')
		plt.plot(um,cs_yoffset(um)/pix_scale,color='green',label='interpolated baseline')
		plt.title('y-offset')
		plt.legend()
		plt.savefig('./psf_parameters_%s.png'%(subband),bbox_inches='tight',dpi=150)
		plt.close()


	scaling_arr = cs_scaling(um)
	x_offset_arr = cs_xoffset(um)
	y_offset_arr = cs_yoffset(um)


	if saveto_psf_gen:
		df = pd.DataFrame(columns =['um','x_offset','y_offset','scaling','mask'])
		df['um'] = um
		df['x_offset'] = x_offset_arr
		df['y_offset'] = y_offset_arr
		df['scaling'] = scaling_arr
		df['mask'] = mask
		df.to_csv(saveto_psf_gen)


	return x_offset_arr,y_offset_arr,scaling_arr,mask



def generate_psf_cube(fn,subband,mask_method,mask_par,aper_coords = None,wcs = None,saveto_psf_gen=None,saveto_psf_cube=None,base_factor=1):
	#Unpack and set up variables.
	data_cube,unc_cube,dq_cube,hdr,um,shp = unpack_hdu(fn)
	psf_cube = np.zeros(np.shape(data_cube))
	

	fwhm_residual = []
	fwhm_bfe_residual = []
	bfe_residual = []
	
	x_offset_arr,y_offset_arr,scaling_arr,mask = get_cube_offsets_scaling(um,data_cube,unc_cube,subband,mask_method,base_factor,aper_coords,wcs,saveto_psf_gen)

	#Over each channel
	for channel, (chan_map, unc_map) in enumerate(zip(data_cube,unc_cube)):	
		print('Generating PSF cube: %d of %d'%(channel,len(data_cube)))

		x_offset_arcsec,y_offset_arcsec,scaling = x_offset_arr[channel],y_offset_arr[channel],scaling_arr[channel]
		
		psf_woffset, pix_scale = generate_single_miri_mrs_psf(subband,channel,
			x_offset_arcsec = x_offset_arcsec,y_offset_arcsec = y_offset_arcsec,shp=np.shape(chan_map))
		psf_woffset /= np.nanmax(psf_woffset)
		
		psf_map = psf_woffset*scaling
		psf_cube[channel] = psf_map

	if saveto_psf_cube:
		hdu = fits.open(fn)
		hdu[1].data = psf_cube
		hdu.writeto(saveto_psf_cube,overwrite=True)

	return psf_cube,x_offset_arr,y_offset_arr,scaling_arr,mask

def get_aper_mask(um,aper_coords,base_factor,wcs,shp):
	#Aperture size is defined by the law et al relation.
	fwhm = base_factor*get_JWST_PSF(um)
	RA, Dec = aper_coords
	aper = define_circular_aperture(RA,Dec,fwhm)
	mask_par = aper.to_pixel(wcs)
	aper_mask = np.mod(np.array(mask_par.to_mask(method='center').to_image(shp),dtype=int),2)
	return aper_mask


def subtract_psf_cube(fn,subband,mask_method,mask_par,psf_filename = None,aper_coords = None,wcs = None,saveto=None,mask_psfsub=False,bfe_factor=0,saveto_psf_gen=None,saveto_psf_cube=None,verbose=False):
	'''
	UPDATE DOCUMENTATION!
	Point Spread Function subtraction for an entire MIRI MRS cube.

	Parameters
	----------
	
	um : 1D array
		Wavelengths in micron.

	data_cube : 3D array
		Intensity cube in MJy sr-1

	unc_cube : 3D array
		Uncertainties in the data_cube.

	subband : string
		MIRI MRS Subband, e.g. 3A, 4C.

	mask_method : string
		Method to use to mask channel maps before fitting the centroid.
		Two method are currently supported:

			SNR
			---

			This method takes a SNR percentage, and cuts all pixels below that percentage. It is easy to use, but can be unreliable if there is bright extended emission.

			APERTURE
			--------

			(Much better) This method places an aperture of 2*FWHM at the coordinates specified, and uses that as a mask.
	
	mask_par : float for mask_method=='SNR' and CircularAperture for mask_method=='APERTURE'.
		Object used to mask the channel map before centroiding. 

		If aper_coords is specified, mask_par will be overwritten with a CircularAperture at the specified radius.

	aper_coords : list of string
		RA, Dec coordinates in format HHhMMmSS.Ss, DDdMMmSS.Ss for a circular aperture.

	wcs : WCS Object
		2D WCS of a channel map.

	saveto : string
		Filename to save the psf parameters.

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
	data_cube,unc_cube,dq_cube,hdr,um,shp = unpack_hdu(fn)
	psfsub_cube = np.zeros(np.shape(data_cube))
	x_offset_arr = []
	y_offset_arr = []
	scaling_arr = []

	fwhm_residual = []
	fwhm_std = []
	fwhm_bfe_residual = []
	residual_channels = []
	fwhm_bfe_std = []
	bfe_residual = []
	bfe_std = []
	shp = np.shape(psfsub_cube[0])

	base_factor = 1

	#If the psfcube already exists, no need to regenerate it.
	if not (os.path.exists(saveto_psf_gen) and os.path.exists(saveto_psf_cube)):
		psf_cube,x_offset_arr,y_offset_arr,scaling_arr,mask = generate_psf_cube(fn,subband,mask_method,mask_par,aper_coords,wcs,saveto_psf_gen,saveto_psf_cube,base_factor)
	else:
		x_offset_arr,y_offset_arr,scaling_arr,mask = get_cube_offsets_scaling(um,data_cube,unc_cube,subband,mask_method,base_factor,aper_coords,wcs,saveto_psf_gen)
		psf_cube = fits.open(saveto_psf_cube)[1].data

	for channel, (chan_map, unc_map) in enumerate(zip(data_cube,unc_cube)):	
		if verbose:
			print('Channel %d of %d'%(channel,len(data_cube)))
		psf_map = psf_cube[channel]
		psfsub_cube[channel] = chan_map - psf_map
		
		fwhm = (base_factor+bfe_factor)*get_JWST_PSF(um[channel])
		RA, Dec = aper_coords
		aper = define_circular_aperture(RA,Dec,fwhm)
		mask_par = aper.to_pixel(wcs)

		aper_mask = get_aper_mask(um[channel],aper_coords,base_factor,wcs,shp)

		aper_mask_bfe = np.mod(np.array(mask_par.to_mask(method='center').to_image(shp),dtype=int),2)

		bfe_mask = aper_mask_bfe - aper_mask

		#Only save the residuals for continuum channels. Line channels arbitrarily increase it, and modifies the statistics.

		if mask[channel] == True:
			residual_channels += [channel]*len(psfsub_cube[channel][aper_mask_bfe==1]/psf_map[aper_mask_bfe==1])
			#Residual with the fwhm only.	
			fwhm_residual += list(psfsub_cube[channel][aper_mask_bfe==1]/psf_map[aper_mask_bfe==1])

			#Residual with the fwhm and the bfe factor.
			fwhm_bfe_residual += list(psfsub_cube[channel][aper_mask_bfe==1]/psf_map[aper_mask_bfe==1])

			#Residual for the bfe area only.
			bfe_residual+= list(psfsub_cube[channel][aper_mask_bfe==1]/psf_map[aper_mask_bfe==1])
		
		if mask_psfsub:
			psfsub_cube[channel][aper_mask_bfe==1] = np.nan

		#This plots the channel maps, the psf and the psfsubtracted for debugging.
		if (0):	
			vmin = -0.05*scaling
			vmax = np.nanmax(psfsub_cube[channel])
			plt.close()
			plt.figure(figsize=(16,6))
			plt.subplot(131)
			plt.imshow(chan_map,vmin=vmin,vmax=vmax,cmap='gist_stern',origin='lower')
			plt.title('%s Channel Map'%(subband))
			plt.colorbar(location='bottom',fraction=0.046)
			plt.subplot(132)
			plt.imshow(psf_map,vmin=vmin,vmax=vmax,cmap='gist_stern',origin='lower')
			plt.title('%s PSF Model'%(subband))
			plt.colorbar(location='bottom',fraction=0.046)
			plt.subplot(133)
			plt.title('%s PSF Subtracted'%(subband))
			plt.imshow(psfsub_cube[channel],vmin=vmin,vmax=vmax,cmap='gist_stern',origin='lower')
			plt.colorbar(location='bottom',fraction=0.046)
			plt.show()


	plt.scatter(residual_channels,fwhm_residual,s=0.1)
	plt.show()


	#Turn this into a plot where you show the average residual per subchannel.
	#This does not need to be anything fancy. Take the median from these arrays, with median scatters.
	if (0):
		plt.close()
		plt.figure(figsize = (10,12))
		plt.suptitle(subband)
		plt.subplot(311)
		x_test = range(len(fwhm_bfe_residual))
		plt.scatter(x_test,fwhm_bfe_residual,label='fwhm bfe residual',s=1)
		plt.errorbar(x_test,fwhm_bfe_residual,yerr=fwhm_bfe_std,linestyle='None',ecolor='black')

		plt.legend()

		plt.subplot(312)
		x_test = range(len(fwhm_residual))
		plt.scatter(x_test,fwhm_residual,label='fwhm residual',s=1)
		plt.errorbar(x_test,fwhm_residual,yerr=fwhm_std,linestyle='None',ecolor='black')

		plt.legend()

		plt.subplot(313)
		x_test = range(len(bfe_residual))
		plt.scatter(x_test,bfe_residual,label='bfe residual',s=1)
		plt.errorbar(x_test,bfe_residual,yerr=bfe_std,linestyle='None',ecolor='black')

		plt.legend()

		plt.show()

	if saveto:
		#This should be the psf_subtracted cube.
		hdu = fits.open(fn).copy()
		hdu[1].data = psfsub_cube
		hdu.writeto(saveto,overwrite=True)

		if (0):
			plt.figure(figsize = (9,4))
			plt.plot(um,fwhm_residual,label='%.2f fwhm'%(base_factor))
			plt.plot(um,fwhm_bfe_residual,label='%.2f fwhm'%(base_factor+bfe_factor))
			plt.plot(um,bfe_residual,label='bfe only')
			plt.legend()
			plt.show()

	return psfsub_cube,x_offset_arr,y_offset_arr,scaling_arr,fwhm_residual,fwhm_bfe_residual,bfe_residual

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

def stripe_correction(hdu,mask,saveto=None):
	'''
	Remove excess striping from IFU cubes due to cosmic ray showers. The JWST pipeline does remove showers, but not completely.

	This function follows the method developed by TEMPLATES (J. Spilker, K.A. Phadke, D. Law)
	https://github.com/JWST-Templates/Notebooks/blob/main/MIRI_MRS_reduction_SPT0418-47_PAH_ch3long.ipynb

	Updates to iterate over channels provided by Lukas Welzel

	Parameters
	----------

	hdu : astropy hdu
		hdu of JWST s3d cube

	mask : 2D array
		Mask over the source, to exclude when estimating the background.

	Return
	------

	cubeavg_wtd : 3D array
		Cube averaged over some channels.

	cubebkg_wtd : 3D array
		Cube of the background estimate, weighted by the uncertainty from the hdu.

	bkgsub_wtd : 3D array
		Background subtracted science cube.
	'''

	sci  = hdu['SCI'].data
	err  = hdu['ERR'].data
	dq   = hdu['DQ'].data
	scihead = hdu['SCI'].header

	# These will be useful
	wts  = err**-2   # w = 1/sigma^2, useful for weighted averages etc.
	scimasked = np.ma.masked_array(sci, mask = (dq>0)) # use DQ array to mask the science data
	# These define the pixel coordinates for the x and y center of the foreground lens,
	# plus a radius based on the Einstein ring size. I (JS) get consistent results whether
	# I get these using the cube's WCS information and ancillary ALMA data, or by collapsing
	# the whole cube over wavelength and visibly seeing the lens+ring.

	# find global peak in cube, assume that we only need to be roughly correct, 
	# and our source is the brightest part of the cube on average
	# use median collapse to decrease impact of outliers

	# Filter out slices along the first axis where all elements are NaN
	valid_slices = ~np.isnan(sci).all(axis=(1, 2))

	# Apply the filter to retain only valid slices
	filtered_sci = sci[valid_slices]

	# Check for columns where the pixel spectrum (across the first axis) is all NaN
	all_nan_columns = np.isnan(sci).all(axis=0)

	# Set these columns to zero in the filtered data
	filtered_sci[:, all_nan_columns] = 0

	# Compute the nanmedian across the first axis
	med_combined = np.nanmedian(filtered_sci, axis=0)

	ind = np.unravel_index(np.argmax(med_combined, axis=None), med_combined.shape)

	# Assign the peak coordinates and the rout value
	xpeak, ypeak = ind
	rout = 1

	#Use the mask specified by the user.
	sourcemask    = np.zeros(sci.shape, dtype=bool)
	if len(np.shape(mask)) == 2:
		sourcemask[:] = mask
		print('Used 2D mask for the 3D spectral cube.')
	elif mask.shape == sci.shape:
		sourcemask = mask
		print('Used 3D mask for the 3D spectral cube.')
	else:
		raise ValueError(f"Mask must be either 2D (y, x) or 3D (z, y, x). Got shape {mask.shape}.")
	
	# Create a copy of the original (DQ masked) data but now include our
	# additional mask from above
	scisourcemask = scimasked.copy()
	scisourcemask.mask += sourcemask

	# We will calculate the stripe template using a running average over the
	# cube. There's an option to use either a straight average or a weighted average,
	# so we'll just make both.
	# These will be the smoothed cubes before stripe removal
	cubeavg     = np.zeros(scisourcemask.shape)
	cubeavg_wtd = np.zeros(scisourcemask.shape)
	# These will contain the estimated stripe templates
	cubebkg     = np.zeros(scisourcemask.shape)
	cubebkg_wtd = np.zeros(scisourcemask.shape)

	# Setup for our background/stripe estimation. Stripes are coherent over tens
	# of wavelength channels (remember cube has been "3d drizzled" so oversamples
	# the detectors spectral resolution). After some experimentation I settled on
	# a 25-channel running average.

	chstep   = 25
	halfstep = int((chstep-1)/2)
	bkg_estimator = MedianBackground() # From photutils
	for chstart in np.arange(halfstep, sci.shape[0]):
		cutout = np.ma.median(scimasked[chstart-halfstep:chstart+chstep+halfstep],axis=0)
		cutout2= np.ma.average(scimasked[chstart-halfstep:chstart+chstep+halfstep],axis=0,weights=wts[chstart-halfstep:chstart+chstep+halfstep])

		# Use photutils to estimate the 2D "background" (striping). The box_size
		# parameter sets the shape of the stripe estimation. Here using (1,shape[1]/2)
		# corresponds to fitting the "background" in a shape that is 1 row tall and
		# half the x-pixels of the cube, which allows enough flexibility to account
		# for the fact that the slices aren't perfectly x-aligned due to the slice
		# curvature on the detector. Also exclude_percentile is very high here because
		# in some rows most of the cube pixels are masked, where our circular source
		# mask is at its widest point.
		bkg = Background2D(cutout,  box_size=(1,int(np.ceil(cutout.shape[1]/2))), mask=(cutout.mask | sourcemask[0]), 
			filter_size=1, bkg_estimator=bkg_estimator, exclude_percentile=75.0, sigma_clip=None)
		bkg2= Background2D(cutout2, box_size=(1,int(np.ceil(cutout.shape[1]/2))), mask=(cutout2.mask | sourcemask[0]), 
			filter_size=1, bkg_estimator=bkg_estimator, exclude_percentile=75.0, sigma_clip=None)

		#print (np.mean(bkg.background_rms))
		# Overwrite our blank cubes with data
		cubeavg[chstart]     = cutout
		cubeavg_wtd[chstart] = cutout2
		cubebkg[chstart]     = bkg.background
		cubebkg_wtd[chstart] = bkg2.background

		# The above fails near the edge channels at the start/end of the cube,
		# pad those slices with the same background for convenience (and remember
		# to ignore edge channels in later analysis)
		cubeavg[0:halfstep]     = cubeavg[halfstep]
		cubeavg_wtd[0:halfstep] = cubeavg_wtd[halfstep]
		cubebkg[0:halfstep]     = cubebkg[halfstep]
		cubebkg_wtd[0:halfstep] = cubebkg[halfstep]
		cubeavg[-halfstep:]     = cubeavg[-halfstep]
		cubeavg_wtd[-halfstep:] = cubeavg_wtd[-halfstep]
		cubebkg[-halfstep:]     = cubebkg[-halfstep]
		cubebkg_wtd[-halfstep:] = cubebkg[-halfstep]

	

	# Finally, also write out the background-subtracted cubes to disk
	bkgsub     = scimasked.data - cubebkg
	bkgsub_wtd = scimasked.data - cubebkg_wtd

	if saveto:
		bkgsubhdu = hdu.copy()
		bkgsubhdu[1].data = bkgsub
		bkgsubhdu.writeto(saveto[0],overwrite=True)

		cubebkghdu = hdu.copy()
		cubebkghdu[1].data = cubebkg
		cubebkghdu.writeto(saveto[1],overwrite=True)


	return cubeavg, cubebkg, bkgsub
