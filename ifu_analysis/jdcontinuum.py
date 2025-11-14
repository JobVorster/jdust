from ifu_analysis.jdfitting import fit_linear_slope
from ifu_analysis.jdutils import unpack_hdu

import pandas as pd 
import numpy as np

from spectres import spectres
from tqdm import tqdm
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
from pybaselines import Baseline
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord

from ifu_analysis.jdutils import get_subcube_name,get_JWST_IFU_um

def multiply_along_axis(A, B, axis):
    return np.swapaxes(np.swapaxes(A, axis, -1) * B, -1, axis)



def read_cont_mask_arr(filename,sep=','):
	'''
	Reads the continuum masks from a file. The file should have two columns 'line-lower' and 'line-upper'.
	Returns a 2D array, for use in get_cont_cube.

	Parameters
	----------

	filename : string
		Filename of the file containing the line masks.

	sep : string
		Separator for pandas.read_csv.

	Returns
	-------

	mask_arr: N x 2 array
		List of [lower,upper] wavelengths to INCLUDE in the new cube.

	'''
	df = pd.read_csv(filename,sep=sep)
	lower = df['line-lower'].values
	upper = df['line-upper'].values
	mask_arr = []
	for l,u in zip(lower,upper):
		mask_arr.append([l,u])
	return mask_arr

def make_spectral_index_map(um,cube,unc_cube):
	'''
	IN DEVELOPMENT! Make a spectral index map.
	'''

	print('WARNING, this function is in development! It does not calculate the spectral index correctly.')
	shp = np.shape(cube[0,:,:])
	spectral_index = np.zeros(shp)
	spectral_index_err = np.zeros(shp)
	for idx in tqdm(range(shp[0])):
		for idy in range(shp[1]):
			flux_arr = cube[:,idx,idy]
			unc_arr = unc_cube[:,idx,idy]
			if sum(np.isfinite(flux_arr)) >0:
				a,da,b,db = fit_linear_slope(um,flux_arr,unc_arr)
				spectral_index[idx,idy] = a
				spectral_index_err[idx,idy] = da  
	return spectral_index,spectral_index_err


def save_cont_cube(cont_cube,unc_cube,dq_cube,original_hdu,filename):
	'''
	Saves a masked intensity, uncertainty and data quality cube to a fits file containting the relevant header information.

	Parameters
	----------

	cont_cube: 3D array
		Array containing the masked intensities.

	unc_cube: 3D array
		Array containing the masked uncertainties.

	dq_cube: 3D array
		Array containing the masked data quality entries.

	original_hdu: HDU
		Hdu of the cube that produced the relevant continuum cube. 

	filename: string
		Filename to save the new fits file to. 

	Notes
	-------

	1. cont_cube, unc_cube, dq_cube must have the same shape as the intensity cube in the original_hdu.

	'''
	if np.shape(cont_cube) != np.shape(unc_cube):
		raise ValueError('Shape of masked intensity cube and masked uncertainty cube not equal.')
	elif np.shape(cont_cube) != np.shape(dq_cube):
		raise ValueError('Shape of masked intensity cube and masked data quality cube not equal.')
	elif np.shape(cont_cube) != np.shape(original_hdu[1].data):
		raise ValueError('Shape of masked intensity cube and unmasked intensity cube not equal.')
	else:
		cont_hdu = original_hdu.copy()
		cont_hdu[1].data = cont_cube
		cont_hdu[2].data = unc_cube
		cont_hdu[3].data = dq_cube
		cont_hdu.writeto(filename)
		print('Continuum cube written successfully.')

def resample_cube(um,cube,cube_unc,resampled_um):
	'''
	Resamples a cube to a new velocity resolutions.

	Parameters
	----------

	um : 1D array
		Wavelengths of each channel in micron.

	cube : 3D array
		Array containing the intensities.

	cube_unc : 3D array
		Array containing the uncertainties on the intensities.

	resampled_um : 1D array
		New wavelength grid to use.

	Returns
	-------

	resampled_cube : 3D array
		Array containing the resampled intensities, the third axis has the length of resampled_um

	resampled_unc : 3D array
		Array containing the uncertainty of the resampled intensities, the third axis has the length of resampled_um
	'''
	shp_2D = np.shape(cube[0,:,:])
	resampled_cube = np.zeros((len(resampled_um),shp_2D[0],shp_2D[1]))
	resampled_unc = np.zeros((len(resampled_um),shp_2D[0],shp_2D[1]))

	for idx in tqdm(range(shp_2D[0])):
		for idy in range(shp_2D[1])	:
			spec_fluxes = cube[:,idx,idy]
			spec_errs = cube_unc[:,idx,idy]
			flux_res, unc_res = spectres(resampled_um, um, spec_fluxes, spec_errs=spec_errs, fill=None, verbose=True)
			resampled_cube[:,idx,idy] = flux_res
			resampled_unc[:,idx,idy] = unc_res
	return resampled_cube, resampled_unc

def radec_to_pixels(ra, dec, wcs):
    """
    Convert RA/Dec coordinates to pixel coordinates.
    
    Parameters:
    -----------
    ra : str or array-like of str
        Right Ascension in sexagesimal format (e.g., "12:34:56.7")
    dec : str or array-like of str
        Declination in sexagesimal format (e.g., "+12:34:56.7")
    wcs : astropy.wcs.WCS object
        World Coordinate System transformation
    
    Returns:
    --------
    x, y : float or array
        Pixel coordinates (0-indexed)
    """

    coords = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame='icrs')
    x, y = wcs.world_to_pixel(coords)
    
    return int(x), int(y)

def continuum_masking_figure(um,flux,flux_masked,baseline,subcube,plot_coords,saveto):
	plt.close()
	plt.figure(figsize = (2*3.46,2*2))
	print('plotting')
	plt.plot(um,flux,label='observation',color='grey',alpha=0.3)
	plt.plot(um,flux_masked,label='lines masked',color='red')
	plt.plot(um,baseline,label='baseline',color='blue')
	plt.show()
	plt.title('%s (%d,%d)'%(subcube,plot_coords[0],plot_coords[1]))
	plt.xlabel(r'Wavelength ($\mu$m)', fontsize = 10)
	plt.ylabel(r'Flux Density (MJy sr$^{-1}$)',fontsize = 10)
	plt.xticks(fontsize = 8)
	plt.yticks(fontsize = 8)
	plt.minorticks_on()
	plt.gca().tick_params(which='both',direction='in')
	plt.legend()
	plt.show()
	plt.savefig(saveto,dpi=200,bbox_inches='tight')


#Continuum estimation.
def automatic_continuum_cube(fn,num_std = 3 ,lam = 1e4,scale=5,saveto='',saveto_plots='',verbose=False,um_cut=27.5):
	data_cube,unc_cube,dq_cube,hdr,um,shp = unpack_hdu(fn)


	cube_shp = np.shape(data_cube)

	subcube = get_subcube_name(fn)

	cont_cube = np.full(cube_shp,np.nan)
	unc_cont_cube = np.full(cube_shp,np.nan)
	dq_cont_cube = np.full(cube_shp,np.nan)

	if (1):
		ra, dec, wcs = ['03h25m38.8898s','+30d44m05.612s', WCS(hdr)]
		ra, dec, wcs = ['03h25m38.7708s','+30d44m08.823s', WCS(hdr)]
		plot_coords = radec_to_pixels(ra, dec, wcs.dropaxis(2))


	for idx in tqdm(range(shp[0])):
		for idy in range(shp[1]):
			um = get_JWST_IFU_um(hdr)
			um_inds = np.where(np.logical_and(um < um_cut,np.isfinite(data_cube[:,idx,idy])))[0]
			flux = data_cube[um_inds,idx,idy].copy()
			flux_unc = unc_cube[um_inds,idx,idy].copy()
			dq = dq_cont_cube[um_inds,idx,idy].copy()

			if (idx==plot_coords[0]) & (idy == plot_coords[1]):

				baseline_fitter = Baseline(um[um_inds], check_finite=True)
				baseline, params = baseline_fitter.fabc(flux, lam=lam,scale=scale,num_std=num_std)

				plt.plot(um[um_inds],flux)
				plt.show()

			if len(um_inds)!=0:
				try:
					baseline_fitter = Baseline(um[um_inds], check_finite=True)
					baseline, params = baseline_fitter.fabc(flux, lam=lam,scale=scale,num_std=num_std)

					mask = params['mask']
					dum = abs(um[0]-um[1])

					flux[~mask] = np.nan
					flux_unc[~mask] = np.nan
					dq[~mask] = np.nan

					cont_cube[um_inds,idx,idy] = flux
					unc_cont_cube[um_inds,idx,idy] = flux_unc
					dq_cont_cube[um_inds,idx,idy] = dq

					if (1):
						

						if (idx==plot_coords[0]) & (idy == plot_coords[1]):
							
							saveto_fig = saveto.split('.fits')[0] + '_%s_cont%d_%d.png'%(subcube,plot_coords[0],plot_coords[1])
							print(saveto_fig)
							continuum_masking_figure(um,data_cube[:,idx,idy],flux,baseline,subcube,plot_coords,saveto_fig)


				except:
					if verbose:
						print('Continuum estimation did not work for pixel (idx,idy) = (%d,%d)'%(idx,idy))
					cont_cube[um_inds,idx,idy] = flux
					unc_cont_cube[um_inds,idx,idy] = flux_unc
					dq_cont_cube[um_inds,idx,idy] = dq
					if (0):
						if (saveto_plots != 0) & (idx %2 == 0) & (idy %2 == 0):
							plt.figure(figsize = (9,4))
							plt.plot(um,data_cube[um_inds,idx,idy],color='grey',alpha=0.2,label='data')
							plt.title('No masking for this pixel')
							plt.xlabel('Wavelength (um)')
							plt.ylabel('Flux Density (MJy sr-1)')
							plt.title('Pixel (x,y) = (%d,%d)'%(idy,idx))
							plt.legend()
							plt.savefig(saveto_plots + '%scontinuumx%dy%d.png'%(subcube,idy,idx),bbox_inches='tight',dpi=75)
							plt.close()



	if saveto:
		hdu = fits.open(fn)
		hdu['SCI'].data = cont_cube
		hdu['ERR'].data = unc_cont_cube
		hdu['DQ'].data = dq_cont_cube
		hdu.writeto(saveto,overwrite=True)
		print('Continuum cube saved!')
	return cont_cube,unc_cont_cube,dq_cont_cube

def cont_integrated_map(um,cont_cube,unc_cont_cube,wcs_2D,saveto=None,n_sigma=5):
	cont_cube[cont_cube < n_sigma*unc_cont_cube] = np.nan
	dum = np.array([um[1]-um[0]]*len(cont_cube))
	cont_map = np.nansum(multiply_along_axis(cont_cube,dum,0),axis = 0)
	if saveto:
		hdr = wcs_2D.to_header()
		hdr['UNIT'] = 'MJy sr-1 um'
		hdr['WAVE'] = (str(np.nanmean(um)), 'MEAN WAVELENGTH OF MAP')
		hdu = fits.PrimaryHDU(data = cont_map,header=hdr)
		hdu.writeto(saveto,overwrite=True)
	return cont_map


def get_cont_cube(data_cube,um,method=None,cont_filename=None,sep=','):
	'''
	Returns a copy of a cube, containing only the specified wavelength ranges. 

	Parameters
	----------
	
	data_cube: 3D array
		Array containing intensities/uncertainties/data quality.

	um: 1D array
		Wavelengths of each channel in micron.

	cont_filename : string
		Filename of the file containing the wavelengths to INCLUDE.
		The file should have two columns 'line-lower' and 'line-upper'. 

	sep : string
		sep : string
		Separator for pandas.read_csv.

	Returns
	-------

	cont_cube: 3D array with same shape as input array.
		Cube with only the data of the specified wavelength range. 

	'''
	if method == 'CHANNELS':
		if not cont_filename:
			raise ValueError('No filename specified for the wavelength ranges to use as continuum.')
		mask_arr = read_cont_mask_arr(cont_filename,sep=',')

		cont_cube = np.zeros(np.shape(data_cube))
		cont_cube[:,:,:] = np.nan
		for um_shaded in mask_arr:
			if (um_shaded[0] >= min(um)) and (um_shaded[1] < max(um)):
				inds = np.where(np.logical_and(um >= um_shaded[0], um< um_shaded[1]))
				cont_cube[inds] = data_cube[inds]
			#else:
			#    print('Range %s, outside of cube wavelength range'%(str(um_shaded)))
		return cont_cube
	elif method == 'AUTOMATIC':
		raise ValueError('Method %s not yet developed.'%(method))
	else:
		raise ValueError('Please specify a valid method.')


