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

def automatic_continuum_estimate(fn,lam =1e4,scale=5,num_std=3,um_cut = 27.5):
	'''
	IN DEVELOPMENT.
	'''

	data_cube,unc_cube,dq_cube,hdr,um,shp = unpack_hdu(fn)

	cont_cube = np.full(np.shape(data_cube),np.nan)
	cont_unc = np.full(np.shape(data_cube),np.nan)

	for idx in tqdm(range(shp[0])):
		for idy in range(shp[1]):
			flux = data_cube[:,idx,idy]
			flux_unc = unc_cube[:,idx,idy]
			inds = um < um_cut

			um[~inds] = np.nan
			flux[~inds] = np.nan
			flux_unc[~inds] = np.nan

			if len(um)!=0:
				try:
					baseline_fitter = Baseline(um, check_finite=True)
					baseline, params = baseline_fitter.fabc(flux, lam=lam,scale=scale,num_std=num_std)
					mask = params['mask']

					flux[~mask] = np.nan
					flux_unc[~mask] = np.nan

					cont_cube[:,idx,idy] = flux
					cont_unc[:,idx,idy] = flux_unc
				except:
					print('Nans detected in pixel (%d,%d), skipping...'%(idx,idy))
	return cont_cube,cont_unc




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


