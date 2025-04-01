import pandas as pd 
import numpy as np

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

def get_cont_cube(data_cube,um,cont_filename,sep=','):
	'''
	Returns a copy of a cube, containing only the specified wavelength ranges. 


	Parameters
	----------
	
	data_cube: 3D array
		Array containing intensities/uncertainties/data quality.

	um: 1D array
		Wavelengths of each channel in micron.

	filename : string
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

