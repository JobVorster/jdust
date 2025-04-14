from ifu_analysis.jdutils import get_JWST_IFU_um, interpolate_nan


from astropy.io import fits
import numpy as np 
from pybaselines import Baseline
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u



def um_to_vlsr(um,lambda0):
	'''
	Convert wavelength to vlsr, with a specified centre wavelength.

	The formula and sign should still be DOUBLE CHECKED!
	
	Parameters
	----------

	um : 1D array
		Wavelength array in microns.

	lambda0 : float
		Centre reference wavelength.

	Returns
	-------

	vlsr : 1D array
		Velocities in units of km s-1
	'''
	c_light = const.c 
	vlsr = c_light*(1-lambda0/um)
	return vlsr.to(u.km*u.s**-1).value


def get_line_cube(fn,lambda0,N_chans):
	'''
	Retrieves a continuum subtracted line cube from a specified cube, at a specific wavelength. The 
	algorithm uses pybaselines pspline_arpls algorithm to estimate the continuum baseline, and then subtracts it.

	The function returns wavelengths, velocities, raw data, continuum estimate, line estimate, uncertainties and data quality cubes.

	Parameters
	----------

	fn : string
		Filename of the IFU cube.

	lambda0 : float
		Reference wavelength.

	N_chans : integer
		Number of channels to take around the reference wavelength.

	Returns
	-------

	um : 1D array
		Wavelengths around the line.

	vlsr : 1D array
		Velocities around the line.

	data_cube : 1D array
		Raw data around the line.

	cont_cube : 1D array
		Continuum estimate.

	line_cube : 1D array
		Continuum subtracted data.

	unc_cube : 1D array
		Uncertainties around the line (untouched by the algorithm at this stage).

	dq_cube : 1D array
		Data quality around the line (untouched by the algorithm at this stage).
	'''

	#For now, we let the user specify the filename, one can add a function later that automatically identifies the relevant channel.
	hdu = fits.open(fn)

	#Unpack the hdu
	hdr = hdu[1].header
	data_cube, unc_cube, dq_cube = [hdu[i].data for i in [1,2,3]]
	um = get_JWST_IFU_um(hdr)
	shp = np.shape(data_cube[0,:,:]) #2D shape
	

	ind0 = np.digitize([lambda0],um)[0]
	if (ind0 == 0) or (ind0 == len(um)):
		raise ValueError('Wavelength %.2f is outside of cube wavelength range (%.2f,%.2f) um.'%(lambda0,min(um),max(um)))
	else:
		#Now only taking N_chans around the centre.
		inds = np.arange(ind0-N_chans//2,ind0 + N_chans//2)
		um = um[inds]
		data_cube = data_cube[inds]
		unc_cube = unc_cube[inds]
		dq_cube = dq_cube[inds]

		vlsr = um_to_vlsr(um,lambda0)
		line_cube = np.full(np.shape(data_cube),np.nan)
		cont_cube = np.full(np.shape(data_cube),np.nan)
		#For each 2D pixel, get the baseline.
		for idx in range(shp[0]):
			for idy in range(shp[1]):

				flux = data_cube[:,idx,idy]
				if sum(np.isfinite(flux)) > 0:
					flux = interpolate_nan(flux)

					baseline_fitter = Baseline(um, check_finite=True)
					baseline,params = baseline_fitter.pspline_arpls(flux)
					#baseline, params = baseline_fitter.fabc(flux, lam=curvature,scale=scale,num_std=num_std)

					#mask = params['mask']

					line_flux = flux - baseline
					line_cube[:,idx,idy] = line_flux
					cont_cube[:,idx,idy] = baseline

		return um, vlsr, data_cube, cont_cube, line_cube, unc_cube, dq_cube









