import astropy.units as u
from scipy.optimize import curve_fit
import astropy.constants as const
import numpy as np



def linear(x,a,b):
	'''
	Linear function.
	
	Parameters
	----------

	x : 1D array
		x-values

	a : float
		slope

	b : float
		intercept

	Returns
	-------

	linear: 1D array
		Linear function.
	'''
	return a*x+b

def solid_angle_small_angle(radius_arcsec):
	'''
	Calculate the solid angle of a circular aperture using small angle approximation.
	Valid when radius is much smaller than 1 radian (~57 degrees).
	
	Parameters
	----------
	radius_arcsec : float
		Radius of the circular aperture in arcseconds
		
	Returns
	-------
	solid_angle : float
		Solid angle in steradians
	'''
	# Convert from arcseconds to radians
	radius_rad = (radius_arcsec * u.arcsec).to(u.rad).value
	
	solid_angle = np.pi * radius_rad**2
	
	return solid_angle  # in steradians

def modified_black_body(um,scaling,T,beta,solid_angle=None,lambda0 = 850):
	'''
	Modified blackbody (MBB) for wavelengths. This function returns in units of Jy if solid_angle is specified or Jy/sr if it is not.

	Parameters
	----------

	um : array
		Wavelengths in microns.

	scaling : float
		Scaling factor (unitless)

	T : float
		Temperature in Kelvin.

	beta : float
		Spectral index of modified black body.

	solid_angle : float (Optional)
		Solid angle of source, result will be returned in Jy if solid_angle is specified.

	lambda0 : float
		Reference wavelength for MBB in micron.

	Returns
	-------

	mbb : array
		Modified blackbody in either Jy or Jy/sr.

	'''
	nu = const.c.value/(um*1e-6)
	nu0 = const.c.value/(lambda0*1e-6)
	mbb = scaling*blackbody_lambda(um, T, solid_angle)*(nu/nu0)**(beta)
	return mbb


def blackbody_lambda(um,T,solid_angle):

	#um in micron.
	#T in Kelvin
	T = T*u.K
	um = um*1e-6*u.m #to meter.
	c_light = const.c #m s-1
	h = const.h #J s
	k_B = const.k_B #J K-1
	kT = k_B*T
	photon_energy = h*c_light/um

	B_lambda = 2*photon_energy*c_light*um**-4/(np.exp(photon_energy/kT)-1)
	B_nu = B_lambda * um**2 / c_light

	#W·m^-2·Hz^-1
	jy_factor = 1e26
	if solid_angle is not None:
		# Multiply by solid angle to get flux density in W·m^-2·Hz^-1
		# Then convert to Jy
		result = B_nu * solid_angle * jy_factor  # Jy
	else:
		# Convert spectral radiance to Jy/sr
		result = B_nu * jy_factor  # Jy/sr
	return result


def fit_linear_slope(um,flux_arr,unc_arr):
	'''
	Fits the linear slope, taking uncertainties into account.
	
	Parameters
	----------

	um : 1D array
		Wavelength array

	flux_arr : 1D array
		Array of fluxes/intensities.

	unc_arr : 1D array
		Uncertainties on the fluxes/intensities.

	Returns
	-------

	a : float
		Slope estimate.

	da : float
		Uncertainty on the slope estimate.

	b : float
		Intercept estimate.
	
	db : float 
		Uncertainty on the intercept estimate.
	'''
	#Flag nans in the array
	valid_inds = np.isfinite(um) & np.isfinite(flux_arr)

	popt,pcov = curve_fit(linear,um[valid_inds],flux_arr[valid_inds],sigma=unc_arr[valid_inds],absolute_sigma=True)
	a = popt[0]
	da = np.sqrt(np.diag(pcov))[0]
	b = popt[1]
	db = np.sqrt(np.diag(pcov))[1]
	return a,da,b,db