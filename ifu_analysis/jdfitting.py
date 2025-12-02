import astropy.units as u
from scipy.optimize import curve_fit
import astropy.constants as const
import numpy as np
from pybaselines import Baseline
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.stats import chi2
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

def parse_absorption(x, tau, interp_um = None,do_flip=False,wavenumber = False):
	'''
	Format absorption from wavenumber to wavelength (micron). Option to interpolate to a specified grid.

	Parameters
	----------
	wavenumber : array
		Wavenumber in cm-1

	tau : array
		Optical depth of absorption.
	
	interp_um : array (optional)
		Wavelength to interpolate to.

	Returns
	-------
	
	um : array
		Wavelength array of absorption, regridded if interp_um is specified.

	tau : array
		Reformatted absorption optical depth.
	'''
	if wavenumber:
		um_to_cm = 1e4
		um = um_to_cm/x
	else: 
		um = x
	if do_flip:
		tau = np.flip(tau,axis=0)
		um = np.flip(um,axis=0)
	if len(interp_um)> 0:
		tau = np.interp(interp_um,um,tau)
		um = interp_um
	return um, tau

def convert_flux_wl(wavelength, flux):
	'''
	Convert flux from units of Jy to W cm-2. Function by Katie Slavicinska.

	Parameters
	----------
	wavelength : array
		Wavelength in microns

	flux : array
		Flux density in Jy.
	
	Returns
	-------

	new_flux : array
		Flux density in W cm-2
	'''
	c = 299792458 #Speed of light, m s-1
	m_to_cm = 1e-2


	#flux is in Jy: 1e-26 W m-2 Hz-1
	wavelength = wavelength*1e-6 #Convert from um to m.

	freq = c/wavelength #Hz
	new_flux = freq * flux #1e-26 W m-2
	
	new_flux = new_flux * (1e-26 * m_to_cm**2) #W cm-2
	return new_flux

def calculate_goodness_of_fit(x, y, yerr, model, popt,pcov):
	'''
	Calculates the Baysian information criterion and Akaike information criterion for data with a model and a fit.
	
	This is useful for model selection. These criteria are most useful if the model and data are in units of order unity.

	Parameters:
	-----------

	x : array
		Independent variable of data.

	y : array
		Dependent variable of data.

	model : function
		Function that specifies a specific model.

	popt : array
		Best fit parameters for the specified model.

	Returns:
	--------

	AIC : integer
		Akaike information criterion
	
	BIC : integer
		Baysian information criterion

	'''


	#Residual
	residual_e = np.array(y - model(x,*popt))

	#Likelihood
	residual_SSE = np.nansum(residual_e**2)

	#Number of data points
	n = len(x)
	#Number of parameters.	
	k = len(popt)

	#Hope this is correct. I copied it from lecture notes. A reference would be helpful.
	AIC = n*np.log(residual_SSE/n) + 2*k
	BIC = n*np.log(residual_SSE/n) + np.log(n)*k

	#chi squared
	chi2 = np.nansum(residual_e**2/yerr**2)

	#reduced_chi squared
	degrees_of_freedom = n-k
	chi2_red = chi2/degrees_of_freedom

	p_value = 1 - chi2.cdf(chi2, degrees_of_freedom)


	#print('FIX: Chi2 in function calculate_goodness_of_fit was replaced by the chi2 p value, variable name is incorrect in most scripts!!')

	return AIC, BIC,chi2,chi2_red

def fit_model(um, flux, flux_unc,model,p0,fit_implementation = 'lmfit',no_negatives=True):
	'''
	Fit the spectra to a model. 

	This function requires some setup to use. The user has to specify the different model options, and add the relevant function.
	The user also has to specify a dictionary p0_dict with the initial guesses for all models. It is handy as it can be set at the top of the script.

	Parameters:
	-----------

	um : array
		Wavelengths in microns.

	flux : array
		Flux density in units the same as the model.

	flux : array
		Flux density uncertainty in units the same as the flux and model.

	fitoption : string
		Which model to fit to.

	p0_dict : dictionary
		Dictionary where p0_dict[fitotion] is an array with the inital guess for each parameter.

	Returns:
	--------

	popt : array
		Best fit for each parameter for the relevant model.

	pcov : 2D array
		Covariance for each parameter.
	'''
	if fit_implementation == 'curve_fit':
		relative_unc = flux_unc/flux #Relative uncertainties.
		popt,pcov = curve_fit(model, um, flux, p0 = p0, sigma = relative_unc)
	elif fit_implementation =='lmfit':

		def fmin(params, xdata,ydata,yunc,model):
			paramvals = []
			for p in params.keys():
				paramvals.append(params[p].value)
			return (ydata - model(xdata,*paramvals))/yunc
		params = Parameters()
		for i,p0val in enumerate(p0):
			if no_negatives:
				params.add('par' + str(i),value= p0val,min=0,max=5000)
			else:
				params.add('par' + str(i),value= p0val,max=5000)
		result = minimize(fmin,params,args=(um,flux,flux_unc,model))
		popt = []
		resultparams = result.params

		for p in resultparams.keys():
			popt.append(resultparams[p].value)
		pcov = result.covar

	return popt,pcov


def load_absorption(wav):
	'''
	Utility function to read all absorptions for the various models.
	Note the very shady use of hardcoded directories. It is because I am using it inside models that will be used inside curve_fit.
	I do not know how to give parameters to functions inside curve_fit that are not fit.
	'''
	absorption_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/absorption/'
	wavenumber_h2o15K, tau_h2o15K = np.transpose(np.genfromtxt(absorption_foldername+ '2023-08-18_hdo-h2o_1-200_thin(118)_154_blcorr.csv', delimiter=','))
	wavenumber_h2o150K, tau_h2o150K = np.transpose(np.genfromtxt(absorption_foldername +'2023-08-18_hdo-h2o_1-200_thin(27)_1500_blcorr.csv', delimiter=','))
	wavenumber_sil, tau_olivine, tau_pyroxene = np.transpose(np.genfromtxt(absorption_foldername + "adwin_silicates.csv", delimiter=",", skip_header=0))

	#Reformat absorption data.
	_, tau_h2o15K = parse_absorption(wavenumber_h2o15K, tau_h2o15K, interp_um = wav,do_flip=True,wavenumber=True)
	_, tau_h2o150K = parse_absorption(wavenumber_h2o150K, tau_h2o150K, interp_um = wav,do_flip=True,wavenumber=True)
	_, tau_olivine = parse_absorption(wavenumber_sil, tau_olivine, interp_um = wav)
	_, tau_pyroxene = parse_absorption(wavenumber_sil, tau_pyroxene, interp_um = wav)
	return tau_h2o15K, tau_h2o150K, tau_olivine, tau_pyroxene




	# blackbody function
def blackbody(wav, temp,scaling):
	'''
	Function by Katie Slavicinska.
	'''
	#FIX!! APRAD IS SET MANUALLY EQUAL TO THE APERTURE SIZE.
	ap_rad = 0.7

	h = 6.626e-34 # W s^2
	c = 2.998e8 # m s^-1
	k = 1.381e-23 # J K^-1
	wav = wav*1e-6 # um --> m
	freq = c * 1/wav # m --> 1/s
	radiance = 2*h*freq**3/(c**2) * (1/(np.exp(h*freq/(k*temp)))) # W m^2 sr-1 Hz-1
	radiance = radiance/1e4 # W cm^2 sr^-1 Hz^-1
	radiance = radiance*freq # W cm^2 sr^-1
	area = np.pi*ap_rad**2/4.25e10 # arcsec^2 --> sr
	radiance = radiance*area # W cm^2
	return radiance*scaling   

#A load of models with combinations of sillicates: pyroxene and olivine, and water: 15 K and 150 K.
def one_blackbody_pyroxene(wav,temp,scaling,pyroxene_scaling):
	tau_h2o15K, tau_h2o150K, tau_olivine, tau_pyroxene = load_absorption(wav)
	return blackbody(wav,temp,scaling)*np.exp(-pyroxene_scaling*tau_pyroxene)

def one_blackbody_olivine(wav,temp,scaling,olivine_scaling):
	tau_h2o15K, tau_h2o150K, tau_olivine, tau_pyroxene = load_absorption(wav)
	return blackbody(wav,temp,scaling)*np.exp(-olivine_scaling*tau_olivine)

def one_blackbody_pyroxene_olivine(wav,temp,scaling,pyroxene_scaling,olivine_scaling):
	tau_h2o15K, tau_h2o150K, tau_olivine, tau_pyroxene = load_absorption(wav)
	abs_tau = olivine_scaling*tau_olivine + pyroxene_scaling*tau_pyroxene
	return blackbody(wav,temp,scaling)*np.exp(-abs_tau)

def one_blackbody_pyroxene_olivine_15K(wav,temp,scaling,pyroxene_scaling,olivine_scaling,h2o15K_scaling):
	tau_h2o15K, tau_h2o150K, tau_olivine, tau_pyroxene = load_absorption(wav)
	abs_tau = olivine_scaling*tau_olivine + pyroxene_scaling*tau_pyroxene + h2o15K_scaling*tau_h2o15K
	return blackbody(wav,temp,scaling)*np.exp(-abs_tau)

def one_blackbody_pyroxene_olivine_150K(wav,temp,scaling,pyroxene_scaling,olivine_scaling,h2o150K_scaling):
	tau_h2o15K, tau_h2o150K, tau_olivine, tau_pyroxene = load_absorption(wav)
	abs_tau = olivine_scaling*tau_olivine + pyroxene_scaling*tau_pyroxene + h2o150K_scaling*tau_h2o150K
	return blackbody(wav,temp,scaling)*np.exp(-abs_tau)

def one_blackbody_pyroxene_olivine_15K_150K(wav,temp,scaling,pyroxene_scaling,olivine_scaling,h2o15K_scaling,h2o150K_scaling):
	tau_h2o15K, tau_h2o150K, tau_olivine, tau_pyroxene = load_absorption(wav)
	abs_tau = olivine_scaling*tau_olivine + pyroxene_scaling*tau_pyroxene + h2o15K_scaling*tau_h2o15K + h2o150K_scaling*tau_h2o150K
	return blackbody(wav,temp,scaling)*np.exp(-abs_tau)

def two_blackbodies(wav,temp1,scaling1,temp2,scaling2):
	return blackbody(wav,temp1,scaling1)+ blackbody(wav,temp2,scaling2)

def two_blackbodies_pyroxene(wav,temp1,scaling1,temp2,scaling2,pyroxene_scaling):
	tau_h2o15K, tau_h2o150K, tau_olivine, tau_pyroxene = load_absorption(wav)
	return (blackbody(wav,temp1,scaling1)+ blackbody(wav,temp2,scaling2))*np.exp(-pyroxene_scaling*tau_pyroxene)

def two_blackbodies_olivine(wav,temp1,scaling1,temp2,scaling2,olivine_scaling):
	tau_h2o15K, tau_h2o150K, tau_olivine, tau_pyroxene = load_absorption(wav)
	return (blackbody(wav,temp1,scaling1)+ blackbody(wav,temp2,scaling2))*np.exp(-olivine_scaling*tau_olivine)

def two_blackbodies_pyroxene_olivine(wav,temp1,scaling1,temp2,scaling2,pyroxene_scaling,olivine_scaling):
	tau_h2o15K, tau_h2o150K, tau_olivine, tau_pyroxene = load_absorption(wav)
	abs_tau = olivine_scaling*tau_olivine + pyroxene_scaling*tau_pyroxene
	return (blackbody(wav,temp1,scaling1)+ blackbody(wav,temp2,scaling2))*np.exp(-abs_tau)

def two_blackbodies_pyroxene_olivine_15K(wav,temp1,scaling1,temp2,scaling2,pyroxene_scaling,olivine_scaling,h2o15K_scaling):
	tau_h2o15K, tau_h2o150K, tau_olivine, tau_pyroxene = load_absorption(wav)
	abs_tau = olivine_scaling*tau_olivine + pyroxene_scaling*tau_pyroxene + h2o15K_scaling*tau_h2o15K
	return (blackbody(wav,temp1,scaling1)+ blackbody(wav,temp2,scaling2))*np.exp(-abs_tau)

def two_blackbodies_pyroxene_olivine_150K(wav,temp1,scaling1,temp2,scaling2,pyroxene_scaling,olivine_scaling,h2o150K_scaling):
	tau_h2o15K, tau_h2o150K, tau_olivine, tau_pyroxene = load_absorption(wav)
	abs_tau = olivine_scaling*tau_olivine + pyroxene_scaling*tau_pyroxene + h2o150K_scaling*tau_h2o150K
	return (blackbody(wav,temp1,scaling1)+ blackbody(wav,temp2,scaling2))*np.exp(-abs_tau)

def two_blackbodies_pyroxene_olivine_15K_150K(wav,temp1,scaling1,temp2,scaling2,pyroxene_scaling,olivine_scaling,h2o15K_scaling,h2o150K_scaling):
	tau_h2o15K, tau_h2o150K, tau_olivine, tau_pyroxene = load_absorption(wav)
	abs_tau = olivine_scaling*tau_olivine + pyroxene_scaling*tau_pyroxene + h2o15K_scaling*tau_h2o15K + h2o150K_scaling*tau_h2o150K  
	return (blackbody(wav,temp1,scaling1)+ blackbody(wav,temp2,scaling2))*np.exp(-abs_tau)


def snippity(um,flux, um_range):
	inds = np.where(np.logical_and(um>um_range[0],um<=um_range[1]))[0]
	
	um_cut = um[inds]
	flux_cut = flux[inds]
	return um_cut,flux_cut

def get_continuum_mask(um,flux,num_std = 6 ,lam = 1e4,scale=5,um_cut=27.5):
	if np.sum(np.isfinite(flux))>0:
		um_use = um
		flux_use = flux


		baseline_fitter = Baseline(um_use)
		baseline, params = baseline_fitter.fabc(flux_use, lam=lam,scale=scale,num_std=num_std)
		mask = params['mask']
		return mask
	else:
		return np.zeros(np.shape(um))+1



def log_likelihood(theta, x, y, yerr):
	#theta: T1, Scaling1, T2, Scaling2, sillicate scaling
	model = 'BB+Sillicate+Water'
	if model == 'BB+Sillicate':
		tbb1, sbb1, tbb2, sbb2, sillicate_scaling, log_f = theta
		model = two_blackbodies_sillicate(x,tbb1,sbb1,tbb2,sbb2,sillicate_scaling)
	elif model == 'BB+Sillicate+Water':
		tbb1, sbb1, tbb2, sbb2, sillicate_scaling,water_scaling, log_f = theta
		model = two_blackbodies_sillicate_water(x,tbb1,sbb1,tbb2,sbb2,sillicate_scaling,water_scaling)
	sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
	return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta):

	model = 'BB+Sillicate+Water'
	if model == 'BB+Sillicate':
		tbb1, sbb1, tbb2, sbb2, sillicate_scaling, log_f = theta

		if 300 < tbb1 < 1500 and 1e-9 < sbb1 < 1e-2 and 30 < tbb2 < 200 and 1e-9 < sbb1 < 1e-2 and 0 < sillicate_scaling < 1e2 and -100 < log_f < 1.0:
			return 0.0
	elif model == 'BB+Sillicate+Water':
		tbb1, sbb1, tbb2, sbb2, sillicate_scaling,water_scaling, log_f = theta
		verbose = False
		if verbose:
			print('tbb1: %.2E'%(tbb1))
			print('sbb1: %.2E'%(sbb1))
			print('tbb2: %.2E'%(tbb2))
			print('sbb2: %.2E'%(sbb2))
			print('sillicate_scaling: %.2E'%(sillicate_scaling))
			print('water_scaling: %.2E'%(water_scaling))
			print('log_f: %.2E'%(log_f))


		tbb1_bounds = [400,2500]
		sbb1_bounds = [0,1e-2]
		tbb2_bounds = [15,400]
		sbb2_bounds = [0,1e-2]
		sillicate_scaling_bounds = [0,1e2]
		water_scaling_bounds = [0,1e2]
		log_f_bounds = [-200,200]

		#One has to watch carefully that your bounds actually occur!
		#If not it can cause a problem where the MCMC is constant for all iterations.
		if tbb1_bounds[0] < tbb1 < tbb1_bounds[1] and sbb1_bounds[0] < sbb1 < sbb1_bounds[1] and tbb2_bounds[0] < tbb2 < tbb2_bounds[1] and 0 < sbb1 < 1e-2 and sbb2_bounds[0] < sillicate_scaling < sbb2_bounds[1] and water_scaling_bounds[0]<water_scaling <water_scaling_bounds[1]  and log_f_bounds[0] < log_f < log_f_bounds[1]:
			if verbose:
				print('Occured!')
				exit()
			return 0.0
	return -np.inf

def log_probability(theta, x, y, yerr):
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + log_likelihood(theta, x, y, yerr)







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