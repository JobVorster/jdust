import os
from spectres import spectres
import numpy as np 
import sys
import matplotlib.pyplot as plt 
import pandas as pd
import json
from pybaselines import Baseline
import emcee
from ifu_analysis.jdspextract import load_spectra,merge_subcubes,read_aperture_ini
from ifu_analysis.jdfitting import read_optool,fit_model,blackbody,prepare_spectra_for_fit,get_continuum_mask,results_to_terminal_str,grab_p0_bounds,calculate_goodness_of_fit
from multiprocessing import Pool
import corner

OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

############################################################
#														   #	
#	 MODEL SPECIFICATIONS								   #
#														   #
############################################################

from collections import OrderedDict

MODEL_SPECIFICATIONS = {
	'CM': {
		'parameters': OrderedDict([
			('T1', {
				'bounds': [0, 1000],
				'prior_type': 'uniform',
				'p0': 70,
				'description': 'Temperature component 1 (K)',
				'latex': ''
			}),
			('O1', {
				'bounds': [1e-10, 0.1],
				'prior_type': 'log_uniform',
				'p0': 1e-2,
				'description': 'BB scaling 1',
				'latex': ''
			}),
			('T2', {
				'bounds': [0, 1000],
				'prior_type': 'uniform',
				'p0': 400,
				'description': 'Temperature component 2 (K)',
				'latex': ''
			}),
			('O2', {
				'bounds': [1e-10, 0.1],
				'prior_type': 'log_uniform',
				'p0': 1e-7,
				'description': 'BB scaling 2',
				'latex': ''
			}),
			('SD_LOS', {
				'bounds': [0, 0.2],
				'prior_type': 'uniform',
				'p0': 1e-3,
				'description': 'Surface density along line of sight',
				'latex': ''
			}),
			('Jv_Scale', {
				'bounds': [1e-10, 0.1],
				'prior_type': 'log_uniform',
				'p0': 1e-9,
				'description': 'Scattering source scaling',
				'latex': ''
			})
		]),
		'model_function': None,  # Will be set after function definition
		'fixed_params': {'Jv_T': 5000.0},
		'description': '',
		'reference': '',
		'metadata': {}
	},
	'CM1BB': {
		'parameters': OrderedDict([
			('T1', {
				'bounds': [0, 1000],
				'prior_type': 'uniform',
				'p0': 70,
				'description': 'Temperature component 1 (K)',
				'latex': ''
			}),
			('O1', {
				'bounds': [1e-10, 0.1],
				'prior_type': 'log_uniform',
				'p0': 1e-2,
				'description': 'BB scaling 1',
				'latex': ''
			}),
			('SD_LOS', {
				'bounds': [0, 0.2],
				'prior_type': 'uniform',
				'p0': 1e-3,
				'description': 'Surface density along line of sight',
				'latex': ''
			}),
			('Jv_Scale', {
				'bounds': [1e-10, 0.1],
				'prior_type': 'log_uniform',
				'p0': 1e-9,
				'description': 'Scattering source scaling',
				'latex': ''
			})
		]),
		'model_function': None,  # Will be set after function definition
		'fixed_params': {'Jv_T': 5000.0},
		'description': '',
		'reference': '',
		'metadata': {}
	}
}

############################################################
#														   #	
#	 MODEL SPECIFICATION HELPER FUNCTIONS				   #
#														   #
############################################################

def get_model_config(model_name):
	"""Get configuration for a specific model."""
	if model_name not in MODEL_SPECIFICATIONS:
		raise ValueError(f"Model {model_name} not found. Available: {list(MODEL_SPECIFICATIONS.keys())}")
	return MODEL_SPECIFICATIONS[model_name]

def get_parameter_names(model_name, grain_sizes, grain_species, include_mfrac=True):
	"""Get ordered list of parameter names for a model."""
	config = get_model_config(model_name)
	param_names = list(config['parameters'].keys())
	
	if include_mfrac:
		for i, gsize in enumerate(grain_sizes):
			for j, gspecies in enumerate(grain_species):
				param_names.append(f'{gspecies}-{gsize}um')
	
	return param_names

def get_bounds(model_name):
	"""Get bounds dictionary for a model."""
	config = get_model_config(model_name)
	return {name: spec['bounds'] for name, spec in config['parameters'].items()}

def get_initial_guess(model_name):
	"""Get initial guess for a model's parameters."""
	config = get_model_config(model_name)
	return [spec['p0'] for spec in config['parameters'].values()]

def theta_to_dict(theta, model_name, Nsizes, Nspecies):
	"""
	Convert theta array to named dictionary.
	
	Parameters:
	-----------
	theta : array
		Parameter array [physical params..., mfrac_independent...]
	model_name : str
		Name of the model
	Nsizes : int
		Number of grain sizes
	Nspecies : int
		Number of grain species
	
	Returns:
	--------
	dict : Parameter dictionary with all parameters unpacked
	"""
	config = get_model_config(model_name)
	param_names = list(config['parameters'].keys())
	
	n_physical = len(param_names)
	
	# Physical parameters
	physical_dict = {}
	for i, name in enumerate(param_names):
		# Map generic names to actual function argument names
		if name == 'T1':
			physical_dict['temp'] = theta[i]
		elif name == 'O1':
			physical_dict['scaling'] = theta[i]
		elif name == 'T2':
			physical_dict['temp2'] = theta[i]
		elif name == 'O2':
			physical_dict['scaling2'] = theta[i]
		else:
			# For other parameters, use the name as-is
			physical_dict[name] = theta[i]
	
	# Add mass fractions
	mfrac_independent = theta[n_physical:]
	mfrac_dependent = 1.0 - np.sum(mfrac_independent)
	mfrac_flat = np.append(mfrac_independent, mfrac_dependent)
	physical_dict['mfrac_arr'] = mfrac_flat.reshape(Nsizes, Nspecies)
	
	# Add fixed parameters
	physical_dict.update(config['fixed_params'])
	
	return physical_dict

def get_default_mfrac_initial_guess(grain_sizes, grain_species):
	"""Get default initial guess for mass fractions (uniform distribution, normalized)."""
	mfrac_arr = np.zeros((len(grain_sizes), len(grain_species)))
	mfrac_arr += 1 / (len(grain_sizes) * len(grain_species))
	
	# Normalize and take first n-1
	mfrac_flat = mfrac_arr.flatten()
	mfrac_flat = mfrac_flat / np.sum(mfrac_flat)
	
	# Only return n-1 mass fractions
	return list(mfrac_flat[:-1])

############################################################
#														   #	
#	 BASIC FUNCTIONS									   #
#														   #
############################################################

def regrid_kappas(wave,kappa_arr,new_wave):
	Nsizes, Nspecies, _ = np.shape(kappa_arr)

	kregrid_arr = np.zeros((Nsizes,Nspecies,len(new_wave)))
	for i in range(Nsizes):
		for j in range(Nspecies):
			kregrid = spectres(new_wave,wave,kappa_arr[i][j])
			kregrid_arr[i][j] = kregrid
	return kregrid_arr

def cavity_model(wav,temp,scaling,temp2,scaling2,surface_density,Jv_T,Jv_Scale):
	#Model CM
	#Regrid to wav with spectres
	kabs = spectres(wav,wave,kappa_abs)
	ksca = spectres(wav,wave,kappa_scat)

	F_source = blackbody(wav,Jv_T,Jv_Scale)

	model = (source_function(wav,temp,scaling,kabs,ksca,F_source)+source_function(wav,temp2,scaling2,kabs,ksca,F_source)) * np.exp(-surface_density*(kabs+ksca))
	return model

def weighted_kappa(mfrac_arr,kappa_arr):
	kappa_nu = np.zeros(np.shape(kappa_arr))[0][0] #kappa as a function of wavelength.
	Nsizes, Nspecies = np.shape(mfrac_arr)
	for i in range(Nsizes):
		for j in range(Nspecies):
			kappa_nu += mfrac_arr[i][j]/np.sum(mfrac_arr.flatten())*kappa_arr[i][j]
	return kappa_nu

def cavity_model_mfrac(wav, kabs_arr, ksca_arr, temp, scaling, temp2, scaling2, 
                       surface_density, Jv_Scale, mfrac_arr, Jv_T=5000.0):
	"""
	Cavity model with mass fraction mixing.
	
	Parameters:
	-----------
	wav : array
		Wavelength array
	kabs_arr : array
		Absorption opacity array (Nsize x Nspecies x Nwave)
	ksca_arr : array
		Scattering opacity array (Nsize x Nspecies x Nwave)
	temp : float
		Temperature component 1
	scaling : float
		BB scaling 1
	temp2 : float
		Temperature component 2
	scaling2 : float
		BB scaling 2
	surface_density : float
		Surface density (SD_LOS)
	Jv_Scale : float
		Scattering source scaling
	mfrac_arr : array
		Mass fraction array (Nsize x Nspecies)
	Jv_T : float
		Scattering temperature (fixed at 5000 K)
	
	Returns:
	--------
	model : array
		Model flux
	"""
	#Model CM
	
	#Before optimisation, one has to regrid the kappas to the relevant grid.
	#kappa_arr: Nsize x Nspecies x Nwave array of opacities.
	#mfrac_arr: Nsize x Nspecies array of mass fractions.

	kabs = weighted_kappa(mfrac_arr, kabs_arr)
	ksca = weighted_kappa(mfrac_arr, ksca_arr)

	# Scattering temperature (fixed parameter)
	F_source = blackbody(wav, Jv_T, Jv_Scale)

	model = (source_function(wav, temp, scaling, kabs, ksca, F_source) + 
	         source_function(wav, temp2, scaling2, kabs, ksca, F_source)) *np.exp(-surface_density * (kabs + ksca))
	return model

def cavity_model_1BB_mfrac(wav, kabs_arr, ksca_arr, temp, scaling,surface_density, Jv_Scale, mfrac_arr, Jv_T=5000.0):
	"""
	Cavity model with mass fraction mixing.
	
	Parameters:
	-----------
	wav : array
		Wavelength array
	kabs_arr : array
		Absorption opacity array (Nsize x Nspecies x Nwave)
	ksca_arr : array
		Scattering opacity array (Nsize x Nspecies x Nwave)
	temp : float
		Temperature component 1
	scaling : float
		BB scaling 1
	surface_density : float
		Surface density (SD_LOS)
	Jv_Scale : float
		Scattering source scaling
	mfrac_arr : array
		Mass fraction array (Nsize x Nspecies)
	Jv_T : float
		Scattering temperature (fixed at 5000 K)
	
	Returns:
	--------
	model : array
		Model flux
	"""
	#Model CM1BB
	
	#Before optimisation, one has to regrid the kappas to the relevant grid.
	#kappa_arr: Nsize x Nspecies x Nwave array of opacities.
	#mfrac_arr: Nsize x Nspecies array of mass fractions.

	kabs = weighted_kappa(mfrac_arr, kabs_arr)
	ksca = weighted_kappa(mfrac_arr, ksca_arr)

	# Scattering temperature (fixed parameter)
	F_source = blackbody(wav, Jv_T, Jv_Scale)

	model = (source_function(wav, temp, scaling, kabs, ksca, F_source)) * np.exp(-surface_density * (kabs + ksca))
	return model

def source_function(wav,temp,scaling,kabs,ksca,F_source):
	return (kabs*blackbody(wav,temp,scaling) + ksca*F_source)/(kabs+ksca)

# Register model functions in MODEL_SPECIFICATIONS
MODEL_SPECIFICATIONS['CM']['model_function'] = cavity_model_mfrac
MODEL_SPECIFICATIONS['CM1BB']['model_function'] = cavity_model_1BB_mfrac

############################################################
#														   #	
#	 USER DEFINED PARAMETERS SECTION 1/2				   #
#														   #
############################################################

distance_dict = {}
distance_dict['L1448MM1'] = 293 #pc
distance_dict['BHR71'] = 176 #pc

COL_WIDTH = 20#for terminal printing

model_str = 'CM' #Options: CM, CM1BB 
source_name = 'L1448MM1' #Options: L1448MM1, BHR71
aperture_name = 'B1'
kp5_filename = "/home/vorsteja/Documents/JOYS/Collaborator_Scripts/Sam_Extinction/KP5_benchmark_RNAAS.csv"
opacity_foldername = '/home/vorsteja/Documents/JOYS/JDust/optool_opacities/'

grain_species = ['olmg50','pyrmg70'] #'qua','for','ens'
Nspecies = len(grain_species)
grain_sizes = [0.1,1.5,6]
Nsizes = len(grain_sizes)

#Wavelength to fit.
fit_wavelengths = [[4.7,5.67],[7.44,14.66],[16,27.5]]

#Wavelength ranges to calculate reduced chi2
chi_ranges = [[4.93,5.6],[7.7,8],[8.5,11.5],[12,14],[15.7,17],[19,20.5]]

############################################################
#														   #	
#	 MCMC PARAMETERS									   #
#														   #
############################################################

def load_initial_guess_from_csv(csv_foldername, source_name, aperture, n_samples=1000):
	"""
	Load initial guess from previous optimization results.
	Takes the average of the last n_samples entries.
	
	Parameters:
	-----------
	csv_foldername : str
		Path to folder containing previous fitting results
	source_name : str
		Source name (e.g., 'L1448MM1')
	aperture : str
		Aperture name (e.g., 'A1', 'A2', etc.)
	n_samples : int
		Number of last samples to average (default: 1000)
	
	Returns:
	--------
	p0_physical : list
		Initial guess for physical parameters [temp, scaling, temp2, scaling2, surface_density, Jv_Scale]
	p0_mfrac : list
		Initial guess for mass fractions (flattened, only n-1 independent fractions)
	"""
	csv_filename = os.path.join(csv_foldername, f'fitting_results_{source_name}_{aperture}.csv')
	
	if not os.path.isfile(csv_filename):
		print(f"Warning: Could not find {csv_filename}")
		print(f"Using default initial guess instead")
		return None, None
	
	try:
		# Load the CSV
		df = pd.read_csv(csv_filename)
		
		# Get last n_samples
		df_tail = df.tail(n_samples)
		
		# Extract physical parameters (note: old CSV has Jv_T, we need to skip it)
		# Expected columns: ID, chi2_red, temp, scaling, temp2, scaling2, surface_density, Jv_T, Jv_Scale, mfracs...
		
		if 'Jv_T' in df_tail.columns:
			# Old format with Jv_T - we skip it
			p0_physical = [
				df_tail['temp'].mean(),
				df_tail['scaling'].mean(),
				df_tail['temp2'].mean(),
				df_tail['scaling2'].mean(),
				df_tail['surface_density'].mean(),
				df_tail['Jv_Scale'].mean()
			]
		else:
			# New format without Jv_T
			p0_physical = [
				df_tail['temp'].mean(),
				df_tail['scaling'].mean(),
				df_tail['temp2'].mean(),
				df_tail['scaling2'].mean(),
				df_tail['surface_density'].mean(),
				df_tail['Jv_Scale'].mean()
			]
		
		# Extract mass fractions - load ALL, normalize, then keep only first n-1
		# Find columns that contain grain species and sizes
		mfrac_cols = [col for col in df_tail.columns if any(species in col for species in grain_species)]
		
		# Load all mass fractions
		p0_mfrac_all = []
		for col in mfrac_cols:
			p0_mfrac_all.append(df_tail[col].mean())
		
		# Normalize so they sum to 1
		p0_mfrac_all = np.array(p0_mfrac_all)
		p0_mfrac_all = p0_mfrac_all / np.sum(p0_mfrac_all)
		
		# Keep only first n-1 (last one will be computed as 1 - sum)
		p0_mfrac = list(p0_mfrac_all[:-1])
		
		print(f"Loaded initial guess from {csv_filename}")
		print(f"  Averaged last {n_samples} samples")
		print(f"  temp = {p0_physical[0]:.2f} K")
		print(f"  scaling = {p0_physical[1]:.2e}")
		print(f"  temp2 = {p0_physical[2]:.2f} K")
		print(f"  scaling2 = {p0_physical[3]:.2e}")
		print(f"  surface_density = {p0_physical[4]:.2e}")
		print(f"  Jv_Scale = {p0_physical[5]:.2e}")
		print(f"  Loaded and normalized {len(mfrac_cols)} mass fractions")
		print(f"  Using {len(p0_mfrac)} independent mass fractions for sampling")
		
		return p0_physical, p0_mfrac
		
	except Exception as e:
		print(f"Error loading initial guess from {csv_filename}: {e}")
		print(f"Using default initial guess instead")
		return None, None

############################################################
#														   #	
#	 MCMC PARAMETERS									   #
#														   #
############################################################

# MCMC settings
NWALKERS = 64  # Number of walkers (should be at least 2*ndim)
NSTEPS = 500000  # Number of steps per walker
NBURN = 100000  # Burn-in steps to discard
THIN = 20  # Thinning factor (keep every Nth sample)
USE_PARALLEL = True  # Use multiprocessing for speed
NCORES = 8  # Number of cores to use if USE_PARALLEL=True

# Initial guess settings
USE_PREVIOUS_RESULTS = True  # Load initial guesses from previous optimization
PREVIOUS_RESULTS_FOLDER = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/cavity_modelling/B2C/'
N_SAMPLES_TO_AVERAGE = 1000  # Number of last samples to average for initial guess

# Outlier removal settings
REMOVE_OUTLIER_CHAINS = True  # Remove chains with anomalously low log probabilities
OUTLIER_SIGMA = 3  # Number of MAD (median absolute deviations) for outlier threshold

############################################################
#														   #	
#	 USER DEFINED PARAMETERS SECTION 2/2				   #
#														   #
############################################################

# Note: Initial guesses and bounds are now defined in MODEL_SPECIFICATIONS
# above. This section is kept for backwards compatibility if needed.

############################################################
#														   #	
#	 INITIALIZE MODELS				 				   	   #
#														   #
############################################################

# Validate that the selected model exists
if model_str not in MODEL_SPECIFICATIONS:
	raise ValueError(f'Model {model_str} not found. Available models: {list(MODEL_SPECIFICATIONS.keys())}. Exiting...')

# Get model configuration
model_config = get_model_config(model_str)
if model_config['model_function'] is None:
	raise ValueError(f'Model function for {model_str} has not been implemented yet. Exiting...')

model = model_config['model_function']
model_parameters = list(model_config['parameters'].keys())

############################################################
#														   #	
#	 INITIALIZE FOLDERNAMES AND APERTURES 				   #
#														   #
############################################################

if source_name == 'L1448MM1':
	input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/extracted_paper/'
	output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/cavity_modelling/'	
	aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt'%(source_name)

	if os.path.isfile(aperture_filename):
		aper_names,aper_sizes,coord_list = read_aperture_ini(aperture_filename)

elif source_name == 'BHR71':
	output_foldername = '/home/vorsteja/Documents/JOYS/JDust/BHR71_scripts_12122025/BHR71/output/'
	fn_band_arr = ['ch1-short','ch1-medium','ch1-long','ch2-short','ch2-medium','ch2-long',
				'ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long'] 
	input_foldername = '/home/vorsteja/Documents/JOYS/JDust/BHR71_scripts_12122025/BHR71_unc/BHR71/spectra_werrors/circle/1.00_arcsec/spectra_sci/'
	aper_names = ['b1','o5','b2', 'b3', 'b4','cr1']

############################################################
#														   #	
#	 INITIALIZE OPACITIES				 				   #
#														   #
############################################################

header,wave,kappa_abs,kappa_scat,g = read_optool(opacity_foldername + '%s_%.1fum.dat'%(grain_species[0],grain_sizes[0]))

kabs_arr = np.zeros((len(grain_sizes),len(grain_species),len(wave)))
ksca_arr = np.zeros((len(grain_sizes),len(grain_species),len(wave)))
	
for i,gsize in enumerate(grain_sizes):
	for j,gspecies in enumerate(grain_species):
		op_fn = opacity_foldername+ '%s_%.1fum.dat'%(gspecies,gsize)
		header,wave,kappa_abs,kappa_scat,g = read_optool(op_fn)
		
		kabs_arr[i][j] = kappa_abs
		ksca_arr[i][j] = kappa_scat

############################################################
#														   #	
#	 EMCEE LOG PROBABILITY FUNCTIONS					   #
#														   #
############################################################

# Global variables that will be set for each aperture
# (needed because emcee passes only theta to log_probability)
_fit_um = None
_fit_flux = None
_fit_unc = None
_rkabs_arr = None
_rksca_arr = None
_bounds = None
_Nsizes = Nsizes
_Nspecies = Nspecies
_n_mfrac_params = Nsizes * Nspecies - 1  # Sample only n-1 mass fractions
_current_model = None  # Will be set to model_str for each aperture

def log_prior(theta):
	"""
	Generic log prior that reads from model specification.
	Automatically applies correct prior type (uniform or log-uniform) per parameter.
	
	theta = [physical_params..., mfrac_independent...]
	Note: mfrac_independent has only n-1 elements; last mass fraction = 1 - sum(mfrac_independent)
	
	Returns:
	--------
	log_prob : float
		Log prior probability. Returns -inf if parameters are out of bounds.
	"""
	theta = np.asarray(theta)  # Ensure theta is a numpy array
	
	# Get model configuration
	config = get_model_config(_current_model)
	param_specs = config['parameters']
	
	log_prob = 0.0
	
	# Check physical parameters
	for i, (name, spec) in enumerate(param_specs.items()):
		value = theta[i]
		bounds = spec['bounds']
		prior_type = spec['prior_type']
		
		# Check bounds
		if not (bounds[0] <= value <= bounds[1]):
			return -np.inf
		
		# Apply appropriate prior
		if prior_type == 'log_uniform':
			# Jeffreys prior: P(x) ∝ 1/x
			# log P(x) = -log(x) - log(log(max/min))
			# The normalization constant is the same for all parameters so we can ignore it in MCMC
			log_prob += -np.log(value)
		elif prior_type == 'uniform':
			# Uniform prior contributes constant (can ignore in MCMC)
			pass
		else:
			raise ValueError(f"Unknown prior type '{prior_type}' for parameter '{name}'")
	
	# Mass fraction checks
	n_physical = len(param_specs)
	mfrac_independent = theta[n_physical:]
	
	# Check that independent mass fractions are positive
	if np.any(mfrac_independent < 0):
		return -np.inf
	
	# Check that sum of independent fractions < 1 (since last = 1 - sum)
	if np.sum(mfrac_independent) >= 1.0:
		return -np.inf
	
	# The dependent fraction is automatically positive if sum < 1
	mfrac_dependent = 1.0 - np.sum(mfrac_independent)
	if mfrac_dependent < 0:  # Redundant but explicit check
		return -np.inf
	
	return log_prob

def log_likelihood(theta):
	"""
	Model-agnostic log likelihood = -0.5 * chi^2
	Works with any model specified in MODEL_SPECIFICATIONS.
	
	theta = [physical_params..., mfrac_independent...]
	Note: mfrac_independent has only n-1 elements; last mass fraction = 1 - sum(mfrac_independent)
	
	Returns:
	--------
	log_likelihood : float
		Log likelihood value. Returns -inf if model evaluation fails.
	"""
	theta = np.asarray(theta)  # Ensure theta is a numpy array
	
	# Convert theta to parameter dictionary
	params = theta_to_dict(theta, _current_model, _Nsizes, _Nspecies)
	
	# Get model function from configuration
	config = get_model_config(_current_model)
	model_func = config['model_function']
	
	if model_func is None:
		raise ValueError(f"Model function not set for {_current_model}")
	
	# Call model with unpacked parameters
	# Note: kabs_arr and ksca_arr are passed as positional args before keyword args
	try:
		model_eval = model_func(
			_fit_um,
			_rkabs_arr,
			_rksca_arr,
			**params  # Unpack all parameters as keyword arguments
		)
	except Exception as e:
		# If model calculation fails, return very low probability
		return -np.inf
	
	# Check for NaN or inf in model
	if not np.all(np.isfinite(model_eval)):
		return -np.inf
	
	# Calculate chi-squared
	residual = _fit_flux - model_eval
	chi2 = np.sum(residual**2 / _fit_unc**2)
	
	# Return log likelihood
	return -0.5 * chi2

def log_probability(theta):
	"""
	Log posterior probability = log prior + log likelihood
	"""
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	
	ll = log_likelihood(theta)
	if not np.isfinite(ll):
		return -np.inf
	
	return lp + ll

############################################################
#														   #	
#	 HELPER FUNCTIONS FOR MCMC							   #
#														   #
############################################################

def initialize_walkers(p0, nwalkers, ndim, bounds, model_name):
	"""
	Initialize walker positions around initial guess p0.
	Uses small perturbations that respect bounds.
	Note: p0 and walkers have only n-1 mass fractions (last one is computed as 1 - sum)
	
	Parameters:
	-----------
	p0 : array
		Initial guess
	nwalkers : int
		Number of walkers
	ndim : int
		Number of dimensions
	bounds : dict
		Bounds dictionary from model specification
	model_name : str
		Name of the model (to get parameter count)
	
	Returns:
	--------
	pos : array
		Initial positions for walkers (nwalkers x ndim)
	"""
	pos = []
	
	# Get number of physical parameters
	config = get_model_config(model_name)
	n_physical = len(config['parameters'])
	param_names = list(bounds.keys())
	
	# Scale of perturbations (relative to parameter values or bounds)
	for i in range(nwalkers):
		while True:
			# Perturb each parameter
			walker = np.zeros(ndim)
			
			# Physical parameters (indices 0 to n_physical-1)
			for j in range(n_physical):
				param_name = param_names[j]
				param_range = bounds[param_name][1] - bounds[param_name][0]
				perturbation = 0.1 * param_range * np.random.randn()
				walker[j] = p0[j] + perturbation
			
			# Independent mass fractions (indices n_physical to ndim-1, which is n-1 fractions)
			# Perturb n-1 fractions, ensure they stay positive and sum < 1
			mfrac_independent = np.array(p0[n_physical:])  # n-1 fractions
			mfrac_perturbed = mfrac_independent * (1 + 0.1 * np.random.randn(len(mfrac_independent)))
			mfrac_perturbed = np.maximum(0, mfrac_perturbed)  # Keep positive
			
			# Rescale if sum >= 1 (to ensure valid simplex)
			if np.sum(mfrac_perturbed) >= 1.0:
				mfrac_perturbed *= 0.99 / np.sum(mfrac_perturbed)
			
			walker[n_physical:] = mfrac_perturbed
			
			# Check if this walker is valid
			if np.isfinite(log_prior(walker)):
				pos.append(walker)
				break
	
	return np.array(pos)

def remove_outlier_chains(sampler, burnin, n_sigma=3):
	"""
	Remove chains with mean log probability far from median.
	Uses MAD (Median Absolute Deviation) for robust outlier detection.
	
	Parameters:
	-----------
	sampler : emcee.EnsembleSampler
		The MCMC sampler after running
	burnin : int
		Number of burn-in steps to discard
	n_sigma : float
		Number of MAD units for outlier threshold (default: 3)
	
	Returns:
	--------
	flat_samples : ndarray
		Flattened samples with outlier chains removed
	flat_log_prob : ndarray
		Flattened log probabilities with outlier chains removed
	n_removed : int
		Number of walkers removed
	"""
	# Get log probabilities (keep chain structure)
	log_prob = sampler.get_log_prob(discard=burnin, flat=False)
	
	# Calculate mean log prob for each walker
	mean_log_prob_per_walker = np.mean(log_prob, axis=0)
	
	# Robust outlier detection using MAD
	median = np.median(mean_log_prob_per_walker)
	mad = np.median(np.abs(mean_log_prob_per_walker - median))
	
	# MAD to standard deviation conversion factor (for normal distribution)
	mad_to_std = 1.4826
	threshold = median - n_sigma * mad_to_std * mad
	
	# Identify good walkers
	good_walkers = mean_log_prob_per_walker > threshold
	n_removed = np.sum(~good_walkers)
	
	print(f"\nOutlier chain removal:")
	print(f"  Median log prob: {median:.2f}")
	print(f"  MAD: {mad:.2f}")
	print(f"  Threshold ({n_sigma}σ): {threshold:.2f}")
	print(f"  Removing {n_removed}/{len(good_walkers)} outlier walkers")
	
	if n_removed > 0:
		print(f"  Outlier walker indices: {np.where(~good_walkers)[0]}")
		print(f"  Outlier mean log probs: {mean_log_prob_per_walker[~good_walkers]}")
	
	# Get samples from good walkers only
	chain = sampler.get_chain(discard=burnin, flat=False)
	log_prob_chain = sampler.get_log_prob(discard=burnin, flat=False)
	
	flat_samples = chain[:, good_walkers, :].reshape(-1, chain.shape[2])
	flat_log_prob = log_prob_chain[:, good_walkers].flatten()
	
	return flat_samples, flat_log_prob, n_removed

def check_burn_in_adequacy(sampler, requested_burnin, param_names):
	"""
	Check if burn-in was adequate based on autocorrelation time.
	
	Parameters:
	-----------
	sampler : emcee.EnsembleSampler
		The MCMC sampler after running
	requested_burnin : int
		The burn-in period that was used
	param_names : list
		Names of parameters
	
	Returns:
	--------
	recommended_burnin : int
		Recommended burn-in based on autocorrelation analysis
	is_adequate : bool
		Whether the requested burn-in was adequate
	"""
	print("\n" + "="*80)
	print("BURN-IN ADEQUACY CHECK")
	print("="*80)
	
	try:
		# Get autocorrelation times (tol=0 to force computation even if not converged)
		tau = sampler.get_autocorr_time(tol=0, quiet=True)
		
		# Recommended burn-in is ~2-3x the longest autocorrelation time
		max_tau = np.max(tau)
		recommended_burnin = int(3 * max_tau)
		
		print(f"Autocorrelation times (in steps):")
		for i, name in enumerate(param_names):
			marker = " ⚠" if tau[i] == max_tau else ""
			print(f"  {name:20s}: {tau[i]:8.1f}{marker}")
		
		print(f"\nMaximum autocorrelation time: {max_tau:.1f} steps")
		print(f"Recommended burn-in (3 × max τ): {recommended_burnin} steps")
		print(f"Actual burn-in used: {requested_burnin} steps")
		
		if recommended_burnin > requested_burnin:
			print(WARNING + f"\n⚠ WARNING: Recommended burn-in ({recommended_burnin}) > actual ({requested_burnin})" + ENDC)
			print(WARNING + "  Chains may not have fully converged. Consider:" + ENDC)
			print(WARNING + f"  1. Increasing burn-in to at least {recommended_burnin} steps" + ENDC)
			print(WARNING + "  2. Running longer chains" + ENDC)
			is_adequate = False
		else:
			print(OKGREEN + f"\n✓ Burn-in appears adequate ({requested_burnin} > {recommended_burnin})" + ENDC)
			is_adequate = True
			
	except Exception as e:
		print(f"Could not compute autocorrelation time: {e}")
		print("This often happens when chains are too short or haven't converged.")
		print("Consider running for more steps.")
		recommended_burnin = requested_burnin
		is_adequate = False
	
	print("="*80)
	
	return recommended_burnin, is_adequate

def calculate_acceptance_fraction(sampler):
	"""Calculate mean acceptance fraction across walkers."""
	return np.mean(sampler.acceptance_fraction)

def print_mcmc_summary(flat_samples, flat_log_prob, param_names_sampled, param_names_all, n_data, ndim, n_removed=0):
	"""
	Print summary statistics from MCMC run.
	
	Parameters:
	-----------
	flat_samples : ndarray
		Flattened MCMC samples (includes reconstructed dependent mass fraction)
	flat_log_prob : ndarray
		Flattened log probabilities
	param_names_sampled : list
		Parameter names that were sampled (n-1 mass fractions)
	param_names_all : list
		All parameter names (all n mass fractions)
	n_data : int
		Number of data points
	ndim : int
		Number of sampled parameters
	n_removed : int
		Number of outlier walkers removed
	"""
	print("\n" + "="*80)
	print("MCMC SUMMARY")
	print("="*80)
	print(f"Total samples (after burn-in, thinning, outlier removal): {len(flat_samples)}")
	if n_removed > 0:
		print(f"Outlier walkers removed: {n_removed}")
	
	# Calculate chi2_red statistics
	dof = n_data - ndim
	chi2_red = -2 * flat_log_prob / dof
	
	print(f"\nReduced χ² statistics:")
	print(f"  Median: {np.median(chi2_red):.3f}")
	print(f"  Mean: {np.mean(chi2_red):.3f}")
	print(f"  Std: {np.std(chi2_red):.3f}")
	
	print("\nParameter estimates (median ± 1σ):")
	print("-"*80)
	
	# Use all parameter names for display (includes reconstructed dependent fraction)
	for i, name in enumerate(param_names_all):
		mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
		q = np.diff(mcmc)
		marker = " (dependent)" if i == len(param_names_all) - 1 else ""
		print(f"{name:20s}: {mcmc[1]:12.6e} +{q[1]:12.6e} -{q[0]:12.6e}{marker}")
	
	print("="*80 + "\n")

############################################################
#														   #	
#	 MAIN FITTING LOOP									   #
#														   #
############################################################

# Distances (specified above).
if source_name not in distance_dict.keys():
	raise ValueError('Distance for this source not found. Please specify a distance. Exiting...')
	exit()
else:
	d = distance_dict[source_name]

# Column names for output - map generic names to output names for backwards compatibility
# Create mapping from generic parameter names to output names
param_name_mapping = {
	'T1': 'temp',
	'O1': 'scaling', 
	'T2': 'temp2',
	'O2': 'scaling2',
	'SD_LOS': 'surface_density',
	'Jv_Scale': 'Jv_Scale',
	'SD_WIND': 'SD_WIND',
	'FF': 'FF'
}

# Get parameter names from model specification
model_param_names = list(get_model_config(model_str)['parameters'].keys())
output_param_names = [param_name_mapping.get(name, name) for name in model_param_names]

# Add mass fraction names
for i, gsize in enumerate(grain_sizes):
	for j, gspecies in enumerate(grain_species):
		output_param_names.append(f'{gspecies}-{gsize}um')

# Column names for CSV output
columns = ['ID', 'chi2_red'] + output_param_names

# Parameter names for MCMC (using generic names from model specification)
param_names_all = get_parameter_names(model_str, grain_sizes, grain_species, include_mfrac=True)
# Only first (n_physical + Nsizes*Nspecies - 1) are sampled
param_names_sampled = param_names_all[:len(model_param_names) + Nsizes*Nspecies - 1]

for aperture in aper_names:
	if aperture != aperture_name:
		continue

	print("\n" + "="*80)
	print(f"PROCESSING: {source_name} - Aperture {aperture}")
	print("="*80)
	
	###############################
	# Load and prepare data
	###############################
	
	if source_name == 'BHR71':
		fn_band_arr = ['ch1-short','ch1-medium','ch1-long','ch2-short','ch2-medium','ch2-long',
					'ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long'] 
		fn_base = [input_foldername + 'BHR71-%s__circle_1.00_arcsec_%s_defringed.txt'%(aperture,x) for x in fn_band_arr]
		um_base_arr = []
		flux_base_arr = []
		unc_base_arr = []
		for fn in fn_base:
			um_base,flux_base,unc_base = np.loadtxt(fn,delimiter=' ').T
			um_base_arr += list(um_base)
			flux_base_arr += list(flux_base)
			unc_base_arr += list(unc_base)
		u_use = np.array(um_base_arr)
		f_use = np.array(flux_base_arr)
		unc_use = np.array(unc_base_arr)
	else:
		fn_base = input_foldername + '/L1448MM1_aper%s.spectra'%(aperture)
		sp_base = merge_subcubes(load_spectra(fn_base))
		u_use,f_use,unc_use = [sp_base[x] for x in ['um', 'flux', 'flux_unc']]
	
	if len(f_use[np.isfinite(f_use)])==0:
		print(f'No data in source {source_name}, aperture {aperture}. Skipping...')
		continue
	
	# Prepare spectrum for fit
	prepared_spectra = prepare_spectra_for_fit(u_use,f_use,unc_use,fit_wavelengths,um_cut=27.5)
	
	spectra_cols = ['um','flux','unc']
	fit_um,fit_flux,fit_unc = [prepared_spectra['fitdata:%s'%(x)] for x in spectra_cols]
	u_use,f_rad,unc_rad = [prepared_spectra['unmasked:%s'%(x)] for x in spectra_cols]
	
	# Regrid opacities to fit wavelengths
	rkabs_arr = regrid_kappas(wave,kabs_arr,fit_um)
	rksca_arr = regrid_kappas(wave,ksca_arr,fit_um)
	
	###############################
	# Set up MCMC
	###############################
	
	# Set global variables for log_probability function
	global _fit_um, _fit_flux, _fit_unc, _rkabs_arr, _rksca_arr, _current_model, _bounds
	
	_fit_um = fit_um
	_fit_flux = fit_flux
	_fit_unc = fit_unc
	_rkabs_arr = rkabs_arr
	_rksca_arr = rksca_arr
	_current_model = model_str  # Set current model for generic prior/likelihood
	
	# Get bounds from model specification
	bounds = get_bounds(model_str)
	_bounds = bounds
	
	# Initial guess - try to load from previous results first
	if USE_PREVIOUS_RESULTS:
		print(f"\nAttempting to load initial guess from previous results...")
		p0_physical_loaded, p0_mfrac_loaded = load_initial_guess_from_csv(
			PREVIOUS_RESULTS_FOLDER, 
			source_name, 
			'B1', 
			n_samples=N_SAMPLES_TO_AVERAGE
		)
		
		if p0_physical_loaded is not None and p0_mfrac_loaded is not None:
			# Successfully loaded
			p0_physical = p0_physical_loaded
			p0_mfrac = p0_mfrac_loaded
			print(f"✓ Using loaded initial guess")
		else:
			# Failed to load, use defaults from MODEL_SPECIFICATIONS
			print(f"✗ Using default initial guess from model specification")
			p0_physical = get_initial_guess(model_str)
			p0_mfrac = get_default_mfrac_initial_guess(grain_sizes, grain_species)
	else:
		# Use default initial guess from MODEL_SPECIFICATIONS
		print(f"\nUsing default initial guess from model specification")
		p0_physical = get_initial_guess(model_str)
		p0_mfrac = get_default_mfrac_initial_guess(grain_sizes, grain_species)
	
	# Combine physical parameters and mass fractions
	p0 = p0_physical + p0_mfrac
	
	# Number of dimensions (only n-1 mass fractions are sampled)
	ndim = len(p0)
	
	print(f"\nMCMC Setup:")
	print(f"  Number of parameters: {ndim} (sampling {Nsizes*Nspecies - 1} of {Nsizes*Nspecies} mass fractions)")
	print(f"  Number of walkers: {NWALKERS}")
	print(f"  Number of steps: {NSTEPS}")
	print(f"  Burn-in steps: {NBURN}")
	print(f"  Initial log probability: {log_probability(p0):.2f}")
	
	# Check if initial guess is valid
	if not np.isfinite(log_probability(p0)):
		print(WARNING + "WARNING: Initial guess has invalid log probability!" + ENDC)
		print(WARNING + "This usually means parameters are outside bounds." + ENDC)
		print("Parameter values:")
		for i, (name, val) in enumerate(zip(param_names_sampled, p0)):
			print(f"  {name}: {val:.6e}")
	
	# Initialize walkers
	print("\nInitializing walkers...")
	pos = initialize_walkers(p0, NWALKERS, ndim, bounds, model_str)
	
	# Check that all initial positions are valid
	for i, p in enumerate(pos):
		if not np.isfinite(log_probability(p)):
			print(f"Warning: Walker {i} has invalid initial position!")
	
	###############################
	# Run MCMC
	###############################
	
	if USE_PARALLEL:
		print(f"Running MCMC with {NCORES} cores...")
		with Pool(NCORES) as pool:
			# Try reducing 'a' from 2.0 to 1.5 or 1.25
			moves = emcee.moves.StretchMove(a=1.5)

			sampler = emcee.EnsembleSampler(NWALKERS, ndim, log_probability, pool=pool, moves=moves)
			
			# Run with progress bar
			sampler.run_mcmc(pos, NSTEPS, progress=True)
	else:
		print("Running MCMC (single core)...")
		sampler = emcee.EnsembleSampler(NWALKERS, ndim, log_probability)
		sampler.run_mcmc(pos, NSTEPS, progress=True)
	
	###############################
	# Analyze results
	###############################
	
	print("\nMCMC complete!")
	print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
	
	# Recommended: 0.2-0.4 for good mixing
	if np.mean(sampler.acceptance_fraction) < 0.15:
		print(WARNING + "Warning: Low acceptance fraction. Consider increasing step size or checking priors." + ENDC)
	elif np.mean(sampler.acceptance_fraction) > 0.6:
		print(WARNING + "Warning: High acceptance fraction. Consider decreasing step size." + ENDC)
	
	# Check burn-in adequacy
	recommended_burnin, is_adequate = check_burn_in_adequacy(sampler, NBURN, param_names_sampled)
	
	# Get samples (discard burn-in, optionally remove outliers)
	if REMOVE_OUTLIER_CHAINS:
		flat_samples_sampled, flat_log_prob, n_removed = remove_outlier_chains(sampler, NBURN, n_sigma=OUTLIER_SIGMA)
		# Also apply thinning manually if needed
		if THIN > 1:
			indices = np.arange(0, len(flat_samples_sampled), THIN)
			flat_samples_sampled = flat_samples_sampled[indices]
			flat_log_prob = flat_log_prob[indices]
	else:
		flat_samples_sampled = sampler.get_chain(discard=NBURN, thin=THIN, flat=True)
		flat_log_prob = sampler.get_log_prob(discard=NBURN, thin=THIN, flat=True)
		n_removed = 0
	
	# Reconstruct the dependent mass fraction for each sample
	mfrac_dependent = 1.0 - np.sum(flat_samples_sampled[:, 6:], axis=1, keepdims=True)
	flat_samples_full = np.hstack([flat_samples_sampled, mfrac_dependent])
	
	# Calculate chi2_red for each sample
	n_data = len(fit_flux)
	dof = n_data - ndim
	chi2_red = -2 * flat_log_prob / dof
	
	# Print summary (use full samples with reconstructed fraction)
	print_mcmc_summary(flat_samples_full, flat_log_prob, param_names_sampled, output_param_names, n_data, ndim, n_removed)
	
	###############################
	# Save results
	###############################
	
	print(f"\nSaving results for {source_name} aperture {aperture}...")
	
	# Create DataFrame with ALL mass fractions (including reconstructed dependent one)
	# Use output_param_names for column names (for backwards compatibility)
	df = pd.DataFrame(flat_samples_full, columns=output_param_names)
	df['chi2_red'] = chi2_red
	df['ID'] = np.arange(len(df))
	
	# Reorder columns to match original format
	df = df[columns]
	
	# Save to CSV
	output_filename = output_foldername + f'fitting_results_{source_name}_{aperture}.csv'
	df.to_csv(output_filename, index=False)
	print(f"Saved to: {output_filename}")
	
	# Save sampler object for later plotting
	sampler_filename = output_foldername + f'sampler_{source_name}_{aperture}.pkl'
	import pickle
	with open(sampler_filename, 'wb') as f:
		pickle.dump({
			'sampler': sampler,
			'param_names_sampled': param_names_sampled,
			'param_names_all': param_names_all,
			'source_name': source_name,
			'aperture': aperture,
			'burnin': NBURN,
			'thin': THIN,
			'ndim': ndim
		}, f)
	print(f"Saved sampler to: {sampler_filename}")
	
	print(OKGREEN + f"Completed {source_name} aperture {aperture}!" + ENDC)

print("\n" + "="*80)
print("ALL APERTURES COMPLETE!")
print("="*80)