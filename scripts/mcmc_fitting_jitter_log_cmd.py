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
import argparse

OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def regrid_kappas(wave,kappa_arr,new_wave):
	Nsizes, Nspecies, _ = np.shape(kappa_arr)

	kregrid_arr = np.zeros((Nsizes,Nspecies,len(new_wave)))
	for i in range(Nsizes):
		for j in range(Nspecies):
			kregrid = spectres(new_wave,wave,kappa_arr[i][j])
			kregrid_arr[i][j] = kregrid
	return kregrid_arr

def weighted_kappa(mfrac_arr,kappa_arr):
	kappa_nu = np.zeros(np.shape(kappa_arr))[0][0] #kappa as a function of wavelength.
	Nsizes, Nspecies = np.shape(mfrac_arr)
	for i in range(Nsizes):
		for j in range(Nspecies):
			kappa_nu += mfrac_arr[i][j]/np.sum(mfrac_arr.flatten())*kappa_arr[i][j]
	return kappa_nu

def blackbody_intensity(wav, temp):
	'''
	Function by Katie Slavicinska.
	'''
	h = 6.626e-34 # W s^2
	c = 2.998e8 # m s^-1
	k = 1.381e-23 # J K^-1
	wav = wav*1e-6 # um --> m
	freq = c * 1/wav # m --> 1/s
	radiance = 2*h*freq**3/(c**2) * (1/(np.exp(h*freq/(k*temp)))) # W m^2 sr-1 Hz-1
	radiance = radiance/1e4 # W cm^2 sr^-1 Hz^-1
	radiance = radiance*freq # W cm^2 sr^-1
	return radiance   


def cavity_model_mfrac(wav,temp,scaling,temp2,scaling2,surface_density,Jv_Scale,mfrac_arr,kabs_arr,ksca_arr):
	#Model CM
	
	#Before optimisation, one has to regrid the kappas to the relevant grid.
	#kappa_arr: Nsize x Nspecies x Nwave array of opacities.
	#mfrac_arr: Nsize x Nspecies array of mass fractions.

	apsize = 0.75
	area = apsize**2/4.25e10 # arcsec^2 --> sr

	scaling *= area
	scaling2 *= area
	Jv_Scale *= area


	kabs = weighted_kappa(mfrac_arr,kabs_arr)
	ksca = weighted_kappa(mfrac_arr,ksca_arr)

	# Fix scattering temperature to 5000 K (not a free parameter)
	Jv_T = 5000.0
	F_source = blackbody_intensity(wav,Jv_T)*Jv_Scale

	#model = (source_function(wav,temp,scaling,kabs,ksca,F_source)+source_function(wav,temp2,scaling2,kabs,ksca,F_source)) * np.exp(-surface_density*(kabs+ksca))
	model = scaling*source_function(wav,temp,kabs,ksca,F_source)* np.exp(-surface_density*(kabs+ksca)) + scaling2*source_function(wav,temp2,kabs,ksca,F_source)*(1-np.exp(-surface_density*(kabs+ksca)))
	return model

def source_function(wav,temp,kabs,ksca,F_source):
	return (kabs*blackbody_intensity(wav,temp) + ksca*F_source)/(kabs+ksca)


############################################################
#														   #	
#	 PARSE COMMAND LINE ARGUMENTS						   #
#														   #
############################################################

parser = argparse.ArgumentParser(description='Run MCMC dust composition fitting on MIRI MRS spectra')
parser.add_argument('source_name', type=str, help='Source name (e.g., BHR71, L1448MM1)')
parser.add_argument('aperture_name', type=str, help='Aperture name (e.g., b1, b2, A1, A2)')
parser.add_argument('--model', type=str, default='CM', choices=['CM', 'CMWIND', 'CM_ANI'], 
					help='Model type (default: CM)')

args = parser.parse_args()

############################################################
#														   #	
#	 USER DEFINED PARAMETERS							   #
#														   #
############################################################

distance_dict = {}
distance_dict['L1448MM1'] = 293 #pc
distance_dict['BHR71'] = 176 #pc

COL_WIDTH = 20#for terminal printing

model_str = args.model
source_name = args.source_name
aperture_name = args.aperture_name

print("\n" + "="*80)
print(f"MCMC FITTING: {source_name} - Aperture {aperture_name}")
print(f"Model: {model_str}")
print("="*80 + "\n")

kp5_filename = "/home/vorsteja/Documents/JOYS/Collaborator_Scripts/Sam_Extinction/KP5_benchmark_RNAAS.csv"
opacity_foldername = '/home/vorsteja/Documents/JOYS/JDust/optool_opacities/'

grain_species = ['olmg50','pyrmg70','for','ens']
Nspecies = len(grain_species)
grain_sizes = [0.1,1.5]
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

# MCMC settings
NWALKERS = 64
NSTEPS = 300000
NBURN = 50000
THIN = 40
USE_PARALLEL = True
NCORES = 8

# Diagnostics - print progress every N steps (no file saving)
PRINT_INTERVAL = 10000

# Outlier removal settings
REMOVE_OUTLIER_CHAINS = True
OUTLIER_SIGMA = 3

############################################################
#														   #	
#	 USER DEFINED PARAMETERS SECTION 2/2				   #
#														   #
############################################################

#Set initial guesses and bounds (now in log-space for scaling parameters)

p0_dict = {}

p0_T1 = 70
p0_log_O1 = np.log10(1e-2)  # log10 of scaling
p0_T2 = 400
p0_log_O2 = np.log10(1e-7)  # log10 of scaling2
p0_log_O_star = np.log10(1e-9)  # log10 of Jv_Scale
p0_log_SD_LOS = np.log10(1e-3)  # log10 of surface_density
p0_log_jitter = np.log10(1e-21 / 1e-20)  # log10 of (jitter in units of 1e-20)

p0_dict['ALL:ALL:CM'] = [p0_T1, p0_log_O1, p0_T2, p0_log_O2, p0_log_SD_LOS, p0_log_O_star, p0_log_jitter]

bounds_dict = {}

uT_emit = 1000
u_log_O_emit = np.log10(0.1)
l_log_O_emit = np.log10(1e-10)
u_log_SD = np.log10(0.2)
l_log_SD = np.log10(1e-10)
u_log_jitter = np.log10(1e-18 / 1e-20)  # Upper bound in units of 1e-20
l_log_jitter = np.log10(1e-24 / 1e-20)  # Lower bound in units of 1e-20

#For model CM (all in log-space now)
bounds_dict['ALL:ALL:CM:T1'] = [0, uT_emit]
bounds_dict['ALL:ALL:CM:log_O1'] = [l_log_O_emit, u_log_O_emit]
bounds_dict['ALL:ALL:CM:log_SD_LOS'] = [l_log_SD, u_log_SD]
bounds_dict['ALL:ALL:CM:log_sigma_jitter'] = [l_log_jitter, u_log_jitter]

#Bounds that are equal to other bounds.
bounds_dict['ALL:ALL:CM:log_O_star'] = bounds_dict['ALL:ALL:CM:log_O1']
bounds_dict['ALL:ALL:CM:T2'] = bounds_dict['ALL:ALL:CM:T1']
bounds_dict['ALL:ALL:CM:log_O2'] = bounds_dict['ALL:ALL:CM:log_O1']

############################################################
#														   #	
#	 INITIALIZE MODELS				 				   	   #
#														   #
############################################################

model_parameters = ['T1','log_O1','T2','log_O2','log_SD_LOS','log_O_star','log_sigma_jitter']
model_uncertainties = ['d%s'%(x) for x in model_parameters]

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
elif source_name =='BHR71':
	input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/extracted_paper_BHR71/'
	output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/cavity_modelling/'	
	aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt'%(source_name)

	if os.path.isfile(aperture_filename):
		aper_names,aper_sizes,coord_list = read_aperture_ini(aperture_filename)
else:
	print(f"ERROR: Unknown source name: {source_name}")
	print("Valid options: L1448MM1, BHR71")
	sys.exit(1)

# Check if requested aperture is valid
if aperture_name not in aper_names:
	print(f"ERROR: Aperture '{aperture_name}' not found for source '{source_name}'")
	print(f"Valid apertures: {aper_names}")
	sys.exit(1)

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

_fit_um = None
_fit_flux = None
_fit_unc = None
_rkabs_arr = None
_rksca_arr = None
_bounds = None
_Nsizes = Nsizes
_Nspecies = Nspecies
_n_mfrac_params = Nsizes * Nspecies - 1

def log_prior(theta):
	theta = np.asarray(theta)
	
	# Extract parameters (now in log-space for scalings)
	temp, log_scaling, temp2, log_scaling2, log_surface_density, log_Jv_Scale, log_sigma_jitter = theta[:7]
	mfrac_independent = theta[7:]
	
	log_prob = 0.0
	
	# Temperature bounds (linear space)
	if not (_bounds['T1'][0] <= temp <= _bounds['T1'][1]):
		return -np.inf
	if not (_bounds['T2'][0] <= temp2 <= _bounds['T2'][1]):
		return -np.inf
	
	# Log-space bounds (no additional Jacobian needed for log-uniform prior)
	if not (_bounds['log_O1'][0] <= log_scaling <= _bounds['log_O1'][1]):
		return -np.inf
	
	if not (_bounds['log_O2'][0] <= log_scaling2 <= _bounds['log_O2'][1]):
		return -np.inf
	
	if not (_bounds['log_O_star'][0] <= log_Jv_Scale <= _bounds['log_O_star'][1]):
		return -np.inf
	
	if not (_bounds['log_SD_LOS'][0] <= log_surface_density <= _bounds['log_SD_LOS'][1]):
		return -np.inf
	
	if not (_bounds['log_sigma_jitter'][0] <= log_sigma_jitter <= _bounds['log_sigma_jitter'][1]):
		return -np.inf
	
	# Mass fraction constraints
	if np.any(mfrac_independent < 0):
		return -np.inf
	
	if np.sum(mfrac_independent) >= 1.0:
		return -np.inf
	
	mfrac_dependent = 1.0 - np.sum(mfrac_independent)
	if mfrac_dependent < 0:
		return -np.inf
	
	return log_prob

def log_likelihood(theta):
	theta = np.asarray(theta)
	
	# Extract parameters and convert from log-space
	temp = theta[0]
	scaling = 10**theta[1]
	temp2 = theta[2]
	scaling2 = 10**theta[3]
	surface_density = 10**theta[4]
	Jv_Scale = 10**theta[5]
	sigma_jitter = 10**theta[6] * 1e-20  # Convert from units of 1e-20
	
	mfrac_independent = theta[7:]
	
	mfrac_dependent = 1.0 - np.sum(mfrac_independent)
	mfrac_flat = np.append(mfrac_independent, mfrac_dependent)
	mfrac_arr = mfrac_flat.reshape(_Nsizes, _Nspecies)
	
	try:
		model_eval = cavity_model_mfrac(_fit_um, temp, scaling, temp2, scaling2,
										surface_density, Jv_Scale, mfrac_arr,
										_rkabs_arr, _rksca_arr)
	except:
		return -np.inf
	
	if not np.all(np.isfinite(model_eval)):
		return -np.inf
	
	# Calculate total uncertainty including jitter
	sigma_total_sq = _fit_unc**2 + sigma_jitter**2
	sigma_total = np.sqrt(sigma_total_sq)
	
	# Calculate residuals
	residual = _fit_flux - model_eval
	
	# Chi-squared likelihood with jitter
	chi2 = np.sum(residual**2 / sigma_total_sq)
	log_normalization = -0.5 * np.sum(np.log(2.0 * np.pi * sigma_total_sq))
	
	log_likelihood_val = -0.5 * chi2 + log_normalization
	
	return log_likelihood_val

def log_probability(theta):
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

def print_checkpoint_diagnostics(sampler, step, param_names, prev_stats=None):
	"""
	Print comprehensive diagnostics at checkpoint
	
	Returns current stats dict for next checkpoint comparison
	"""
	print("\n" + "="*80)
	print(f"CHECKPOINT DIAGNOSTICS - Step {step:,}")
	print("="*80)
	
	# Get chain (use last 20% of samples to avoid burn-in issues)
	try:
		n_samples = min(step, 50000)  # Use last 50k samples or all if fewer
		chain = sampler.get_chain(discard=max(0, step - n_samples), flat=True)
		log_prob = sampler.get_log_prob(discard=max(0, step - n_samples), flat=True)
	except:
		# If can't get chain, use all samples
		chain = sampler.get_chain(flat=True)
		log_prob = sampler.get_log_prob(flat=True)
	
	if len(chain) == 0:
		print("No samples available yet")
		print("="*80 + "\n")
		return None
	
	# Acceptance fraction
	acc_frac = np.mean(sampler.acceptance_fraction)
	print(f"Acceptance fraction: {acc_frac:.3f}", end="")
	if 0.2 <= acc_frac <= 0.5:
		print(" ✓")
	elif 0.15 <= acc_frac < 0.2 or 0.5 < acc_frac <= 0.6:
		print(" ⚠")
	else:
		print(" ✗")
	
	# Log probability statistics
	log_prob_p16, log_prob_p50, log_prob_p84 = np.percentile(log_prob, [16, 50, 84])
	log_prob_iqr = log_prob_p84 - log_prob_p16
	
	print(f"\nLog Probability:")
	print(f"  16th percentile: {log_prob_p16:10.2f}")
	print(f"  50th percentile: {log_prob_p50:10.2f}")
	print(f"  84th percentile: {log_prob_p84:10.2f}")
	print(f"  IQR: {log_prob_iqr:10.2f}")
	
	# Parameter statistics
	print(f"\nParameter Statistics (16th | 50th | 84th percentiles):")
	print("-"*80)
	
	current_stats = {}
	
	for i, name in enumerate(param_names):
		if i >= chain.shape[1]:
			break
			
		p16, p50, p84 = np.percentile(chain[:, i], [16, 50, 84])
		iqr = p84 - p16
		
		# Store for next comparison
		current_stats[name] = {'p16': p16, 'p50': p50, 'p84': p84, 'iqr': iqr}
		
		# Format output based on parameter type
		if 'log' in name.lower():
			# Log-space parameters
			print(f"{name:20s}: {p16:8.3f} | {p50:8.3f} | {p84:8.3f}  (IQR: {iqr:7.3f})", end="")
		elif 'temp' in name.lower() or name.startswith('T'):
			# Temperatures (usually 10-1000 K)
			print(f"{name:20s}: {p16:8.1f} | {p50:8.1f} | {p84:8.1f}  (IQR: {iqr:7.1f})", end="")
		else:
			# Mass fractions and other 0-1 parameters
			print(f"{name:20s}: {p16:8.5f} | {p50:8.5f} | {p84:8.5f}  (IQR: {iqr:7.5f})", end="")
		
		# Compare to previous checkpoint if available
		if prev_stats is not None and name in prev_stats:
			prev_iqr = prev_stats[name]['iqr']
			iqr_change = iqr - prev_iqr
			iqr_percent_change = 100 * iqr_change / prev_iqr if prev_iqr != 0 else 0
			
			if abs(iqr_percent_change) < 1:
				# Converged (IQR stable)
				print(f"  ✓ ({iqr_percent_change:+.1f}%)")
			elif iqr_percent_change < 0:
				# Narrowing (good - converging)
				print(f"  ↓ ({iqr_percent_change:+.1f}%)")
			else:
				# Widening (still exploring)
				print(f"  ↑ ({iqr_percent_change:+.1f}%)")
		else:
			print()  # No previous data
	
	# Autocorrelation estimate (if enough samples)
	if len(chain) > 100:
		try:
			tau = sampler.get_autocorr_time(tol=0, quiet=True)
			max_tau = np.max(tau)
			max_tau_idx = np.argmax(tau)
			max_tau_param = param_names[max_tau_idx] if max_tau_idx < len(param_names) else "unknown"
			
			print(f"\nAutocorrelation Time:")
			print(f"  Max τ: {max_tau:,.0f} steps ({max_tau_param})")
			print(f"  Current samples / max τ: {len(chain) / max_tau:.1f}×")
			
			if len(chain) / max_tau > 50:
				print("  ✓ Well sampled (>50× τ)")
			elif len(chain) / max_tau > 10:
				print("  ⚠ Adequately sampled (>10× τ)")
			else:
				print("  ✗ Under-sampled (<10× τ)")
		except:
			print(f"\nAutocorrelation Time: Cannot estimate yet (need more samples)")
	
	# Summary
	print(f"\n{'='*80}\n")
	
	return current_stats

def initialize_walkers(p0, nwalkers, ndim, bounds):
	pos = []
	
	for i in range(nwalkers):
		while True:
			walker = np.zeros(ndim)
			
			# Perturb first 7 physical parameters
			for j in range(7):
				param_range = bounds[list(bounds.keys())[j]][1] - bounds[list(bounds.keys())[j]][0]
				perturbation = 0.1 * param_range * np.random.randn()
				walker[j] = p0[j] + perturbation
			
			# Perturb mass fractions
			mfrac_independent = np.array(p0[7:])
			mfrac_perturbed = mfrac_independent * (1 + 0.1 * np.random.randn(len(mfrac_independent)))
			mfrac_perturbed = np.maximum(0, mfrac_perturbed)
			
			if np.sum(mfrac_perturbed) >= 1.0:
				mfrac_perturbed *= 0.99 / np.sum(mfrac_perturbed)
			
			walker[7:] = mfrac_perturbed
			
			if np.isfinite(log_prior(walker)):
				pos.append(walker)
				break
	
	return np.array(pos)

def remove_outlier_chains(sampler, burnin, n_sigma=3):
	log_prob = sampler.get_log_prob(discard=burnin, flat=False)
	mean_log_prob_per_walker = np.mean(log_prob, axis=0)
	
	median = np.median(mean_log_prob_per_walker)
	mad = np.median(np.abs(mean_log_prob_per_walker - median))
	
	mad_to_std = 1.4826
	threshold = median - n_sigma * mad_to_std * mad
	
	good_walkers = mean_log_prob_per_walker > threshold
	n_removed = np.sum(~good_walkers)
	
	print(f"\nOutlier chain removal:")
	print(f"  Median log prob: {median:.2f}")
	print(f"  MAD: {mad:.2f}")
	print(f"  Threshold ({n_sigma}σ): {threshold:.2f}")
	print(f"  Removing {n_removed}/{len(good_walkers)} outlier walkers")
	
	if n_removed > 0:
		print(f"  Outlier walker indices: {np.where(~good_walkers)[0]}")
	
	chain = sampler.get_chain(discard=burnin, flat=False)
	log_prob_chain = sampler.get_log_prob(discard=burnin, flat=False)
	
	flat_samples = chain[:, good_walkers, :].reshape(-1, chain.shape[2])
	flat_log_prob = log_prob_chain[:, good_walkers].flatten()
	
	return flat_samples, flat_log_prob, n_removed

def check_burn_in_adequacy(sampler, requested_burnin, param_names):
	print("\n" + "="*80)
	print("BURN-IN ADEQUACY CHECK")
	print("="*80)
	
	try:
		tau = sampler.get_autocorr_time(tol=0, quiet=True)
		
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
			is_adequate = False
		else:
			print(OKGREEN + f"\n✓ Burn-in appears adequate ({requested_burnin} > {recommended_burnin})" + ENDC)
			is_adequate = True
			
	except Exception as e:
		print(f"Could not compute autocorrelation time: {e}")
		recommended_burnin = requested_burnin
		is_adequate = False
	
	print("="*80)
	
	return recommended_burnin, is_adequate

def print_mcmc_summary(flat_samples, flat_log_prob, param_names_sampled, param_names_all, n_data, ndim, n_removed=0):
	print("\n" + "="*80)
	print("MCMC SUMMARY")
	print("="*80)
	print(f"Total samples (after burn-in, thinning, outlier removal): {len(flat_samples)}")
	if n_removed > 0:
		print(f"Outlier walkers removed: {n_removed}")
	
	print(f"\nLog probability statistics:")
	print(f"  Median: {np.median(flat_log_prob):.3f}")
	print(f"  Mean: {np.mean(flat_log_prob):.3f}")
	print(f"  Std: {np.std(flat_log_prob):.3f}")
	
	# Extract median jitter (convert back from log and units)
	jitter_idx = 6
	median_log_jitter = np.median(flat_samples[:, jitter_idx])
	median_jitter = 10**median_log_jitter * 1e-20
	
	print(f"\nJitter parameter:")
	print(f"  Median σ_jitter: {median_jitter:.3e}")
	print(f"  Jitter/median(obs_unc): {median_jitter / np.median(_fit_unc):.2f}x")
	
	print("\nParameter estimates (median ± 1σ):")
	print("-"*80)
	
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

# Distances
if source_name not in distance_dict.keys():
	raise ValueError('Distance for this source not found. Please specify a distance. Exiting...')
	exit()
else:
	d = distance_dict[source_name]

# Parameter names - stored in log space
param_names_all = ['temp','log_scaling','temp2','log_scaling2','log_surface_density','log_Jv_Scale','log_sigma_jitter']
for i,gsize in enumerate(grain_sizes):
	for j,gspecies in enumerate(grain_species):
		param_names_all.append(f'{gspecies}-{gsize}um')

param_names_sampled = param_names_all[:7 + Nsizes*Nspecies - 1]

# Process only the requested aperture
aperture = aperture_name

print("\n" + "="*80)
print(f"PROCESSING: {source_name} - Aperture {aperture}")
print("="*80)

###############################
# Load and prepare data
###############################

fn_base = input_foldername + '/%s_aper%s.spectra'%(source_name,aperture)
sp_base = merge_subcubes(load_spectra(fn_base))
u_use,f_use,unc_use = [sp_base[x] for x in ['um', 'flux', 'flux_unc']]

if len(f_use[np.isfinite(f_use)])==0:
	print(f'No data in source {source_name}, aperture {aperture}. Exiting...')
	sys.exit(1)

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

_fit_um = fit_um
_fit_flux = fit_flux
_fit_unc = fit_unc
_rkabs_arr = rkabs_arr
_rksca_arr = rksca_arr

# Get bounds for this model
bounds = {}
for param in model_parameters:
	key = f'ALL:ALL:{model_str}:{param}'
	if key in bounds_dict:
		bounds[param] = bounds_dict[key]
	else:
		print(f"Warning: No bounds found for {param}, using defaults")
		bounds[param] = [-10, 10] if 'log' in param else [0, 1e10]

_bounds = bounds

# Initial guess
mfrac_arr = np.zeros((len(grain_sizes),len(grain_species)))
mfrac_arr += 1/(len(grain_sizes)*len(grain_species))

mfrac_flat = mfrac_arr.flatten()
mfrac_flat = mfrac_flat / np.sum(mfrac_flat)

p0_physical = p0_dict[f'ALL:ALL:{model_str}']
p0_mfrac = list(mfrac_flat[:-1])

# Combine physical parameters and mass fractions
p0 = p0_physical + p0_mfrac

# Number of dimensions
ndim = len(p0)

print(f"\nMCMC Setup:")
print(f"  Number of parameters: {ndim} (sampling {Nsizes*Nspecies - 1} of {Nsizes*Nspecies} mass fractions)")
print(f"  Number of walkers: {NWALKERS}")
print(f"  Number of steps: {NSTEPS}")
print(f"  Burn-in steps: {NBURN}")
print(f"  Likelihood: Chi-squared (Gaussian) with jitter term")
print(f"  Parameterization: Log-space for scalings, linear for temperatures")
print(f"  Initial log probability: {log_probability(p0):.2f}")

if not np.isfinite(log_probability(p0)):
	print(WARNING + "WARNING: Initial guess has invalid log probability!" + ENDC)

# Initialize walkers
print("\nInitializing walkers...")
pos = initialize_walkers(p0, NWALKERS, ndim, bounds)

###############################
# Run MCMC
###############################

# Initialize previous stats for IQR tracking
prev_checkpoint_stats = None

if USE_PARALLEL:
	print(f"Running MCMC with {NCORES} cores, printing diagnostics every {PRINT_INTERVAL} steps...")
	with Pool(NCORES) as pool:
		moves = emcee.moves.StretchMove(a=1.5)
		sampler = emcee.EnsembleSampler(NWALKERS, ndim, log_probability, pool=pool, moves=moves)
		
		# Run in chunks to show progress and diagnostics
		for i in range(0, NSTEPS, PRINT_INTERVAL):
			end = min(i + PRINT_INTERVAL, NSTEPS)
			n_steps_to_run = end - i
			
			# Run the chunk
			if i == 0:
				sampler.run_mcmc(pos, n_steps_to_run, progress=True)
			else:
				sampler.run_mcmc(None, n_steps_to_run, progress=True)
			
			# Print diagnostics
			prev_checkpoint_stats = print_checkpoint_diagnostics(
				sampler, end, param_names_sampled, prev_checkpoint_stats
			)
			
else:
	print(f"Running MCMC (single core), printing diagnostics every {PRINT_INTERVAL} steps...")
	sampler = emcee.EnsembleSampler(NWALKERS, ndim, log_probability)
	
	for i in range(0, NSTEPS, PRINT_INTERVAL):
		end = min(i + PRINT_INTERVAL, NSTEPS)
		n_steps_to_run = end - i
		
		# Run the chunk
		if i == 0:
			sampler.run_mcmc(pos, n_steps_to_run, progress=True)
		else:
			sampler.run_mcmc(None, n_steps_to_run, progress=True)
		
		# Print diagnostics
		prev_checkpoint_stats = print_checkpoint_diagnostics(
			sampler, end, param_names_sampled, prev_checkpoint_stats
		)

###############################
# Analyze results
###############################

print("\n" + "="*80)
print("MCMC COMPLETE - FINAL ANALYSIS")
print("="*80)
print(f"Total steps completed: {NSTEPS:,}")
print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

if np.mean(sampler.acceptance_fraction) < 0.15:
	print(WARNING + "  ⚠️ Low acceptance fraction - consider adjusting proposal" + ENDC)
elif np.mean(sampler.acceptance_fraction) > 0.6:
	print(WARNING + "  ⚠️ High acceptance fraction - chains moving slowly" + ENDC)
else:
	print(OKGREEN + "  ✓ Acceptance fraction in good range" + ENDC)

# Check burn-in adequacy
recommended_burnin, is_adequate = check_burn_in_adequacy(sampler, NBURN, param_names_sampled)

# Get samples
if REMOVE_OUTLIER_CHAINS:
	flat_samples_sampled, flat_log_prob, n_removed = remove_outlier_chains(sampler, NBURN, n_sigma=OUTLIER_SIGMA)
	if THIN > 1:
		indices = np.arange(0, len(flat_samples_sampled), THIN)
		flat_samples_sampled = flat_samples_sampled[indices]
		flat_log_prob = flat_log_prob[indices]
else:
	flat_samples_sampled = sampler.get_chain(discard=NBURN, thin=THIN, flat=True)
	flat_log_prob = sampler.get_log_prob(discard=NBURN, thin=THIN, flat=True)
	n_removed = 0

# Reconstruct the dependent mass fraction
mfrac_dependent = 1.0 - np.sum(flat_samples_sampled[:, 7:], axis=1, keepdims=True)
flat_samples_full = np.hstack([flat_samples_sampled, mfrac_dependent])

# Print summary
n_data = len(fit_flux)
print_mcmc_summary(flat_samples_full, flat_log_prob, param_names_sampled, param_names_all, n_data, ndim, n_removed)

###############################
# Save results
###############################

print(f"\nSaving results for {source_name} aperture {aperture}...")

# Create DataFrame
df = pd.DataFrame(flat_samples_full, columns=param_names_all)
df['log_prob'] = flat_log_prob
df['ID'] = np.arange(len(df))

# Reorder columns to have ID and log_prob first
df = df[['ID', 'log_prob'] + param_names_all]

# Save to CSV
output_filename = output_foldername + f'fitting_results_{source_name}_{aperture}.csv'
df.to_csv(output_filename, index=False)
print(f"Saved to: {output_filename}")

print(OKGREEN + f"\nCompleted {source_name} aperture {aperture}!" + ENDC)
print("="*80)