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

def cavity_model_mfrac(wav,temp,scaling,temp2,scaling2,surface_density,Jv_Scale,mfrac_arr,kabs_arr,ksca_arr):
	#Model CM
	
	#Before optimisation, one has to regrid the kappas to the relevant grid.
	#kappa_arr: Nsize x Nspecies x Nwave array of opacities.
	#mfrac_arr: Nsize x Nspecies array of mass fractions.

	kabs = weighted_kappa(mfrac_arr,kabs_arr)
	ksca = weighted_kappa(mfrac_arr,ksca_arr)

	# Fix scattering temperature to 5000 K (not a free parameter)
	Jv_T = 5000.0
	F_source = blackbody(wav,Jv_T,Jv_Scale)

	#model = (source_function(wav,temp,scaling,kabs,ksca,F_source)+source_function(wav,temp2,scaling2,kabs,ksca,F_source)) * np.exp(-surface_density*(kabs+ksca))
	model = source_function(wav,temp,scaling,kabs,ksca,F_source)* np.exp(-surface_density*(kabs+ksca)) + source_function(wav,temp2,scaling2,kabs,ksca,F_source)*(1-np.exp(-surface_density*(kabs+ksca)))
	return model

def source_function(wav,temp,scaling,kabs,ksca,F_source):
	return (kabs*blackbody(wav,temp,scaling) + ksca*F_source)/(kabs+ksca)


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

def load_initial_guess_from_csv(csv_foldername, source_name, aperture, n_samples=1000):
	"""
	Load initial guess from previous optimization results.
	Takes the average of the last n_samples entries.
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
		
		return p0_physical, p0_mfrac
		
	except Exception as e:
		print(f"Error loading initial guess from {csv_filename}: {e}")
		print(f"Using default initial guess instead")
		return None, None

# MCMC settings
NWALKERS = 32
NSTEPS = 700000
NBURN = 200000
THIN = 20
USE_PARALLEL = True
NCORES = 8

# Initial guess settings
USE_PREVIOUS_RESULTS = False
PREVIOUS_RESULTS_FOLDER = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/cavity_modelling/B2C/'
N_SAMPLES_TO_AVERAGE = 1000

# Outlier removal settings
REMOVE_OUTLIER_CHAINS = True
OUTLIER_SIGMA = 3

############################################################
#														   #	
#	 USER DEFINED PARAMETERS SECTION 2/2				   #
#														   #
############################################################

#Set initial guesses and bounds

p0_dict = {}

p0_T1 = 70
p0_O1 = 1e-2
p0_T2 = 400
p0_O2 = 1e-7
p0_T_star = 5000
p0_O_star = 1e-9
p0_SD_LOS = 1e-3
p0_SD_WIND = 1e-7
p0_FF = 0.5

p0_dict['ALL:ALL:CMWIND'] = [p0_T1,p0_O1,p0_T2,p0_O2,p0_SD_LOS,p0_O_star,p0_SD_WIND]
p0_dict['ALL:ALL:CM'] =  [p0_T1,p0_O1,p0_T2,p0_O2,p0_SD_LOS,p0_O_star]
p0_dict['ALL:ALL:CM_ANI'] =  [p0_T1,p0_T2,p0_FF,p0_SD_LOS]

bounds_dict = {}

uT_emit = 1000
uO_emit = 0.1
uSD = 0.2
uT_star = 5001
lT_star = 4999
utheta = 180
uFF = 0.5

#For model CM
bounds_dict['ALL:ALL:CM:T1'] = [0,uT_emit]
bounds_dict['ALL:ALL:CM:O1'] = [0,uO_emit]
bounds_dict['ALL:ALL:CM:SD_LOS'] = [0,uSD]

#Bounds that are equal to other bounds.
bounds_dict['ALL:ALL:CM:O_star'] = bounds_dict['ALL:ALL:CM:O1']
bounds_dict['ALL:ALL:CM:T2'] = bounds_dict['ALL:ALL:CM:T1']
bounds_dict['ALL:ALL:CM:O2'] = bounds_dict['ALL:ALL:CM:O1']

############################################################
#														   #	
#	 INITIALIZE MODELS				 				   	   #
#														   #
############################################################

#if model_str == 'CM':
#	model = cavity_model

#else:
#	raise ValueError('Please choose a valid model. Exiting...')
#	exit()
model_parameters = ['T1','O1','T2','O2','SD_LOS','O_star']
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
'''elif source_name == 'BHR71':
	output_foldername = '/home/vorsteja/Documents/JOYS/JDust/BHR71_scripts_12122025/BHR71/output/'
	fn_band_arr = ['ch1-short','ch1-medium','ch1-long','ch2-short','ch2-medium','ch2-long',
				'ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long'] 
	input_foldername = '/home/vorsteja/Documents/JOYS/JDust/BHR71_scripts_12122025/BHR71_unc/BHR71/spectra_werrors/circle/1.00_arcsec/spectra_sci/'
	aper_names = ['b1', 'b2', 'b3', 'b4', 'cr1', 'o5'] + ['cl1','cl2','cl3','cr2','cr3','cr4']'''


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
	
	temp, scaling, temp2, scaling2, surface_density, Jv_Scale = theta[:6]
	mfrac_independent = theta[6:]
	
	log_prob = 0.0
	
	if not (_bounds['T1'][0] <= temp <= _bounds['T1'][1]):
		return -np.inf
	if not (_bounds['T2'][0] <= temp2 <= _bounds['T2'][1]):
		return -np.inf
	
	if not (_bounds['O1'][0] <= scaling <= _bounds['O1'][1]):
		return -np.inf
	else:
		log_prob += -np.log(scaling)
	
	if not (_bounds['O2'][0] <= scaling2 <= _bounds['O2'][1]):
		return -np.inf
	else:
		log_prob += -np.log(scaling2)
	
	if not (_bounds['O_star'][0] <= Jv_Scale <= _bounds['O_star'][1]):
		return -np.inf
	else:
		log_prob += -np.log(Jv_Scale)
	
	if not (_bounds['SD_LOS'][0] <= surface_density <= _bounds['SD_LOS'][1]):
		return -np.inf
	
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
	
	temp, scaling, temp2, scaling2, surface_density, Jv_Scale = theta[:6]
	mfrac_independent = theta[6:]
	
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
	
	residual = _fit_flux - model_eval
	chi2 = np.sum(residual**2 / _fit_unc**2)
	
	return -0.5 * chi2

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

def initialize_walkers(p0, nwalkers, ndim, bounds):
	pos = []
	
	for i in range(nwalkers):
		while True:
			walker = np.zeros(ndim)
			
			for j in range(6):
				param_range = bounds[list(bounds.keys())[j]][1] - bounds[list(bounds.keys())[j]][0]
				perturbation = 0.1 * param_range * np.random.randn()
				walker[j] = p0[j] + perturbation
			
			mfrac_independent = np.array(p0[6:])
			mfrac_perturbed = mfrac_independent * (1 + 0.1 * np.random.randn(len(mfrac_independent)))
			mfrac_perturbed = np.maximum(0, mfrac_perturbed)
			
			if np.sum(mfrac_perturbed) >= 1.0:
				mfrac_perturbed *= 0.99 / np.sum(mfrac_perturbed)
			
			walker[6:] = mfrac_perturbed
			
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
	
	dof = n_data - ndim
	chi2_red = -2 * flat_log_prob / dof
	
	print(f"\nReduced χ² statistics:")
	print(f"  Median: {np.median(chi2_red):.3f}")
	print(f"  Mean: {np.mean(chi2_red):.3f}")
	print(f"  Std: {np.std(chi2_red):.3f}")
	
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

# Column names for output
columns = ['ID','chi2_red','temp','scaling','temp2','scaling2','surface_density','Jv_Scale']
for i,gsize in enumerate(grain_sizes):
	for j,gspecies in enumerate(grain_species):
		columns.append(f'{gspecies}-{gsize}um')

# Parameter names
param_names_all = ['temp','scaling','temp2','scaling2','surface_density','Jv_Scale']
for i,gsize in enumerate(grain_sizes):
	for j,gspecies in enumerate(grain_species):
		param_names_all.append(f'{gspecies}-{gsize}um')

param_names_sampled = param_names_all[:6 + Nsizes*Nspecies - 1]

# Process only the requested aperture
aperture = aperture_name

print("\n" + "="*80)
print(f"PROCESSING: {source_name} - Aperture {aperture}")
print("="*80)

###############################
# Load and prepare data
###############################

if source_name == 'BHR71_Lukasz':
	if aperture in  ['b1', 'b2', 'b3', 'b4', 'cr1', 'o5']:
		fn_base = [input_foldername + 'BHR71-%s__circle_1.00_arcsec_%s_defringed.txt'%(aperture,x) for x in fn_band_arr]
		um_base_arr = []
		flux_base_arr = []
		unc_base_arr = []
		for fn in fn_base:
			#Extract wavelengths, fluxes and uncertainties.
			um_base,flux_base,unc_base = np.loadtxt(fn,delimiter=' ').T
		
			um_base_arr += list(um_base)
			flux_base_arr += list(flux_base)
			unc_base_arr += list(unc_base)

		u_use = np.array(um_base_arr)
		f_use = np.array(flux_base_arr)
		unc_use = np.array(unc_base_arr)
	elif aperture in ['cl1','cl2','cl3','cr2','cr3','cr4']:
		fn = input_foldername + 'extra_apertures/' + 'BHR71-%s_sci_circle_1.00_arcsec_final.txt'%(aperture)
		u_use,f_use,unc_use = np.loadtxt(fn,delimiter=' ').T
else:
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
		bounds[param] = [0, 1e10]

_bounds = bounds

# Initial guess
if USE_PREVIOUS_RESULTS:
	print(f"\nAttempting to load initial guess from previous results...")
	p0_physical_loaded, p0_mfrac_loaded = load_initial_guess_from_csv(
		PREVIOUS_RESULTS_FOLDER, 
		source_name, 
		'B1', 
		n_samples=N_SAMPLES_TO_AVERAGE
	)
	
	if p0_physical_loaded is not None and p0_mfrac_loaded is not None:
		p0_physical = p0_physical_loaded
		p0_mfrac = p0_mfrac_loaded
		print(f"✓ Using loaded initial guess")
	else:
		print(f"✗ Using default initial guess")
		mfrac_arr = np.zeros((len(grain_sizes),len(grain_species)))
		mfrac_arr += 1/(len(grain_sizes)*len(grain_species))
		
		mfrac_flat = mfrac_arr.flatten()
		mfrac_flat = mfrac_flat / np.sum(mfrac_flat)
		
		p0_physical = p0_dict[f'ALL:ALL:{model_str}']
		p0_mfrac = list(mfrac_flat[:-1])
else:
	print(f"\nUsing default initial guess")
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
print(f"  Initial log probability: {log_probability(p0):.2f}")

if not np.isfinite(log_probability(p0)):
	print(WARNING + "WARNING: Initial guess has invalid log probability!" + ENDC)

# Initialize walkers
print("\nInitializing walkers...")
pos = initialize_walkers(p0, NWALKERS, ndim, bounds)

###############################
# Run MCMC
###############################

if USE_PARALLEL:
	print(f"Running MCMC with {NCORES} cores...")
	with Pool(NCORES) as pool:
		moves = emcee.moves.StretchMove(a=1.5)
		sampler = emcee.EnsembleSampler(NWALKERS, ndim, log_probability, pool=pool, moves=moves)
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

if np.mean(sampler.acceptance_fraction) < 0.15:
	print(WARNING + "Warning: Low acceptance fraction." + ENDC)
elif np.mean(sampler.acceptance_fraction) > 0.6:
	print(WARNING + "Warning: High acceptance fraction." + ENDC)

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
mfrac_dependent = 1.0 - np.sum(flat_samples_sampled[:, 6:], axis=1, keepdims=True)
flat_samples_full = np.hstack([flat_samples_sampled, mfrac_dependent])

# Calculate chi2_red
n_data = len(fit_flux)
dof = n_data - ndim
chi2_red = -2 * flat_log_prob / dof

# Print summary
print_mcmc_summary(flat_samples_full, flat_log_prob, param_names_sampled, param_names_all, n_data, ndim, n_removed)

###############################
# Save results
###############################

print(f"\nSaving results for {source_name} aperture {aperture}...")

# Create DataFrame
df = pd.DataFrame(flat_samples_full, columns=param_names_all)
df['chi2_red'] = chi2_red
df['ID'] = np.arange(len(df))

# Reorder columns
df = df[['ID', 'chi2_red'] + param_names_all]

# Save to CSV
output_filename = output_foldername + f'fitting_results_{source_name}_{aperture}.csv'
df.to_csv(output_filename, index=False)
print(f"Saved to: {output_filename}")

print(OKGREEN + f"\nCompleted {source_name} aperture {aperture}!" + ENDC)
print("="*80)