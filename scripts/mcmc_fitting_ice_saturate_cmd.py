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
from scipy.interpolate import interp1d
from scipy.stats import norm
import corner
import argparse
import numpy as np 
from spectres import spectres
from scipy.interpolate import interp1d

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

def cavity_model_mfrac(wav,temp,scaling,temp2,scaling2,surface_density,Jv_Scale,mfrac_arr,kabs_arr,ksca_arr,tau_ice=None):
	#Model CM
	#
	# Geometry: point source --> warm slab (optically thick) --> cold slab --> observer
	#
	# kappa_arr : Nsize x Nspecies x Nwave array of opacities
	# mfrac_arr : Nsize x Nspecies array of mass fractions
	# tau_ice   : total ice optical depth on wav grid (sum_k N_k * sigma_k).
	#             None means no ice. Ice enters only as extinction, no ice emission.
	#
	# Warm source function includes scattering from the point source (Jv_Scale).
	# Cold source function is a pure blackbody because:
	#   (1) The optically thick warm slab blocks the point source from the cold dust.
	#   (2) The warm slab is geometrically compact (Omega_warm ~ 1e-5 of aperture),
	#       so it subtends a negligible solid angle as seen from the cold dust and
	#       does not significantly illuminate it via scattering.
	# Both slabs are extincted by tau_total = tau_dust + tau_ice.

	kabs = weighted_kappa(mfrac_arr,kabs_arr)
	ksca = weighted_kappa(mfrac_arr,ksca_arr)

	# Total optical depth of cold slab: dust + ice
	tau_dust = surface_density * (kabs + ksca)
	if tau_ice is None:
		tau_total = tau_dust
	else:
		tau_total = tau_dust + tau_ice

	# Warm source function: thermal emission + scattering from point source
	# Jv_Scale encodes the solid angle of the protostar as seen from the warm dust,
	# the phase function and broadband extinction along that path.
	Jv_T = 5000.0
	F_source = blackbody_intensity(wav, Jv_T)*Jv_Scale
	S_warm = source_function(wav, temp, kabs, ksca, F_source)

	# Cold source function: pure blackbody, no scattering term
	S_cold = blackbody_intensity(wav, temp2)

	# Two-slab RT: warm slab behind, cold slab in front, both sharing tau_total
	model = S_warm*scaling * np.exp(-tau_total) + S_cold*scaling2*(1 - np.exp(-tau_total))
	return model

def source_function(wav,temp,kabs,ksca,F_source):
	return (kabs*blackbody_intensity(wav,temp) + ksca*F_source)/(kabs+ksca)

def make_prerun_diagnostic_plot(u_use, f_rad, unc_rad, fit_um, fit_flux, fit_unc,
                                saturated_mask, p0, source_name, aperture, model_str,
                                output_foldername, cavity_model_mfrac,
                                mfrac_arr_p0, rkabs_arr_p0, rksca_arr_p0,
                                rice_sigma_arr_p0=None, ice_params_p0=None):
    '''
    Generate a pre-run diagnostic plot before MCMC sampling begins.

    Shows the full spectrum with the fit wavelength grid and saturated channels
    highlighted (Panel 1), and the initial guess model overlaid on the full
    spectrum (Panel 2). Saves to output_foldername and calls plt.show().

    Parameters
    ----------
    u_use : array
        Full wavelength array in um (unmasked, surface brightness units).
    f_rad : array
        Full flux array in MJy sr^-1.
    unc_rad : array
        Full uncertainty array in MJy sr^-1.
    fit_um : array
        Wavelength array used for fitting (line-masked, wavelength-cut).
    fit_flux : array
        Flux array used for fitting.
    fit_unc : array
        Uncertainty array used for fitting.
    saturated_mask : boolean array
        True where fit_flux / fit_unc < Nsigma_sat, on fit_um grid.
    p0 : array
        Initial guess parameter vector.
    source_name : str
        Source name for title and filename.
    aperture : str
        Aperture name for title and filename.
    model_str : str
        Model name string for title.
    output_foldername : str
        Folder to save the diagnostic figure.
    cavity_model_mfrac : callable
        Model function with signature (wav, temp, scaling, temp2, scaling2,
        surface_density, Jv_Scale, mfrac_arr, kabs_arr, ksca_arr, tau_ice).
    mfrac_arr_p0 : array
        Nsizes x Nspecies mass fraction array from p0.
    rkabs_arr_p0 : array
        Absorption opacity array regridded to u_use.
    rksca_arr_p0 : array
        Scattering opacity array regridded to u_use.
    rice_sigma_arr_p0 : list of arrays, optional
        Ice cross-sections regridded to u_use. None means no ice.
    ice_params_p0 : array, optional
        Ice column density initial guesses. None means no ice.
    '''
    from scipy.interpolate import interp1d
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    C_DATA    = 'steelblue'
    C_MODEL   = 'tomato'
    C_FIT     = 'black'
    C_SAT     = 'steelblue'
    ALPHA_FULL = 0.35
    LW_THIN   = 0.6
    LW_THICK  = 1.0

    FONTSIZE_LABEL  = 8
    FONTSIZE_LEGEND = 7
    FONTSIZE_TICK   = 7
    FONTSIZE_TITLE  = 8

    # --- Evaluate initial guess model on u_use grid ---
    # Regrid opacities to u_use
    from spectres import spectres as _spectres
    import numpy as _np

    def _regrid_kappas_to(wave_orig, kappa_arr, new_wave):
        Nsizes, Nspecies, _ = _np.shape(kappa_arr)
        out = _np.zeros((Nsizes, Nspecies, len(new_wave)))
        for i in range(Nsizes):
            for j in range(Nspecies):
                out[i][j] = _spectres(new_wave, wave_orig, kappa_arr[i][j])
        return out

    # Build ice tau on u_use grid
    if rice_sigma_arr_p0 is not None and ice_params_p0 is not None:
        tau_ice_full = np.zeros(len(u_use))
        for k, (sigma, N_ice) in enumerate(zip(rice_sigma_arr_p0, ice_params_p0)):
            # sigma may be on fit_um — interpolate to u_use
            interp = interp1d(fit_um, sigma, bounds_error=False, fill_value=0.0)
            tau_ice_full += N_ice * interp(u_use)
    else:
        tau_ice_full = np.zeros(len(u_use))

    # Regrid kappas to u_use
    # Use global wave (optool grid) — passed implicitly via closure in original script
    # Here we interpolate rkabs_arr_p0 from fit_um to u_use
    Nsizes_p0, Nspecies_p0, _ = np.shape(rkabs_arr_p0)
    kabs_full = np.zeros((Nsizes_p0, Nspecies_p0, len(u_use)))
    ksca_full = np.zeros((Nsizes_p0, Nspecies_p0, len(u_use)))
    for i in range(Nsizes_p0):
        for j in range(Nspecies_p0):
            kabs_full[i][j] = interp1d(fit_um, rkabs_arr_p0[i][j],
                                        bounds_error=False, fill_value='extrapolate')(u_use)
            ksca_full[i][j] = interp1d(fit_um, rksca_arr_p0[i][j],
                                        bounds_error=False, fill_value='extrapolate')(u_use)

    temp, scaling, temp2, scaling2, surface_density, Jv_Scale = p0[:6]

    try:
        model_full = cavity_model_mfrac(
            u_use, temp, scaling, temp2, scaling2,
            surface_density, Jv_Scale, mfrac_arr_p0,
            kabs_full, ksca_full, tau_ice_full
        )
    except Exception as e:
        print('Warning: could not evaluate initial guess model on full grid: %s' % e)
        model_full = np.full(len(u_use), np.nan)

    # --- Identify fit and non-fit channels on u_use grid ---
    fit_um_set = set(np.round(fit_um, 6))
    in_fit = np.array([round(w, 6) in fit_um_set for w in u_use])

    # --- Find saturated channels on u_use grid ---
    # Interpolate saturation mask from fit_um to u_use (nearest neighbour)
    sat_interp = interp1d(fit_um, saturated_mask.astype(float),
                          kind='nearest', bounds_error=False, fill_value=0.0)
    sat_full = sat_interp(u_use).astype(bool)

    # --- Build plot arrays ---
    # Detected on fit grid: in_fit and not saturated
    detected_full = in_fit & ~sat_full
    undetected_full = in_fit & sat_full
    outside_full = ~in_fit

    f_det   = f_rad.copy().astype(float)
    f_det[~detected_full] = np.nan

    f_out   = f_rad.copy().astype(float)
    f_out[~outside_full] = np.nan

    unc_det   = unc_rad.copy().astype(float)
    unc_det[~detected_full] = np.nan

    unc_undet = unc_rad.copy().astype(float)
    unc_undet[~undetected_full] = np.nan

    # --- Find contiguous saturated blocks on fit_um for upper limit markers ---
    blocks   = []
    in_block = False
    flagged  = saturated_mask
    for idx in range(len(flagged)):
        if flagged[idx] and not in_block:
            block_start = idx
            in_block    = True
        elif not flagged[idx] and in_block:
            blocks.append((block_start, idx - 1))
            in_block = False
    if in_block:
        blocks.append((block_start, len(flagged) - 1))

    # --- Figure ---
    panel_height = 3.46 * 0.5
    fig, axes = plt.subplots(2, 1, figsize=(3.46 * 2, panel_height * 2), sharex=True)
    fig.subplots_adjust(hspace=0.08)
    fig.suptitle('%s  |  Aperture %s  |  Model %s' % (source_name, aperture, model_str),
                 fontsize=FONTSIZE_TITLE)

    for ax_idx, ax in enumerate(axes):

        # Outside fit range — faint
        ax.plot(u_use, f_out,
                color=C_DATA, lw=LW_THIN, alpha=ALPHA_FULL, zorder=2)

        # Detected in fit range
        ax.plot(u_use, f_det,
                color=C_DATA, lw=LW_THIN, alpha=0.9, zorder=4,
                label=r'$F_\nu$ (fit range)' if ax_idx == 0 else None)

        # Saturation floor — grey where detected, black thick where undetected
        sat_floor_det   = Nsigma_sat * unc_det
        sat_floor_undet = Nsigma_sat * unc_undet

        ax.plot(u_use, sat_floor_det,
                color='grey', lw=0.5, alpha=0.8, zorder=3,
                label=r'$N_{\sigma}\,\sigma_{\rm pipe}$' if ax_idx == 0 else None)

        ax.plot(u_use, sat_floor_undet,
                color=C_FIT, lw=LW_THICK, alpha=1.0, zorder=3)

        # Upper limit markers and fills for saturated blocks
        for block_start, block_end in blocks:
            block_um    = fit_um[block_start:block_end + 1]
            block_floor = Nsigma_sat * fit_unc[block_start:block_end + 1]

            ax.fill_between(block_um, block_floor * 1e-3, block_floor,
                            color=C_SAT, alpha=0.08, zorder=1)

            block_len     = block_end - block_start + 1
            arrow_indices = [block_start + block_len // 2] if block_len < 20 else [block_start, block_end]
            for aidx in arrow_indices:
                ax.plot(fit_um[aidx], Nsigma_sat * fit_unc[aidx],
                        marker='v', ms=2.5, color=C_SAT, alpha=0.9, zorder=5)

        # Initial guess model — panel 2 only
        if ax_idx == 1:
            ax.plot(u_use, model_full,
                    color=C_MODEL, lw=LW_THICK, alpha=0.9, zorder=6,
                    label='Initial guess')

        ax.set_yscale('log')
        ax.set_xlim(4.7, 27.5)
        ax.set_ylabel(r'$I_\nu$ (MJy sr$^{-1}$)', fontsize=FONTSIZE_LABEL)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.legend(fontsize=FONTSIZE_LEGEND, loc='upper left', ncol=3, frameon=False)

    axes[0].set_title('Data + fit range + saturation mask', fontsize=FONTSIZE_LABEL, pad=2)
    axes[1].set_title('Initial guess model', fontsize=FONTSIZE_LABEL, pad=2)
    axes[-1].set_xlabel(r'Wavelength ($\mu$m)', fontsize=FONTSIZE_LABEL)
    axes[-1].tick_params(labelsize=FONTSIZE_TICK)

    savepath = output_foldername + 'prerun_diagnostic_%s_%s.png' % (source_name, aperture)
    plt.savefig(savepath, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()
    print('Saved diagnostic plot to %s' % savepath)


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
			
			mfrac_independent = np.array(p0[6:6 + _n_mfrac_params])
			mfrac_perturbed = mfrac_independent * (1 + 0.1 * np.random.randn(len(mfrac_independent)))
			mfrac_perturbed = np.maximum(0, mfrac_perturbed)
			
			if np.sum(mfrac_perturbed) >= 1.0:
				mfrac_perturbed *= 0.99 / np.sum(mfrac_perturbed)
			
			walker[6:6 + _n_mfrac_params] = mfrac_perturbed

			# Perturb ice column densities around p0 values
			if USE_ICE:
				for k in range(Nice):
					idx = 6 + _n_mfrac_params + k
					lo, hi = _ice_N_bounds[k]
					param_range = hi - lo
					walker[idx] = p0[idx] + 0.1 * param_range * np.random.randn()
					walker[idx] = np.clip(walker[idx], lo, hi)
			
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

def print_mcmc_summary(flat_samples, flat_log_prob, param_names_sampled, param_names_all, chi2_red, ndim, n_removed=0):
	print("\n" + "="*80)
	print("MCMC SUMMARY")
	print("="*80)
	print(f"Total samples (after burn-in, thinning, outlier removal): {len(flat_samples)}")
	if n_removed > 0:
		print(f"Outlier walkers removed: {n_removed}")
	
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
#                                                          #
#   ICE LOADING AND REGRIDDING                            #
#                                                          #
############################################################

def load_ice_absorbance(filename, input_units='wavenumber', delimiter=' ', comment='#'):
	"""
	Load a lab ice absorbance file and return wavelength in micron
	and cross-section sigma in cm^2 molecule^-1.

	Parameters
	----------
	filename    : str   path to absorbance file
	input_units : str   'um' for micron, 'wavenumber' for cm^-1
	delimiter   : str   column delimiter
	comment     : str   comment character

	Returns
	-------
	wav_um  : 1D array  wavelength in micron, monotonically increasing
	sigma   : 1D array  cross-section in cm^2 molecule^-1
	"""
	data = np.loadtxt(filename, delimiter=delimiter if delimiter != ' ' else None,
					  comments=comment)
	col0 = data[:, 0]
	sigma = data[:, 1]

	if input_units == 'wavenumber':
		# wavenumber (cm^-1) -> micron, reverses order
		wav_um = 1e4 / col0
	elif input_units == 'um':
		wav_um = col0
	else:
		raise ValueError(f"ice_input_units must be 'um' or 'wavenumber', got '{input_units}'")
	return wav_um, sigma

def regrid_ice(wav_um, sigma, new_wave):
	"""
	Regrid ice cross-section onto new_wave using linear interpolation.
	Outside the lab wavelength range the cross-section is set to zero
	(no absorption where not measured).

	Parameters
	----------
	wav_um   : 1D array  lab wavelength grid in micron
	sigma    : 1D array  cross-section on lab grid
	new_wave : 1D array  target wavelength grid in micron

	Returns
	-------
	sigma_regridded : 1D array  cross-section on new_wave
	"""
	interp_func = interp1d(wav_um, sigma, kind='linear',
						   bounds_error=False, fill_value=0.0)
	return interp_func(new_wave)

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
ice_absorption_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/absorption/'


DO_CRYSTALS = False
UNC_MULTIPLYER = 1
Nsigma_sat = 2.5

if DO_CRYSTALS:
	grain_species = ['olmg50','pyrmg70','for','ens']
else:
	grain_species = ['olmg50','pyrmg70']
Nspecies = len(grain_species)
grain_sizes = [0.1,1.5]
Nsizes = len(grain_sizes)

#Wavelength to fit.
fit_wavelengths = [[4.7,14.66],[16,27.5]]

#If you want to regrid spectra to finer resolution before fitting. WARNING: This arbitrarily reduces uncertainties on measurments, giving over-accurate MCMC constraints.
spectral_resolution = None 

############################################################
#														   #	
#	 ICE PARAMETERS                                        #
#                                                          #                                                         #
#   Set ice_species = [] to run without ice                #
#                                                          #
############################################################

# Initial guess for ice scaling factors — physically motivated literature values
# for Class I protostars. Override per species as needed.
p0_ice_defaults = {
	'H2O 15K'  : 0.5,
	'H2O 150K'  : 0.5
}

ice_species   = ['H2O 15K', 'H2O 150K']
ice_filenames = [
	#ice_absorption_foldername + 'LEIDEN_pure_water_ice_15K_wavenumber.csv',
	#ice_absorption_foldername + 'LEIDEN_pure_water_ice_75K_wavenumber.csv'

	ice_absorption_foldername  + '2023-08-18_hdo-h2o_1-200_thin(27)_1500_blcorr.csv',
	ice_absorption_foldername + '2023-08-18_hdo-h2o_1-200_thin(118)_154_blcorr.csv'

]

# Input wavelength units for each file: 'um' or 'wavenumber'
ice_input_units = ['wavenumber', 'wavenumber']

# Bounds
ice_N_lower = 0.0
ice_N_upper = 30

# Per-species overrides — leave empty dict {} to use global bounds for all
ice_N_bounds_dict = {
	# 'H2O'  : [0.0, 1e19],
	# 'CO2'  : [0.0, 5e17],
	# 'CO'   : [0.0, 5e17],
	# 'NH4+' : [0.0, 5e17],
}

# Delimiter for ice files (' ' for whitespace, ',' for csv, etc.)
ice_file_delimiter = ','

# Comment character in ice files
ice_file_comment = '#'


# Load all ice species
ice_wav_list   = []   # lab wavelength grids
ice_sigma_list = []   # lab cross-sections

USE_ICE = len(ice_species) > 0
Nice = len(ice_species)

if USE_ICE:
	print(f"\nLoading {Nice} ice species:")
	for k, (name, fn, units) in enumerate(zip(ice_species, ice_filenames, ice_input_units)):
		wav_ice, sigma_ice = load_ice_absorbance(fn, input_units=units,
												 delimiter=ice_file_delimiter,
												 comment=ice_file_comment)
		ice_wav_list.append(wav_ice)
		ice_sigma_list.append(sigma_ice)
		print(f"  [{k}] {name:8s}  λ = {wav_ice.min():.2f} – {wav_ice.max():.2f} μm   "
			  f"peak σ = {np.max(sigma_ice):.2e} cm²/mol   file: {fn}")
	print()
else:
	print("\nNo ice species specified. Running without ice.\n")

# Ice column density bounds per species
ice_N_bounds = []
for name in ice_species:
	if name in ice_N_bounds_dict:
		ice_N_bounds.append(ice_N_bounds_dict[name])
	else:
		ice_N_bounds.append([ice_N_lower, ice_N_upper])

############################################################
#														   #	
#	 MCMC PARAMETERS									   #
#														   #
############################################################

# MCMC settings
NWALKERS = 64
if DO_CRYSTALS:
	NSTEPS = 1000000
else:
	NSTEPS = 700000
NBURN = 300000
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
#	 Fitting initial conditions and bounds				   #
#														   #
############################################################

#Set initial guesses and bounds

p0_dict = {}

p0_T1 = 70
p0_O1 = 1e-6
p0_T2 = 400
p0_O2 = 1e-7
p0_T_star = 5000
p0_O_star = 1e-5
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
	#PSF Unsubtracted:
	#input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/BASE/'
	#PSF Subtracted:
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


#Global variables for use in MCMC.
_fit_um = None
_fit_flux = None
_fit_unc = None
_rkabs_arr = None
_rksca_arr = None
_bounds = None
_Nsizes = Nsizes
_Nspecies = Nspecies
_n_mfrac_params = Nsizes * Nspecies - 1
_rice_sigma_arr = None   # list of Nice arrays, each on fit_um grid — set after regridding
_ice_N_bounds   = None   # list of [lower, upper] per species — set after regridding
_tau_ice        = None   # ice optical depth array on fit_um grid — set after regridding
_apsize = None
_saturated_mask = None   # boolean mask for saturated channels (SNR < 1)

def log_prior(theta):
	theta = np.asarray(theta)
	
	temp, scaling, temp2, scaling2, surface_density, Jv_Scale = theta[:6]
	mfrac_independent = theta[6:6 + _n_mfrac_params]
	
	log_prob = 0.0
	
	#Uniform
	if not (_bounds['T1'][0] <= temp <= _bounds['T1'][1]):
		return -np.inf
	if not (_bounds['T2'][0] <= temp2 <= _bounds['T2'][1]):
		return -np.inf
	
	#Log uniform
	if not (_bounds['O1'][0] <= scaling <= _bounds['O1'][1]):
		return -np.inf
	else:
		log_prob += -np.log(scaling)
	
	#Log uniform
	if not (_bounds['O2'][0] <= scaling2 <= _bounds['O2'][1]):
		return -np.inf
	else:
		log_prob += -np.log(scaling2)
	
	#Log uniform
	if not (_bounds['O_star'][0] <= Jv_Scale <= _bounds['O_star'][1]):
		return -np.inf
	else:
		log_prob += -np.log(Jv_Scale)
	
	#Uniform
	if not (_bounds['SD_LOS'][0] <= surface_density <= _bounds['SD_LOS'][1]):
		return -np.inf
	
	#Make sure the mass fractions stay between 0 and 1.
	if np.any(mfrac_independent < 0):
		return -np.inf
	
	if np.sum(mfrac_independent) >= 1.0:
		return -np.inf
	
	mfrac_dependent = 1.0 - np.sum(mfrac_independent)
	if mfrac_dependent < 0:
		return -np.inf

	# Ice scaling priors - linear uniform
	if USE_ICE:
		N_ice_params = theta[6 + _n_mfrac_params:]
		for k, N_ice in enumerate(N_ice_params):
			lo, hi = _ice_N_bounds[k]
			if not (lo <= N_ice <= hi):
				return -np.inf
			else:
				log_prob += -np.log(N_ice)

	return log_prob

def log_likelihood(theta):
	theta = np.asarray(theta)
	
	temp, scaling, temp2, scaling2, surface_density, Jv_Scale = theta[:6]
	mfrac_independent = theta[6:6 + _n_mfrac_params]
	
	mfrac_dependent = 1.0 - np.sum(mfrac_independent)
	mfrac_flat = np.append(mfrac_independent, mfrac_dependent)
	mfrac_arr = mfrac_flat.reshape(_Nsizes, _Nspecies)

	# Build ice optical depth array from sampled column densities
	if USE_ICE:
		N_ice_params = theta[6 + _n_mfrac_params:]
		tau_ice = np.zeros(len(_fit_um))
		for k, N_ice in enumerate(N_ice_params):
			tau_ice += N_ice * _rice_sigma_arr[k]

	else:
		tau_ice = np.zeros(len(_fit_um))

	try:
		model_eval = cavity_model_mfrac(_fit_um, temp, scaling, temp2, scaling2,
										surface_density, Jv_Scale, mfrac_arr,
										_rkabs_arr, _rksca_arr, tau_ice)
	except:
		return -np.inf
	
	if not np.all(np.isfinite(model_eval)):
		return -np.inf

	residual = _fit_flux - model_eval

	# Normal channels — standard Gaussian likelihood
	log_like_normal = -0.5 * np.sum(
		(residual[~_saturated_mask]**2 / _fit_unc[~_saturated_mask]**2)
	)

	# Saturated channels — one-sided likelihood
	# Model must predict flux below observed floor, penalized for predicting above
	log_like_saturated = np.sum(
		norm.logsf(Nsigma_sat * _fit_unc[_saturated_mask],
				   loc=model_eval[_saturated_mask],
				   scale=_fit_unc[_saturated_mask])
	)

	return log_like_normal + log_like_saturated

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

# Add ice column density parameter names
if USE_ICE:
	for name in ice_species:
		param_names_all.append(f'N_{name}')

param_names_sampled = param_names_all[:6 + Nsizes*Nspecies - 1]
if USE_ICE:
	param_names_sampled += [f'N_{name}' for name in ice_species]

# Process only the requested aperture
aperture = aperture_name
apsize = aper_sizes[np.where(aper_names==aperture)][0]

arcsec2_to_ster = 2.35e-11
aparea = np.pi*apsize**2*arcsec2_to_ster # arcsec^2 --> sr

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
prepared_spectra = prepare_spectra_for_fit(u_use,f_use,unc_use,fit_wavelengths,um_cut=27.5,spectral_resolution=spectral_resolution)

spectra_cols = ['um','flux','unc']
fit_um,fit_flux,fit_unc = [prepared_spectra['fitdata:%s'%(x)] for x in spectra_cols]

fit_unc *= UNC_MULTIPLYER


u_use,f_rad,unc_rad = [prepared_spectra['unmasked:%s'%(x)] for x in spectra_cols]

unc_rad *= UNC_MULTIPLYER


#Convert flux densities to surface brightness.
f_rad /= aparea
unc_rad /= aparea

fit_flux /= aparea
fit_unc /= aparea

# Define saturated mask — channels where SNR < Nsigma_sat (silicate feature floor)
saturated_mask = (fit_flux / fit_unc) < Nsigma_sat
_saturated_mask = saturated_mask
print(f"\nSaturated channels: {np.sum(saturated_mask)} of {len(fit_flux)}")
print(f"Saturated wavelength range: {fit_um[saturated_mask].min():.2f} - {fit_um[saturated_mask].max():.2f} um")

# Regrid opacities to fit wavelengths
rkabs_arr = regrid_kappas(wave,kabs_arr,fit_um)
rksca_arr = regrid_kappas(wave,ksca_arr,fit_um)

# Regrid ice cross-sections to fit wavelengths
if USE_ICE:
	rice_sigma_arr = []
	print("Regridding ice cross-sections to fit wavelength grid...")
	for k, (name, wav_ice, sigma_ice) in enumerate(zip(ice_species, ice_wav_list, ice_sigma_list)):
		sigma_regrid = regrid_ice(wav_ice, sigma_ice, fit_um)

		plt.plot(fit_um,sigma_regrid)
		plt.show()

		rice_sigma_arr.append(sigma_regrid)
		n_nonzero = np.sum(sigma_regrid > 0)
		print(f"  {name:8s}: {n_nonzero}/{len(fit_um)} wavelength channels have non-zero absorbance")
	print()
else:
	rice_sigma_arr = []

###############################
# Set up MCMC
###############################

_fit_um = fit_um
_fit_flux = fit_flux
_fit_unc = fit_unc
_rkabs_arr = rkabs_arr
_rksca_arr = rksca_arr
_rice_sigma_arr = rice_sigma_arr
_ice_N_bounds = ice_N_bounds
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


p0_ice = []
if USE_ICE:
	for k, name in enumerate(ice_species):
		p0_ice.append(p0_ice_defaults.get(name, 1e16))

# Combine physical parameters, mass fractions and ice column densities
p0 = p0_physical + p0_mfrac + p0_ice


# Number of dimensions
ndim = len(p0)

print(f"\nMCMC Setup:")
print(f"  Number of parameters: {ndim}")
print(f"    Physical + scaling : 6")
print(f"    Mass fractions     : {Nsizes*Nspecies - 1} (sampled) + 1 (dependent)")
if USE_ICE:
	print(f"    Ice N columns      : {Nice}  ({', '.join(ice_species)})")
print(f"  Number of walkers: {NWALKERS}")
print(f"  Number of steps: {NSTEPS}")
print(f"  Burn-in steps: {NBURN}")
print(f"  Initial log probability: {log_probability(p0):.2f}")

if not np.isfinite(log_probability(p0)):
	print(WARNING + "WARNING: Initial guess has invalid log probability!" + ENDC)

# Build p0 mfrac_arr for diagnostic
mfrac_p0_flat = np.array(p0_mfrac + [1.0 - sum(p0_mfrac)])
mfrac_arr_p0  = mfrac_p0_flat.reshape(Nsizes, Nspecies)

make_prerun_diagnostic_plot(
    u_use, f_rad, unc_rad, fit_um, fit_flux, fit_unc,
    saturated_mask, p0, source_name, aperture, model_str,
    output_foldername, cavity_model_mfrac,
    mfrac_arr_p0, rkabs_arr, rksca_arr,
    rice_sigma_arr_p0=rice_sigma_arr if USE_ICE else None,
    ice_params_p0=p0_ice if USE_ICE else None
)


# Initialize walkers
print("\nInitializing walkers...")
pos = initialize_walkers(p0, NWALKERS, ndim, bounds)

###############################
# Run MCMC
###############################

if USE_PARALLEL:
	print(f"Running MCMC with {NCORES} cores...")
	with Pool(NCORES) as pool:
		moves = emcee.moves.StretchMove(a=1.3)
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
mfrac_dependent = 1.0 - np.sum(flat_samples_sampled[:, 6:6 + _n_mfrac_params], axis=1, keepdims=True)
# Assemble full samples: physical + all mass fracs + ice
flat_samples_full = np.hstack([
	flat_samples_sampled[:, :6 + _n_mfrac_params],   # physical + independent mfracs
	mfrac_dependent,                                   # dependent mfrac
])
if USE_ICE:
	flat_samples_full = np.hstack([
		flat_samples_full,
		flat_samples_sampled[:, 6 + _n_mfrac_params:],  # ice column densities
	])

# Calculate chi2_red using only normal (non-saturated) channels
# flat_log_prob includes the one-sided saturated likelihood which is not chi2
n_data_normal = len(fit_flux) - np.sum(_saturated_mask)
dof = n_data_normal - ndim

def log_likelihood_normal_only(theta):
	theta = np.asarray(theta)
	temp, scaling, temp2, scaling2, surface_density, Jv_Scale = theta[:6]
	mfrac_independent = theta[6:6 + _n_mfrac_params]
	mfrac_dependent = 1.0 - np.sum(mfrac_independent)
	mfrac_flat = np.append(mfrac_independent, mfrac_dependent)
	mfrac_arr = mfrac_flat.reshape(_Nsizes, _Nspecies)
	if USE_ICE:
		N_ice_params = theta[6 + _n_mfrac_params:]
		tau_ice = np.zeros(len(_fit_um))
		for k, N_ice in enumerate(N_ice_params):
			tau_ice += N_ice * _rice_sigma_arr[k]
	else:
		tau_ice = np.zeros(len(_fit_um))
	model_eval = cavity_model_mfrac(_fit_um, temp, scaling, temp2, scaling2,
									surface_density, Jv_Scale, mfrac_arr,
									_rkabs_arr, _rksca_arr, tau_ice)
	residual = _fit_flux[~_saturated_mask] - model_eval[~_saturated_mask]
	return -0.5 * np.sum(residual**2 / _fit_unc[~_saturated_mask]**2)

chi2_red = np.array([-2 * log_likelihood_normal_only(s) / dof for s in flat_samples_sampled])

# Print summary
print_mcmc_summary(flat_samples_full, flat_log_prob, param_names_sampled, param_names_all, chi2_red, ndim, n_removed)

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
if DO_CRYSTALS:
	output_filename = output_foldername + f'fitting_results_crys_{source_name}_{aperture}.csv'
else:
	output_filename = output_foldername + f'fitting_results_nocrys_{source_name}_{aperture}.csv'
df.to_csv(output_filename, index=False)
print(f"Saved to: {output_filename}")

print(OKGREEN + f"\nCompleted {source_name} aperture {aperture}!" + ENDC)
print("="*80)