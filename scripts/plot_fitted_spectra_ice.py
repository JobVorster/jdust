import matplotlib.pyplot as plt 
from ifu_analysis.jdfitting import read_optool,prepare_spectra_for_fit
from ifu_analysis.jdspextract import load_spectra,merge_subcubes,read_aperture_ini
import numpy as np
from spectres import spectres
import pandas as pd
import os
from scipy.interpolate import interp1d

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

def load_ice_absorbance(filename, input_units='wavenumber', delimiter=' ', comment='#'):
	"""
	Load a lab ice absorbance file and return wavelength in micron
	and cross-section sigma in cm^2 molecule^-1.
	"""
	data = np.loadtxt(filename, delimiter=delimiter if delimiter != ' ' else None,
					  comments=comment)
	col0 = data[:, 0]
	sigma = data[:, 1]

	if input_units == 'wavenumber':
		wav_um = 1e4 / col0
	elif input_units == 'um':
		wav_um = col0
	else:
		raise ValueError(f"ice_input_units must be 'um' or 'wavenumber', got '{input_units}'")
	return wav_um, sigma

def regrid_ice(wav_um, sigma, new_wave):
	"""
	Regrid ice cross-section onto new_wave using linear interpolation.
	Outside the lab wavelength range the cross-section is set to zero.
	"""
	interp_func = interp1d(wav_um, sigma, kind='linear',
						   bounds_error=False, fill_value=0.0)
	return interp_func(new_wave)

def blackbody_intensity(wav, temp):
	h = 6.626e-34
	c = 2.998e8
	k = 1.381e-23
	wav = wav*1e-6
	freq = c * 1/wav
	radiance = 2*h*freq**3/(c**2) * (1/(np.exp(h*freq/(k*temp))))
	radiance = radiance/1e4
	radiance = radiance*freq
	return radiance

def cavity_model_mfrac(wav, temp, scaling, temp2, scaling2, surface_density, Jv_Scale,
					   mfrac_arr, kabs_arr, ksca_arr, tau_ice=None):
	kabs = weighted_kappa(mfrac_arr, kabs_arr)
	ksca = weighted_kappa(mfrac_arr, ksca_arr)

	tau_dust = surface_density * (kabs + ksca)
	if tau_ice is None:
		tau_total = tau_dust
	else:
		tau_total = tau_dust + tau_ice

	Jv_T = 5000.0
	F_source = blackbody_intensity(wav, Jv_T) * Jv_Scale
	S_warm = source_function(wav, temp, kabs, ksca, F_source)
	S_cold = blackbody_intensity(wav, temp2)

	model = S_warm*scaling * np.exp(-tau_total) + S_cold*scaling2 * (1 - np.exp(-tau_total))
	return model

def source_function(wav, temp, kabs, ksca, F_source):
	return (kabs * blackbody_intensity(wav, temp) + ksca * F_source) / (kabs + ksca)

############################################################
#														   #	
#	 INITIALIZE OPACITIES				 				   #
#														   #
############################################################

opacity_foldername = '/home/vorsteja/Documents/JOYS/JDust/optool_opacities/'

grain_species = ['olmg50','pyrmg70','for','ens'] 
Nspecies = len(grain_species)
grain_sizes = [0.1,1.5]
Nsizes = len(grain_sizes)

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
#	 ICE PARAMETERS                                        #
#														   #
############################################################

h2o_foldername = '/home/vorsteja/Documents/JOYS/Collaborator_Scripts/Contsub/'

ice_species   = ['H2O 15K', 'H2O 150K']
ice_filenames = [
	h2o_foldername + '2023-08-18_hdo-h2o_1-200_thin(118)_154_blcorr.csv',
	h2o_foldername + '2023-08-18_hdo-h2o_1-200_thin(27)_1500_blcorr.csv'
]
ice_input_units  = ['wavenumber', 'wavenumber']
ice_file_delimiter = ','
ice_file_comment   = '#'

USE_ICE = len(ice_species) > 0
Nice    = len(ice_species)

# Load ice cross-sections
ice_wav_list   = []
ice_sigma_list = []

if USE_ICE:
	for name, fn, units in zip(ice_species, ice_filenames, ice_input_units):
		wav_ice, sigma_ice = load_ice_absorbance(fn, input_units=units,
												 delimiter=ice_file_delimiter,
												 comment=ice_file_comment)
		ice_wav_list.append(wav_ice)
		ice_sigma_list.append(sigma_ice)

############################################################
#														   #	
#	 READ DATA							 				   #
#														   #
############################################################

fit_wavelengths = [[4.7,14.66],[16,27.5]]

source_name = 'L1448MM1'
aperture = 'B1'

if source_name == 'L1448MM1':
	input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/extracted_paper/'
	output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/cavity_modelling/'	
	aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt'%(source_name)
	fn_base = input_foldername + '/L1448MM1_aper%s.spectra'%(aperture)
	sp_base = merge_subcubes(load_spectra(fn_base))
	u_use,f_use,unc_use = [sp_base[x] for x in ['um', 'flux', 'flux_unc']]

	if os.path.isfile(aperture_filename):
		aper_names,aper_sizes,coord_list = read_aperture_ini(aperture_filename)

if source_name == 'BHR71':
	input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/extracted_paper_BHR71'
	output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/cavity_modelling/'	
	aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt'%(source_name)
	fn_base = input_foldername + '/%s_aper%s.spectra'%(source_name,aperture)
	sp_base = merge_subcubes(load_spectra(fn_base))
	u_use,f_use,unc_use = [sp_base[x] for x in ['um', 'flux', 'flux_unc']]

	if os.path.isfile(aperture_filename):
		aper_names,aper_sizes,coord_list = read_aperture_ini(aperture_filename)

if source_name == 'BHR71_Lukasz':
	input_foldername = '/home/vorsteja/Documents/JOYS/JDust/BHR71_scripts_12122025/BHR71_unc/BHR71/spectra_werrors/circle/1.00_arcsec/spectra_sci/'
	fn_band_arr = ['ch1-short','ch1-medium','ch1-long','ch2-short','ch2-medium','ch2-long',
				'ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long'] 
	if aperture in  ['b1', 'b2', 'b3', 'b4', 'cr1', 'o5']:
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

source_name = source_name.split('_')[0]

# Prepare spectrum for fit
prepared_spectra = prepare_spectra_for_fit(u_use,f_use,unc_use,fit_wavelengths,um_cut=27.5,spectral_resolution=None)

spectra_cols = ['um','flux','unc']
fit_um,fit_flux,fit_unc = [prepared_spectra['fitdata:%s'%(x)] for x in spectra_cols]
u_use,f_rad,unc_rad = [prepared_spectra['unmasked:%s'%(x)] for x in spectra_cols]

# Convert flux densities to surface brightness (matching MCMC script)
apsize = aper_sizes[np.where(aper_names==aperture)][0]
arcsec2_to_ster = 2.35e-11
aparea = np.pi * apsize**2 * arcsec2_to_ster  # arcsec^2 --> sr

f_rad    /= aparea
unc_rad  /= aparea
fit_flux /= aparea
fit_unc  /= aparea

# Regrid opacities to fit wavelengths
rkabs_arr = regrid_kappas(wave,kabs_arr,fit_um)
rksca_arr = regrid_kappas(wave,ksca_arr,fit_um)

# Regrid opacities to full wavelength grid
rkabs_arr_all = regrid_kappas(wave,kabs_arr,u_use)
rksca_arr_all = regrid_kappas(wave,ksca_arr,u_use)

# Regrid ice cross-sections to both grids
if USE_ICE:
	rice_sigma_arr     = []  # on fit_um
	rice_sigma_arr_all = []  # on u_use
	for wav_ice, sigma_ice in zip(ice_wav_list, ice_sigma_list):
		rice_sigma_arr.append(regrid_ice(wav_ice, sigma_ice, fit_um))
		rice_sigma_arr_all.append(regrid_ice(wav_ice, sigma_ice, u_use))

mfrac_arr = np.zeros((len(grain_sizes),len(grain_species)))


############################################################
#														   #	
#	 LOAD BEST FITS							 			   #
#														   #
############################################################

best_fit_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/cavity_modelling/'

fn = best_fit_foldername + 'summary_percentiles.csv'

df = pd.read_csv(fn)
ind = np.where(np.logical_and(df['sourcename'] == source_name,df['aperture']==aperture))[0][0]
best_fit = df.iloc[ind]
print(best_fit)

temp = best_fit['temp_50']
scaling = 10**best_fit['scaling_50']
temp2 = best_fit['temp2_50']
scaling2 = 10**best_fit['scaling2_50']
Jv_Scale = 10**best_fit['Jv_Scale_50']
surface_density = 10**best_fit['surface_density_50']
Jv_T = 5000

for i,size in enumerate(grain_sizes):
	for j,comp in enumerate(grain_species):
		mfrac_arr[i][j] = best_fit['%s-%sum_50'%(comp,size)]

# Load ice column densities from best fit
N_ice_best = []
if USE_ICE:
	for name in ice_species:
		col = f'N_{name}_50'
		if col in best_fit.index:
			N_ice_best.append(best_fit[col])
		else:
			print(f"Warning: column '{col}' not found in summary CSV, defaulting N=0 for {name}")
			N_ice_best.append(0.0)

# Build tau_ice on both wavelength grids
if USE_ICE:
	tau_ice_fit = np.zeros(len(fit_um))
	tau_ice_all = np.zeros(len(u_use))
	for k, N_ice in enumerate(N_ice_best):
		tau_ice_fit += N_ice * rice_sigma_arr[k]
		tau_ice_all += N_ice * rice_sigma_arr_all[k]
else:
	tau_ice_fit = None
	tau_ice_all = None


############################################################
#														   #	
#	 PLOTTING								 			   #
#														   #
############################################################

fig = plt.figure(figsize=(16, 18))
gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 0.8, 1], hspace=0.35)
ax     = fig.add_subplot(gs[0])
ax_mid = fig.add_subplot(gs[1])
ax_ice = fig.add_subplot(gs[2])
ax2    = fig.add_subplot(gs[3])


model_all_um = cavity_model_mfrac(u_use, temp, scaling, temp2, scaling2, surface_density, Jv_Scale,
								  mfrac_arr, rkabs_arr_all, rksca_arr_all, tau_ice=tau_ice_all)

# Model without ice for comparison
model_all_um_noice = cavity_model_mfrac(u_use, temp, scaling, temp2, scaling2, surface_density, Jv_Scale,
										mfrac_arr, rkabs_arr_all, rksca_arr_all, tau_ice=None)

param_str = (f"temp = {temp:.2f}\n"
			 f"scaling = {scaling:.2e}\n"
			 f"temp2 = {temp2:.2f}\n"
			 f"scaling2 = {scaling2:.2e}\n"
			 f"surface_density = {surface_density:.2e}\n"
			 f"Jv_T = {Jv_T:.2f}\n"
			 f"Jv_Scale = {Jv_Scale:.2e}")
if USE_ICE:
	for name, N in zip(ice_species, N_ice_best):
		param_str += f"\nN_{name} = {N:.2e}"


kabs = weighted_kappa(mfrac_arr, rkabs_arr_all)
ksca = weighted_kappa(mfrac_arr, rksca_arr_all)

# Top plot - SED
ax.plot(u_use, model_all_um, color='red', label='Model (with ice)\n' + param_str,zorder=1)
if USE_ICE:
	ax.plot(u_use, model_all_um_noice, color='orange', linestyle=':', label='Model (no ice)', alpha=0.7)
ax.plot(fit_um, fit_flux, color='blue', label='Fitted Data',alpha=0.5,zorder=0)
ax.fill_between(fit_um, fit_flux-fit_unc, fit_flux, color='blue', alpha=0.5)
ax.fill_between(fit_um, fit_flux, fit_flux+fit_unc, color='blue', alpha=0.5)

T1 = temp
O1 = scaling
T2 = temp2
O2 = scaling2

sigma_LOS = surface_density
T_star = Jv_T
O_star = Jv_Scale

# Component decomposition matching MCMC model exactly:
# warm:  S_warm * scaling * exp(-tau_total)   where S_warm = (kabs*BB_warm + ksca*F_scat) / (kabs+ksca)
# cold:  BB_cold * scaling2 * (1-exp(-tau_total))   -- pure blackbody, no scattering
tau_total_all = surface_density * (kabs + ksca)
if tau_ice_all is not None:
	tau_total_all = tau_total_all + tau_ice_all

F_source   = blackbody_intensity(u_use, T_star) * O_star
S_warm     = (kabs * blackbody_intensity(u_use, T1) + ksca * F_source) / (kabs + ksca)

comp1      = S_warm * O1 * np.exp(-tau_total_all)          # warm component (extincted)
comp2      = blackbody_intensity(u_use, T2) * O2 * (1 - np.exp(-tau_total_all))  # cold component
scattering = ksca * F_source / (kabs + ksca) * O1 * np.exp(-tau_total_all)       # scattering contribution only

wout_ext   = comp1 + comp2  # full model = sum of components

ax.fill_between(u_use, y1=wout_ext, y2=model_all_um, color='brown', alpha=0.3,
				label=r'$\Sigma_{\rm LOS} =$' + '%.2f'%(sigma_LOS*1e3) + r' mg cm$^{-2}$')
ax.plot(u_use, comp1,      color='green', linestyle='dashed', label='%d K'%(T1))
ax.plot(u_use, comp2,      color='red',   linestyle='dashed', label='%d K'%(T2))
ax.plot(u_use, scattering, color='blue',  linestyle='dashed', label='%d K'%(T_star))

ax.set_ylim(1e-12,1e-8)
ax.plot(u_use, f_rad, color='grey', label='Data', alpha=0.3)
ax.set_yscale('log')
ax.legend(fontsize=7)
ax.set_xlabel('Wavelength (μm)')
ax.set_ylabel('Flux')

# Middle plot - Residuals
model_eval = cavity_model_mfrac(fit_um, temp, scaling, temp2, scaling2, surface_density, Jv_Scale,
								mfrac_arr, rkabs_arr, rksca_arr, tau_ice=tau_ice_fit)
relative_unc = (fit_flux - model_eval) / (3*fit_unc)
ax_mid.plot(fit_um, relative_unc, 'o-', color='blue', markersize=3)
ax_mid.set_xlabel('Wavelength (μm)')
ax_mid.set_ylabel('F - Model / uncertainty')
ax_mid.grid(True, alpha=0.3)

# Ice optical depth plot
if USE_ICE and tau_ice_all is not None:
	colors_ice = ['purple', 'magenta', 'teal', 'olive']
	tau_ice_per_species_all = [N * sigma for N, sigma in zip(N_ice_best, rice_sigma_arr_all)]
	for k, (name, tau_k) in enumerate(zip(ice_species, tau_ice_per_species_all)):
		ax_ice.plot(u_use, tau_k, color=colors_ice[k % len(colors_ice)],
					label=f'{name}  N={N_ice_best[k]:.2e}')
	ax_ice.plot(u_use, tau_ice_all, color='black', linestyle='--', label='Total ice τ')
	ax_ice.set_xlabel('Wavelength (μm)')
	ax_ice.set_ylabel(r'Ice $\tau_\nu$')
	ax_ice.legend(fontsize=8)
	ax_ice.grid(True, alpha=0.3)
else:
	ax_ice.text(0.5, 0.5, 'No ice species loaded', transform=ax_ice.transAxes,
				ha='center', va='center', fontsize=12, color='grey')
	ax_ice.set_axis_off()

# Bottom plot - Mass fractions
mfrac_flat = np.array(mfrac_arr.flatten())
mfrac_norm = mfrac_flat / np.nansum(mfrac_flat)

mfrac_rearranged = []
labels = []
colors_list = ['C0', 'C1', 'C2', 'C3', 'C4']
bar_colors = []

for i_spec, species in enumerate(grain_species):
	for i_size, size in enumerate(grain_sizes):
		idx = i_size * Nspecies + i_spec
		mfrac_rearranged.append(mfrac_norm[idx])
		labels.append(f'{species}\n{size:.1f}μm')
		bar_colors.append(colors_list[i_spec])

x_pos = np.arange(len(mfrac_rearranged))

ax2.bar(x_pos, mfrac_rearranged, color=bar_colors, edgecolor='black', linewidth=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Mass Fraction')
ax2.set_ylim(1e-3,1)
ax2.set_xlabel('Grain Species & Size')
ax2.grid(axis='y', alpha=0.3)

plt.show()