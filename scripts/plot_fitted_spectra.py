import matplotlib.pyplot as plt 
from ifu_analysis.jdfitting import read_optool,fit_model,blackbody,prepare_spectra_for_fit,get_continuum_mask,results_to_terminal_str,grab_p0_bounds,calculate_goodness_of_fit
from ifu_analysis.jdspextract import load_spectra,merge_subcubes,read_aperture_ini
import numpy as np
from spectres import spectres
import pandas as pd
import os

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

def cavity_model_mfrac(wav,temp,scaling,temp2,scaling2,surface_density,Jv_Scale,mfrac_arr,kabs_arr,ksca_arr):
	#Model CM
	
	#Before optimisation, one has to regrid the kappas to the relevant grid.
	#kappa_arr: Nsize x Nspecies x Nwave array of opacities.
	#mfrac_arr: Nsize x Nspecies array of mass fractions.

	kabs = weighted_kappa(mfrac_arr,kabs_arr)
	ksca = weighted_kappa(mfrac_arr,ksca_arr)

	apsize = 0.75 #apsize in arcsec
	area = apsize**2/4.25e10 # arcsec^2 --> sr

	# Fix scattering temperature to 5000 K (not a free parameter)
	Jv_T = 5000.0
	F_source = blackbody(wav,Jv_T,Jv_Scale)

	#F_source2 = 0*source_function(wav,temp,scaling,kabs,ksca,F_source)

	#model = (source_function(wav,temp,scaling,kabs,ksca,F_source)+source_function(wav,temp2,scaling2,kabs,ksca,F_source)) * np.exp(-surface_density*(kabs+ksca))
	model = source_function(wav,temp,scaling,kabs,ksca,F_source)* np.exp(-surface_density*(kabs+ksca)) + source_function(wav,temp2,scaling2,kabs,ksca,F_source)*(1-np.exp(-surface_density*(kabs+ksca)))
	return model

def source_function(wav,temp,scaling,kabs,ksca,F_source):
	return (kabs*blackbody(wav,temp,scaling) + ksca*F_source)/(kabs+ksca)

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
#	 READ DATA							 				   #
#														   #
############################################################



#Read in base spectra (without line masking)
#Plot regridded line masked spectra
#Read in fitting results.



#Wavelength to fit.
fit_wavelengths = [[4.7,14.66],[16,27.5]]


source_name = 'BHR71'
aperture = 'B1'

if source_name == 'L1448MM1':
	#PSF Unsubtracted:
	#input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/BASE/'
	#PSF Subtracted:
	input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/extracted_paper/'
	output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/cavity_modelling/'	
	aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt'%(source_name)
	fn_base = input_foldername + '/L1448MM1_aper%s.spectra'%(aperture)
	sp_base = merge_subcubes(load_spectra(fn_base))
	u_use,f_use,unc_use = [sp_base[x] for x in ['um', 'flux', 'flux_unc']]

	if os.path.isfile(aperture_filename):
		aper_names,aper_sizes,coord_list = read_aperture_ini(aperture_filename)

if source_name == 'BHR71':
	#PSF Unsubtracted:
	#input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/BASE/'
	#PSF Subtracted:
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
			#Extract wavelengths, fluxes and uncertainties.
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

# Regrid opacities to fit wavelengths
rkabs_arr = regrid_kappas(wave,kabs_arr,fit_um)
rksca_arr = regrid_kappas(wave,ksca_arr,fit_um)

# Regrid opacities to fit wavelengths
rkabs_arr_all = regrid_kappas(wave,kabs_arr,u_use)
rksca_arr_all = regrid_kappas(wave,ksca_arr,u_use)

mfrac_arr = np.zeros((len(grain_sizes),len(grain_species)))




############################################################
#														   #	
#	 LOAD BEST FITS							 			   #
#														   #
############################################################

best_fit_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/cavity_modelling/' #/MCMC_No_Jitter/
#best_fit_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/cavity_modelling/MCMC_No_Jitter/' 

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


############################################################
#														   #	
#	 PLOTTING								 			   #
#														   #
############################################################



fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
ax = fig.add_subplot(gs[0])
ax_mid = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])



model_all_um = cavity_model_mfrac(u_use,temp,scaling,temp2,scaling2,surface_density,Jv_Scale,mfrac_arr,rkabs_arr_all,rksca_arr_all)
param_str = (f"temp = {temp:.2f}\n"
			 f"scaling = {scaling:.2e}\n"
			 f"temp2 = {temp2:.2f}\n"
			 f"scaling2 = {scaling2:.2e}\n"
			 f"surface_density = {surface_density:.2e}\n"
			 f"Jv_T = {Jv_T:.2f}\n"
			 f"Jv_Scale = {Jv_Scale:.2e}")


kabs = weighted_kappa(mfrac_arr,rkabs_arr_all)
ksca = weighted_kappa(mfrac_arr,rksca_arr_all)


# Top plot - SED
ax.plot(u_use, model_all_um, color='red', label='Model \n' + param_str)
ax.plot(fit_um, fit_flux, color='blue', label='Fitted Data')
ax.fill_between(fit_um, fit_flux-fit_unc,fit_flux, color='blue',alpha=0.5)
ax.fill_between(fit_um, fit_flux,fit_flux+fit_unc, color='blue',alpha=0.5)



T1 = temp
O1 = scaling
T2 = temp2
O2 = scaling2

sigma_LOS = surface_density
T_star = Jv_T
O_star = Jv_Scale

comp1 = kabs*blackbody(u_use,T1,O1)/(kabs+ksca)
comp2 = kabs*blackbody(u_use,T2,O2)/(kabs+ksca)

F_source = blackbody(u_use,T_star,O_star)
scattering =  2*ksca*F_source/(kabs+ksca)

wout_ext = comp1+comp2+scattering

ax.fill_between(u_use,y1=wout_ext,y2= model_all_um,color='brown',alpha=0.3,label=r'$\Sigma_{\rm LOS} =$' + '%.2f'%(sigma_LOS*1e3) + r' mg cm$^{-2}$')
ax.plot(u_use,comp1,color='green',linestyle='dashed',label='%d K'%(T1))
ax.plot(u_use,comp2,color='red',linestyle='dashed',label='%d K'%(T2))
ax.plot(u_use,scattering,color='blue',linestyle='dashed',label='%d K'%(T_star))

ax.set_ylim(1e-23,1e-18)
ax.plot(u_use, f_rad, color='grey', label='Data', alpha=0.3)
ax.set_yscale('log')
ax.legend()
ax.set_xlabel('Wavelength (μm)')
ax.set_ylabel('Flux')


model_eval = cavity_model_mfrac(fit_um,temp,scaling,temp2,scaling2,surface_density,Jv_Scale,mfrac_arr,rkabs_arr,rksca_arr)

# Middle plot - Relative uncertainties
relative_unc = (fit_flux-model_eval) / (3*fit_unc)
ax_mid.plot(fit_um, relative_unc , 'o-', color='blue', markersize=3)
ax_mid.set_xlabel('Wavelength (μm)')
ax_mid.set_ylabel('F - Model / uncertainty')
ax_mid.grid(True, alpha=0.3)

# Bottom plot - rearranged to group by species
mfrac_flat = np.array(mfrac_arr.flatten())
mfrac_norm = mfrac_flat / np.nansum(mfrac_flat)

# Rearrange data to group by species instead of by size
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


#ax2.plot(lastchi2/np.nanmax(lastchi2),label='%.3f, chi2'%(change))

#temp_arr = np.array(pars_arr)[:,0][-10000:]


#ax2.plot(temp_arr/np.nanmax(temp_arr),label='temp')
#ax2.legend()
ax2.bar(x_pos, mfrac_rearranged, color=bar_colors, edgecolor='black', linewidth=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Mass Fraction')
ax2.set_ylim(1e-3,1)
ax2.set_xlabel('Grain Species & Size')
ax2.grid(axis='y', alpha=0.3)

plt.show()