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

def cavity_model_mfrac(wav,temp,scaling,temp2,scaling2,surface_density,Jv_T,Jv_Scale,mfrac_arr,kabs_arr,ksca_arr):
	#Model CM
	
	#Before optimisation, one has to regrid the kappas to the relevant grid.
	#kappa_arr: Nsize x Nspecies x Nwave array of opacities.
	#mfrac_arr: Nsize x Nspecies array of mass fractions.

	kabs = weighted_kappa(mfrac_arr,kabs_arr)
	ksca = weighted_kappa(mfrac_arr,ksca_arr)

	F_source = blackbody(wav,Jv_T,Jv_Scale)

	model = (source_function(wav,temp,scaling,kabs,ksca,F_source)+source_function(wav,temp2,scaling2,kabs,ksca,F_source)) * np.exp(-surface_density*(kabs+ksca))
	return model



def source_function(wav,temp,scaling,kabs,ksca,F_source):
	return (kabs*blackbody(wav,temp,scaling) + ksca*F_source)/(kabs+ksca)




############################################################
#														   #	
#	 USER DEFINED PARAMETERS SECTION 1/2				   #
#														   #
############################################################

distance_dict = {}
distance_dict['L1448MM1'] = 293 #pc
distance_dict['BHR71'] = 176 #pc


COL_WIDTH = 20#for terminal printing

model_str = 'CM' #Options: CM, CMWIND, CM_ANI 
source_name = 'L1448MM1' #Options: L1448MM1, BHR71
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
	

	#19 January 2026, also in cavity_model.py
	#Before running this script again:
		#Make the setup to save the chi2 and other things about specific models.
		#Run alot of models fitting for exploratory data analysis.

		
	#Experiment:
		#Stitching makes some artifacts 
		#Make sure all the relevant wavelengths are included with the spectral extraction.
		#One can even make the current apertures larger for S/N.
		#Make the A3 + apertures larger for S/N.

		#Why does B2 not use the ch4long for stitching but A2 does? -- it is because the ch4long cube is masked from psf subtraction.
			# Fix --> Move the apertures so that you are outside the masked regions.

	#Make model changes for 3D radiative transfer modelling.

	#Another important experiment:
		#Rebin the spectrum to constant resolving power -- after flagging.
		#Rewrite the model to do the MCMC-type fitting of Olofsson, J et al. 2010
		#Generate 0.1 um, 1.5 um and 6 um opacities for Olivine, Pyroxene, Forsterite, Enstatite and Sillica.
		#Also opacities for most important ices (just search them on a JOYS paper, and download from the LAMBDA database).
		#Copy the cavity model code to change it to the mcmc-like with mass fractions 
			# -- keep the functional form as is, but just let the relevant kappas vary with mass fraction.
		#See if you can produce better fits.
		#P.S. Check if the spectral regridding is correct.
		#One has to think about the screening opacity -- is it KP5?
		#Look for crystalinity fractions!

		#Then think about composition -- one can make different composition "suites".



############################################################
#														   #	
#	 INITIALIZE MODELS				 				   	   #
#														   #
############################################################

#FYI:
#Steps when adding a new model:
#Define the model as a function in the format (wavelength, *pars)
#Choose a modelstring for it.
#Name all parameters, and add the model to the INITIALIZE MODELS section.
#Set an initial guess (p0) and optional bounds in the second USER DEFINED PARAMETERS SECTION.


#'model' becomes the model function, while 'model_parameters' are the user defined names of the relevant parameters.
#The code will break if the number of parameters and names do not match up.

if model_str == 'CM':
	model = cavity_model
	model_parameters = ['T1','O1','T2','O2','SD_LOS','T_star','O_star']

else:
	raise ValueError('Please choose a valid model. Exiting...')
	exit()

model_uncertainties = ['d%s'%(x) for x in model_parameters]
############################################################
#														   #	
#	 INITIALIZE FOLDERNAMES AND APERTURES 				   #
#														   #
############################################################

#People save their spectra in different formats. The important thing is to define the file and aperture names here.
#This is a bit gross.
#Make sure you are successfully reading your spectra before continuing.

if source_name == 'L1448MM1':

	input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/extracted_paper/'
	output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/cavity_modelling/'	
	#Define aperture filename (for aperture names and radii).
	aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt'%(source_name)

	#aper_names = ['C3']

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
#	 USER DEFINED PARAMETERS SECTION 2/2				   #
#														   #
############################################################

#Set initial guesses.

#Models:
#CMWIND: Cavity model with wind extinction
#CM: Cavity without wind extinction.

#Keys have the format [SOURCE]:[APERTURE]:[MODEL]
#APERTURE and SOURCE have a all option, the script will check 
#if there are others, but setting them to ALL, will be the default.

#Model parameters
#CMWIND: 'T1','O1','T2','O2','SD_LOS','T_star','O_star','SD_WIND'
#CM: 'T1','O1','T2','O2','SD_LOS','T_star','O_star'


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

p0_dict['ALL:ALL:CMWIND'] = [p0_T1,p0_O1,p0_T2,p0_O2,p0_SD_LOS,p0_T_star,p0_O_star,p0_SD_WIND]
p0_dict['ALL:ALL:CM'] =  [p0_T1,p0_O1,p0_T2,p0_O2,p0_SD_LOS,p0_T_star,p0_O_star]
p0_dict['ALL:ALL:CM_ANI'] =  [p0_T1,p0_T2,p0_FF,p0_SD_LOS,p0_T_star]


#Bounds work in the same way, except that you have to
#specify the parameter name as well 
#[SOURCE]:[APERTURE]:[MODEL]:[PAR]
#with PAR matching the corresponding 'model_parameters' entry.

bounds_dict = {}

uT_emit = 1000
uO_emit = 0.1
uSD = 0.2
uT_star = 5001 #Fix the scattering temperature (its degenerate with the scaling).
lT_star = 4999
utheta = 180
uFF = 0.5

#For model CM
##########################################################################
bounds_dict['ALL:ALL:CM:T1'] = [0,uT_emit]
bounds_dict['ALL:ALL:CM:O1'] = [0,uO_emit]
bounds_dict['ALL:ALL:CM:SD_LOS'] = [0,uSD]
bounds_dict['ALL:ALL:CM:T_star'] = [lT_star,uT_star]

#Bounds that are equal to other bounds.
bounds_dict['ALL:ALL:CM:O_star'] = bounds_dict['ALL:ALL:CM:O1']
bounds_dict['ALL:ALL:CM:T2'] = bounds_dict['ALL:ALL:CM:T1']
bounds_dict['ALL:ALL:CM:O2'] = bounds_dict['ALL:ALL:CM:O1']

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

mfrac_arr = np.zeros((len(grain_sizes),len(grain_species)))
mfrac_arr += 1/(len(grain_sizes)*len(grain_species))

#Now I have arrays that are of the right shape.

############################################################
#														   #	
#	 FITTING AND OUTPUT				 				   	   #
#														   #
############################################################

#Distances (specified above).
if source_name not in distance_dict.keys():
	raise ValueError('Distance for this source not found. Please specify a distance. Exiting...')
	exit()
else:
	d = distance_dict[source_name]

for aperture in aper_names:
	#This needs some attention. Its the extraction of data for a specific source.
	if source_name == 'BHR71':
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
	else:
		#Filenames of spectra extracted and saved with the jdspextract functions.
		fn_base = input_foldername + '/L1448MM1_aper%s.spectra'%(aperture)

		#Formatting: Merge the arrays of the stiched spectra.
		sp_base = merge_subcubes(load_spectra(fn_base))

		#Extract wavelengths, fluxes and uncertainties.
		u_use,f_use,unc_use = [sp_base[x] for x in ['um', 'flux', 'flux_unc']]
	if len(f_use[np.isfinite(f_use)])==0:
		print('No data in source %s, aperture %s. Continuing...'%(source_name,aperture))
		continue


	#Prepare spectrum for fit.
	#This includes:
	#Conversion from Jy to W cm-2
	#Line masking.
	#Selecting fit wavelengths.
	#NaN flagging.
	prepared_spectra = prepare_spectra_for_fit(u_use,f_use,unc_use,fit_wavelengths,um_cut=27.5)
	

	spectra_cols = ['um','flux','unc']
	fit_um,fit_flux,fit_unc = [prepared_spectra['fitdata:%s'%(x)] for x in spectra_cols]
	u_use,f_rad,unc_rad = [prepared_spectra['unmasked:%s'%(x)] for x in spectra_cols]
	u_noline,f_noline,unc_noline = [prepared_spectra['linemasked:%s'%(x)] for x in spectra_cols]


	rkabs_arr = regrid_kappas(wave,kabs_arr,fit_um)
	rksca_arr = regrid_kappas(wave,ksca_arr,fit_um)

	rkabs_arr_all = regrid_kappas(wave,kabs_arr,u_use)
	rksca_arr_all = regrid_kappas(wave,ksca_arr,u_use)
	do_plot = False

	print('Fitting for source %s aperture %s'%(source_name,aperture))

	if do_plot:

		plt.ion()
		fig = plt.figure(figsize=(16, 14))
		gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
		ax = fig.add_subplot(gs[0])
		ax_mid = fig.add_subplot(gs[1])
		ax2 = fig.add_subplot(gs[2])

	temp = 70
	scaling = 1e-9
	temp2 = 400 
	scaling2 = 1e-7
	surface_density = 1e-3
	Jv_T = 5000
	Jv_Scale = 1e-9


	N_iter = 500000



	pars = [temp,scaling,temp2,scaling2,surface_density,Jv_T,Jv_Scale] + list(mfrac_arr.flatten())
	pars_history = np.zeros((len(pars),N_iter))
	chi2_red_arr = [1e4]
	chi2_red_better = [1e4]
	pars_arr = [pars]
	k_params = 0
	not_better_count = 0

	change = 1e4

	while k_params < N_iter:
		k_params += 1
		ind_par = np.random.randint(len(pars))
		
		multiplyer = np.random.uniform(0.99,1.01)
		if ind_par == 0:
			temp *= multiplyer
		elif ind_par == 1:
			scaling *= multiplyer
		elif ind_par == 2:
			temp2 *= multiplyer
		elif ind_par == 3:
			scaling2 *= multiplyer
		elif ind_par == 4:
			surface_density *= multiplyer
		elif ind_par == 5:
			Jv_T *= 1
		elif ind_par == 6:
			Jv_Scale *= multiplyer
		elif ind_par > 6:
			ind_par1 = ind_par- 7
			ind_size = ind_par1//Nspecies
			ind_species = ind_par1%Nspecies
			mfrac_arr[ind_size][ind_species] *= multiplyer
		
		model_eval = cavity_model_mfrac(fit_um,temp,scaling,temp2,scaling2,surface_density,Jv_T,Jv_Scale,mfrac_arr,rkabs_arr,rksca_arr)
		
		# Inside the loop
		if k_params % 100000 == 0:
			print('Iteration %d of %d'%(k_params,N_iter))

			if do_plot:
				ax.clear()
				ax_mid.clear()
				ax2.clear()
				
				model_all_um = cavity_model_mfrac(u_use,temp,scaling,temp2,scaling2,surface_density,Jv_T,Jv_Scale,mfrac_arr,rkabs_arr_all,rksca_arr_all)
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


				lastchi2 = chi2_red_better[-100:]
				change = min(lastchi2) - max(lastchi2)


				ax.set_ylim(1e-23,1e-18)
				ax.plot(u_use, f_rad, color='grey', label='Data', alpha=0.3)
				ax.set_yscale('log')
				ax.set_xscale('log')
				ax.legend()
				ax.set_xlabel('Wavelength (μm)')
				ax.set_ylabel('Flux')
				ax.set_title(f'Iteration {k_params}, χ²_red = {chi2_red_arr[-1]:.4f}, Delta χ²_red = {change:.4f}, Not better count = {not_better_count:.1f}', 
							 fontsize=10, loc='left')
				
				# Middle plot - Relative uncertainties
				relative_unc = (fit_flux-model_eval) / fit_flux
				ax_mid.plot(fit_um, relative_unc * 100, 'o-', color='blue', markersize=3)
				ax_mid.set_xscale('log')
				ax_mid.set_xlabel('Wavelength (μm)')
				ax_mid.set_ylabel('Relative Error (%)')
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
				
				plt.draw()
				plt.pause(0.001)
			
		#Residual
		residual_e = np.array(fit_flux - model_eval)
		#Number of data points
		n = len(fit_um)
		#Number of parameters.	
		k = len(pars)
		#chi squared
		chi2 = np.nansum(residual_e**2/fit_unc**2)
		#reduced_chi squared
		degrees_of_freedom = n-k
		chi2_red = chi2/degrees_of_freedom
		chi2_red_arr.append(chi2_red)
		
		if k_params > 10:
			delta_chi2 = chi2_red - chi2_red_arr[-2]
			if delta_chi2 < 0:
				not_better_count = 0
				pars = [temp,scaling,temp2,scaling2,surface_density,Jv_T,Jv_Scale] + list(mfrac_arr.flatten())
				pars_arr.append(pars)
				chi2_red_better.append(chi2_red)
			else:
				not_better_count +=1
				if ind_par == 0:
					temp /= multiplyer
				elif ind_par == 1:
					scaling /= multiplyer
				elif ind_par == 2:
					temp2 /= multiplyer
				elif ind_par == 3:
					scaling2 /= multiplyer
				elif ind_par == 4:
					surface_density /= multiplyer
				elif ind_par == 5:
					Jv_T /= multiplyer
				elif ind_par == 6:
					Jv_Scale /= multiplyer
				elif ind_par > 6:
					ind_par1 = ind_par - 7
					ind_size = ind_par1//Nspecies
					ind_species = ind_par1%Nspecies
					mfrac_arr[ind_size][ind_species] /= multiplyer
				


	print('Saving for source %s aperture %s'%(source_name,aperture))
	columns = ['ID','chi2_red','temp','scaling','temp2','scaling2','surface_density','Jv_T','Jv_Scale','olmg50-0.1um','olmg50-1.5um','olmg50-6.0um','pyrmg70-0.1um','pyrmg70-1.5um','pyrmg70-6.0um']
	df = pd.DataFrame(columns = columns)
	for i,col in enumerate(columns):
		if i == 0:
			inarr = np.arange(len(np.array(pars_arr)[:,0]))
		elif i == 1:
			inarr = chi2_red_better
		else:
			inarr = np.array(pars_arr)[:,i-2]
		df[col] = inarr
	df.to_csv(output_foldername + 'fitting_results_%s_%s.csv'%(source_name,aperture))
	if do_plot:
		# After the loop, turn off interactive mode
		plt.ioff()
		plt.show()

