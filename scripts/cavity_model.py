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

def cavity_model(wav,temp,scaling,temp2,scaling2,surface_density,Jv_T,Jv_Scale):
	#Model CM
	#Regrid to wav with spectres
	kabs = spectres(wav,wave,kappa_abs)
	ksca = spectres(wav,wave,kappa_scat)

	F_source = blackbody(wav,Jv_T,Jv_Scale)

	model = (source_function(wav,temp,scaling,kabs,ksca,F_source)+source_function(wav,temp2,scaling2,kabs,ksca,F_source)) * np.exp(-surface_density*(kabs+ksca))
	return model

def cavity_model_windextinction(wav,temp,scaling,temp2,scaling2,surface_density,Jv_T,Jv_Scale,sigma_wind):
	#Model CMWIND
	#Regrid to wav with spectres
	kabs = spectres(wav,wave,kappa_abs)
	ksca = spectres(wav,wave,kappa_scat)

	F_source = blackbody(wav,Jv_T,Jv_Scale)*np.exp(-sigma_wind*(kabs+ksca))

	model = (source_function(wav,temp,scaling,kabs,ksca,F_source)+source_function(wav,temp2,scaling2,kabs,ksca,F_source)) * np.exp(-surface_density*(kabs+ksca))
	return model

def source_function(wav,temp,scaling,kabs,ksca,F_source):
	return (kabs*blackbody(wav,temp,scaling) + ksca*F_source)/(kabs+ksca)

def source_function_intensity(wav,temp,kabs,ksca,Jv):
	return (kabs*blackbody(wav,temp,1) + ksca*Jv)/(kabs+ksca)

def henyey_greenstein(theta,gsca):
	return 0.5*(1-gsca**2)/(1+gsca**2 - 2*gsca*np.cos(np.deg2rad(theta)))**1.5


def cavity_model_anisotropic(wav,temp,temp2,scaling2,surface_density,Jv_T):
	#Model CMWIND
	#Regrid to wav with spectres
	kabs = spectres(wav,wave,kappa_abs)
	ksca = spectres(wav,wave,kappa_scat)
	kext = kabs + ksca
	gsca = spectres(wav,wave,g)

	Jv_Omega = np.pi*R_star_r**2

	I_source = blackbody(wav,Jv_T,1)#*henyey_greenstein(theta_sca,gsca) *np.exp(-sigma_wind*(kabs+ksca))

	Jv = I_source/(np.pi*4)*Jv_Omega

	model = (aperture_area*source_function_intensity(wav,temp,kabs,ksca,Jv) + scaling2/aperture_area*source_function_intensity(wav,temp2,kabs,ksca,Jv))*np.exp(-surface_density*kext)
	return model






############################################################
#														   #	
#	 USER DEFINED PARAMETERS SECTION 1/2				   #
#														   #
############################################################

distance_dict = {}
distance_dict['L1448MM1'] = 293 #pc
distance_dict['BHR71'] = 176 #pc


COL_WIDTH = 20#for terminal printing


#IN PROGRESS:
#CM_ANI IS THE DEVELOPMENT FUNCTION.
global theta_sca
theta_sca = 90 #For the anisotropic scattering models.

au_per_Rsol = 214.94252873563218
r = 500 #au
r*= au_per_Rsol
R_star = 10 #R_sol
global R_star_r
global aperture_area
R_star_r = R_star/r



aperture_radius = 1 #arcsec
aperture_radius /= 206265 #radian
aperture_area = np.pi*aperture_radius**2/(3e18)**-2
#END IN PROGRESS

model_str = 'CM' #Options: CM, CMWIND, CM_ANI 
source_name = 'L1448MM1' #Options: L1448MM1, BHR71
kp5_filename = "/home/vorsteja/Documents/JOYS/Collaborator_Scripts/Sam_Extinction/KP5_benchmark_RNAAS.csv"
opacity_foldername = '/home/vorsteja/Documents/JOYS/JDust/optool_opacities/'

op_fn = opacity_foldername + sys.argv[1]
if sys.argv[1] =='':
	op_fn = ''
#op_fn = '' #If None then it is KP5.

#Wavelength to fit.
fit_wavelengths = [[4.92,27.5]]

#Wavelength ranges to calculate reduced chi2
chi_ranges = [[4.93,5.6],[7.7,8],[8.5,11.5],[12,14],[15.7,17],[19,20.5]]
	
	#19 January 2026, also in B2C_WIND.py
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

if model_str == 'CMWIND':

	model = cavity_model_windextinction
	model_parameters = ['T1','O1','T2','O2','SD_LOS','T_star','O_star','SD_WIND']

elif model_str == 'CM':
	model = cavity_model
	model_parameters = ['T1','O1','T2','O2','SD_LOS','T_star','O_star']

elif model_str == 'CM_ANI':
	model = cavity_model_anisotropic
	model_parameters = ['T1','T2','FF','SD_LOS','T_star']
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
	if os.path.isfile(aperture_filename):
		aper_names,aper_sizes,coord_list = read_aperture_ini(aperture_filename)

elif source_name == 'BHR71':
	output_foldername = '/home/vorsteja/Documents/JOYS/JDust/BHR71_scripts_12122025/BHR71/output/'

	fn_band_arr = ['ch1-short','ch1-medium','ch1-long','ch2-short','ch2-medium','ch2-long',
				'ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long'] 
	input_foldername = '/home/vorsteja/Documents/JOYS/JDust/BHR71_scripts_12122025/BHR71_unc/BHR71/spectra_werrors/circle/1.00_arcsec/spectra_sci/'
	aper_names = ['b1', 'b2', 'b3', 'b4', 'cr1', 'o5']

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


#For model CMWIND
##########################################################################
bounds_dict['ALL:ALL:CMWIND:T1'] = [0,uT_emit]
bounds_dict['ALL:ALL:CMWIND:O1'] = [0,uO_emit]
bounds_dict['ALL:ALL:CMWIND:SD_LOS'] = [0,uSD]
bounds_dict['ALL:ALL:CMWIND:T_star'] = [lT_star,uT_star]

#Bounds that are equal to other bounds.
bounds_dict['ALL:ALL:CMWIND:O_star'] = bounds_dict['ALL:ALL:CMWIND:O1']
bounds_dict['ALL:ALL:CMWIND:SD_WIND'] = bounds_dict['ALL:ALL:CMWIND:SD_LOS']
bounds_dict['ALL:ALL:CMWIND:T2'] = bounds_dict['ALL:ALL:CMWIND:T1']
bounds_dict['ALL:ALL:CMWIND:O2'] = bounds_dict['ALL:ALL:CMWIND:O1']


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

#For model CM_ANI
##########################################################################
bounds_dict['ALL:ALL:CM_ANI:T1'] = [0,uT_emit]
bounds_dict['ALL:ALL:CM_ANI:SD_LOS'] = [0,uSD]
bounds_dict['ALL:ALL:CM_ANI:FF'] = [0,uFF]
bounds_dict['ALL:ALL:CM_ANI:T_star'] = [lT_star,uT_star]

#Bounds that are equal to other bounds.
bounds_dict['ALL:ALL:CM_ANI:T2'] = bounds_dict['ALL:ALL:CM:T1']




############################################################
#														   #	
#	 INITIALIZE OPACITIES				 				   #
#														   #
############################################################
#These are globals, because they are often re-used in optimization.
global kappa_abs
global kappa_scat
global g

if op_fn:
	header,wave,kappa_abs,kappa_scat,g = read_optool(op_fn)
	opacity_str = op_fn.split('/')[-1].split('.dat')[0]
else:
	#If no opacity filename is specified, use KP5.
	file = kp5_filename  # OpTool version
	opacity_str = 'KP5'
	df=pd.read_csv(file)

	wave=df['wavelength'].values
	kappa_abs = df['kabs'].values
	kappa_scat = df['ksca'].values
	g = df['gsca'].values


############################################################
#														   #	
#	 FITTING AND OUTPUT				 				   	   #
#														   #
############################################################

title_columns = ['Source','Aperture'] + model_parameters

format_spec = f"{{:<{COL_WIDTH}}}"
title_print = ''.join([format_spec]*(len(model_parameters)+2)).format(*title_columns)

dustmodel_print = '#'*10 + '\t' + 'Solving for dust model: ' + opacity_str+ '\t' + '#'*10 

print('#'* int(1.2*len(dustmodel_print)))
print(dustmodel_print)
print('#'* int(1.2*len(dustmodel_print)))


print(title_print)

print('#'* len(title_print))

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
		fn_base = input_foldername + 'L1448MM1_aper%s.spectra'%(aperture)

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


	

	p0, single_bound_dict = grab_p0_bounds(source_name,aperture,model_parameters,model_str,p0_dict,bounds_dict)
	

	#aperture_radius *= d #au
	#aperture_area = np.pi*aperture_radius**2 #au^2
	#aperture_area *= 1.496e+13**2 #cm^2
	#mass = Sigma_abs * aperture_area * 5.267e-31


	#Fitting with lmfit.
	popt,pcov = fit_model(fit_um, fit_flux, fit_unc,model,p0,model_parameters,single_bound_dict)
	
	title_print = results_to_terminal_str(source_name,aperture,popt,pcov,model_parameters,COL_WIDTH)
	print(title_print)



	#Plot the results:
	plt.close()
	plt.figure(figsize = (10,8))
	plt.subplot(211)
	plt.plot(fit_um,fit_flux,label='continuum',color='black')
	#plt.scatter(u_use,f_rad,color='blue',s=0.2)
	plt.xlim(4.9,27)
	plt.plot(u_use,model(u_use,*popt),label='model',color='red')
	for fill_range in fit_wavelengths:
		plt.axvspan(fill_range[0],fill_range[1],facecolor='grey',alpha=0.3,edgecolor='grey')
	plt.suptitle('%s, Aperture: %s'%(source_name,aperture))
	#plt.title(r'$T_{\rm emit} =$' +'%.1f K '%(popt[0]) + r'$\Omega_{\rm emit}$ =' + '%.2E ster '%(popt[1]) + r'$\Sigma_{\rm d} =$' + '%.1E g cm-2 '%(popt[2]) + r'$T_{scat} =$' + '%d K '%(popt[3]) + r'$\Omega_{\rm scat} = $' + '%.2E ster'%(popt[-1]))
	plt.yscale('log')
	
	plt.xlabel('Wavelength (um)')
	plt.ylabel('Flux Density (W cm-2)')

	ylims = plt.gca().get_ylim()
	if model_str =='CM':


		kabs = spectres(u_use,wave,kappa_abs)
		ksca = spectres(u_use,wave,kappa_scat)


		#[p0_T1,p0_O1,p0_T2,p0_O2,p0_SD_LOS,p0_T_star,p0_O_star]

		#Plot the different components.
		T1 = popt[0]
		O1 = popt[1]
		T2 = popt[2]
		O2 = popt[3]

		sigma_LOS = popt[4]
		T_star = popt[5]
		O_star = popt[6]

		comp1 = kabs*blackbody(u_use,T1,O1)/(kabs+ksca)
		comp2 = kabs*blackbody(u_use,T2,O2)/(kabs+ksca)

		F_source = blackbody(u_use,T_star,O_star)
		scattering =  2*ksca*F_source/(kabs+ksca)

		wout_ext = comp1+comp2+scattering

		plt.fill_between(u_use,y1=wout_ext,y2= model(u_use,*popt),color='brown',alpha=0.3,label=r'$\Sigma_{\rm LOS} =$' + '%.2f'%(sigma_LOS*1e3) + r' mg cm$^{-2}$')
		plt.plot(u_use,comp1,color='green',linestyle='dashed',label='%d K'%(T1))
		plt.plot(u_use,comp2,color='red',linestyle='dashed',label='%d K'%(T2))
		plt.plot(u_use,scattering,color='blue',linestyle='dashed',label='%d K'%(T_star))
	plt.legend()
	plt.ylim([ylims[0],1.2*np.nanmax(wout_ext)])

	plt.subplot(212)

	plt.errorbar(fit_um,100*(fit_flux - model(fit_um,*popt))/fit_flux,yerr=100*np.abs(fit_unc/fit_flux),color='black',linestyle=None,zorder=0,lw=0.1)
	plt.scatter(fit_um,100*(fit_flux - model(fit_um,*popt))/fit_flux,color='red',s=0.2,zorder=1)
	plt.xlabel('Wavelength (um)')
	plt.ylabel('Residual Error %')
	plt.xlim(4.9,27)
	plt.axhline(0,color='grey',linestyle='dashed')
	plt.ylim(-100,100)

	plot_dir = output_foldername + opacity_str + '/'
	if not os.path.isdir(plot_dir):
		os.mkdir(plot_dir)


	chi2_red_dict = {}
	x = fit_um
	y = fit_flux
	yerr = fit_unc

	_, _,_,chi2_red = calculate_goodness_of_fit(x, y, yerr, model, popt,pcov)

	chi2_red_dict['all'] = chi2_red

	

	for um_min,um_max in chi_ranges:
		um_mid = (um_min+um_max)/2
		inds_chi = np.where(np.logical_and(fit_um>um_min,fit_um<um_max))

		x = fit_um[inds_chi]
		y = fit_flux[inds_chi]
		yerr = fit_unc[inds_chi]

		_, _,_,chi2_red = calculate_goodness_of_fit(x, y, yerr, model, popt,pcov)
		plt.axvspan(um_min,um_max,facecolor='green',alpha=0.3,edgecolor='grey')
		chi2_red_dict[str(round(um_mid,1))] = chi2_red

	chi2_str = ''
	for key1 in chi2_red_dict.keys():
		val1 = chi2_red_dict[key1]
		chi2_str += r'$\chi_{\rm red}$' + '(%s)=%.2f'%(key1,val1) + '\t'

	plt.text(5.3,75,chi2_str,fontsize=8)	

	show_apertures = []
	if aperture in show_apertures:
		plt.show()
	else:
		plt.savefig(plot_dir + '%s_%s_%s_%s.png'%(source_name,aperture,model_str,opacity_str),bbox_inches='tight',dpi=200)



#Saving columns:
#SOURCE, APERTURE, PSFSUB (Y/N), CORE1, CFRAC1, CORE2, CFRAC2, CORE3, CFRAC3, MANTLE1, MFRAC1, MANTLE2, MFRAC2, MANTLE3, MFRAC3, AMIN, AMAX, SLOPE, POROSITY, CHI2RED_ALL, [CHI2RED_XX] 

############################################################
#														   #	
#	 MCMC THINGS NOT WORKING YET				 		   #
#														   #
############################################################


'''
def log_likelihood(theta, x, y, yerr):
	#theta: T1, Scaling1, T2, Scaling2, sillicate scaling
	popt = theta[:-1]
	log_f = theta[-1]

	model = cavity_model(x,*popt)
	sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
	return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta):
	# print(theta) # Keep this commented unless actively debugging
	
	T1, O1, T2, O2, Sigma, Tscat, Oscat, log_f = theta
	
	#Define prior for each quantity.
		
	return log_p


	

def log_probability(theta, x, y, yerr):
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + log_likelihood(theta, x, y, yerr)

	do_mcmc = False
	if do_mcmc:
		x = fit_um
		y = fit_flux
		yerr= fit_unc

		print('Fit results: '+ str(popt))

		residual_e = y - model(x,*popt)
		residual_SSE = np.nansum(residual_e**2)

		print('residual_SSE: %.2E, log_f = %.2f'%(residual_SSE,np.log(residual_SSE)))
		#[10,1e-9,10,1e-9,1,1,1] this is the multiplication function for the initial conditions.

		N_walkers = 21

		pos = np.array(list(popt) + [np.log(residual_SSE)])+ np.array(list(popt)+[np.log(residual_SSE)]) * 5 * np.random.randn(N_walkers, len(popt) +1)
		nwalkers, ndim = pos.shape

		sampler = emcee.EnsembleSampler(
		nwalkers, ndim, log_probability, args=(x, y, yerr))
		sampler.run_mcmc(pos, 10000 , progress=True)
		plt.close()
		labels = ["T1", "O1", "T2", "O2", "Sigma",'Tscat', "Oscat",'log_f']
		fig, axes = plt.subplots(len(labels), figsize=(15, 7), sharex=True)
		samples = sampler.get_chain()
			

		T1_bounds = [30,120]
		O1_bounds = [0,1e-2]
		T2_bounds = [200,600]
		S2_bounds = [0,1e-2]
		Sigma = [0,1]
		Tscat_bounds = [500,2500]
		S2_bounds = [0,1e-2]
		log_f_bounds = [-200,200]

		bound_arr = [T1_bounds,O1_bounds,T2_bounds,S2_bounds,Sigma,Tscat_bounds,S2_bounds,log_f_bounds]


		for i in range(ndim):
			ax = axes[i]
			bounds = bound_arr[i]
			ax.plot(samples[:, :, i], "k", alpha=0.3,label='walkers')
			ax.set_xlim(0, len(samples))
			ax.set_ylabel(labels[i])
			#ax.axhspan(bounds[0], bounds[1], facecolor='grey', alpha=0.5,label='prior bounds')
			ax.set_ylim(np.nanmin(samples[:, :, i]),np.nanmax(samples[:, :, i]))
			#ax.yaxis.set_label_coords(-0.1, 0.5)
		axes[-1].set_xlabel("step number");

		plt.show()

		tau = sampler.get_autocorr_time()
		print('Autocorrelation time: '+ str(tau))

		flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
		print(flat_samples.shape)

		
		plt.close()
		fig = corner.corner(
			flat_samples, labels=labels) #, truths=[m_true, b_true, np.log(f_true)]
		plt.show()


'''