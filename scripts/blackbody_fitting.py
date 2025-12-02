from ifu_analysis.jdspextract import load_spectra,merge_subcubes,read_aperture_ini
from ifu_analysis.jdutils import define_circular_aperture
from ifu_analysis.jdfitting import *
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from pybaselines import Baseline
import corner
import emcee
from astropy.wcs import WCS
from astropy.io import fits
import pandas as pd

def parse_fit_results(source_name,aperture,which_data,fitoption,popt,pcov,goodness_of_fit,fit_successful = False):
	
	#fitoptions = ['BB','BB-SP','BB-SO','BB-SPO',
	#		  'BB-SPO-WW', 'BB-SPO-WH','BB-SPO-WWH',
	#		  '2BB','2BB-SP','2BB-SO','2BB-SPO',
	#		  '2BB-SPO-WW','2BB-SPO-WH','2BB-SPO-WWH']


	col_arr = [
	'Source Name', #Ind 0
	'Aperture', #Ind 1
	'PSF Subtracted (T/F)', #Ind 2
	'Model', #Ind 3
	'Fit Successful (T/F)', #Ind 4
	'BB1T', #Ind 5
	'lsqunc BB1T', #Ind 6
	'BB1S', #Ind 7
	'lsqunc BB1S', #Ind 8
	'BB2T', #Ind 9
	'lsqunc BB2T', #Ind 10
	'BB2S', #Ind 11
	'lsqunc BB2S', #Ind 12
	'Sil Oliv', #Ind 13
	'lsqunc Sil Oilv', #Ind 14
	'Sil Pyro', #Ind 15
	'lsqunc Sil Pyro', #Ind 16
	'H2O 15K', #Ind 17
	'lsqunc H2O 15K', #Ind 18
	'H2O 150K', #Ind 19
	'lsqunc H2O 150K', #Ind 20
	'chi2', #Ind 21
	'chi2_reduced', #Ind 22 
	'AIC', #Ind 23
	'BIC' #Ind 24
	]

	fit_results = ['N/A']*len(col_arr)

	#Metadata
	fit_results[0] = source_name
	fit_results[1] = aperture
	if which_data == 'PSFSUB':
		fit_results[2] = True
	else:
		fit_results[2] = False
	fit_results[3] = fitoption

	if not fit_successful:
		fit_results[4] = False
		return np.array(fit_results,dtype=str)
	else:

		#Then the fit worked.
		fit_results[4] = True

		#Turn the covariance matrix into the uncertainties --> This assumes a zero covariance.
		#We will follow up the chosen models with MCMC simulations in any way to get true uncertainties.
		lsqunc = np.sqrt(np.diag(pcov)) 

		#We can save the goodness of fit measures already:
		AIC, BIC,chi2,chi2_red = goodness_of_fit
		fit_results[21] = chi2
		fit_results[22] = chi2_red
		fit_results[23] = AIC
		fit_results[24] = BIC

		#Split the model code.
		#If it has length 1, it is either 1 BB or 2 BB.
		descr = fitoption.split('-')
		
		#This procedure is the careful mapping of popt indices for each model type. 
		#This function needs to be rewritten if the model types are changed.
		#Is also extremely error prone if indices are mistakenly used.
		#Probably this would work better with a dictionary or something.

		#First, there will always be at least 1 BB.
		fit_results[5] = popt[0]
		fit_results[6] = lsqunc[0]
		fit_results[7] = popt[1]
		fit_results[8] = lsqunc[1]

		#If there are two blackbodies.
		if descr[0] == '2BB':
			fit_results[9] = popt[2]
			fit_results[10] = lsqunc[2]
			fit_results[11] = popt[3]
			fit_results[12] = lsqunc[3]

		if len(descr) == 1:
			#If there are no other model components, just return the results.
			return np.array(fit_results,dtype=str)
		else:
			#The second is always sillicates (there are no options with water and no sillicates).
			#So it is either Olivine only (code SO), Pyroxene only (code SP) or both (code SPO).

			#Now the popt index depends on whether its one or two blackbodies.
			if descr[1] == 'SO':
				if descr[0] == 'BB':
					fit_results[13] = popt[2]
					fit_results[14] = lsqunc[2]
				else:
					fit_results[13] = popt[4]
					fit_results[14] = lsqunc[4]
			#At least the logic is exactly the same for the pyroxene case.
			elif descr[1] =='SP':
				if descr[0] == 'BB':
					fit_results[15] = popt[2]
					fit_results[16] = lsqunc[2]
				else:
					fit_results[15] = popt[4]
					fit_results[16] = lsqunc[4]
			elif descr[1] =='SPO':
				if descr[0] == 'BB':
					fit_results[13] = popt[2]
					fit_results[14] = lsqunc[2]
					fit_results[15] = popt[3]
					fit_results[16] = lsqunc[3]
				else:
					fit_results[13] = popt[4]
					fit_results[14] = lsqunc[4]
					fit_results[15] = popt[5]
					fit_results[16] = lsqunc[5]
			else:
				raise ValueError('Something went wrong in the model codes when parsing the fit results. Exiting...')


			#Gross. Now it will only get worse. Time to check if there is water.
			#This should really be rewritten. But I am just trying to get it working now while my 7 month old is sleeping.

			if len(descr) ==2:
				return np.array(fit_results,dtype=str)
			else:
				#The third is always water, and the second is always sillicates.
				#The water can be warm (code WW), hot (code WH) or both (code WWH).
				#The second is always sillicates
				#So it is either Olivine only (code SO), Pyroxene only (code SP) or both (code SPO).
				#Then there is whether there are 2 BBs or not --> Twelve options in all. 

				#BUT, we can take advantage of that the sillicates and BBs are already set, and that we know the relative water postions.
				if descr[2] == 'WW':
					fit_results[17] = popt[-1]
					fit_results[18] = lsqunc[-1]
				elif descr[2] == 'WH':
					fit_results[19] = popt[-1]
					fit_results[20] = lsqunc[-1]
				else:
					fit_results[17] = popt[-2]
					fit_results[18] = lsqunc[-2]
					fit_results[19] = popt[-1]
					fit_results[20] = lsqunc[-1]
				#BOOM! Done.
				#TODO: Write code that does not look like a 12-year old wrote it.

				return np.array(fit_results,dtype=str)



def plot_fit(um, flux, flux_unc,fitoption,fit_models_dict,popt,pcov, fit_wavelengths,N_sigma_clip, source_name, aperture,which_data,ylim=[1e-22,5e-17],AIC =None,BIC = None,cont_mask=None,saveto=None):
	aanda_col_width = 3.46457 #inches
	model = fit_models_dict[fitoption]
	plt.close()
	plt.figure(figsize = (2*aanda_col_width,1.5*aanda_col_width))

	plt.subplot(211)
	if len(cont_mask) > 0:
		plt.plot(um[cont_mask],flux[cont_mask],color = 'black',label ='Continuum',zorder=0)
		plt.scatter(um[~cont_mask],flux[~cont_mask],color = 'blue',label ='Lines',s=0.3,zorder=1)
	else:
		plt.plot(um,flux,color = 'black',label ='data')
	plt.plot(um,flux_unc*N_sigma_clip,color='grey',label = '%d'%(N_sigma_clip) + r'$\sigma_{F_{\nu}}$')
	plt.plot(um,model(um,*popt),color='magenta',lw = 2,label=fitoption)

	plt.grid(alpha=0.2,linestyle='dotted')
	plt.xlabel(r'Wavelength ($\mu$m)')
	plt.ylabel(r'Intensity (W cm-2)')
	plt.title('%s, Aperture: %s, Option: %s'%(source_name,aperture,which_data))
	plt.ylim(ylim)
	plt.yscale('log')

	#Plot the model subcomponents.
	plot_subcomponents(um, flux, flux_unc,fitoption,fit_models_dict,popt,pcov)


	plt.legend(fontsize = 8)
	

	for fill_range in fit_wavelengths:
		plt.axvspan(fill_range[0],fill_range[1],facecolor='grey',alpha=0.3,edgecolor='grey')

	

	plt.subplot(212)
	if len(cont_mask) > 0:
		plt.scatter(um[cont_mask], (flux[cont_mask] - model(um[cont_mask],*popt))/flux[cont_mask],zorder=1,color='black',s=0.5,label='AIC: %d, BIC: %d'%(AIC,BIC))
		plt.errorbar(um[cont_mask], (flux[cont_mask] - model(um[cont_mask],*popt))/flux[cont_mask],yerr = flux_unc[cont_mask]/flux[cont_mask],ecolor='black',zorder=0,linestyle='None')
	else:
		plt.scatter(um, (flux - model(um,*popt))/flux,zorder=1,color='black',s=0.5,label='AIC: %d, BIC: %d'%(AIC,BIC))
		plt.errorbar(um, (flux - model(um,*popt))/flux,yerr = flux_unc/flux,ecolor='black',zorder=0,linestyle='None')
	plt.legend()
	plt.ylabel('Relative Error |Obs - Model|/Data')
	if np.sum(np.isfinite(flux))>0:
		plt.hlines(0, min(um),max(um),color='grey',linestyle='dashed')
	plt.xlabel(r'Wavelength ($\mu$m)')
	
	for fill_range in fit_wavelengths:
		plt.axvspan(fill_range[0],fill_range[1],facecolor='grey',alpha=0.3,edgecolor='grey')

	plt.ylim(+3,-3)
	plt.grid(alpha=0.2,linestyle='dotted')

	if saveto:
		plt.savefig(saveto,bbox_inches='tight',dpi=200)
	else:
		plt.show()

def extinction(wave1,av): 
  extinc=np.interp(wave1,wave,Av)*av
  return (extinc)
				    

#Here I need to put all the fitting subcomponents plotting.
#WRITE THIS FUNCTION.
#This function also needs some customization by the user.
def plot_subcomponents(um, flux, flux_unc,fitoption,fit_models_dict,popt,pcov):
	model = fit_models_dict[fitoption]

	#need some sort of algorithm to plot fitting parameters.


	if fitoption == 'BB+Sillicate':
		T1, S1, T2, S2, Sil_scaling = popt
		#BB1
		plt.plot(um,blackbody(um,T1,S1),color='green',linestyle='dashed',label='BB1, T = %d K, S = %.2E'%(T1,S1))
		#BB2
		plt.plot(um,blackbody(um,T2,S2),color='red',linestyle='dashed',label='BB2, T = %d K, S = %.2E'%(T2,S2))
		#Total BB
		plt.plot(um,two_blackbodies(um,T1,S1,T2,S2),label='Total emission')

		#Sillicate absorption
		#Only one absorption, so no need to plot it.
		plt.plot(um,two_blackbodies_sillicate(um,*popt),label='Sil Scaling = %.2E'%(Sil_scaling),color='brown',linestyle='dashed')

		#What to plot here.
	elif fitoption == 'BB+Sillicate+Water':
		return None
	else:
		ValueError('Please specify a valid fitoption to plot.')


input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/spectral_extraction/'
absorption_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/absorption/'
fn_h2o15K = '2023-08-18_hdo-h2o_1-200_thin(118)_154_blcorr.csv'
fn_h2o150K = '2023-08-18_hdo-h2o_1-200_thin(27)_1500_blcorr.csv'
fn_sillicates = "adwin_silicates.csv"


#TODO to continue:
	#Add flagging/fitting wavelengths --> This needs to be discussed in the meeting.
	#Why are there gaps in the spectra for certain apertures in the PSF subbed cubes?


source_name = 'L1448MM1'
do_sigma_clip = True
do_extinction_correction = True
N_sigma_clip = 3


#Wavelength to fit.
fit_wavelengths = [[4.92,5.6],[5.61,6],[6.5,6.56],[7.3,7.605],[7.75,14.5],[16,27.5]]

#Data subgroups to iterate over.
which_datas = ['NOSUB','PSFSUB']



#Which model options to fit.
fitoptions = ['BB','BB-SP','BB-SO','BB-SPO',
			  'BB-SPO-WW', 'BB-SPO-WH','BB-SPO-WWH',
			  '2BB','2BB-SP','2BB-SO','2BB-SPO',
			  '2BB-SPO-WW','2BB-SPO-WH','2BB-SPO-WWH']
fitmodels = [blackbody, one_blackbody_pyroxene, one_blackbody_olivine, one_blackbody_pyroxene_olivine, 
			one_blackbody_pyroxene_olivine_15K, one_blackbody_pyroxene_olivine_150K, one_blackbody_pyroxene_olivine_15K_150K,
			two_blackbodies, two_blackbodies_pyroxene, two_blackbodies_olivine, two_blackbodies_pyroxene_olivine,
			two_blackbodies_pyroxene_olivine_15K, two_blackbodies_pyroxene_olivine_150K, two_blackbodies_pyroxene_olivine_15K_150K]

fit_models_dict = {}
for fitoption, fitmodel in zip(fitoptions,fitmodels):
	fit_models_dict[fitoption] = fitmodel

#Initial guesses for the different fit options.

T1_guess = 800
S1_guess = 1e-5
T2_guess = 300
S2_guess = 1e-5
Sil_guess = 1e-1
Wat_guess = 1e-1


#This is a lazy little way to code the initial conditions.
#Its based on the names of the models.
#The name always goes blackbody type - Sillicates - Water
#Based on the number of blackbodies there is a p0 guess.
#Then based on the length of the sillicate and water, there is either one or two guesses added.
p0_dict = {}
for fitoption in fitoptions:
	p0 = []
	descr = fitoption.split('-')
	if descr[0] == 'BB':
		p0 += [T1_guess, S1_guess]
	else:
		p0 += [T1_guess,S1_guess,T2_guess,S2_guess]
	if len(descr) >1:
		p0 += [Sil_guess]*(len(descr[1])-1)
	if len(descr) == 3:
		p0 += [Wat_guess]*(len(descr[2])-1)
	else:
		ValueError('There is some sort of issue with the way the models are named. Either change the model name, or the initial guess initialization.')
	p0_dict[fitoption] = p0



#Define aperture filename (for aperture names and radii).
aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt'%(source_name)

#Filename where to save results (with subfiles for each which_data and fitoption precreated).
output_foldername ='/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/spectral_extraction/spectra_bb/'


#Filename of extinction corrections.
if do_extinction_correction:
	fn_extinction = '/home/vorsteja/Documents/JOYS/Collaborator_Scripts/Gabriella_L1448MM/Av_map_fit_SNR30_KP5.fits'

#Columns for saving the fit results.
col_arr = [
	'Source Name', #Ind 0
	'Aperture', #Ind 1
	'PSF Subtracted (T/F)', #Ind 2
	'Model', #Ind 3
	'Fit Successful (T/F)', #Ind 4
	'BB1T', #Ind 5
	'lsqunc BB1T', #Ind 6
	'BB1S', #Ind 7
	'lsqunc BB1S', #Ind 8
	'BB2T', #Ind 9
	'lsqunc BB2T', #Ind 10
	'BB2S', #Ind 11
	'lsqunc BB2S', #Ind 12
	'Sil Oliv', #Ind 13
	'lsqunc Sil Oilv', #Ind 14
	'Sil Pyro', #Ind 15
	'lsqunc Sil Pyro', #Ind 16
	'H2O 15K', #Ind 17
	'lsqunc H2O 15K', #Ind 18
	'H2O 150K', #Ind 19
	'lsqunc H2O 150K', #Ind 20
	'chi2', #Ind 21
	'chi2_reduced', #Ind 22 
	'AIC', #Ind 23
	'BIC' #Ind 24
	]

def list_to_str(list1,sep=','):
	st = ''
	for i,item in enumerate(list1):
		if i != len(list1)-1:
			st += item +sep
		else:
			st += item +'\n'
	return st

results_filename = output_foldername + 'fit_results.csv'
if os.path.exists(results_filename):
  os.remove(results_filename)

f = open(results_filename,'w+')
f.write(list_to_str(col_arr))
f.close()

extinction_arr = []


if os.path.isfile(aperture_filename):
	aper_names,aper_sizes,coord_list = read_aperture_ini(aperture_filename)
	for aper_name, aper_size, coords in zip(aper_names,aper_sizes,coord_list):
		if do_extinction_correction:
			hdu_ext = fits.open(fn_extinction)
			wcs = WCS(hdu_ext[0].header)
			ext = hdu_ext[0].data 



			shp = np.shape(ext)
			aper = define_circular_aperture(coords[0],coords[1],aper_size) #Skyaperture 
			mask_par = aper.to_pixel(wcs)
			aper_mask = np.mod(np.array(mask_par.to_mask(method='center').to_image(shp),dtype=int),2)
			inds = np.where(aper_mask ==1)
			extinction_arr.append(np.nanmean(ext[inds]))
if do_extinction_correction:
	print(extinction_arr)	


for fitoption in fitoptions:

	descr = fitoption.split('-')
	#Only do 2BB fits.
	#if descr[0] == 'BB':
	#	continue	

	for which_data in which_datas:

		#Only psfsubbed:
		if which_data == 'NOSUB':
			continue

		for i_ap, (aperture, ap_rad) in enumerate(zip(aper_names,aper_sizes)):

			#if ('A' in aperture) or ('C' in aperture):
			#	continue

			print('Doing Source: %s, Fit option: %s, Aperture: %s, Model: %s'%(source_name,which_data,aperture,fitoption))

			#Filenames of spectra extracted and saved with the jdspextract functions.
			fn_base = input_foldername + 'BASE/L1448MM1_aper%s.spectra'%(aperture)
			fn_psfmodel = input_foldername + 'PSFMODEL/L1448MM1_aper%s.spectra'%(aperture)
			fn_psfsub = input_foldername + 'PSFSUB/L1448MM1_aper%s.spectra'%(aperture)

			#Formatting: Merge the arrays of the stiched spectra.
			sp_base = merge_subcubes(load_spectra(fn_base))
			sp_psfmodel = merge_subcubes(load_spectra(fn_psfmodel))
			sp_psfsub = merge_subcubes(load_spectra(fn_psfsub))

			#Extract wavelengths, fluxes and uncertainties.
			um_base,flux_base,unc_base = [sp_base[x] for x in ['um', 'flux', 'flux_unc']]
			um_psfmodel,flux_psfmodel,unc_psfmodel = [sp_psfmodel[x] for x in ['um', 'flux', 'flux_unc']]
			um_psfsub,flux_psfsub,unc_psfsub = [sp_psfsub[x] for x in ['um', 'flux', 'flux_unc']]

			#Sigma clipping.
			#TODO: Do the sigma clipping at the stitching to have correct uncertainties.
			if do_sigma_clip:
				inds_base = np.where(flux_base > N_sigma_clip*unc_base)
				inds_psfmodel = np.where(flux_psfmodel > N_sigma_clip*unc_psfmodel)
				inds_psfsub = np.where(flux_psfsub > N_sigma_clip*unc_psfsub)

				#Flag sigma_clipped measurements.
				if which_data =='NOSUB':
					u_use = um_base[inds_base]
					f_use = flux_base[inds_base]
					unc_use = unc_base[inds_base]
				elif which_data =='PSFSUB':
					u_use = um_psfsub[inds_psfsub]
					f_use = flux_psfsub[inds_psfsub]
					unc_use = unc_psfsub[inds_psfsub]
				else:
					raise ValueError('Please specify which data would be used (BASE/PSFSUB).')
					exit()
			else:
				if which_data =='NOSUB':
					u_use = um_base
					f_use = flux_base
					unc_use = unc_base
				elif which_data =='PSFSUB':
					u_use = um_psfsub
					f_use = flux_psfsub
					unc_use = unc_psfsub
				else:
					raise ValueError('Please specify which data would be used (BASE/PSFSUB).')
					exit()

			if do_extinction_correction:
				ext_app = extinction_arr[i_ap]

				file = "/home/vorsteja/Documents/JOYS/Collaborator_Scripts/Sam_Extinction/KP5_benchmark_RNAAS.csv"  # OpTool version

				df=pd.read_csv(file)

				wave=df['wavelength']
				Av=df['Av']
				ext_um = extinction(u_use,ext_app)
				f_use *= np.exp(ext_um/1.086)
				unc_use	*= np.exp(ext_um/1.086)

			


			#Convert from Jy to W cm-2
			f_rad = convert_flux_wl(u_use,f_use)
			unc_rad = convert_flux_wl(u_use,unc_use)

			#The MIRI MRS detector is bad above 27.5 um, flag all data above it.
			um_cut = 27.5
			um_inds = np.where(np.logical_and(u_use < um_cut,np.isfinite(f_rad)))[0]

			u_use = u_use[um_inds]
			f_rad = f_rad[um_inds]
			unc_rad = unc_rad[um_inds]


			# importing H2O ice data
			#I reimport at every loop because the length changes due to interpolation.
			wavenumber_h2o15K, tau_h2o15K = np.transpose(np.genfromtxt(absorption_foldername+ fn_h2o15K, delimiter=','))
			wavenumber_h2o150K, tau_h2o150K = np.transpose(np.genfromtxt(absorption_foldername +fn_h2o150K, delimiter=','))
			wavenumber_sil, tau_olivine, tau_pyroxene = np.transpose(np.genfromtxt(absorption_foldername + fn_sillicates, delimiter=",", skip_header=0))

			#Reformat absorption data.
			_, tau_h2o15K = parse_absorption(wavenumber_h2o15K,tau_h2o15K,interp_um = u_use,do_flip=True,wavenumber=True)
			_, tau_h2o150K = parse_absorption(wavenumber_h2o150K,tau_h2o150K,interp_um = u_use,do_flip=True,wavenumber=True)
			_, tau_olivine = parse_absorption(wavenumber_sil,tau_olivine,interp_um = u_use)
			_, tau_pyroxene = parse_absorption(wavenumber_sil,tau_pyroxene,interp_um = u_use)

			#Mask emission lines.
			cont_mask = get_continuum_mask(u_use,f_rad)

			u_noline = np.array(u_use)
			f_noline = np.array(f_rad)
			unc_noline = np.array(unc_rad)

			if np.sum(np.isfinite(f_rad))>0:
				u_noline[~cont_mask] = np.nan
				f_noline[~cont_mask] = np.nan
				unc_noline[~cont_mask] = np.nan

			#We will only fit within use specified ranges. Our model does not aim to explain all data.
			fit_um = []
			fit_flux = []
			fit_unc = []

			for fill_range in fit_wavelengths:
				plt.axvspan(fill_range[0],fill_range[1],facecolor='grey',alpha=0.3,edgecolor='grey')

				#Cuts out data of specific range.
				um_cut,flux_cut = snippity(u_noline,f_noline,fill_range)
				um_cut,unc_cut = snippity(u_noline,unc_noline,fill_range)

				#Appends to master array.
				fit_um+=list(um_cut)
				fit_flux+=list(flux_cut)
				fit_unc+=list(unc_cut)

			fit_um = np.array(fit_um)
			fit_flux = np.array(fit_flux)
			fit_unc = np.array(fit_unc)


			#Only run code if data exists.
			if np.sum(np.isfinite(fit_flux))>0:

				#Ignore nan values.
				valid = ~np.isnan(fit_flux)
				fit_um = fit_um[valid]
				fit_flux = fit_flux[valid]
				fit_unc = fit_unc[valid]
				
				model = fit_models_dict[fitoption]

				p0 = p0_dict[fitoption]

				popt,pcov = fit_model(fit_um, fit_flux, fit_unc,model,p0)

				try:
					popt,pcov = fit_model(fit_um, fit_flux, fit_unc,model,p0)
					goodness_of_fit = calculate_goodness_of_fit(fit_um, fit_flux,fit_unc, model, popt,pcov)
					AIC, BIC,chi2,chi2_red = goodness_of_fit
					print('Fit successfull (reduced chi2: %.2f) with popt: %s'%(chi2_red,str(['%.2E'%(x) for x in popt])))
					fit_results = parse_fit_results(source_name,aperture,which_data,fitoption,popt,pcov,goodness_of_fit,fit_successful = True)

					f = open(results_filename,'a')
					f.write(list_to_str(fit_results))
					f.close()

					saveto = output_foldername + '%s_%s_%s_%s_fit.png'%(source_name,aperture,which_data,fitoption)
					plot_fit(u_use, f_rad, unc_rad,fitoption,fit_models_dict,popt,pcov, fit_wavelengths,N_sigma_clip, source_name, aperture,which_data,AIC=AIC,BIC=BIC,cont_mask=cont_mask,saveto=saveto)
				except:
					fit_results = parse_fit_results(source_name,aperture,which_data,fitoption,popt=[],pcov=[],goodness_of_fit=[],fit_successful = False)
					f = open(results_filename,'a')
					f.write(list_to_str(fit_results))
					f.close()
					print('No fit was obtained. Continuing to next aperture model combination...')
					continue
				



'''
#DO NOT DELETE.
#MCMC IMPLEMENTATION.
do_mcmc = False
if do_mcmc:
	x = fit_um
	y = fit_flux
	yerr= fit_unc

	print('Fit results: '+ str(popt))

	residual_e = y - two_blackbodies_sillicate_water(x,*popt)
	residual_SSE = np.nansum(residual_e**2)

	print('residual_SSE: %.2E, log_f = %.2f'%(residual_SSE,np.log(residual_SSE)))
	#[10,1e-9,10,1e-9,1,1,1] this is the multiplication function for the initial conditions.

	N_walkers = 21

	pos = np.array(list(popt) + [np.log(residual_SSE)])+ np.array([10,1e-9,10,1e-9,1,1,1]) * 5 * np.random.randn(N_walkers, len(popt) +1)
	nwalkers, ndim = pos.shape

	sampler = emcee.EnsembleSampler(
	nwalkers, ndim, log_probability, args=(x, y, yerr))
	sampler.run_mcmc(pos, 1000 , progress=True)
	plt.close()
	labels = ["tbb1", "sbb1", "tbb2", "sbb2", "sill_scaling",'water_scaling', "logf"]
	fig, axes = plt.subplots(len(labels), figsize=(15, 7), sharex=True)
	samples = sampler.get_chain()
		

	tbb1_bounds = [400,2500]
	sbb1_bounds = [0,1e-2]
	tbb2_bounds = [15,400]
	sbb2_bounds = [0,1e-2]
	sillicate_scaling_bounds = [0,1e2]
	water_scaling_bounds = [0,1e2]
	log_f_bounds = [-200,200]

	bound_arr = [tbb1_bounds,sbb1_bounds,tbb2_bounds,sbb2_bounds,sillicate_scaling_bounds,water_scaling_bounds,log_f_bounds]


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




	exit()
'''