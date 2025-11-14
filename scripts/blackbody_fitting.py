from ifu_analysis.jdspextract import load_spectra,merge_subcubes,read_aperture_ini
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from pybaselines import Baseline

input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/spectral_extraction/'
absorption_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/absorption/'

# importing H2O ice data
h2o_slav_15k = np.transpose(np.genfromtxt(absorption_foldername+ '2023-08-18_hdo-h2o_1-200_thin(118)_154_blcorr.csv', delimiter=','))
#Convert from wavenumber (cm-1) to micron
h2o_slav_15k[0] = 1e4/h2o_slav_15k[0]
h2o_slav_150k = np.transpose(np.genfromtxt(absorption_foldername +'2023-08-18_hdo-h2o_1-200_thin(27)_1500_blcorr.csv', delimiter=','))
#Convert from wavenumber (cm-1) to micron
h2o_slav_150k[1] = 1e4/h2o_slav_150k[1]

h2o_slav_15k = np.flip(h2o_slav_15k,axis=1)
h2o_slav_150k = np.flip(h2o_slav_150k,axis=1)


methanol_15k = np.transpose(np.genfromtxt(absorption_foldername+ 'methanol_1146_15.0K.csv', delimiter=','))
methanol_15k[0] = 1e4/methanol_15k[0]
methanol_15k = np.flip(methanol_15k,axis=1)


# importing silicates
adwin_silicates = np.transpose(np.genfromtxt(absorption_foldername + "adwin_silicates.csv", delimiter=",", skip_header=0))




#['BASE','PSFMODEL','PSFSUB']

# convert flux from Jy to W/cm^2
def convert_flux_wl(spec):
	'''
	Function by Katie Slavicinska.
	'''
	wl = spec[0]
	flux = spec[1]
	f = 2.998e8/(wl*10**(-6))
	new_flux = f * flux / 1e30
	return new_flux

	# blackbody function
def blackbody(wav, temp,scaling):
	'''
	Function by Katie Slavicinska.
	'''
	#FIX!! APRAD IS SET MANUALLY NOW.
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

def two_blackbodies(wav,temp1,scaling1,temp2,scaling2):
	return blackbody(wav,temp1,scaling1)+ blackbody(wav,temp2,scaling2)

def two_blackbodies_sillicate(wav,temp1,scaling1,temp2,scaling2,sillicate_scaling):
	absorption_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/absorption/'
	adwin_silicates = np.transpose(np.genfromtxt(absorption_foldername + "adwin_silicates.csv", delimiter=",", skip_header=0))
	# interpolating silicate data to observed miri wavelengths
	sil = 'PYROXENE'
	if sil =='OLIVINE':
		sillicate = np.interp(wav, adwin_silicates[0], adwin_silicates[1])
	if sil == 'PYROXENE':
		sillicate = np.interp(wav, adwin_silicates[0], adwin_silicates[2])
	return (blackbody(wav,temp1,scaling1)+ blackbody(wav,temp2,scaling2))*np.exp(-sillicate*sillicate_scaling)

def two_blackbodies_sillicate_water(wav,temp1,scaling1,temp2,scaling2,sillicate_scaling,water_scaling):
	absorption_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/absorption/'
	adwin_silicates = np.transpose(np.genfromtxt(absorption_foldername + "adwin_silicates.csv", delimiter=",", skip_header=0))

	h2o_slav_15k = np.transpose(np.genfromtxt(absorption_foldername+ '2023-08-18_hdo-h2o_1-200_thin(118)_154_blcorr.csv', delimiter=','))
	h2o_slav_150k = np.transpose(np.genfromtxt(absorption_foldername +'2023-08-18_hdo-h2o_1-200_thin(27)_1500_blcorr.csv', delimiter=','))

	h2o_slav_15k[0] = 1e4/h2o_slav_15k[0]
	h2o_slav_150k[0] = 1e4/h2o_slav_150k[0]
	h2o_slav_15k = np.flip(h2o_slav_15k,axis=1)
	h2o_slav_150k = np.flip(h2o_slav_150k,axis=1)


	# interpolating silicate data to observed miri wavelengths
	sil = 'PYROXENE'
	if sil =='OLIVINE':
		sillicate = np.interp(wav, adwin_silicates[0], adwin_silicates[1])
	elif sil == 'PYROXENE':
		sillicate = np.interp(wav, adwin_silicates[0], adwin_silicates[2])

	wat = 'WARM'
	if wat == 'WARM':
		h2o_slav = np.interp(wav, h2o_slav_15k[0], h2o_slav_15k[1])
	elif wat == 'HOT':
		h2o_slav = np.interp(wav, h2o_slav_150k[0], h2o_slav_150k[1])

	return (blackbody(wav,temp1,scaling1)+ blackbody(wav,temp2,scaling2))*np.exp(-sillicate*sillicate_scaling)*np.exp(-h2o_slav*water_scaling)



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
source_name = 'L1448MM1'

#TODO to continue:
	#Add flagging/fitting wavelengths --> This needs to be discussed in the meeting.
	#Why are there gaps in the spectra for certain apertures in the PSF subbed cubes?

fit_wavelengths = [[4.92,5.6],[5.61,6],[6.5,6.56],[7.3,7.605],[7.75,14.5],[16,27.5]]



aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt'%(source_name)
output_foldername ='/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/spectral_extraction/spectra_bb/'

if os.path.isfile(aperture_filename):
	aper_names,aper_sizes,coord_list = read_aperture_ini(aperture_filename)
for fitoption in ['BB+Sillicate','BB+Sillicate+Water']:
	for which_data in ['NOSUB','PSFSUB']:

		for aperture, ap_rad in zip(aper_names,aper_sizes):


			print('Doing Source: %s, Option: %s, Aperture: %s, Model: %s'%(source_name,which_data,aperture,fitoption))

			fn_base = input_foldername + 'BASE/L1448MM1_aper%s.spectra'%(aperture)
			fn_psfmodel = input_foldername + 'PSFMODEL/L1448MM1_aper%s.spectra'%(aperture)
			fn_psfsub = input_foldername + 'PSFSUB/L1448MM1_aper%s.spectra'%(aperture)

			sp_base,sp_psfmodel,sp_psfsub = merge_subcubes(load_spectra(fn_base)),merge_subcubes(load_spectra(fn_psfmodel)),merge_subcubes(load_spectra(fn_psfsub))



			um_base,flux_base,unc_base = [sp_base[x] for x in ['um', 'flux', 'flux_unc']]
			um_psfmodel,flux_psfmodel,unc_psfmodel = [sp_psfmodel[x] for x in ['um', 'flux', 'flux_unc']]
			um_psfsub,flux_psfsub,unc_psfsub = [sp_psfsub[x] for x in ['um', 'flux', 'flux_unc']]


			inds_base = np.where(flux_base > 3*unc_base)
			inds_psfmodel = np.where(flux_psfmodel > 3*unc_psfmodel)
			inds_psfsub = np.where(flux_psfsub > 3*unc_psfsub)
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


			f_rad = convert_flux_wl([u_use,f_use])
			unc_rad = convert_flux_wl([u_use,unc_use])

			um_cut = 27.5
			um_inds = np.where(np.logical_and(u_use < um_cut,np.isfinite(f_rad)))[0]

			u_use = u_use[um_inds]
			f_rad = f_rad[um_inds]
			unc_rad = unc_rad[um_inds]


			sillicate = np.interp(u_use, adwin_silicates[0], adwin_silicates[2])

			h2o_slav = np.interp(u_use, h2o_slav_15k[0], h2o_slav_15k[1])

			#plt.plot(u_use,np.exp(-sillicate))
			#plt.plot(u_use,np.exp(-h2o_slav))
			#plt.show()
			#exit()
		
			plt.close()
			plt.figure(figsize = (12,8))
			plt.subplot(211)
			plt.scatter(u_use,f_rad,color='orange',s=1,zorder=1,label='Data',alpha=0.5)

			cont_mask = get_continuum_mask(u_use,f_rad)

			if np.sum(np.isfinite(f_rad))>0:
				u_use[~cont_mask] = np.nan
				f_rad[~cont_mask] = np.nan
				unc_rad[~cont_mask] = np.nan

			plt.scatter(u_use,f_rad,color='red',s=1,zorder=1,label='Data Line Masked')
			plt.plot(u_use,3*unc_rad,color='grey',linestyle='dotted',label='Detection Limit')
			plt.ylim(1e-22,5e-17)
			


			fit_um = []
			fit_flux = []
			fit_unc = []

			for fill_range in fit_wavelengths:
				plt.axvspan(fill_range[0],fill_range[1],facecolor='grey',alpha=0.3,edgecolor='grey')
				um_cut,flux_cut = snippity(u_use,f_rad,fill_range)
				um_cut,unc_cut = snippity(u_use,unc_rad,fill_range)

				fit_um+=list(um_cut)
				fit_flux+=list(flux_cut)
				fit_unc+=list(unc_cut)

			fit_um = np.array(fit_um)
			fit_flux = np.array(fit_flux)
			fit_unc = np.array(fit_unc)

			if np.sum(np.isfinite(fit_flux))>0:

				valid = ~np.isnan(fit_flux)
			
				

				if fitoption =='BB+Sillicate':

					#T1, BB1 Scaling, T2, BB2 Scaling, Sil Scaling
					p0 = [700, 1e-9, 50, 1e-3, 0]
					#bounds = [(0,0,0,0,0,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf)]
					popt,pcov = curve_fit(two_blackbodies_sillicate,fit_um[valid],fit_flux[valid],p0=p0,sigma = fit_unc/fit_flux)
					plt.plot(u_use,two_blackbodies_sillicate(u_use,*popt),label='BB, T1=%d, S1=%.1E,T2=%d, S2=%.1E, Sil=%.1E'%(popt[0],popt[1],popt[2],popt[3],popt[4]))

					plt.plot(u_use,blackbody(u_use,popt[0],popt[1]),color='green',linestyle='dashed',label='BB1')
					plt.plot(u_use,blackbody(u_use,popt[2],popt[3]),color='yellow',linestyle='dashed',label='BB2')
					plt.plot(u_use,blackbody(u_use,popt[0],popt[1])+ blackbody(u_use,popt[2],popt[3]),color='magenta',label='BB Total',alpha=0.7,lw=1)

					plt.plot(u_use,(blackbody(u_use,popt[0],popt[1])+ blackbody(u_use,popt[2],popt[3]))*np.exp(-sillicate*popt[4]),color='brown',linestyle='dashed',label='Sillicate PYROXENE')
			




				elif fitoption == 'BB+Sillicate+Water':

					#T1, BB1 Scaling, T2, BB2 Scaling, Sil Scaling, Water Scaling 
					p0 = [700, 1e-9, 50, 1e-3, 1e-2, 1e-2]
					#bounds = [(0,0,0,0,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf)]
					popt,pcov = curve_fit(two_blackbodies_sillicate_water,fit_um[valid],fit_flux[valid],p0=p0)
					plt.plot(u_use,two_blackbodies_sillicate_water(u_use,*popt),label='BB, T1=%d, S1=%.1E,T2=%d, S2=%.1E, Sil=%.1E, Wat=%.1E'%(popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]))

					plt.plot(u_use,blackbody(u_use,popt[0],popt[1]),color='green',linestyle='dashed',label='BB1')
					plt.plot(u_use,blackbody(u_use,popt[2],popt[3]),color='yellow',linestyle='dashed',label='BB2')
					plt.plot(u_use,blackbody(u_use,popt[0],popt[1])+ blackbody(u_use,popt[2],popt[3]),color='magenta',label='BB Total',alpha=0.7,lw=1)

					plt.plot(u_use,(blackbody(u_use,popt[0],popt[1])+ blackbody(u_use,popt[2],popt[3]))*np.exp(-sillicate*popt[4]),color='brown',linestyle='dashed',label='Sillicate PYROXENE')
					plt.plot(u_use,(blackbody(u_use,popt[0],popt[1])+ blackbody(u_use,popt[2],popt[3]))*np.exp(-h2o_slav*popt[5]),color='blue',linestyle='dashed',label='Warm H2O')

			plt.grid(alpha=0.2,linestyle='dotted')
			plt.xlabel(r'Wavelength ($\mu$m)')
			plt.ylabel('Intensity (W cm-2)')
			plt.title('L1448MM Aperture: %s, Option: %s'%(aperture,which_data))
			plt.yscale('log')
			plt.legend(fontsize = 8)

			plt.subplot(212)


			

			if fitoption == 'BB+Sillicate':
				if np.sum(np.isfinite(fit_flux))>0:
					residual_e = fit_flux[valid] - two_blackbodies_sillicate(fit_um[valid],*popt)
					residual_SSE = np.nansum(residual_e**2)
					n = len(fit_flux[valid])
					k = len(popt)

					AIC = n*np.log(residual_SSE/n) + 2*k
					BIC = n*np.log(residual_SSE/n) + np.log(n)*k
				else:
					AIC = 0
					BIC = 0


				plt.scatter(u_use, (f_rad - two_blackbodies_sillicate(u_use,*popt))/f_rad,zorder=1,color='black',s=0.5,label='AIC: %d, BIC: %d'%(AIC,BIC))
				plt.errorbar(u_use, (f_rad - two_blackbodies_sillicate(u_use,*popt))/f_rad,yerr = unc_rad/f_rad,ecolor='black',zorder=0,linestyle='None')

			elif fitoption == 'BB+Sillicate+Water':
				if np.sum(np.isfinite(fit_flux))>0:
					residual_e = fit_flux[valid] - two_blackbodies_sillicate_water(fit_um[valid],*popt)
					residual_SSE = np.nansum(residual_e**2)
					n = len(fit_flux[valid])
					k = len(popt)

					AIC = n*np.log(residual_SSE/n) + 2*k
					BIC = n*np.log(residual_SSE/n) + np.log(n)*k
				else:
					AIC = 0
					BIC = 0



				plt.scatter(u_use, (f_rad - two_blackbodies_sillicate_water(u_use,*popt))/f_rad,zorder=1,color='black',s=0.5,label='AIC: %d, BIC: %d'%(AIC,BIC))
				plt.errorbar(u_use, (f_rad - two_blackbodies_sillicate_water(u_use,*popt))/f_rad,yerr = unc_rad/f_rad,ecolor='black',zorder=0,linestyle='None')

			plt.legend()
			plt.ylabel('Relative Error |Obs - Model|/Data')
			if np.sum(np.isfinite(f_rad))>0:
				plt.hlines(0, min(u_use),max(u_use),color='grey',linestyle='dashed')
			plt.xlabel(r'Wavelength ($\mu$m)')
			
			for fill_range in fit_wavelengths:
				plt.axvspan(fill_range[0],fill_range[1],facecolor='grey',alpha=0.3,edgecolor='grey')

			plt.ylim(+3,-3)
			plt.grid(alpha=0.2,linestyle='dotted')


			plt.savefig(output_foldername + '%s/%s/'%(fitoption,which_data) +  'L1448MM1_Ap%s.png'%(aperture),bbox_inches='tight',dpi=200)


