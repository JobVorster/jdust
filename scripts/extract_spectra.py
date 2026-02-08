from ifu_analysis.jdspextract import *
from ifu_analysis.jdutils import *
from ifu_analysis.jdcontinuum import get_cont_cube
from astropy.wcs import WCS
from glob import glob
import os.path
from scipy.ndimage import median_filter
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt


do_flooring = False #Whether to set low flux measurements to the 3 sigma level.
do_zero_pointing = False
spectral_extraction_method = 'PIPELINE' #or NAIVE or PIPELINE
#sources: BHR71, HH211, L1448MM1, SerpensSMM1
source_name = 'L1448MM1'

fn_band_arr = ['ch1-short','ch1-medium','ch1-long','ch2-short','ch2-medium','ch2-long',
'ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long'] 

#or_output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/spectral_extraction/'
or_output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/'
aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt'%(source_name)

#BASE, BASE_CONT, PSFSUB, PSFSUB_CONT, PSFMODEL
for which_spectra in ['BASE','PSFMODEL']: #'BASE', 'PSFSUB', 'PSFMODEL'
	if which_spectra =='BASE':
		foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/'
		filenames = [foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr.fits'%(x) for x in fn_band_arr]
	elif which_spectra =='BASE_CONT':
		foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/'
		filenames = [foldername + 'L1448-mm_%s_cont_stripecorr.fits'%(x) for x in fn_band_arr]
	elif which_spectra == 'PSFSUB':
		foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/PSF_Subtraction/'
		filenames = [foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_psfsub.fits'%(x) for x in fn_band_arr]
	elif which_spectra =='PSFSUB_CONT':
		foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/PSF_Subtraction/'
		filenames = [foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_psfsub_contcube.fits'%(x) for x in fn_band_arr]
	elif which_spectra =='PSFMODEL':
		foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/PSF_Subtraction/'
		filenames = [foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_PSFcube.fits'%(x) for x in fn_band_arr]
	else:
		raise ValueError('Please specify a valid option to extract spectra (BASE, BASE_CONT, PSFSUB, PSFSUB_CONT, PSFMODEL)')
		exit()


	output_foldername = or_output_foldername + which_spectra + '/'


	subcubes = []
	for filename in filenames:
		subcubes.append(get_subcube_name(filename))
	subcubes = np.array(subcubes)


	if os.path.isfile(aperture_filename):
		aper_names,aper_sizes,coord_list = read_aperture_ini(aperture_filename)
		#plot_apertures(aper_names,aper_sizes,coord_list,ax1,wcs,color='White')
		#plt.savefig('./output-files/%s_continuum_SNR.jpg'%(source_name),dpi=300,bbox_inches='tight')

	if os.path.isfile(aperture_filename):
		for aper_name,aper_size,(RA_centre,Dec_centre) in zip(aper_names,aper_sizes,coord_list):
			print('Doing aperture %s of %s'%(aper_name,source_name))
			results = unstitched_spectrum_from_cube_list(filenames,RA_centre,Dec_centre,aper_size,method = spectral_extraction_method)

			if (1):
				
				plt.close()
				plt.figure(figsize = (16,4))
				for j in range(len(results['subcube_name'])):
					
					inds = np.where(results['um'][j] < 27.5)
					plt.plot(results['um'][j][inds],results['flux'][j][inds],label=subcubes[j])
					plt.plot(results['um'][j][inds],results['flux_unc'][j][inds],color='grey')
				plt.xlabel('Wavelength (um)')
				plt.suptitle('Source: %s, Aperture: %s UNSTITCHED'%(source_name,aper_name))
				plt.ylabel('Flux Density (Jy)')
				#plt.xscale('log')
				plt.yscale('log')
				plt.legend()
				plt.minorticks_on()
				plt.grid(which ='both',alpha=0.3)
				plt.savefig(output_foldername + '/%s_aper%s_spectrum_unstitched.jpg'%(source_name,aper_name),dpi=300,bbox_inches='tight')

			print('Aperture: %s, which_spectra: %s'%(aper_name,which_spectra))
			results_stitched = stitch_subcubes(results)
			results_merged = merge_subcubes(results_stitched,do_zero_pointing = do_zero_pointing)
			save_spectra(results,output_foldername + '/%s_aper%s_unstitched.spectra'%(source_name,aper_name))


			#The merged spectra gives a problem with saving.
			save_spectra(results_stitched,output_foldername +'/%s_aper%s.spectra'%(source_name,aper_name))

			
			if (1):
				plt.close()
				plt.figure(figsize = (16,4))
				for j in range(len(results_stitched['subcube_name'])):
					inds = np.where(results_stitched['um'][j] < 27)
					plt.plot(results_stitched['um'][j][inds],results_stitched['flux'][j][inds],label=subcubes[j])
					plt.plot(results_stitched['um'][j][inds],results_stitched['flux_unc'][j][inds],color='grey')
				plt.xlabel('Wavelength (um)')
				plt.suptitle('Source: %s, Aperture: %s STITCHED'%(source_name,aper_name))
				plt.ylabel('Flux Density (Jy)')
				plt.legend()
				#plt.xscale('log')
				plt.yscale('log')
				plt.minorticks_on()
				plt.grid(which ='both',alpha=0.3)
				plt.savefig(output_foldername + '/%s_aper%s_spectrum.jpg'%(source_name,aper_name),dpi=300,bbox_inches='tight')