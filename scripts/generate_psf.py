from ifu_analysis.jdpsfsub import subtract_psf_cube, get_aper_mask
from ifu_analysis.jdutils import unpack_hdu

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import pandas as pd
from astropy.wcs import WCS

#NOTES FOR WRITING THE PSF SUBTRACTION SECTION
#TODO: Delete as they are done.
#Write code to generate the psf subtracted and unsubtracted continuum maps.
#Fix the whole residual plot thing --> Maybe have the psf-subbed cube, and a mask in a different HDU.

#DONE:
	#Check how the line flagging works.
		#One could also implement a spectral extraction + baseline_classification
		#The problem is that the offsets are strange in line channels.
			#Do we want to track the offset variations precisely? Or just to some linear/polynomial order?
			#Spectral extraction is done with 2D gaussian fitting. It is not real spectral extraction (we do no aperture corrections). Rather we want to have a model of the cube PSF.
		#Then one has a continuum psf model.
	#Check the subtraction in line flagged channels.
	#Make sure the generate psf saves.
	

#WRITING NOTES: Delete as they are done.
#PROBLEM: We saw drifts in the centroid over channels. 
	#REASON: Sub-pixel changes in peak Line emission, wavelength dependent extinction (e.g. sillicate dip).
	#One interesting thing is that the centroid does not move in the y direction, but only in the x-direction along the jet.
	#This means that the drift is due to sub-pixel changes in the peak emission due to wavelength dependent extinction+emission.

#SIMPLEST FOR POSITION: STAR IS AT A SPECIFIC POSITION.
	#Even though the star is at a specific position, the emission peak is not determined by the star alone, but by the star-disk-envelope system.
	#Strong PSF is unresolved, but not a point source.
	#PLOTTING: One can make a peak position as a function of wavelength.
	#PLOTTING: One can make the scaling as a function of wavelength, with the continuum classified, and the baseline used.
#SIMPLEST FOR SCALING: ESTIMATE CONTINUUM CURVE.


#Make this a loop through all subbands, with the right SNR_percentile.

#
subband_arr = ['1A','1B','1C','2A','2B','2C','3A','3B','3C','4A','4B','4C']

fn_band_arr = ['ch1-short','ch1-medium','ch1-long','ch2-short','ch2-medium','ch2-long','ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long']

#Single subband
subband_arr = ['4C']
fn_band_arr = ['ch4-long']

fn_arr = ['/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/L1448-mm_%s_s3d_LSRcorr_stripecorr.fits'%(x) for x in fn_band_arr]
cont_fn_arr = ['/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/L1448-mm_%s_cont.fits'%(x) for x in fn_band_arr]
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/PSFSub_Docs/L1448MM1/'




dosave = True
doplot = True

mask_method = 'APERTURE'
mask_par = None
aper_coords = ['03h25m38.8898s','+30d44m05.612s']

mask_psfsub = True
bfe_factor = 0.35
base_factor = 1
mask_ratio = 0.5

for filename,cont_filename,subband in zip(fn_arr,cont_fn_arr,subband_arr):
	
	#Fix this filename.
	if dosave:
		fn = filename.split('/')[-1].split('.fits')[0]
		saveto = output_foldername + fn + '_psfsub.fits'
		saveto_psf_gen = output_foldername + fn + '_psf_options.csv'
		saveto_psf_cube = output_foldername + fn + '_PSFcube.fits'
	else:
		saveto = None

	print('Doing Subband %s'%(subband))

	data_cube,unc_cube,dq_cube,hdr,um,shp = unpack_hdu(filename)
	wcs = WCS(hdr)
	wcs = wcs.dropaxis(2)


	psfsub_cube,x_offset_arr,y_offset_arr,scaling_arr,fwhm_residual,fwhm_bfe_residual,bfe_residual = subtract_psf_cube(fn=filename,subband=subband,
		mask_method=mask_method,mask_par=mask_par,aper_coords = aper_coords,wcs = wcs,
		saveto=saveto,mask_psfsub=mask_psfsub,bfe_factor=bfe_factor,
		saveto_psf_gen=saveto_psf_gen,saveto_psf_cube=saveto_psf_cube,verbose=True)