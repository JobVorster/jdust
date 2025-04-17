from ifu_analysis.jdpsfsub import subtract_psf_cube
from ifu_analysis.jdutils import unpack_hdu

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import pandas as pd
from astropy.wcs import WCS


#Make this a loop through all subbands, with the right SNR_percentile.
subband_arr = ['3A','3B','3C','4A','4B','4C']
fn_band_arr = ['ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long']
fn_arr = ['/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_IFUAlign/L1448-mm_%s_s3d_LSRcorr.fits'%(x) for x in fn_band_arr]
SNR_percentile_arr = [99]*3 + [97.5]*3 
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/PSF_Subtraction/'

dosave = True
doplot = False

mask_method = 'APERTURE'
mask_par = None
aper_coords = ['03h25m38.8898s','+30d44m05.612s']


for filename,subband,SNR_percentile in zip(fn_arr,subband_arr,SNR_percentile_arr):

	#Fix this filename.
	if dosave:
		fn = filename.split('/')[-1].split('.fits')[0]
		saveto = output_foldername + fn + '_psf_options.csv'
	else:
		saveto = None

	print('Doing Subband %s'%(subband))

	data_cube,unc_cube,dq_cube,hdr,um,shp = unpack_hdu(filename)
	wcs = WCS(hdr)
	wcs = wcs.dropaxis(2)

	#psfsub_cube,x_offset_arr,y_offset_arr,scaling_arr = subtract_psf_cube(data_cube,unc_cube,subband,SNR_percentile)

	psfsub_cube,x_offset_arr,y_offset_arr,scaling_arr = subtract_psf_cube(um,data_cube,unc_cube,subband,
		mask_method,mask_par,aper_coords = aper_coords,wcs = wcs,saveto=saveto)

	#I still need to check if the psfsub works as it should.
	if doplot:
		plt.figure(figsize = (16,5))
		plt.subplot(131)
		plt.plot(um,x_offset_arr)
		plt.xlabel('Wavelength (um)')
		plt.ylabel('X Offset')
		plt.subplot(132)
		plt.plot(um,y_offset_arr)
		plt.xlabel('Wavelength (um)')
		plt.ylabel('Y Offset')
		plt.subplot(133)
		plt.plot(um,scaling_arr)
		plt.xlabel('Wavelength (um)')
		plt.ylabel('Scaling')
		plt.show()

	if dosave:
		hdu = fits.open(filename).copy()
		hdu[1].data = psfsub_cube
		hdu.writeto(output_foldername + fn + '_PSFsub.fits',overwrite=True)
	