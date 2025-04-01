from jdust_utils import *
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np


#Make this a loop through all subbands, with the right SNR_percentile.
subband_arr = ['4C']#['3A','3B','3C','4A','4B','4C']
fn_band_arr = ['ch4-long']#['ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long']
fn_arr = ['/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_IFUAlign/L1448-mm_%s_s3d_LSRcorr.fits'%(x) for x in fn_band_arr]
SNR_percentile_arr = [98.5]#[99]*3 + [97.5]*2 + [96] 
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/jwst-ifu-analysis/output-files/PSF_Subtraction/'

for filename,subband,SNR_percentile in zip(fn_arr,subband_arr,SNR_percentile_arr):

	print('Doing Subband %s'%(subband))

	hdu = fits.open(filename)
	data_cube = hdu[1].data
	unc_cube  =hdu[2].data
	psfsub_cube,x_offset_arr,y_offset_arr,scaling_arr = subtract_psf_cube(data_cube,unc_cube,subband,SNR_percentile)

	hdu[1].data = psfsub_cube

	#Fix this filename.
	fn = filename.split('/')[-1].split('.fits')[0]

	hdu.writeto(output_foldername + fn + '_PSFsub.fits')
