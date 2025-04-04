from ifu_analysis.jdpsfsub import subtract_psf_cube
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import pandas as pd


#Make this a loop through all subbands, with the right SNR_percentile.
subband_arr = ['3A','3B','3C','4A','4B','4C']
fn_band_arr = ['ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long']
fn_arr = ['/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_IFUAlign/L1448-mm_%s_s3d_LSRcorr.fits'%(x) for x in fn_band_arr]
SNR_percentile_arr = [99]*3 + [97.5]*3 
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/PSF_Subtraction/'

for filename,subband,SNR_percentile in zip(fn_arr,subband_arr,SNR_percentile_arr):

	print('Doing Subband %s'%(subband))

	hdu = fits.open(filename)
	data_cube = hdu[1].data
	unc_cube  =hdu[2].data
	psfsub_cube,x_offset_arr,y_offset_arr,scaling_arr = subtract_psf_cube(data_cube,unc_cube,subband,SNR_percentile)

	hdu[1].data = psfsub_cube

	#Fix this filename.
	fn = filename.split('/')[-1].split('.fits')[0]

	#hdu.writeto(output_foldername + fn + '_PSFsub.fits')
	df = pd.DataFrame(columns =['x_offset','y_offset','scaling'])
	df['x_offset'] = x_offset_arr
	df['y_offset'] = y_offset_arr
	df['scaling'] = scaling_arr
	df.to_csv(output_foldername + fn + '_psf_options.csv')
