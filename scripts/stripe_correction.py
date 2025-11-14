from ifu_analysis.jdpsfsub import stripe_correction
from ifu_analysis.jdutils import get_JWST_PSF,define_circular_aperture,unpack_hdu,get_JWST_IFU_um,get_subcube_name
from ifu_analysis.jdcontinuum import automatic_continuum_cube
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
from pybaselines import Baseline
from tqdm import tqdm



sub_names =['short','medium','long']
subchannel_arr =  ['ch1-%s'%(x) for x in sub_names]+['ch2-%s'%(x) for x in sub_names]+['ch3-%s'%(x) for x in sub_names] +['ch4-%s'%(x) for x in sub_names]

subchannel_arr = ['ch4-long']

output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/'
saveto_plots = output_foldername

for subchannel in subchannel_arr:
	print('Doing subchannel %s'%(subchannel))
	hdu_fn = '/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_IFUAlign/L1448-mm_%s_s3d_LSRcorr.fits'%(subchannel)
	
	output_data = output_foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr.fits'%(subchannel)
	output_bkg = output_foldername + 'L1448-mm_%s_stripebkg.fits'%(subchannel)
	output_cont = output_foldername + 'L1448-mm_%s_cont.fits'%(subchannel)
	output_cont_stripecorr = output_foldername + 'L1448-mm_%s_cont_stripecorr.fits'%(subchannel)

	hdu = fits.open(hdu_fn)
	data_cube,unc_cube,dq_cube,hdr,um,shp = unpack_hdu(hdu_fn)
	wcs = WCS(hdr)
	wcs = wcs.dropaxis(2)
	#aper_coords = ['03h25m38.8898s','+30d44m05.612s']


	

	cont_cube,unc_cont_cube,dq_cont_cube = automatic_continuum_cube(hdu_fn,saveto = output_cont,saveto_plots =saveto_plots)
	n_sigma = 5
	cont_cube[cont_cube > n_sigma*unc_cont_cube] = np.nan
	cont_map = np.nansum(cont_cube,axis = 0)

	#This is for masking an aperture.
	#fwhm = 2*get_JWST_PSF(max(um))
	#RA, Dec = aper_coords
	#aper = define_circular_aperture(RA,Dec,fwhm)
	#mask_par = aper.to_pixel(wcs)
	#mask = np.mod(np.array(mask_par.to_mask(method='center').to_image(shp),dtype=int)+1,2)
	#mask = np.abs(mask-1)

	mask = np.zeros(np.shape(cont_cube), dtype=bool)
	mask[:,cont_map ==0] = True
	mask[np.isnan(cont_cube)] = True 
	cubeavg_wtd, cubebkg_wtd, bkgsub_wtd = stripe_correction(hdu,mask,saveto=[output_data,output_bkg])

	#Stripe correct continuum cubes.
		#Open
		#Subtract striped background
		#Save

	hdu_cont = fits.open(output_cont)
	hdu_cont[1].data = cont_cube - cubebkg_wtd
	hdu_cont.writeto(output_cont_stripecorr,overwrite=True)

