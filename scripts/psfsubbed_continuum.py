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


#subchannel_arr = ['ch3-long']
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/PSF_Subtraction/'
saveto_plots = None
verbose =True
do_cont_cube = True

for subchannel in subchannel_arr:
	print('Doing subchannel %s'%(subchannel))
	hdu_fn = output_foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_psfsub.fits'%(subchannel)

	saveto = output_foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_psfsub_contcube.fits'%(subchannel)

	if do_cont_cube:
		cont_cube,unc_cont_cube,dq_cont_cube = automatic_continuum_cube(fn=hdu_fn,saveto = saveto,verbose=True)
	else:
		cont_cube,unc_cont_cube,dq_cont_cube = [fits.open(saveto)[i].data for i in [1,2,3]]

	fn_no_psfsub = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/L1448-mm_%s_cont.fits'%(subchannel)
	cont_cube_nosub,unc_cont_cube_nosub,dq_cont_cube_nosub = [fits.open(fn_no_psfsub)[i].data for i in [1,2,3]]



	n_sigma = 5
	cont_cube[cont_cube < n_sigma*unc_cont_cube] = np.nan
	cont_map = np.nansum(cont_cube,axis = 0)


	n_sigma = 5
	cont_cube_nosub[cont_cube_nosub < n_sigma*unc_cont_cube_nosub] = np.nan
	cont_map_nosub = np.nansum(cont_cube_nosub,axis = 0)

	plt.figure(figsize = (16,9))
	
	plt.subplot(121)
	plt.imshow(cont_map_nosub,origin='lower',cmap='gist_stern')
	plt.title('Nosub')
	plt.subplot(122)
	plt.imshow(cont_map,origin='lower',cmap='gist_stern')
	plt.title('Sub')
	plt.show()
