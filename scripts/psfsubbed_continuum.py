from ifu_analysis.jdpsfsub import stripe_correction
from ifu_analysis.jdutils import get_JWST_PSF,define_circular_aperture,unpack_hdu,get_JWST_IFU_um,get_subcube_name
from ifu_analysis.jdcontinuum import automatic_continuum_cube,cont_integrated_map
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
verbose =False
do_cont_cube = True
do_cont_cube_nosub = True

save_moment_maps = True

for subchannel in subchannel_arr:
	print('Doing subchannel %s'%(subchannel))
	hdu_fn = output_foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_psfsub.fits'%(subchannel)
	hdr = fits.open(hdu_fn)[1].header
	um = get_JWST_IFU_um(hdr)
	wcs = WCS(hdr)
	wcs_2D = wcs.dropaxis(2)

	saveto = output_foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_psfsub_contcube.fits'%(subchannel)
	saveto_2D = output_foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_psfsub_cont2D.fits'%(subchannel)
	saveto_nosub = output_foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_contcube.fits'%(subchannel)
	saveto_nosub_2D = output_foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_cont2D.fits'%(subchannel)

	if do_cont_cube:
		cont_cube,unc_cont_cube,dq_cont_cube = automatic_continuum_cube(fn=hdu_fn,saveto = saveto,verbose=verbose)
	else:
		cont_cube,unc_cont_cube,dq_cont_cube = [fits.open(saveto)[i].data for i in [1,2,3]]

	fn_no_psfsub = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/L1448-mm_%s_s3d_LSRcorr_stripecorr.fits'%(subchannel)

	if do_cont_cube_nosub:
		cont_cube_nosub,unc_cont_cube_nosub,dq_cont_cube_nosub = automatic_continuum_cube(fn=fn_no_psfsub,saveto=saveto_nosub,verbose=verbose)
	else:
		cont_cube_nosub,unc_cont_cube_nosub,dq_cont_cube_nosub = [fits.open(fn_no_psfsub)[i].data for i in [1,2,3]]

	cont_map = cont_integrated_map(um,cont_cube,unc_cont_cube,wcs_2D,saveto=saveto_2D,n_sigma=5)
	cont_map_nosub = cont_integrated_map(um,cont_cube_nosub,unc_cont_cube_nosub,wcs_2D,saveto=saveto_nosub_2D,n_sigma=5)




	if (0):

		plt.figure(figsize = (16,9))
		
		plt.subplot(121)
		plt.imshow(cont_map_nosub,origin='lower',cmap='gist_stern')
		plt.colorbar(location='top')
		plt.title('Nosub')
		plt.subplot(122)
		plt.imshow(cont_map,origin='lower',cmap='gist_stern')
		plt.colorbar(location='top')
		plt.title('Sub')
		plt.show()
