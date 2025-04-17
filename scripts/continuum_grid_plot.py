from ifu_analysis.jdvis import annotate_imshow, generate_image_grid,align_axes
from ifu_analysis.jdutils import get_subcube_name,get_JWST_IFU_um,get_JWST_PSF

import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
import numpy as np

#generate_image_grid(shp,figsize,wcs_arr=None)
#align_axes(img_arr,ax_arr,wcs_arr,reference_ax = 0)


output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/publication_figures/'
filenames = glob('/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448-mm_*_PSFsub_stripe_continuum.fits')
filenames.sort()
shp = (2,3)
figsize = (14,8)

distance = 240
linear_scale = 300

wcs_arr = []
hdr_arr = []
img_arr = []
subcube_arr = []
um_arr = []
for i,fn in enumerate(filenames):
	
	hdu = fits.open(fn)
	hdr = hdu[0].header
	wcs = WCS(hdr)
	subchannel = get_subcube_name(fn)
	hdu_cube = fits.open('/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/PSF_Subtraction/L1448-mm_%s_s3d_LSRcorr_PSFsub.fits'%(subchannel))
	cube_hdr = hdu_cube[1].header
	um = get_JWST_IFU_um(cube_hdr)
	mean_um = np.mean(um)
	um_arr.append(mean_um)
	

	subcube_arr.append(subchannel)
	img_arr.append(hdu[0].data)
	hdr_arr.append(hdr)
	wcs_arr.append(wcs)

	#if i == 2:
	#	break

print([len(x) for x in [wcs_arr,hdr_arr,img_arr,subcube_arr]])

fig, ax_arr = generate_image_grid(shp,figsize,wcs_arr)
ax_arr = ax_arr.flat
ax_arr = align_axes(img_arr,ax_arr,wcs_arr,reference_ax = 0)


colorbar_label = r'Flux Density (MJy sr$^{-1}$ $\mu$m)'
source_name = 'L1448MM1'
img_type = 'Continuum'
imshow_cmap = 'gist_stern'
for i in range(len(img_arr)):
	if i != 3:
		hide_ticks = True
	else:
		hide_ticks = False

	img,ax,hdr,subchannel,mean_um = img_arr[i],ax_arr[i],hdr_arr[i],subcube_arr[i],um_arr[i]
	fwhm = get_JWST_PSF(mean_um)

	vmax = np.percentile(img,[99.5])[0]

	ax.imshow(img,origin='lower',cmap=imshow_cmap,vmin=0,vmax=vmax)
	
	ax = annotate_imshow(ax,hdr,hide_ticks=hide_ticks,source_name=source_name,wavelength=r'%.1f $\mu$m'%(mean_um),img_type=img_type,
		beam_fwhm = fwhm,linear_scale = linear_scale, distance = distance,
		add_colorbar=True,colorbar_label=colorbar_label,dogrid=True)
fig.subplots_adjust(wspace=-0.3,hspace=0.2)
fig.savefig(output_foldername + '%s_ch3_ch4.jpg'%(source_name),dpi=300,bbox_inches='tight')

