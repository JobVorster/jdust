from ifu_analysis.jdvis import annotate_imshow, generate_image_grid,align_axes
from ifu_analysis.jdutils import get_subcube_name,get_JWST_IFU_um,get_JWST_PSF

import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
import numpy as np

#generate_image_grid(shp,figsize,wcs_arr=None)
#align_axes(img_arr,ax_arr,wcs_arr,reference_ax = 0)


subband_arr = ['1A','1B','1C','2A','2B','2C','3A','3B','3C','4A','4B','4C']

fn_band_arr = ['ch1-short','ch1-medium','ch1-long','ch2-short','ch2-medium','ch2-long',
'ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long'] 


source_name = 'L1448MM1'
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/publication_figures/'
input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/PSF_Subtraction/continuum_maps/'
cube_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/PSF_Subtraction/'


filenames = [input_foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_psfsub_cont2D.fits'%(x) for x in fn_band_arr]
cube_filenames = [cube_foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_PSFcube.fits'%(x) for x in fn_band_arr]



output_figure_name = output_foldername + '%s_ch3_ch4_PSFSUB_STRIPE.jpg'%(source_name)


filenames.sort()
shp = (4,3)
figsize = (14,16)

distance = 240
linear_scale = 300

wcs_arr = []
hdr_arr = []
img_arr = []
subcube_arr = []
um_arr = []
for i,fn in enumerate(filenames):
	
	hdu = fits.open(fn)
	
	subchannel = get_subcube_name(fn)

	#Sorry -- this just gets the wavelength. One has to specify the path to a cube. 
	#This needs to be fixed so that the wavelength goes into the header.
	hdr_3D =fits.open(cube_filenames[i])[1].header
	hdr = hdu[0].header
	wcs = WCS(hdr)
	mean_um = np.mean(get_JWST_IFU_um(hdr_3D))
	um_arr.append(mean_um)
	

	subcube_arr.append(subchannel)
	img_arr.append(hdu[0].data)
	hdr_arr.append(hdr)
	wcs_arr.append(wcs)

	#if i == 2:
	#	break

#This was just for if one does not work.
#um_arr.append(um_arr[-1])
#subcube_arr.append(subcube_arr[-1])
#img_arr.append(img_arr[-1])
#hdr_arr.append(hdr_arr[-1])
#wcs_arr.append(wcs_arr[-1])


fig, ax_arr = generate_image_grid(shp,figsize,wcs_arr)
ax_arr = ax_arr.flat
print([len(i) for i in [wcs_arr,hdr_arr,img_arr,subcube_arr,um_arr,ax_arr]])
ax_arr = align_axes(img_arr,ax_arr,wcs_arr,reference_ax = 0)

star_pos = ['03h25m38.8898s','+30d44m05.612s']
colorbar_label = r'Flux Density (MJy sr$^{-1}$ $\mu$m)'

img_type = 'Continuum'
imshow_cmap = 'magma'
for i in range(len(img_arr)):
	if i != 999:
		hide_ticks = True
	else:
		hide_ticks = False

	img,ax,hdr,subchannel,mean_um = img_arr[i],ax_arr[i],hdr_arr[i],subcube_arr[i],um_arr[i]
	fwhm = get_JWST_PSF(mean_um)
	if i > 8:
		vmax = np.percentile(img,[99.9])[0]
	elif i in [3,4]:
		vmax = 15
	else:
		vmax = 250
	ax.set_facecolor('black')
	ax.imshow(img,origin='lower',cmap=imshow_cmap,vmin=0,vmax=vmax)
	
	ax = annotate_imshow(ax,hdr,hide_ticks=hide_ticks,source_name=source_name,wavelength=r'%.1f $\mu$m'%(mean_um),img_type=img_type,
		beam_fwhm = fwhm,linear_scale = linear_scale, distance = distance,
		add_colorbar=True,colorbar_label=colorbar_label,dogrid=True,star_pos=star_pos)

fig.subplots_adjust(wspace=-0.3,hspace=0.3)
fig.savefig(output_figure_name,dpi=300,bbox_inches='tight')

