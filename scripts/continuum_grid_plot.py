from ifu_analysis.jdvis import annotate_imshow, generate_image_grid,align_axes
from ifu_analysis.jdutils import get_subcube_name

import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.wcs import WCS
from glob import glob

#generate_image_grid(shp,figsize,wcs_arr=None)
#align_axes(img_arr,ax_arr,wcs_arr,reference_ax = 0)

filenames = glob('/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448-mm_*_PSFsub_stripe_continuum.fits')
filenames.sort()
shp = (2,3)
figsize = (16,12)



wcs_arr = []
hdr_arr = []
img_arr = []
subcube_arr = []
for i,fn in enumerate(filenames):
	hdu = fits.open(fn)
	hdr = hdu[0].header
	wcs = WCS(hdr)
	subchannel = get_subcube_name(fn)

	subcube_arr.append(subchannel)
	img_arr.append(hdu[0].data)
	hdr_arr.append(hdr)
	wcs_arr.append(wcs)

	#if i == 2:
	#	break

print([len(x) for x in [wcs_arr,hdr_arr,img_arr,subcube_arr\]])

fig, ax_arr = generate_image_grid(shp,figsize,wcs_arr)
ax_arr = ax_arr.flat
ax_arr = align_axes(img_arr,ax_arr,wcs_arr,reference_ax = 0)



for i in range(len(img_arr)):
	img,ax,hdr,subchannel = img_arr[i],ax_arr[i],hdr_arr[i],subcube_arr[i]
	ax.imshow(img,origin='lower')
	
#ax,hdr,
#	beam=None,RA_format = 'hh:mm:ss.sss',Dec_format = 'dd:mm:ss.s',
#	source_name=None,wavelength=None,img_type=None,fontdict={'va': 'center','ha': 'center','fontsize':12,'weight':'bold','color':'white'},
#	linear_scale = None, distance = None,
#	add_colorbar=False,colorbar_label=None

	ax = annotate_imshow(ax,hdr,source_name='L1448MM1',wavelength=subchannel,img_type='Continuum',
		add_colorbar=True,colorbar_label='Flux Density (MJy sr-1 um)',dogrid=True)

plt.show()

