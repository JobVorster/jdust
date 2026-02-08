from ifu_analysis.jdvis import annotate_imshow,plot_apertures
from ifu_analysis.jdutils import get_subcube_name,get_JWST_IFU_um,get_JWST_PSF
from ifu_analysis.jdspextract import read_aperture_ini

import os
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
import numpy as np

source_name = 'L1448MM1'

cont_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/PSF_Subtraction/continuum_maps/'
fn_cont = cont_foldername+'L1448-mm_ch3-medium_s3d_LSRcorr_stripecorr_psfsub_cont2D.fits'

line_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/line_cubes/'
fn_co2 = line_foldername + 'combined_maps/co2/L1448-mm_CO2 _14.9803439087585um.fits'

fn_h2 = line_foldername + 'h2/L1448-mm_H2 _12.2786120207857um.fits'

#Define aperture filename (for aperture names and radii).
aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt'%(source_name)

if os.path.isfile(aperture_filename):
	aper_names,aper_sizes,aper_coords = read_aperture_ini(aperture_filename)

map_arr = [fn_cont,fn_co2,fn_h2]
hdu_cont, hdu_co2, hdu_h2 = [fits.open(x) for x in map_arr]
hdu_arr = [hdu_cont, hdu_co2, hdu_h2]
map_cont, map_co2, map_h2 = [hdu[0].data for hdu in hdu_arr]

hdr = hdu_cont[0].header
wcs_2D = WCS(hdr)
hide_ticks = False
do_apers = True
mean_um = round(float(hdr['WAVE']),2)
distance = 240
linear_scale = 300
fwhm = get_JWST_PSF(mean_um)

star_pos = ['03h25m38.8898s','+30d44m05.612s']
colorbar_label = r'Flux Density (MJy sr$^{-1}$ $\mu$m)'

img_type = 'Continuum'
imshow_cmap = 'inferno'
h2_color = 'cyan'


extra_annotations=''

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(1,1,1,projection = wcs_2D)
ax.set_facecolor('black')
map_cont[map_cont ==0] = np.nan
ax.imshow(map_cont,origin='lower',vmin=0,vmax=250,cmap = imshow_cmap)
#ax.contour(map_cont,cmap=imshow_cmap)

ax = annotate_imshow(ax,hdr,hide_ticks=hide_ticks,source_name=source_name,wavelength=r'%.1f $\mu$m'%(mean_um),img_type=img_type,
		beam_fwhm = fwhm,linear_scale = linear_scale, distance = distance,
		add_colorbar=True,colorbar_label=colorbar_label,dogrid=True,star_pos=star_pos)

cont_alpha = 0.5

map_co2[map_co2==0] = np.nan
#ax.contour(np.log(map_co2),colors='cyan',alpha=cont_alpha)

#Add annotation to the figure.
option = 'top_left'

xlim,ylim = ax.get_xlim(),ax.get_ylim()
xextent = np.diff(xlim)[0]
yextent = np.diff(ylim)[0]

if option == 'top_left':
	fontdict={'va': 'center','ha': 'left','fontsize':12,'color':'Green'}
	extent_perc = 0.2
	xorigin = xlim[0]+1.6*extent_perc*xextent
	yorigin = ylim[1]-0.7*extent_perc*yextent
	#ax.text(xorigin,yorigin,r'ln(CO$_2$ 14.98 $\mu$m)',**fontdict)


map_h2[map_h2==0] = np.nan
ax.contour(map_h2,colors='grey',alpha=cont_alpha)

if option == 'top_left':
	fontdict={'va': 'center','ha': 'left','fontsize':12,'color':'grey'}
	extent_perc = 0.2
	xorigin = xlim[0]+1.6*extent_perc*xextent
	yorigin = ylim[1]-1.*extent_perc*yextent
	ax.text(xorigin,yorigin,r'H$_2$ 12.28 $\mu$m',**fontdict)
if do_apers:
	apers_string = 'apers'
	plot_apertures(aper_names,aper_sizes,aper_coords,ax,wcs_2D,color='white',aper_alpha=0.4)
else:
	apers_string = ''

plt.savefig('/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/publication_figures/multi_line_%s_%s.png'%(source_name,apers_string),bbox_inches='tight',dpi=200)

