from ifu_analysis.jdvis import generate_image_grid,get_image_world_extent,get_image_pixel_extent

from ifu_analysis.jdutils import get_JWST_IFU_um,make_moment_map,get_wcs_arr,get_JWST_PSF,get_subcube_name
from ifu_analysis.jdcontinuum import get_cont_cube,make_spectral_index_map,resample_cube, sav_gol_continuum

from astropy.io import fits
from astropy.wcs import WCS 
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.colors import LogNorm
from scipy.ndimage import median_filter
import astropy.units as u
import numpy as np
from spectres import spectres
import pandas as pd
from photutils.aperture import CircularAperture


do_plot_psfsub_parameters = True

fn_cont_mask = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/cont-mask-L1448MM1.txt'
#psf_sub_foldername = '/home/vorsteja/Documents/JOYS/JWST_cubes/DELETEch3and4/'
psf_sub_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/PSF_Subtraction/'
#
df_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/PSF_Subtraction/'

df_filenames = glob(df_foldername + '*.csv')
df_filenames.sort()
filenames = glob(psf_sub_foldername + '*.fits')
filenames.sort()
wcs_arr = get_wcs_arr(filenames)

shp = (2,3)
figsize = (16,10)

#fig, axs = generate_image_grid(shp,figsize,wcs_arr)
#axs = axs.flat

if do_plot_psfsub_parameters:
	plt.figure(figsize = (16,5))
for i,fn in enumerate(filenames):
	df = pd.read_csv(df_filenames[i])
	x_offset_arcsec, y_offset_arcsec,scaling = [df[x].values for x in ['x_offset','y_offset','scaling']]

	hdu = fits.open(fn)

	hdr = hdu[1].header
	um = get_JWST_IFU_um(hdr)

	inds = np.arange(0,len(x_offset_arcsec),1)

	x_pix_scale = hdr['CDELT1']*3600 #arcsec per pixel
	y_pix_scale = hdr['CDELT2']*3600 #arcsec per pixel

	if do_plot_psfsub_parameters:
		plt.subplot(131)
		plt.plot(um[inds],x_offset_arcsec/x_pix_scale)
		plt.xlabel('um')
		plt.ylabel('x offset pixel')
		plt.subplot(132)
		plt.plot(um[inds],y_offset_arcsec/y_pix_scale)
		plt.xlabel('um')
		plt.ylabel('y offset pixel')
		plt.subplot(133)
		plt.plot(um[inds],scaling)
		plt.xlabel('um')
		plt.ylabel('scaling (MJy sr-1)')


	x_offset_subchannel = round(50+np.nanmedian(x_offset_arcsec/x_pix_scale))
	y_offset_subchannel = round(50+np.nanmedian(y_offset_arcsec/y_pix_scale))

	beam_size = get_JWST_PSF(np.mean(um))/(x_pix_scale+y_pix_scale)**0.5
	N_beams = 5

	aper = CircularAperture((x_offset_subchannel,y_offset_subchannel),r=N_beams*beam_size)

	


	beam_size = get_JWST_PSF(np.mean(um))*u.arcsec
	pixscale = hdr['CDELT1']*u.degree

	data_cube = hdu[1].data	
	unc_cube = hdu[2].data
	dq_cube = hdu[3].data

	cont_cube = get_cont_cube(data_cube,um,fn_cont_mask,sep=',')
	cont_unc = get_cont_cube(unc_cube,um,fn_cont_mask,sep=',')
	cont_dq = get_cont_cube(dq_cube,um,fn_cont_mask,sep=',')


	resample_factor = None
	chan_lower = 0
	if resample_factor:
		new_wavs = np.linspace(min(um),max(um),len(um)//resample_factor)
		cont_cube,cont_unc = resample_cube(um,cont_cube,cont_unc,new_wavs)
		chan_upper = len(new_wavs)
	else:
		chan_upper = len(um)
	

	mom0,mom0_unc = make_moment_map(cont_cube,cont_unc,chan_lower,chan_upper,order=0)

	aper_mask = np.array(aper.to_mask(method='center').to_image(shape=np.shape(mom0)),dtype=int)
	if i == 0:
		world_bl,world_br,world_tl,world_tr = get_image_world_extent(mom0,wcs_arr[i])
		xlim,ylim = get_image_pixel_extent(wcs_arr[i],world_bl,world_br,world_tl,world_tr)
	else:
		xlim,ylim = get_image_pixel_extent(wcs_arr[i],world_bl,world_br,world_tl,world_tr)

	vmin = 40 * np.mean(mom0_unc)
	vmax = 700 * np.mean(mom0_unc)

	inds = np.where(aper_mask==1)

	mom0[inds] = 0

	#axs[i].imshow(mom0,vmin=vmin, vmax=vmax,cmap='gist_stern')
	#axs[i].scatter([x_offset_subchannel],[y_offset_subchannel],marker='x',color='magenta',s=200)
	#axs[i].set_title(get_subcube_name(fn))
	#axs[i].imshow(mom0/mom0_unc,vmin=30,vmax=500)
	#aper.plot(ax = axs[i],edgecolor='magenta')

	cont_min, cont_max = 70,300
	contour_levels = [i for i in np.logspace(np.log10(cont_min),np.log10(cont_max),5)]

	#axs[i].contour(mom0/mom0_unc,levels = contour_levels,cmap='Greys')
	#axs[i].set_xlim(xlim)
	#axs[i].set_ylim(ylim)
	#_ = axs[i].add_artist(ellipse_artist)  
#fig.tight_layout()
plt.show()