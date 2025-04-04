import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture

def plot_unstitched_spectra(results,umlim=None):
	'''
	Plot spectra which are not yet stiched to a single array.

	Parameters
	----------
	
	results : dictionary
		Containing:

			source_name : string
				Name of the source from the filename.

			RA_centre, Dec_centre, aper_size : Saved as specified above.

			subcube_name : list of string
				Names of each subcube (used in stitching).

			um : list of arrays
				2D list containing wavelengths of each subcube.

			flux : list of arrays
				2D list containing the flux densities of each subcube.

			flux_unc : list of arrays
				2D list containing the uncertainty on flux densities of each subcube.

	umlim : float
		Cutoff wavelength limit for the plotting.

	Returns
	-------

	None
		Makes plot.
	'''
	subcubes  = results['subcube_name']



	for i,subcube in enumerate(subcubes):

		if umlim:
			inds = results['um'][i]<umlim
		else:
			inds = [True]*len(results['um'][i])

		um = results['um'][i][inds]
		flux = results['flux'][i][inds]
		flux_unc = results['flux_unc'][i][inds]
		plt.errorbar(um,flux,yerr = flux_unc,marker='o',label='%s'%(subcube))
	plt.xlabel(r'Wavelength ($\mu$m)')
	plt.minorticks_on()
	plt.grid(which ='both',alpha=0.3)
	plt.ylabel('Flux Density (Jy)')
	plt.legend()


def get_image_world_extent(img,wcs_2D):
	'''
	Retrieve the corners of an image in world coordinates.

	Parameters
	----------

	img : 2D array
		Image

	wcs_2D : wcs object
		The wcs corresponding to the image.

	Returns
	-------
	
	world_bl : 1D array
		Coordinates of the bottom left corner in world coordinates.

	world_br : 1D array
		Coordinates of the bottom right corner in world coordinates.

	world_tl : 1D array
		Coordinates of the top left corner in world coordinates.

	world_tr : 1D array
		Coordinates of the top right corner in world coordinates.

	'''
	shp = np.shape(img)
	x_inds = np.arange(0,shp[0])
	y_inds = np.arange(0,shp[1])

	#The four corners are defined as follows:
	bottom_left = [x_inds[0],y_inds[0]]
	bottom_right = [x_inds[-1],y_inds[0]]
	top_left = [x_inds[0],y_inds[-1]]
	top_right = [x_inds[-1],y_inds[-1]]

	world_bl = np.array(wcs_2D.wcs_pix2world(bottom_left[0],bottom_left[1] ,1))
	world_br = np.array(wcs_2D.wcs_pix2world(bottom_right[0],bottom_right[1] ,1))
	world_tl = np.array(wcs_2D.wcs_pix2world(top_left[0],top_left[1] ,1))
	world_tr = np.array(wcs_2D.wcs_pix2world(top_right[0],top_right[1] ,1))
	return world_bl,world_br,world_tl,world_tr

def get_image_pixel_extent(wcs_2D,world_bl,world_br,world_tl,world_tr):

	print('HI, PLEASE MAKE A DOCSTRING FOR get_image_pixel_extent')
	x_arr = []
	y_arr = []

	#This is so lazy, sorry. Clean it up if you want!

	x_pix,y_pix = np.array(wcs_2D.wcs_world2pix(world_bl[0],world_bl[1],1))
	x_arr.append(x_pix)
	y_arr.append(y_pix)

	x_pix,y_pix = np.array(wcs_2D.wcs_world2pix(world_br[0],world_br[1],1))
	x_arr.append(x_pix)
	y_arr.append(y_pix)

	x_pix,y_pix = np.array(wcs_2D.wcs_world2pix(world_tl[0],world_tl[1],1))
	x_arr.append(x_pix)
	y_arr.append(y_pix)

	x_pix,y_pix = np.array(wcs_2D.wcs_world2pix(world_tr[0],world_tr[1],1))
	x_arr.append(x_pix)
	y_arr.append(y_pix)

	xlim = [min(x_arr),max(x_arr)]
	ylim = [min(y_arr),max(y_arr)]
	return xlim,ylim


def generate_image_grid(shp,figsize,wcs_arr=None):
	'''
	Generate a grid of 2D images of specified shape and figsize, with the option to set the wcs projection for each subplot.

	Parameters
	----------

	shp : tuple
		A tuple (Nrows, Ncols) setting the subplot grid.

	figsize : tuple
		Figure size

	wcs_arr : 1D array of wcs instance
		Projections for each plot.

	Returns
	-------

	fig : Figure instance
		The figure object.

	axs : array of Axes
		An array of axes objects for further changes.
	'''
	fig = plt.figure(figsize = figsize)
	if len(shp) == 2:
		Nrows, Ncols = shp
		for index in range(1,Nrows*Ncols+1):
			if wcs_arr:
				fig.add_subplot(Nrows,Ncols,index,projection = wcs_arr[index-1])
			else:
				fig.add_subplot(Nrows,Ncols,index)
	else:
		ValueError('Incorrect shape for grid, shape should either be (Nrows,Ncols)!')

	axs = np.reshape(fig.axes,shp)
	return fig, axs
	
def make_snr_figures(mom0,mom0_unc,wcs_2D,contour_levels=None,contour_cmap='Greys_r',contour_alpha=0.3):
	'''
	Make a three panel figure with a moment map, and its uncertainties. Mostly for data inspection.

	Parameters
	----------

	mom0 : 2D array
		The moment map.

	mom0_unc : 2D array
		Uncertainties on the moment map.

	wcs_2D : WCS object
		WCS of the map.

	contour_levels : list
		List of values for plt.contour, the values are relevant for the SNR!!

	contour_cmap : string
		Contour colormap.

	contour_alpha : float
		Opacity of the contours.

	Returns
	-------

	fig : Figure instance
		The figure instance from the plotting.

	ax1, ax2, ax3 : Axes instances
		The axes instances from the plotting.

	Notes
	-----

	TO IMPLEMENT: Saving, and cleaner axis labels. 


	'''
	fig = plt.figure(figsize = (16,7))

	ax1 = fig.add_subplot(1, 3, 1, projection=wcs_2D)  
	ax2 = fig.add_subplot(1, 3, 2, projection=wcs_2D)  
	ax3 = fig.add_subplot(1, 3, 3, projection=wcs_2D) 

	#plt.subplot(131,projection=wcs_2D)
	im1 = ax1.imshow(mom0,origin='lower',vmax = 10000)

	cbar1 = fig.colorbar(im1, ax=ax1, location='top', fraction=0.046, pad=0)
	cbar1.set_label(r'$I_{\nu}$ (MJy sr$^{-1}$)')


	if contour_levels:
		ax1.contour(mom0/mom0_unc,levels=contour_levels ,cmap=contour_cmap,alpha=contour_alpha)

	#plt.subplot(132,projection=wcs_2D)
	im2 = ax2.imshow(mom0_unc,origin='lower')
	
	cbar2 = fig.colorbar(im2, ax=ax2, location='top', fraction=0.046, pad=0)
	cbar2.set_label(r'$\delta I_{\nu}$ (MJy sr$^{-1}$)')


	#plt.subplot(133,projection=wcs_2D)
	im3 = ax3.imshow(mom0/mom0_unc,origin='lower')
	cbar3 = fig.colorbar(im3, ax=ax3, location='top', fraction=0.046, pad=0)
	cbar3.set_label('S/N (unitless)')

	return fig, ax1, ax2, ax3

def plot_apertures(aper_names,aper_sizes,aper_coords,ax,wcs_2D,color='white',aper_alpha=0.4):
	'''
	Plot N apertures with names onto a WCS Axes.

	Parameters
	----------

	aper_names : 1D list of strings
		Names of apertures.

	aper_sizes : 1D list of floats
		Sizes of apertures (in arcsec).

	aper_coords : N x 2 list of strings [RA_centre, Dec_centre]

		RA_centre: string
		J2000 Right Ascension in the format XXhXXmXX.XXs

		Dec_centre: string
			J2000 Declination in the format +XXdXXmXX.XXs where the sign can be + or -
	
	ax : WCSAxes instance
		Axis to plot on.

	wcs_2D : WCS instance
		WCS of the plot.

	color : string
		Color of the plotted apertures.

	aper_alpha : float
		Opacity of the apertures.

	Returns
	-------

	None
		Plots apertures on the plot instance.
	'''
	for name, size, (RA,Dec) in zip(aper_names,aper_sizes,aper_coords):
		aper = define_circular_aperture(RA,Dec,size)
		aper = aper.to_pixel(wcs_2D)
		aper.plot(ax=ax,color=color,zorder=5,alpha=aper_alpha)
		x,y = aper.positions
		ax.text(x,y,name,va='center',ha='center',color=color)