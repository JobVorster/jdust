import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture,CircularAperture
from astropy.visualization.wcsaxes import WCSAxes,add_beam
from astropy.wcs import WCS
import astropy.units as u

################################
# Constants					   #
################################

pc_to_au = 206265
deg_to_arcsec = 3600


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


def get_image_world_extent(img,wcs_2D,filter_nans=True):
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
	ylim = [min(y_arr),0.88*max(y_arr)]
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

def annotate_imshow(ax,hdr,hide_ticks=False,do_minor_ticks=False,
	beam_fwhm=None,RA_format = 'hh:mm:ss.ss',Dec_format = 'dd:mm:ss.s',
	source_name=None,wavelength=None,img_type=None,fontdict={'va': 'center','ha': 'left','fontsize':12,'color':'white'},
	linear_scale = None, distance = None,
	add_colorbar=False,colorbar_label=None,dogrid=False):
	'''
	INSERT DOCSTRING HERE!!
	'''
	print('(annotate_imshow) Please test and add a docstring to this function!!')


	if not isinstance(ax, WCSAxes):
		raise ValueError('Axis is not of type WCSAxes, but some other axis type. Initialize axis with projection=wcs to use this function.')

	xlim,ylim = ax.get_xlim(),ax.get_ylim()
	xextent = np.diff(xlim)[0]
	yextent = np.diff(ylim)[0]

	ra,dec = ax.coords


	ra.display_minor_ticks(do_minor_ticks)
	dec.display_minor_ticks(do_minor_ticks)

	ra.set_ticks(direction='in',color='white',size = 3,width=1,spacing = 1*u.arcsec)
	dec.set_ticks(direction='in',color='white',size = 3,width=1,spacing=1*u.arcsec)

	if hide_ticks:
		ra.set_ticklabel_visible(False)
		dec.set_ticklabel_visible(False)
		ra.set_axislabel('')
		dec.set_axislabel('')
	else:
		ra.set_major_formatter(RA_format)
		dec.set_major_formatter(Dec_format)

		ra.set_axislabel('Right Ascension (J2000)',fontsize = 13)
		dec.set_axislabel('Declination (J2000)',fontsize = 13)

		ra.set_ticklabel(size = 11,exclude_overlapping=False)
		dec.set_ticklabel(size = 11,exclude_overlapping=False)



		

	wcs = WCS(hdr)

	#Make sure the wcs is 2D.
	if wcs.world_n_dim == 3:
		wcs.dropaxis(2)

	#Beam fwhm must be in arcsec.
	if beam_fwhm:
		add_beam(ax,major=beam_fwhm*u.arcsec,minor=beam_fwhm*u.arcsec,angle=0,fc=None,ec='white',fill=None)

	#Annotation of title etc.
	if source_name or wavelength or img_type:
		annotate_str = ''
		if source_name:
			annotate_str += source_name + '\n'
		if wavelength:
			annotate_str += wavelength + '\n'
		if img_type : 
			annotate_str += img_type + '\n'

		#Add annotation to the figure.
		extent_perc = 0.2
		xorigin = xlim[0]+0.3*extent_perc*xextent
		yorigin = ylim[1]-extent_perc*yextent
		ax.text(xorigin,yorigin,annotate_str,**fontdict)

	#Scalebar.
	if linear_scale or distance:
		if not (distance and linear_scale):
			raise ValueError('Please specify the distance and linear scale to show linear scale bar.')
		else:
			#Calculate from header.
			pixel_scale = hdr['CDELT1'] #deg per pixel.
			distance *= pc_to_au
			theta = np.rad2deg(linear_scale/distance)
			npixels = theta/pixel_scale

			lin_str = '%d au'%(linear_scale)

			extent_perc = 0.10
			xorigin = xlim[1]-2*extent_perc*xextent
			yorigin = ylim[1]-extent_perc*yextent

			xplot = [xorigin,xorigin+npixels]
			yplot = [yorigin,yorigin]

			ax.plot(xplot,yplot-yextent*0.04,color='white',linewidth=2.5)
			xtext = np.mean(xplot)
			ytext = yorigin
			ax.text(xtext,ytext,lin_str,color='white',va='center',ha='center',fontsize = 10)

	if add_colorbar:
		cbar_tick_fontsize = 10
		im = ax.get_images()[-1]
		cbar = plt.colorbar(im, ax=ax,location='top',fraction=0.046,pad = 0)
		cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
		if colorbar_label:
			cbar.set_label(colorbar_label,fontsize = 12)


	if dogrid:
		ax.coords.grid(color='white', alpha=0.15, linestyle='solid')

	return ax

def align_axes(img_arr,ax_arr,wcs_arr,reference_ax = 0):
	'''
	Align a set of axes based on the image extent of a reference image. The function does not reproject,
	it merely sets the xlim and ylim.

	Parameters
	----------

	img_arr : 1D array of 2D array
		Array of 2D images to plot.

	ax_arr : 1D array of WCSAxes
		Array of axes to align.

	wcs_arr : 1D array of WCS
		2D WCS object of each image.
	
	reference_ax : integer
		Index of the image to use as the reference.

	Returns
	-------

	ax_arr : 1D array of WCSAxes
		Array of aligned axes.
	'''
	for ax in ax_arr:
		if not isinstance(ax, WCSAxes):
			raise ValueError('Axis is not of type WCSAxes, but some other axis type. Initialize axis with projection=wcs to use this function.')

	len_arr = [len(x) for x in [img_arr,ax_arr,wcs_arr]]
	if np.mean(len_arr) != len(img_arr):
		raise ValueError('Error: img_arr, ax_arr and wcs_arr should all be the same length!')

	corners = get_image_world_extent(img_arr[reference_ax],wcs_arr[reference_ax])

	for i in range(len(img_arr)):
		xlim,ylim = get_image_pixel_extent(wcs_arr[i],*corners)
		ax_arr[i].set_xlim(xlim)
		ax_arr[i].set_ylim(ylim)
	return ax_arr


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