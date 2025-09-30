"""
Dust with JWST utils scripts.
@author: Job Vorster, 20 March 2025. 
Email: jobvorster8@gmail.com
"""
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture
from spectres import spectres #Resampling arxiv link: https://arxiv.org/abs/1705.05165
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

def unpack_hdu(fn):
	'''
	Utility function to unpack often used variables from an IFU cube.

	Parameters
	----------

	fn : string
		Filename of the IFU cube.

	Returns
	-------

	data_cube : 3D array
		Science data.

	unc_cube : 3D array
		Uncertainties on the science data.

	dq_cube : 3D array
		Data quality of the science data.

	hdr : astropy Header
		Header of the science hdu.

	um : 1D array
		Wavelengths in microns. 

	shp : tuple
		Shape of the 2D map.

	'''
	hdu = fits.open(fn)

	#Unpack the hdu
	hdr = hdu[1].header
	data_cube, unc_cube, dq_cube = [hdu[i].data for i in [1,2,3]]
	um = get_JWST_IFU_um(hdr)
	shp = np.shape(data_cube[0,:,:]) #2D shape

	return data_cube,unc_cube,dq_cube,hdr,um,shp

def get_JWST_IFU_um(header):
	'''
	Returns a wavelength array from an IFU cube ImageHDU header (typically index 1 for MIRI MRS cubes).
	Based on the formula from the pipeline documentation:
	"wave = (np.arange(hdr['NAXIS3']) + hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']"
	URL: https://jwst-docs.stsci.edu/jwst-calibration-status/miri-calibration-status/miri-mrs-calibration-status#gsc.tab=0
	
	Parameters
	----------

	header : dictionary
		The header of the ImageHDU of the IFU cube. Make sure the NAXIS3, CRPIX3, CRVAL3, and CDELT3 columns are located in the header.

	Returns
	-------

	um : 1D array
		Array of wavelengths in microns.
	'''
	n_chan = np.arange(0,header['NAXIS3'],1)
	crpix = header['CRPIX3']
	crval = header['CRVAL3']
	cdelt = header['CDELT3']
	um = (n_chan + crpix - 1) * cdelt + crval
	return um

def interpolate_nan(array_like):
	'''
	If an array containts nan values, fill the nan values with interpolations from surrounding entries.

	Parameters
	----------

	array_like : array
		An array containing nans.

	Returns
	-------
	
	array : array
		The same array with the nans interpolated.
	'''
	array = array_like.copy()

	nans = np.isnan(array)

	def get_x(a):
		return a.nonzero()[0]

	array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])

	return array

def is_nan_map(array):
	'''
	Checks whether an N-dimensional array consists of ONLY nans.

	Parameters
	----------

	array: N-D array
		Any array.

	Returns
	-------

	is_nan_map: boolean
		True, if the map is only nans.
	'''
	if np.sum(np.isfinite(array.flatten()))==0:
		return True 
	else:
		return False


def make_moment_map(data_cube,unc_cube,chan_lower,chan_upper,order=0):
	'''
	Makes a moment map from a data cube. The order can be specified as 0, 1, 2 (ONLY 0 IMPLEMENTED)!

	Parameters
	----------

	data_cube : 3D array
		A spectral cube.

	unc_cube : 3D array
		The corresponding uncertainty cube.

	chan_lower : integer
		Lower limit for the channels for the moment map.

	chan_upper : integer
		Upper limit for the channels for the moment map.

	order : integer
		Order of the moment map.

	Returns
	-------

	mom0 : 2D array
		The moment map.

	mom0_unc : 2D array
		The corresponding uncertainty to the moment map.
	'''
	if order == 0:
		mom0 = np.nansum(data_cube[chan_lower:chan_upper+1,:,:],axis=0)
		mom0_unc = np.sqrt(np.nansum(unc_cube[chan_lower:chan_upper+1,:,:]**2,axis=0))
		return mom0,mom0_unc
	else:
		raise ValueError('Higher order moments have not yet been implemented yet, only moment 0.')

def get_wcs_arr(filenames):
	'''
	Get an array of 2D wcs projections for a list of cubes. This function assumes hdu[1] contains the science data, 
	and that there is a spectral axis that needs to be dropped.

	Parameters
	----------

	filenames : list of string
		Filenames of fits files.

	Returns
	-------

	wcs_arr : list of wcs
		List of projections.
	'''
	wcs_arr = []
	for fn in filenames:
		hdu = fits.open(fn)

		hdr = hdu[1].header

		wcs = WCS(hdr)
		wcs_2D = wcs.dropaxis(2)
		wcs_arr.append(wcs_2D)
	return wcs_arr


def get_subcube_name(filename):
	'''
	Get the name for the MIRI MRS subcube, assuming naming convention 'SOURCE_ch4-short_s3d_LSRcorr.fits'

	Parameters
	----------

	filename : string
		Cube filename

	Returns
	-------

	subcube_name : string
		Name of the subcube (e.g. ch4-short)

	'''
	subcube_name = filename.split('/')[-1].split('_')[1]
	return subcube_name

def get_source_name(filename):
	'''
	Get the name for the MIRI MRS source, assuming naming convention 'SOURCE_ch4-short_s3d_LSRcorr.fits'

	Parameters
	----------

	filename : string
		Cube filename

	Returns
	-------

	source_name : string
		Name of the source (e.g. HH211)

	'''
	source_name = filename.split('/')[-1].split('_')[0]
	return source_name

def define_circular_aperture(RA_centre, Dec_centre,size_arcsec):
	'''
	Short function to make a circular aperture of a specific radius.

	Parameters
	----------

	RA_centre: string
		J2000 Right Ascension in the format XXhXXmXX.XXs

	Dec_centre: string
		J2000 Declination in the format +XXdXXmXX.XXs where the sign can be + or -
	
	Returns
	-------
	
	aper: SkyAperture object
		photutils circular aperture with the relevant sky coordinates.

	'''
	position = SkyCoord(ra=RA_centre, dec=Dec_centre, unit='arcsec')
	aper = SkyCircularAperture(position, size_arcsec*u.arcsec)
	return aper

def get_JWST_PSF(um):
	'''
	Returns the JWST PSF from Law, D.R. et. al. 2023 for a specified wavelength.

	Parameters
	----------

	um: float
		Wavelength in microns.

	Returns
	-------
	
	fwhm: float 
		Full width at half maximum in arcsec.
	
	'''
	fwhm = 0.033*um+0.106
	return fwhm

def read_aperture_ini(filename,sep=','):
	'''
	Read an aperture ini file. Returns the names, sizes (in arcsec) and position of the aperture.

	Parameters
	----------

	filename : string
		Filename of the ini file.

	sep : string
		Separator for pandas.read_csv.

	Returns
	-------

	aper_names : 1D list of string
		Names of apertures.

	aper_sizes : 1D list of float
		Aperture sizes in arcsec.

	coord_list : N x 2 list of string.

		RA_centre: string
		J2000 Right Ascension in the format XXhXXmXX.XXs

		Dec_centre: string
		J2000 Declination in the format +XXdXXmXX.XXs where the sign can be + or -
	'''
	
	df = pd.read_csv(filename,sep=sep)
	aper_names = df['Name'].values
	aper_sizes = df['Size(arcsec)'].values
	ras = df['R.A.(J2000)'].values
	decs = df['Dec.(J2000)'].values
	coord_list = [[x,y] for (x,y) in zip(ras,decs)]

	return aper_names,aper_sizes,coord_list