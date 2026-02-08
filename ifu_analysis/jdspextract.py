from jwst.datamodels import MirMrsApcorrModel,IFUCubeModel
from astropy.io import fits
from astropy.wcs import WCS
from ifu_analysis.jdutils import *
import numpy as np
import asdf
from jwst.extract_1d.ifu import ifu_extract1d
from photutils.aperture import aperture_photometry
import json
import matplotlib.pyplot as plt



def jpipe_extract_spectrum(filename,aperture,bkg_sigma_clip=10):
	'''
	Extracts a spectrum from an IFU cube, and its uncertainty with a specified aperture using the JWST pipeline function call.
	The function makes a aperture correction file, and a reference file which is then used with extended emission settings

	Parameters
	----------

	filename : string
		Filename of the MIRI MRS IFU cube.

	aperture : CircularSkyAperture instance
		A circular sky aperture from Photutils.

	bkg_sigma_clip : float
		The number of standard deviations to clip for background.
	

	Returns
	-------

	flux_arr : 1D array
		Array containing summed intensities (in units of flux density, not intensity)

	unc_arry : 1D array
		Array containing uncertainties on summed intensities (in units of flux density, not intensity)

	'''
	#####
	input_model = IFUCubeModel(filename)
	prefix = filename.split('/')[-1].split('.fits')[0]

	print('Extracting spectrum from file %s'%(prefix))

	hdr = fits.open(filename)[1].header
	wcs = WCS(hdr).dropaxis(2)

	#Reference file.
	um = get_JWST_IFU_um(hdr) #If there is non-linearities in wavelength, this might give problems.
	radius = np.array([aperture.r.value]*len(um)) #(arcsec)
	inner_bkg = 1.1*radius #Not relevant for extended emission.
	outer_bkg=2*radius #Not relevant for extended emission.


	subtract_background=False #Not relevant for extended emission.
	method='exact'
	subpixels=10 #Not relevant for extended emission.
	
	tree = {
		"meta": {
			"subtract_background": subtract_background,
			"method": method,
			"subpixels": subpixels
		},
		'data': {
		"wavelength": um,
		"radius": radius,
		"inner_bkg": inner_bkg,
		"outer_bkg": outer_bkg,
		'axis_ratio' : um,
		'axis_pa' : um
		}
	}

	#Change this to a random temp file.
	fname_reffile = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/temp/temp_%s_ref.asdf'%(prefix)
	# Create the ASDF file object from our data tree
	af = asdf.AsdfFile(tree,lazy_load=False)
	af.validate()
	af.write_to(fname_reffile)
	af.close()
	

	#Aperture correction file.
	apcorr = MirMrsApcorrModel(input_model,validate_on_assignment=True)
	apcorr.schema_url = 'http://stsci.edu/schemas/jwst_datamodel/mirmrs_apcorr.schema'
	fname_apcorr = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/temp/temp_%s_apcorr.asdf'%(prefix)
	apcorr.to_asdf(fname_apcorr)

	#Pixel coordinates.
	pixaper = aperture.to_pixel(wcs)
	center_xy = pixaper.positions

	ref_file = fname_reffile
	source_type='POINT'
	bkg_sigma_clip = bkg_sigma_clip
	apcorr_ref_file= None#fname_apcorr
	subtract_background=subtract_background
	ifu_rfcorr=True
	ifu_autocen=False
	spectra=ifu_extract1d(input_model=input_model, 
				  ref_file=ref_file, 
				  source_type=source_type, 
				  subtract_background=subtract_background, 
				  bkg_sigma_clip=bkg_sigma_clip, 
				  apcorr_ref_file=apcorr_ref_file, 
				  center_xy=center_xy, 
				  ifu_autocen=ifu_autocen, 
				  ifu_rfcorr=ifu_rfcorr, 
				  ifu_rscale=None, 
				  ifu_covar_scale=1.0)

	spec_table = spectra.spec[0].spec_table

	um = spec_table['wavelength']
	flux_arr = spec_table['flux']  
	unc_arr = spec_table['flux_error']  # or whatever column name is used for flux

	#print(spec_table.columns)
	#ColDefs(
	#name = 'WAVELENGTH'; format = 'D'; unit = 'um'
	#name = 'FLUX'; format = 'D'; unit = 'Jy'
	#name = 'FLUX_ERROR'; format = 'D'; unit = 'Jy'
	#name = 'FLUX_VAR_POISSON'; format = 'D'; unit = 'Jy^2'
	#name = 'FLUX_VAR_RNOISE'; format = 'D'; unit = 'Jy^2'
	#name = 'FLUX_VAR_FLAT'; format = 'D'; unit = 'Jy^2'
	#name = 'SURF_BRIGHT'; format = 'D'; unit = 'MJy/sr'
	#name = 'SB_ERROR'; format = 'D'; unit = 'MJy/sr'
	#name = 'SB_VAR_POISSON'; format = 'D'; unit = '(MJy/sr)^2'
	#name = 'SB_VAR_RNOISE'; format = 'D'; unit = '(MJy/sr)^2'
	#name = 'SB_VAR_FLAT'; format = 'D'; unit = '(MJy/sr)^2'
	#name = 'DQ'; format = 'J'; bzero = 2147483648
	#name = 'BACKGROUND'; format = 'D'; unit = 'MJy/sr'
	#name = 'BKGD_ERROR'; format = 'D'; unit = 'MJy/sr'
	#name = 'BKGD_VAR_POISSON'; format = 'D'; unit = '(MJy/sr)^2'
	#name = 'BKGD_VAR_RNOISE'; format = 'D'; unit = '(MJy/sr)^2'
	#name = 'BKGD_VAR_FLAT'; format = 'D'; unit = '(MJy/sr)^2'
	#name = 'NPIXELS'; format = 'D'
	if (0):
		plt.figure(figsize = (16,16))
		#plt.subplot(221)
		plt.plot(um,spec_table['SURF_BRIGHT']  ,label='SURF_BRIGHT')
		plt.plot(um,5*spec_table['SB_ERROR'],label='SB_ERROR')
		plt.legend()
		#plt.subplot(222)
		#plt.plot(um,spec_table['BACKGROUND'],label='background')
		#plt.legend()
		#plt.subplot(223)
		#plt.plot(um,spec_table['NPIXELS'],label='Npixles')
		#plt.legend()
		#plt.subplot(224)
		#plt.plot(um,spec_table['SB_ERROR'],label='SB_ERROR')
		#plt.legend()
		plt.show()


	

	return np.array(um),np.array(flux_arr),np.array(unc_arr)


def extract_spectrum(cube,unc_cube,aperture,wcs):
	'''
	Extracts a spectrum from an IFU cube, and its uncertainty with a specified aperture.

	Parameters
	----------

	cube : 3D array
		Array containing the intensities.

	unc_cube : 3D array
		Uncertainties corresponding to the cube intensities.

	aperture : SkyAperture object
		photutils aperture defined in celestial coordinates.

	sky_wcs : WCS object 
		astropy WCS object containing only sky-coords (i.e. 2D).

	Returns
	-------

	flux_arr : 1D array
		Array containing summed intensities (in units of flux density, not intensity)

	unc_arry : 1D array
		Array containing uncertainties on summed intensities (in units of flux density, not intensity)

	'''
	#####
	pixaper = aperture.to_pixel(wcs)
	center_xy = pixaper.positions


	#####


	flux_arr = []
	unc_arr = []
	for channel_map,unc_map in zip(cube,unc_cube):
		if is_nan_map(channel_map):
			flux_arr.append(np.nan)
			unc_arr.append(np.nan)
		else:
			table = aperture_photometry(data=channel_map,apertures=aperture,error=unc_map,wcs=wcs)
			flux_arr.append(np.array(table['aperture_sum'])[0])
			unc_arr.append(np.array(table['aperture_sum_err'])[0])
	return np.array(flux_arr),np.array(unc_arr)

def unstitched_spectrum_from_cube_list(filenames,RA_centre,Dec_centre,aper_size,method='PIPELINE'):
	'''
	Extract spectrum from a list of JWST IFU cube filenames. 

	The function assumes:
	1. All the files are the same source.
	2. A naming convention 'SOURCE_SUBCHANNEL_s3d_LSRcorr.fits'
	3. The file has data, uncertainty and data quality cubes.

	Parameters
	----------

	filenames : list of string
		List of filenames with the format specified above.

	RA_centre : string
		Right Ascension of centre of aperture in the format 'XXhXXmXX.XXs'.

	Dec_centre : string
		Declination of centre of aperture in the format 'XXdXXmXX.XXs'.

	aper_size : float
		Size of circular aperture in arcsec.

	method : string (Options: 'NAIVE' or 'PIPELINE')
		How to extract the spectrum.
		'NAIVE':
			Uses aperture photometry per channel with photutils, does not correct for 

	
	Returns
	-------

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
	'''
	results = {}
	results['source_name'] = ''
	results['RA_centre'] = RA_centre
	results['Dec_centre'] = Dec_centre
	results['aper_size'] = aper_size

	results['subcube_name'] = []
	results['um'] = []
	results['flux'] = []
	results['flux_unc'] = []
	
	for i,cube_filename in enumerate(filenames):
		if i == 0:
			source_name = get_source_name(cube_filename)
			results['source_name'] = source_name

		
		subcube_name = get_subcube_name(cube_filename)
		hdu = fits.open(cube_filename)
		header = hdu[1].header
		um = get_JWST_IFU_um(header)

		data_cube = hdu[1].data
		
		unc_cube = hdu[2].data
		wcs = WCS(header)

		#Make sky-only wcs
		wcs_2D = wcs.dropaxis(2)
		Sky_aper = define_circular_aperture(RA_centre,Dec_centre,aper_size)

		if method == 'NAIVE':

			flux_arr, unc_arr = extract_spectrum(data_cube,unc_cube,Sky_aper,wcs_2D)
			flux_arr /= 1e6 #convert to Jy instead of MJy
			unc_arr /= 1e6
		elif method == 'PIPELINE':
			um, flux_arr, unc_arr = jpipe_extract_spectrum(cube_filename,Sky_aper)
		
		else:
			raise ValueError('Please specify which method to use for spectral extraction.')

		results['subcube_name'].append(subcube_name)
		results['flux'].append(flux_arr)
		results['flux_unc'].append(unc_arr)
		results['um'].append(um)
	
	return results

def save_spectra(results, filename):
	'''
	Save the spectra dictionary to a file using JSON format.
	
	The function converts numpy arrays to lists for JSON serialization 
	to ensure compatibility and ease of reading.

	Parameters
	----------
	results : dictionary
		Dictionary containing spectral data with numpy arrays.
	filename : string
		Path and name of the file to save the spectra data.
	
	Returns
	-------
	None
		Saves the spectra data to the specified file.
	'''
	# Create a copy of the results to avoid modifying the original
	serializable_results = results.copy()
	
	# Convert numpy arrays to lists for JSON serialization
	serializable_results['um'] = [um.tolist() for um in results['um']]
	serializable_results['flux'] = [flux.tolist() for flux in results['flux']]
	serializable_results['flux_unc'] = [flux_unc.tolist() for flux_unc in results['flux_unc']]
	
	# Save to file using JSON
	with open(filename, 'w') as f:
		json.dump(serializable_results, f)

def load_spectra(filename):
	'''
	Load spectra data from a JSON file and convert lists back to numpy arrays.
	
	This function is the inverse of save_spectra. It reads the JSON file
	and converts the serialized lists back to numpy arrays.
	
	Parameters
	----------
	filename : string
		Path and name of the file containing the saved spectra data.
	
	Returns
	-------
	results : dictionary
		Dictionary containing spectral data with numpy arrays.
	'''
	
	# Load the JSON data from file
	with open(filename, 'r') as f:
		loaded_results = json.load(f)
	
	# Convert lists back to numpy arrays
	loaded_results['um'] = [np.array(um) for um in loaded_results['um']]
	loaded_results['flux'] = [np.array(flux) for flux in loaded_results['flux']]
	loaded_results['flux_unc'] = [np.array(flux_unc) for flux_unc in loaded_results['flux_unc']]
	
	return loaded_results

def sort_subcubes(subcubes):
	'''
	Process and reorder array items by channel dynamically.

	Parameters
	----------
	arr : array-like
		Input array containing channel-specific items.
		Expected format: ['chX-long', 'chX-medium', 'chX-short', ...]
		Where X is the channel number.

	Returns
	-------
	list
		Reordered array with items sorted for each channel:
		- Sorted from short to long within each channel
		- Channels ordered sequentially
	'''
	# Extract unique channel numbers dynamically
	channels = sorted(set(item.split('ch')[1].split('-')[0] for item in subcubes))
	
	result = []
	for ch in channels:
		# Filter for current channel
		channel_items = [item for item in subcubes if f'ch{ch}-' in item]
		
		# Reorder: short, medium, long
		channel_order = sorted(channel_items, key=lambda x: x.split('-')[-1],reverse=True)
		
		# Add to result
		result.extend(channel_order)
	
	return result

def find_neighbouring_subcubes(results):
	'''
	Identify and analyze neighboring subcubes based on their micrometre (um) ranges.

	This function processes a dictionary of subcube results to find adjacent subcubes 
	and their overlapping micrometre ranges. It sorts the subcubes and generates 
	pairs of neighboring subcubes along with their boundary ranges.

	Parameters:
	-----------

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

	Returns:
	--------

	pair : dictionary
		Containing:

			name_pairs : list of string
				List of adjacent subcube name pairs (format: 'subcube1:subcube2')
				
			um_pairs : list of [float,float]
				List of corresponding micrometre range boundaries (format: 'lower_boundary:upper_boundary')

	Notes:
	------
	- Subcubes are first sorted to determine their relative positions
	- The function captures the maximum of the lower subcube and the minimum of the 
	  upper subcube to define neighboring ranges
	'''

	#Get names of subcubes
	subcubes = results['subcube_name'].copy()
	subcubes = np.array(subcubes)

	#Save the um range for each subcube.
	um_range = {}
	for i,subcube in enumerate(subcubes):
		um_range[subcube] = [min(results['um'][i]),max(results['um'][i])]

	#Sort the subcubes in order.
	sorted_subcubes = sort_subcubes(subcubes)

	#Now neigbouring ranges should have the overlap, for each pair.
	pairs = {}
	pairs['name_pairs'] = []
	pairs['um_pairs'] = []
	for i,subcube in enumerate(sorted_subcubes):
		if (i > 0) and (i < len(sorted_subcubes)):
			lower = sorted_subcubes[i-1]
			upper = sorted_subcubes[i]
			pairs['name_pairs'].append('%s:%s'%(lower,upper))
			pairs['um_pairs'].append('%.6f:%.6f'%(um_range[lower][1],um_range[upper][0]))
	return pairs

def stitch_subcubes(results, method='ADD',baseband = 'ch3-short'):
	'''
	Stitch together overlapping spectra from neighboring subcubes.
	
	This function processes spectra from different subcubes, identifies overlapping regions,
	and applies a scaling factor to align the flux values across subcubes. The function
	currently supports multiplicative scaling, with placeholders for additive and
	mean-based approaches.
	
	Parameters:
	-----------
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
	method : string, optional
		Method used for stitching the spectra. Default is 'ADD'.
		Options:
			'MULTIPLY': Scales the flux values of the higher wavelength subcube
					   by multiplying with a factor derived from overlapping regions.
			'ADD': Adds an offset to align spectra.
			'MEAN': Takes the mean of overlapping regions (not implemented).
				
	Returns:
	--------
	results : dictionary
		The input dictionary with updated flux and flux_unc values for stitched spectra.
		The structure is preserved, but flux values are scaled to create a continuous spectrum.
		
	Notes:
	------
	- The function uses the 'spectres' function to resample spectra to the same wavelength grid
	- For 'MULTIPLY' method, the scaling factor is calculated as the mean ratio of fluxes in overlapping regions
	- The uncertainty on the scaling factor is estimated from the standard deviation of ratio values
	- The function processes subcubes sequentially from lower to higher wavelengths
	'''
	
	if baseband:
		base_inds  = np.where(np.array(results['subcube_name']) == baseband)[0][0]		
		um_base, flux_base, unc_base = [results[x][base_inds].copy() for x in ['um', 'flux', 'flux_unc']]


	# Find neighboring subcubes
	pairs = find_neighbouring_subcubes(results)
	
	# Print available subcube names for reference
	#print("Available subcube names:", np.unique(results['subcube_name']))
	
	# Process each pair of neighboring subcubes
	new_results = results.copy()

	#for debugging
	do_stitch_plot = False
	if do_stitch_plot:
		plt.close()
		inds = np.where(np.array(results['subcube_name']) == 'ch3-short')[0][0]		
		um_plot, flux_plot, unc_plot = [results[x][inds] for x in ['um', 'flux', 'flux_unc']]
		plt.plot(um_plot,flux_plot,label='baseband')

	for name_pair, um_pair in zip(pairs['name_pairs'], pairs['um_pairs']):
		# Extract neighboring subchannels
		chlow, chhigh = name_pair.split(':')
		um_lim_high, um_lim_low = np.array(um_pair.split(':'), dtype='float')
		
		# Extract the relevant spectra
		indlow = np.where(np.array(results['subcube_name']) == chlow)[0][0]
		indhigh = np.where(np.array(results['subcube_name']) == chhigh)[0][0]
		um_low, flux_low, unc_low = [results[x][indlow] for x in ['um', 'flux', 'flux_unc']]
		um_high, flux_high, unc_high = [results[x][indhigh] for x in ['um', 'flux', 'flux_unc']]
		
		# Mask all data outside of the overlap
		flux_low, unc_low = flux_low[um_low > um_lim_low], unc_low[um_low > um_lim_low]
		um_low = um_low[um_low > um_lim_low]
		flux_high, unc_high = flux_high[um_high < um_lim_high], unc_high[um_high < um_lim_high]
		um_high = um_high[um_high < um_lim_high]
		
		# Resample the chlow spectra to the spectral resolution of chhigh
		flux_low, unc_low = spectres(um_high, um_low, flux_low, spec_errs=unc_low, fill=None, verbose=True)
		



		# Apply the selected stitching method
		if method == 'MULTIPLY':
			vals = flux_low/flux_high
			factor = abs(np.nanmean(vals))
			factor_unc = abs(np.nanstd(vals))
			new_results['flux'][indhigh] *= factor
			new_results['flux_unc'][indhigh] *= factor
		elif method == 'ADD':
			vals = flux_low - flux_high
			factor = np.nanmedian(vals)
			factor_unc = np.nanstd(vals)

			#Calculate relative uncertainties.
			rel_unc = new_results['flux_unc'][indhigh]/new_results['flux'][indhigh]

			#Keep the relative uncertainties constant because of stitching.
			#NOT IMPLEMENTED, ARTIFICIALLY MAKES UNCERTAINTIES TOO LOW.
			#unc_factor = rel_unc*(new_results['flux'][indhigh] + [factor]*len(rel_unc)) - new_results['flux_unc'][indhigh]

			#new_results['flux_unc'][indhigh] += unc_factor

			if 'ch4' in name_pair:
				print('Stitching is only being done for CH4!!')
				new_results['flux'][indhigh] += factor

			print('name_pair: %s, factor: %.3E'%(name_pair,factor))


		elif method == 'MEAN':
			# Not implemented
			vals = (flux_low + flux_high) / 2
			raise ValueError('Method not yet implemented.')
		
	if do_stitch_plot:
		plot_res = merge_subcubes(new_results)
		plt.plot(plot_res['um'],plot_res['flux'],label ='post-stitch')
		plt.title('%s'%(str(name_pair)))
		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel('Wavelength (um)')
		plt.ylabel('Flux Density (Jy)')
		plt.legend()
		plt.show()

	
	if baseband:
		plot_res = merge_subcubes(new_results)
		um_plot, flux_plot, unc_plot = [plot_res[x] for x in ['um', 'flux', 'flux_unc']]

		um_inds,comm1,comm2 = np.intersect1d(um_base,um_plot,return_indices=True)
		offset = np.nanmedian(flux_base[comm1]-flux_plot[comm2])

		for j in range(len(new_results['flux'])):
			new_results['flux'][j] += offset


		#print('%s:%s, factor =%.2E+-%.2E' % (chlow, chhigh, factor, factor_unc))
	return new_results

def merge_subcubes(results,do_zero_pointing=False):
	'''
	Merge all subcubes in the results dictionary into single arrays.
	
	This function takes a results dictionary containing multiple subcubes and merges
	the wavelength, flux, and uncertainty arrays from all subcubes into single,
	continuous arrays. The function preserves all metadata from the original dictionary.
	
	Parameters:
	-----------
	results : dictionary
		Containing:
			source_name : string
				Name of the source from the filename.
			RA_centre, Dec_centre, aper_size : Saved as specified.
			subcube_name : list of string
				Names of each subcube.
			um : list of arrays
				2D list containing wavelengths of each subcube.
			flux : list of arrays
				2D list containing the flux densities of each subcube.
			flux_unc : list of arrays
				2D list containing the uncertainty on flux densities of each subcube.
	
	Returns:
	--------
	merged_results : dictionary
		A new dictionary containing:
			source_name : string
				Name of the source from the filename.
			RA_centre, Dec_centre, aper_size : Preserved from input.
			um : array
				1D array containing all wavelengths from all subcubes.
			flux : array
				1D array containing all flux densities from all subcubes.
			flux_unc : array
				1D array containing all flux uncertainties from all subcubes.
			subcube_indices : dictionary
				Dictionary mapping each subcube name to the indices of its data in the merged arrays.
	
	Notes:
	------
	- The function assumes that subcubes have been properly stitched before merging
	- The output arrays are sorted by wavelength
	- The function preserves the correspondence between wavelength, flux, and uncertainty
	- The subcube_indices dictionary allows retrieval of data from specific subcubes if needed
	'''
	# Initialize the merged results dictionary with metadata
	merged_results = {key: results[key] for key in results if key not in ['um', 'flux', 'flux_unc', 'subcube_name']}
	
	# Initialize empty lists for merging
	all_um = np.array([])
	all_flux = np.array([])
	all_flux_unc = np.array([])
	subcube_indices = {}
	
	# Merge all subcubes
	for i, subcube_name in enumerate(results['subcube_name']):
		# Get current start index
		start_idx = len(all_um)
		
		# Append data from this subcube
		all_um = np.append(all_um, results['um'][i])
		all_flux = np.append(all_flux, results['flux'][i])
		all_flux_unc = np.append(all_flux_unc, results['flux_unc'][i])
		
		# Store indices for this subcube
		end_idx = len(all_um)
		subcube_indices[subcube_name] = (start_idx, end_idx)
	
	# Sort all arrays by wavelength
	sort_indices = np.argsort(all_um)
	all_um = all_um[sort_indices]
	all_flux = all_flux[sort_indices]
	all_flux_unc = all_flux_unc[sort_indices]
	
	# Update the subcube_indices after sorting
	for subcube_name in subcube_indices:
		start_idx, end_idx = subcube_indices[subcube_name]
		indices = np.arange(start_idx, end_idx)
		new_indices = np.where(np.isin(sort_indices, indices))[0]
		subcube_indices[subcube_name] = (min(new_indices), max(new_indices) + 1)
	
	# Add merged arrays to the results dictionary
	merged_results['um'] = all_um
	merged_results['flux'] = all_flux
	merged_results['flux_unc'] = all_flux_unc
	merged_results['subcube_indices'] = subcube_indices

	if do_zero_pointing:
		if min(all_flux) < 0:
			merged_results['flux'] = all_flux + abs(min(all_flux))

	return merged_results

def read_spectra(filename):
	'''
	Read spectra dictionary from a previously saved JSON file.
	
	Converts lists back to numpy arrays to restore the original format.

	Parameters
	----------
	filename : string
		Path and name of the file containing spectra data.
	
	Returns
	-------
	results : dictionary
		Dictionary containing spectral data with numpy arrays.
	'''
	# Read from file
	with open(filename, 'r') as f:
		results = json.load(f)
	
	# Convert lists back to numpy arrays
	results['um'] = [np.array(um) for um in results['um']]
	results['flux'] = [np.array(flux) for flux in results['flux']]
	results['flux_unc'] = [np.array(flux_unc) for flux_unc in results['flux_unc']]
	
	return results