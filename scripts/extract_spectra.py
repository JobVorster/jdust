from ifu_analysis.jdspextract import *
from ifu_analysis.jdutils import *
from ifu_analysis.jdcontinuum import get_cont_cube
from astropy.wcs import WCS
from glob import glob
import os
import os.path
from scipy.ndimage import median_filter
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt


# ============================================================================
# USER-DEFINED PARAMETERS
# ============================================================================

# Source configuration
source_name = 'BHR71'

# Processing options
do_flooring = False  # Whether to set low flux measurements to the 3 sigma level
do_zero_pointing = False
spectral_extraction_method = 'PIPELINE'  # Options: 'NAIVE' or 'PIPELINE'

# Spectra types to process
# Options: 'BASE', 'BASE_CONT', 'PSFSUB', 'PSFSUB_CONT', 'PSFMODEL'
spectra_types_to_process = ['PSFSUB', 'BASE', 'PSFMODEL']

# Input folders for each spectra type
# Each folder should contain all 12 band files for that spectra type
# Set to None for spectra types you don't want to process
spectra_input_folders = {
    'BASE': '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/BHR71/ifualign/CUBES/BASE/',
    'PSFSUB': '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/BHR71/ifualign/CUBES/PSFSUB/',
    'PSFMODEL': '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/BHR71/ifualign/CUBES/PSFMODEL/',
}

# Output folder (spectra type subfolders will be created automatically)
base_output_folder = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/BHR71/ifualign/PSF_Subtraction/spectra/'

# Aperture file
aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt' % (source_name)

# Band names (must have exactly 12 bands)
fn_band_arr = ['ch1-short', 'ch1-medium', 'ch1-long', 'ch2-short', 'ch2-medium', 'ch2-long',
               'ch3-short', 'ch3-medium', 'ch3-long', 'ch4-short', 'ch4-medium', 'ch4-long']

baseband = 'ch3-short'

# ============================================================================
# END USER-DEFINED PARAMETERS
# ============================================================================


def check_existing_files(output_folder, source_name, aper_names):
	"""
	Check if spectral extraction files already exist in the output folder.
	Returns True if user wants to proceed (overwrite), False otherwise.
	"""
	if not os.path.exists(output_folder):
		return True
	
	# Check for existing spectrum files
	existing_files = []
	for aper_name in aper_names:
		spectrum_file = os.path.join(output_folder, '%s_aper%s.spectra' % (source_name, aper_name))
		if os.path.exists(spectrum_file):
			existing_files.append(spectrum_file)
	
	if existing_files:
		print("\n" + "="*70)
		print("WARNING: Existing spectral extraction files found in:")
		print(output_folder)
		print("\nExisting files:")
		for f in existing_files:
			print("  - " + os.path.basename(f))
		print("="*70)
		response = input("\nDo you want to overwrite these files? (yes/no): ").strip().lower()
		return response in ['yes', 'y']
	
	return True


def get_filenames_for_spectra_type(which_spectra, source_name, fn_band_arr, spectra_input_folders):
	"""
	Get the appropriate filenames based on the spectra type.
	Searches the specified folder for files containing the source name and each band name.
	Validates that all 12 bands are present.
	"""
	if which_spectra not in spectra_input_folders:
		raise ValueError(f'Spectra type {which_spectra} not found in spectra_input_folders dictionary')
	
	foldername = spectra_input_folders[which_spectra]
	
	if foldername is None:
		raise ValueError(f'No input folder specified for spectra type {which_spectra}')
	
	if not os.path.exists(foldername):
		raise FileNotFoundError(f'Input folder does not exist: {foldername}')
	
	# Get all .fits files in the folder
	all_fits_files = glob(os.path.join(foldername, '*.fits'))
	
	if len(all_fits_files) == 0:
		raise FileNotFoundError(f'No .fits files found in folder: {foldername}')
	
	# Find files for each band
	filenames = []
	missing_bands = []
	
	for band in fn_band_arr:
		# Look for files containing both the source name and band name
		matching_files = [f for f in all_fits_files if band in os.path.basename(f)]
		
		if len(matching_files) == 0:
			missing_bands.append(band)
		elif len(matching_files) == 1:
			filenames.append(matching_files[0])
		else:
			# Multiple matches - try to find the most appropriate one
			# Prefer exact band match
			print(f"WARNING: Multiple files found for band {band}:")
			for f in matching_files:
				print(f"  - {os.path.basename(f)}")
			print(f"Using: {os.path.basename(matching_files[0])}")
			filenames.append(matching_files[0])
	
	# Check if all bands were found
	if missing_bands:
		raise FileNotFoundError(
			f'Missing bands for {which_spectra} in folder {foldername}:\n' +
			f'Missing bands: {", ".join(missing_bands)}\n' +
			f'Expected 12 bands: {", ".join(fn_band_arr)}'
		)
	
	if len(filenames) != 12:
		raise ValueError(f'Expected 12 band files, found {len(filenames)} for spectra type {which_spectra}')
	
	print(f"Found all 12 bands for {which_spectra} in {foldername}")
	
	return filenames


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

# Read aperture information
if not os.path.isfile(aperture_filename):
	raise FileNotFoundError(f"Aperture file not found: {aperture_filename}")

aper_names, aper_sizes, coord_list = read_aperture_ini(aperture_filename)

# Process each spectra type
for which_spectra in spectra_types_to_process:
	print("\n" + "="*70)
	print(f"Processing spectra type: {which_spectra}")
	print("="*70)
	
	# Get filenames for this spectra type
	filenames = get_filenames_for_spectra_type(which_spectra, source_name, fn_band_arr, spectra_input_folders)
	
	# Set up output folder and create if it doesn't exist
	output_foldername = os.path.join(base_output_folder, which_spectra, '')
	os.makedirs(output_foldername, exist_ok=True)
	print(f"Output folder: {output_foldername}")
	
	# Check if files already exist and get user confirmation
	if not check_existing_files(output_foldername, source_name, aper_names):
		print(f"Skipping {which_spectra} - user chose not to overwrite existing files.\n")
		continue
	
	# Get subcube names
	subcubes = []
	for filename in filenames:
		subcubes.append(get_subcube_name(filename))
	subcubes = np.array(subcubes)
	
	# Process each aperture
	for aper_name, aper_size, (RA_centre, Dec_centre) in zip(aper_names, aper_sizes, coord_list):
		print('Doing aperture %s of %s' % (aper_name, source_name))
		results = unstitched_spectrum_from_cube_list(filenames, RA_centre, Dec_centre, aper_size, method=spectral_extraction_method)
		
		# Plot unstitched spectrum
		if (1):
			plt.close()
			plt.figure(figsize=(16, 4))
			for j in range(len(results['subcube_name'])):
				inds = np.where(results['um'][j] < 27.5)
				plt.plot(results['um'][j][inds], results['flux'][j][inds], label=subcubes[j])
				plt.plot(results['um'][j][inds], results['flux_unc'][j][inds], color='grey')
			plt.xlabel('Wavelength (um)')
			plt.suptitle('Source: %s, Aperture: %s UNSTITCHED' % (source_name, aper_name))
			plt.ylabel('Flux Density (Jy)')
			plt.yscale('log')
			plt.legend()
			plt.minorticks_on()
			plt.grid(which='both', alpha=0.3)
			plt.savefig(output_foldername + '%s_aper%s_spectrum_unstitched.jpg' % (source_name, aper_name), dpi=300, bbox_inches='tight')
		
		print('Aperture: %s, which_spectra: %s' % (aper_name, which_spectra))
		results_stitched = stitch_subcubes(results,baseband =baseband)
		results_merged = merge_subcubes(results_stitched, do_zero_pointing=do_zero_pointing)
		save_spectra(results, output_foldername + '%s_aper%s_unstitched.spectra' % (source_name, aper_name))
		
		# The merged spectra gives a problem with saving.
		save_spectra(results_stitched, output_foldername + '%s_aper%s.spectra' % (source_name, aper_name))
		
		# Plot stitched spectrum
		if (1):
			plt.close()
			plt.figure(figsize=(16, 4))
			for j in range(len(results_stitched['subcube_name'])):
				inds = np.where(results_stitched['um'][j] < 27)
				plt.plot(results_stitched['um'][j][inds], results_stitched['flux'][j][inds], label=subcubes[j])
				plt.plot(results_stitched['um'][j][inds], results_stitched['flux_unc'][j][inds], color='grey')
			plt.xlabel('Wavelength (um)')
			plt.suptitle('Source: %s, Aperture: %s STITCHED' % (source_name, aper_name))
			plt.ylabel('Flux Density (Jy)')
			plt.legend()
			plt.yscale('log')
			plt.minorticks_on()
			plt.grid(which='both', alpha=0.3)
			plt.savefig(output_foldername + '%s_aper%s_spectrum.jpg' % (source_name, aper_name), dpi=300, bbox_inches='tight')

print("\n" + "="*70)
print("All processing complete!")
print("="*70)