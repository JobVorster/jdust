from jdust_utils import *
from astropy.wcs import WCS
from glob import glob
import os.path
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from lmfit import Model
from scipy.signal import savgol_filter

def get_continuum(flux,filter_size=80):
	return savgol_filter(flux,window_length=filter_size,polyorder=3)


#sources: BHR71, HH211, L1448MM1, SerpensSMM1
source_name = 'L1448MM1'

foldername = '/home/vorsteja/Documents/JOYS/JWST_cubes/%s/'%(source_name)
spectra_foldername = '/home/vorsteja/Documents/JOYS/JDust/jwst-ifu-analysis/output-files/L1448MM1_Spectra/'
cont_filename = '/home/vorsteja/Documents/JOYS/JDust/jwst-ifu-analysis/input-files/cont-mask-%s.txt'%(source_name)

h2_lines = '/home/vorsteja/Documents/JOYS/JDust/jwst-ifu-analysis/input-files/emission-lines/h2_lines.txt'

df = pd.read_csv(h2_lines)
h2_name,h2_um = df['Line'].values,df['Wavelength'].values


spectra_filenames = glob(spectra_foldername + '*.spectra')
spectra_filenames.sort()

for test_spectra in spectra_filenames:
	if 'unstitched' in test_spectra:
		continue

	spectra = merge_subcubes(read_spectra(test_spectra))
	
	aper_name = test_spectra.split('/')[-1].split('.')[0].split('aper')[-1]
	#'source_name', 'RA_centre', 'Dec_centre', 'aper_size', 'um', 'flux', 'flux_unc', 'subcube_indices'

	um,flux,flux_unc = [spectra[x] for x in ['um', 'flux', 'flux_unc']]
	solid_angle = solid_angle_small_angle(spectra['aper_size'])
	mask = um<28
	um = um[mask]
	flux = flux[mask]
	flux_unc = flux_unc[mask]


	filter_size = 200
	filtered = np.array(get_continuum(flux,filter_size=filter_size))
	mask = np.logical_and(um<27,um>16)
	um = um[mask]
	flux = flux[mask]
	flux_unc = flux_unc[mask]
	filtered = filtered[mask]

	lambda0 = 850


	#(um,scaling,T,beta,solid_angle=None,lambda0 = 850)
	# create model
	fmodel = Model(modified_black_body)
	# create parameters -- these are named from the function arguments --
	# giving initial values
	params = fmodel.make_params(solid_angle=solid_angle, lambda0=lambda0)

	# fix solid_angle and lambda0:
	params['solid_angle'].vary = False
	params['lambda0'].vary = False

	scaling_val = 1
	if not scaling_val:
		params['scaling'].min = 0.9
		params['scaling'].max = 1.1
	else:
		params['scaling'].vary = False
		params['scaling'].value = scaling_val

	params['scaling'].min = 0.9
	params['scaling'].max = 1.1

	params['T'].min = 30
	params['T'].max = 300

	beta_val = None#-2
	if not beta_val:
		params['beta'].value = -2
		params['beta'].min = -4
		params['beta'].max = +3
	else:
		params['beta'].vary = False
		params['beta'].value = beta_val

	T_val = None
	if not T_val:
		params['T'].value = 70
		params['T'].min = 30
		params['T'].max = 150
	else:
		params['T'].vary = False
		params['T'].value = T_val

	
	result = fmodel.fit(filtered, params, um=um)
	print(result.params)
	plt.subplot(211)
	plt.plot(um,flux,label='Raw')
	plt.plot(um,filtered,label='Sav Gol Filter')
	mbb = modified_black_body(um,result.params['scaling'].value,result.params['T'].value,result.params['beta'].value,solid_angle=solid_angle,lambda0 = lambda0).value
	plt.plot(um,mbb,label='MBB')
	plt.legend()
	plt.subplot(212)
	plt.plot(um,filtered - mbb,label='Residual')
	plt.legend()
	plt.show()

	
	if (0):
		plt.figure(figsize = (16,5))
		plt.suptitle('%s, Aperture %s'%(source_name,aper_name))
		plt.subplot(131)
		plt.plot(um,flux,label='Data')
		plt.plot(um,filtered,label='median filter (%d channels)'%(filter_size))
		plt.xlabel('Wavelength')
		plt.ylabel('Flux Density (Jy)')
		plt.legend()
		plt.subplot(132)
		plt.plot(um,flux-filtered)
		plt.xlabel('Wavelength')
		plt.ylabel('Residual (Jy)')
		#exit()


		#fig = plt.figure(figsize = (16,4))
		#ax = plt.gca()
		#ax.plot(um,flux,label='data')
		#ax.plot(um,filtered,label='median_filter')

		fit_pars = fit_modified_blackbody(um, filtered, flux_unc, solid_angle=solid_angle_small_angle(spectra['aper_size']), lambda0=850, 
								T_guess=30, beta_guess=1.5, 
								T_bounds=(10, 100), beta_bounds=(0, 3))
		plt.subplot(133)
		plot_modified_blackbody_fit(plt.gca(),um, filtered, flux_unc, fit_pars, 
									 solid_angle=solid_angle_small_angle(spectra['aper_size']), lambda0=850, 
									 figsize=(10, 6), title=None)

		plt.show()
