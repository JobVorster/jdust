from ifu_analysis.jdutils import get_JWST_IFU_um, interpolate_nan, unpack_hdu,get_subcube_name,get_JWST_PSF


from astropy.io import fits
import numpy as np 
from pybaselines import Baseline
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
import pandas as pd
from glob import glob
from scipy.optimize import curve_fit

def gaussian(x,A,mu,sigma):
	return A*np.exp(-(x-mu)**2/(2*sigma**2))

def um_to_vlsr(um,lambda0):
	'''
	Convert wavelength to vlsr, with a specified centre wavelength.

	The formula and sign should still be DOUBLE CHECKED!
	
	Parameters
	----------

	um : 1D array
		Wavelength array in microns.

	lambda0 : float
		Centre reference wavelength.

	Returns
	-------

	vlsr : 1D array
		Velocities in units of km s-1
	'''
	c_light = const.c 
	vlsr = c_light*((um-lambda0)/lambda0)
	return vlsr.to(u.km*u.s**-1).value

def strip_spaces(str1):
	numerics = ''
	non_spacesnumerics = ''
	for c in str1:
		if c != ' ':
			if c == 'I':
				numerics += c 
			else:
				non_spacesnumerics += c
	return non_spacesnumerics + ' ' + numerics

def line_figure(line_map,line_um,line_name,wcs):
	beam_size = get_JWST_PSF(line_um)
	fig = plt.figure(figsize = (4,4))
	ax = plt.subplot(projection=wcs)
	shp = np.shape(line_map)
	ax.imshow(line_map,origin='lower',cmap='gist_stern')
	#ax.text(0.1*shp[0],0.65*shp[1],'%.2f'%(line_um) + r' $\mu$m',fontsize = 12,color='white')
	#ax.text(0.1*shp[0],0.72*shp[1],'%s'%(line_name) ,fontsize = 12,color='white', fontweight='bold')

	ra,dec = ax.coords
	ra.set_axislabel('')
	dec.set_axislabel('')
	return fig,ax

def get_line_parameters(fn):
	#Columns, separated by tabs: Wv, microns	A, sec-1	E_u/k	g_u	Species	Transition, v	Transition, J
	df = pd.read_csv(fn,sep='\t')
	cols = df.columns

	inds = np.where(np.logical_and(df['Wv, microns'].values > 4.90027,df['Wv, microns'].values < 27.5))

	if 'Wv, microns' in cols:
		waves = df['Wv, microns'].values[inds]
	else:
		waves = []

	if 'A, sec-1' in cols:
		a_coeff = df['A, sec-1'].values[inds]
	else:
		a_coeff = []

	if 'E_u/k' in cols:
		E_upper = df['E_u/k'].values[inds]
	else:
		E_upper = []

	if 'g_u' in cols:
		g_upper = df['g_u'].values[inds]
	else:
		g_upper = []

	if 'Species' in cols:
		species = [strip_spaces(x) for x in df['Species'].values[inds]]
	else:
		species = []

	if 'Transition, v' in cols:
		vtrans = df['Transition, v'].values[inds]
	else:
		vtrans = []

	if 'Transition, J' in cols:
		jtrans = df['Transition, J'].values[inds]
	else:
		jtrans = []

	return waves, species, a_coeff, E_upper, g_upper,vtrans,jtrans


def choose_subchannel(lambda0,foldername,selecter,header_hdu):
	filenames = glob(foldername + selecter)
	filenames.sort()



	for fn in filenames:
		subcube = get_subcube_name(fn)
		um = get_JWST_IFU_um(fits.open(fn)[header_hdu].header)
		if (lambda0 > min(um)) and (lambda0 < max(um)):
			return fn,subcube


def get_line_cube(fn,lambda0,N_chans,store_type='DATA',N_sigma=5):
	'''
	Retrieves a continuum subtracted line cube from a specified cube, at a specific wavelength. The 
	algorithm uses pybaselines pspline_arpls algorithm to estimate the continuum baseline, and then subtracts it.

	The function returns wavelengths, velocities, raw data, continuum estimate, line estimate, uncertainties and data quality cubes.

	Parameters
	----------

	fn : string
		Filename of the IFU cube.

	lambda0 : float
		Reference wavelength.

	N_chans : integer
		Number of channels to take around the reference wavelength.

	Returns
	-------

	um : 1D array
		Wavelengths around the line.

	vlsr : 1D array
		Velocities around the line.

	data_cube : 1D array
		Raw data around the line.

	cont_cube : 1D array
		Continuum estimate.

	line_cube : 1D array
		Continuum subtracted data.

	unc_cube : 1D array
		Uncertainties around the line (untouched by the algorithm at this stage).

	dq_cube : 1D array
		Data quality around the line (untouched by the algorithm at this stage).
	'''

	data_cube,unc_cube,dq_cube,hdr,um,shp = unpack_hdu(fn)

	ind0 = np.digitize([lambda0],um)[0]
	if (ind0 == 0) or (ind0 == len(um)):
		raise ValueError('Wavelength %.2f is outside of cube wavelength range (%.2f,%.2f) um.'%(lambda0,min(um),max(um)))
	else:
		#Now only taking N_chans around the centre.
		inds = np.arange(max(ind0-N_chans//2,0),min(ind0 + N_chans//2,len(um)))
		um = um[inds]
		data_cube = data_cube[inds]
		unc_cube = unc_cube[inds]
		dq_cube = dq_cube[inds]

		vlsr = um_to_vlsr(um,lambda0)
		line_cube = np.full(np.shape(data_cube),np.nan)
		cont_cube = np.full(np.shape(data_cube),np.nan)
		#For each 2D pixel, get the baseline.
		for idx in range(shp[0]):
			for idy in range(shp[1]):

				flux = data_cube[:,idx,idy]
				if sum(np.isfinite(flux)) > 0:
					flux = interpolate_nan(flux)

					baseline_fitter = Baseline(um, check_finite=True)
					baseline,params = baseline_fitter.pspline_arpls(flux)
					#baseline, params = baseline_fitter.fabc(flux, lam=curvature,scale=scale,num_std=num_std)

					#mask = params['mask']

					line_flux = flux - baseline
					popt = None
					if store_type == 'DATA':
						line_cube[:,idx,idy] = line_flux
					elif store_type == 'MODEL':
						line_ind = np.digitize([lambda0],um)[0]
						#Check if there is a detection around our wavelength.
						if (line_flux[line_ind - 1] >= N_sigma*unc_cube[line_ind - 1,idx,idy]) |(line_flux[line_ind] >= N_sigma*unc_cube[line_ind,idx,idy])|(line_flux[line_ind + 1] >= N_sigma*unc_cube[line_ind + 1,idx,idy]):
							print('TRYING FITTING')
							try:
								neighbouring_points = [line_flux[line_ind + x] for x in [-1,0,1]]
								p0 = [max(neighbouring_points),um[line_ind],0.3]
								popt,pcov = curve_fit(gaussian,um,line_flux,p0=p0) #,sigma=unc_cube[:,idx,idy]/line_flux

								line_cube[:,idx,idy] = gaussian(um,*popt)
								print('FITTING SUCCEEDED')
							except:
								print('FITTING FAILED')
								line_cube[:,idx,idy] = [np.nan]*len(um)

								if (1):
									plt.figure(figsize = (16,9))
									plt.subplot(121)
									plt.plot(um,flux,label='data',marker='o')
									plt.title('Pixel (%s,%s)'%(idy,idx))
									plt.plot(um,baseline,label='baseline')
									plt.vlines(um[line_ind],ymin=min(flux),ymax = max(flux),color='grey',linestyle='dashed')
									plt.legend()
									plt.xlabel('um')
									plt.ylabel('flux')
									plt.subplot(122)
									plt.plot(um,line_flux,label='cont subtracted',marker='o')
									plt.plot(um,unc_cube[:,idx,idy],label='unc')
									plt.plot(um,5*unc_cube[:,idx,idy],label='5x unc')

									if popt is not None:
										plt.plot(um,gaussian(um,*popt),color='red',label='model')

									plt.vlines(um[line_ind],ymin=min(line_flux),ymax = max(line_flux),color='grey',linestyle='dashed')
									plt.legend()
									plt.xlabel('um')
									plt.ylabel('flux')
									plt.show()


						else:
							line_cube[:,idx,idy] = [np.nan]*len(um)

					cont_cube[:,idx,idy] = baseline





					



		return um, vlsr, data_cube, cont_cube, line_cube, unc_cube, dq_cube









