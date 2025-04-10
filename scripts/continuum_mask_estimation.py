from ifu_analysis.jdspextract import load_spectra,merge_subcubes
from ifu_analysis.jdutils import get_JWST_IFU_um
from ifu_analysis.jdpsfsub import stripe_correction

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from pybaselines import Baseline
from astropy.io import fits
from tqdm import tqdm
from matplotlib.colors import LogNorm
#fn = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_Spectra/L1448MM1_aperPS.spectra'

#results = merge_subcubes(load_spectra(fn))

#um = results['um']
#flux = results['flux']
#flux_unc = results['flux_unc']

subchannel = 'ch4-medium'

hdu_fn = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/PSF_Subtraction/L1448-mm_%s_s3d_LSRcorr_PSFsub.fits'%(subchannel)
#hdu_fn = '/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_IFUAlign/L1448-mm_%s_s3d_LSRcorr.fits'%(subchannel)
hdu = fits.open(hdu_fn)

hdr = hdu[1].header


data_cube = hdu[1].data
unc_cube = hdu[2].data
shp = np.shape(data_cube[0,:,:])

cont_map = np.full(shp,np.nan)
ch_count_map = np.full(shp,np.nan)


sum_data = False
n_sigma = 3

for idx in tqdm(range(shp[0])):
	for idy in range(shp[1]):

		um = get_JWST_IFU_um(hdr)
		flux = data_cube[:,idx,idy]
		flux_unc = unc_cube[:,idx,idy]
		inds = um < 27.5

		um = np.array(um[inds])
		flux = np.array(flux[inds])
		
		um = um[np.isfinite(flux)]
		flux = flux[np.isfinite(flux)]

		if len(um)!=0:
			try:
				lam,p = 1e4,0.01

				baseline_fitter = Baseline(um, check_finite=True)
				baseline, params = baseline_fitter.fabc(flux, lam=1e4,scale=3,num_std=3)
				mask = params['mask']
				dum = abs(um[0]-um[1])

				flux_sum = flux[mask]
				unc_sum = flux_unc[mask]

				if (idx == 29) & (idy == 26):
					plt.figure(figsize = (16,5))
					plt.plot(um,flux,alpha=0.3,color='Grey',label='data')
					plt.plot(um[mask],flux_sum,color='red',label='data classified as baseline')
					plt.xlabel('Wavelength (um)')
					plt.ylabel('Flux Density (MJy sr-1)')
					plt.title('Spectra from pixel (x,y) = (%d,%d) of cube %s'%(idx,idy,subchannel))
					plt.legend()
					plt.show()
				if sum_data:
					flux_sum = flux_sum[flux_sum>n_sigma*unc_sum]
				else:
					flux_sum = baseline[baseline > n_sigma*flux_unc]

				cont_map[idx,idy] = np.sum(flux_sum*dum)
			except:
				continue

#cont_map[cont_map < 4] = 0

mask = np.zeros(np.shape(cont_map))
mask[cont_map >40] = 1 
plt.figure(figsize = (10,5))
plt.subplot(121)
plt.imshow(cont_map,cmap='gist_stern',vmin=0)
plt.title('%s Without striping correction'%(subchannel))
plt.colorbar(location='top',fraction = 0.046)
plt.contour(mask)


cubeavg_wtd, cubebkg_wtd, bkgsub_wtd = stripe_correction(hdu,mask)


cont_map = np.full(shp,np.nan)

cont_map_low =  np.full(shp,np.nan)
cont_map_high =  np.full(shp,np.nan)

ch_count_map = np.full(shp,np.nan)
for idx in tqdm(range(shp[0])):
	for idy in range(shp[1]):

		um = get_JWST_IFU_um(hdr)
		flux = bkgsub_wtd[:,idx,idy]
		flux_unc = unc_cube[:,idx,idy]
		inds = um < 27.5
		um_mid = np.mean(um)

		um = np.array(um[inds])
		flux = np.array(flux[inds])
		
		um = um[np.isfinite(flux)]
		flux = flux[np.isfinite(flux)]

		if len(um)!=0:
			try:
				lam,p = 1e4,0.01

				baseline_fitter = Baseline(um, check_finite=True)
				baseline, params = baseline_fitter.fabc(flux, lam=lam,scale=3,num_std=3)
				mask = params['mask']
				dum = abs(um[0]-um[1])

				flux_sum = flux[mask]
				unc_sum = flux_unc[mask]


				if sum_data:
					flux_sum = flux_sum[flux_sum>n_sigma*unc_sum]
				else:
					flux_sum = baseline[baseline > n_sigma*flux_unc]

				cont_map[idx,idy] = np.sum(flux_sum*dum)

				cont_map_low[idx,idy] = np.sum(flux_sum[um<um_mid]*dum)
				cont_map_high[idx,idy] = np.sum(flux_sum[um>um_mid]*dum)

			except:
				continue

plt.subplot(122)
plt.imshow(cont_map,cmap='gist_stern',vmin=0)
plt.title('%s With striping correction'%(subchannel))
plt.colorbar(location='top',fraction = 0.046)
plt.contour(cont_map)
plt.show()
exit()

plt.figure(figsize=(16,16))
plt.imshow(cont_map,cmap='gist_stern',vmin=0)
plt.colorbar(location='top',fraction = 0.046)
plt.contour(cont_map)
plt.show()


plt.subplot(323)
plt.imshow(data_cube[226],cmap='gist_stern',vmin=0,vmax=90)
plt.colorbar(location='top',fraction = 0.046)

plt.subplot(324)
plt.imshow(cubebkg_wtd[226],cmap='gist_stern',vmin=0,vmax=90)
plt.colorbar(location='top',fraction = 0.046)

plt.subplot(325)
plt.imshow(bkgsub_wtd[226],cmap='gist_stern',vmin=0,vmax=90)
plt.colorbar(location='top',fraction = 0.046)


#plt.subplot(223)
#plt.imshow(cont_map_high/cont_map_low,cmap='gist_stern',vmin=0)
#plt.colorbar(location='top',fraction = 0.046)#
#plt.subplot(224)


plt.show()


exit()

plt.contour(cont_map,cmap='Greys')
plt.show()
exit()
#spline = CubicSpline(um_fit,baseline)

cont_channels = ['mask']

mask = params['mask']

plt.figure(figsize = (10,6))
plt.plot(um[mask],flux[mask],color='blue',marker='o')
plt.plot(um,flux,color='black',alpha=0.5,zorder=0)
plt.plot(um,baseline,color='red',zorder=1)
plt.show()


#Aperture correction
#Smoothing all cubes to the same resolution, with webbpsf.