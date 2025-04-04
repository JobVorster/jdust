from ifu_analysis.jdspextract import load_spectra,merge_subcubes
from ifu_analysis.jdutils import get_JWST_IFU_um

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


hdu_fn = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/PSF_Subtraction/L1448-mm_ch4-short_s3d_LSRcorr_PSFsub.fits'
hdu = fits.open(hdu_fn)

hdr = hdu[1].header


data_cube = hdu[1].data
unc_cube = hdu[2].data
shp = np.shape(data_cube[0,:,:])

cont_map = np.zeros(shp)

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
				baseline, params = baseline_fitter.fabc(flux, lam=lam,scale=3,num_std=3)
				mask = params['mask']
				dum = abs(um[0]-um[1])

				flux_sum = flux[mask]
				unc_sum = flux_unc[mask]


				flux_sum = flux_sum[flux_sum>5*unc_sum]

				cont_map[idx,idy] = np.nansum(flux_sum*dum)
			except:
				continue
plt.imshow(cont_map,cmap='gist_stern')
#plt.contour(cont_map,cmap='Greys')
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
