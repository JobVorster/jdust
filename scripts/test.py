from ifu_analysis.jdcontinuum import automatic_continuum_estimate
from ifu_analysis.jdpsfsub import stripe_correction
from ifu_analysis.jdutils import get_JWST_IFU_um

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS

subchannel = 'ch4-short'

fn = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/PSF_Subtraction/L1448-mm_%s_s3d_LSRcorr_PSFsub.fits'%(subchannel)

#fn = '/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_IFUAlign/L1448-mm_%s_s3d_LSRcorr.fits'%(subchannel)

saveto = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448-mm_%s_s3d_PSFsub_cont.fits'%(subchannel)
saveto_stripe = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448-mm_%s_s3d_PSFsub_stripe.fits'%(subchannel)
saveto_cont = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448-mm_%s_PSFsub_stripe_continuum.fits'%(subchannel)
hdu = fits.open(fn)
um = get_JWST_IFU_um(hdu[1].header)
wcs = WCS(hdu[1].header)
wcs = wcs.dropaxis(2)
num_std = 5

cont_cube,cont_unc = automatic_continuum_estimate(fn,saveto=saveto,num_std=num_std)

cont_cube[cont_cube < num_std*cont_unc] = np.nan

mom0 =np.nansum(cont_cube,axis = 0)


mask = np.zeros(np.shape(mom0))
masklevel = 12000
mask[mom0 > masklevel] = 1

plt.figure(figsize = (16,5))
plt.subplot(121)
plt.imshow(mom0,vmin = np.nanmean(cont_unc))
plt.contour(mom0,levels=[masklevel],cmap='Greys')
plt.show()

p = stripe_correction(hdu,mask,saveto = saveto_stripe)

cont_cube,cont_unc = automatic_continuum_estimate(saveto_stripe,saveto=saveto,num_std=num_std)

cont_cube[cont_cube < num_std*cont_unc] = np.nan

dum = abs(um[0] - um[1])
mom0 =np.nansum(cont_cube*dum,axis = 0)
spectra = np.nanmean(cont_cube,axis=(1,2))

header2D = wcs.to_header()
mom0hdu = fits.PrimaryHDU(data = mom0,header = header2D)
mom0hdu.writeto(saveto_cont,overwrite=True)

plt.figure(figsize = (16,5))
plt.subplot(121)
plt.imshow(mom0,vmin = np.nanmean(cont_unc))
plt.contour(mom0,levels=[masklevel],cmap='Greys')
plt.subplot(122)
plt.plot(spectra)
plt.show()