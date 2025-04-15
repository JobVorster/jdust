from ifu_analysis.jdcontinuum import automatic_continuum_estimate
from ifu_analysis.jdpsfsub import stripe_correction

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

subchannel = 'ch4-long'

fn = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/PSF_Subtraction/L1448-mm_%s_s3d_LSRcorr_PSFsub.fits'%(subchannel)

#fn = '/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_IFUAlign/L1448-mm_%s_s3d_LSRcorr.fits'%(subchannel)

saveto = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448-mm_%s_s3d_PSFsub_cont.fits'%(subchannel)
saveto_stripe = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448-mm_%s_s3d_PSFsub_stripe.fits'%(subchannel)
hdu = fits.open(fn)

num_std = 15

cont_cube,cont_unc = automatic_continuum_estimate(fn,saveto=saveto,num_std=num_std)

cont_cube[cont_cube < num_std*cont_unc] = np.nan

mom0 =np.nansum(cont_cube,axis = 0)


mask = np.zeros(np.shape(mom0))
masklevel = 10000
mask[mom0 > masklevel] = 1

plt.figure(figsize = (16,5))
plt.subplot(121)
plt.imshow(mom0,vmin = np.nanmean(cont_unc))
plt.contour(mom0,levels=[masklevel],cmap='Greys')
plt.show()

p = stripe_correction(hdu,mask,saveto = saveto_stripe)

cont_cube,cont_unc = automatic_continuum_estimate(saveto_stripe,saveto=saveto,num_std=num_std)

cont_cube[cont_cube < num_std*cont_unc] = np.nan

mom0 =np.nansum(cont_cube,axis = 0)
spectra = np.nanmean(cont_cube,axis=(1,2))



plt.figure(figsize = (16,5))
plt.subplot(121)
plt.imshow(mom0,vmin = np.nanmean(cont_unc))
plt.contour(mom0,levels=[masklevel],cmap='Greys')
plt.subplot(122)
plt.plot(spectra)
plt.show()