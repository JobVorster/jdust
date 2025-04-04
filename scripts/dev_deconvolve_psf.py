from ifu_analysis.jdpsfsub import generate_single_miri_mrs_psf
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import richardson_lucy
from skimage.filters import unsharp_mask

fn = '/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_IFUAlign/L1448-mm_ch3-short_s3d_LSRcorr.fits'

subband = '3A'
channel = 200

hdu = fits.open(fn)

chmap = hdu[1].data[channel]
chmap[np.isnan(chmap)] = 0
shp = np.shape(chmap)

PSF_map, pix_scale = generate_single_miri_mrs_psf(subband,channel,fov_pixels=30)

deconv = richardson_lucy(chmap,PSF_map,num_iter=1000,clip = False)
PSF_map /= np.nanmax(PSF_map)

chmap /= np.nanmax(chmap)

plt.subplot(131)
plt.imshow(chmap)
plt.subplot(132)
plt.imshow(PSF_map)
plt.subplot(133)
plt.imshow(deconv)

plt.show()