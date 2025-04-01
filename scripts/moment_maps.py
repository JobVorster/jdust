from jdust_utils import *
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from scipy.ndimage import median_filter



#sources: BHR71, HH211, L1448MM1, SerpensSMM1
source_name = 'L1448MM1'
#foldername = '/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_IFUAlign/'
foldername = '/home/vorsteja/Documents/JOYS/JDust/jwst-ifu-analysis/output-files/PSF_Subtraction/'
#foldername = '/home/vorsteja/Documents/JOYS/JWST_cubes/%s/'%(source_name)
cont_filename = '/home/vorsteja/Documents/JOYS/JDust/jwst-ifu-analysis/input-files/cont-mask-%s.txt'%(source_name)

filenames = glob(foldername + '*.fits')
#filenames.sort()
subcubes = []
for filename in filenames:
	subcubes.append(get_subcube_name(filename))
subcubes = np.array(subcubes)
ind = np.where(subcubes =='ch4-short')[0][0]
hdu = fits.open(filenames[ind])
wcs = WCS(hdu[1].header)
wcs = wcs.dropaxis(2)
data_cube = hdu[1].data
unc_cube = hdu[2].data


um = get_JWST_IFU_um(hdu[1].header)
print(f"Data cube wavelength range: {min(um)} to {max(um)}")
cont_cube = get_cont_cube(data_cube,um,cont_filename,sep=',')
unc_cube = get_cont_cube(unc_cube,um,cont_filename,sep=',')

mom0,mom0_unc = make_moment_map(cont_cube,unc_cube,0,len(um))

mom0 = median_filter(mom0,size = 2)

cont_min, cont_max = 50,1000
contour_levels = [i for i in np.logspace(np.log10(cont_min),np.log10(cont_max),10)]
fig, ax1, ax2, ax3 = make_snr_figures(mom0,mom0_unc,wcs,contour_levels=contour_levels,contour_cmap='Reds')

####################################################
fig.suptitle(source_name)
# Set RA and Dec format with higher precision
ax1.coords[0].set_major_formatter('hh:mm:ss.sss')  # RA with milliarcsec precision
ax1.coords[1].set_major_formatter('dd:mm:ss.ss')   # Dec with centiarcsec precision
ax2.coords[0].set_major_formatter('hh:mm:ss.sss')
ax2.coords[1].set_major_formatter('dd:mm:ss.ss')
ax3.coords[0].set_major_formatter('hh:mm:ss.sss')
ax3.coords[1].set_major_formatter('dd:mm:ss.ss')
####################################################
plt.show()
