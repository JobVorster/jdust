import matplotlib.pyplot as plt 
import numpy as np
from astropy.io import fits 

fn_co2 = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/line_cubes/co2/L1448-mm_CO2 _14.9838297057067um.fits'
fn_cont = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/PSF_Subtraction/continuum_maps/L1448-mm_ch3-medium_s3d_LSRcorr_stripecorr_psfsub_cont2D.fits'

co2map = fits.open(fn_co2)[0].data
contmap = fits.open(fn_cont)[0].data

plt.scatter(contmap,co2map)
plt.show()
