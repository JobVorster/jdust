import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from astropy.io import fits

input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/'
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/publication_figures/'

fn_bkg = input_foldername + 'L1448-mm_ch3-long_stripebkg.fits'
fn_cont = input_foldername + 'L1448-mm_ch3-long_cont.fits'
aanda_col_width = 3.46457 #inches
cont_cube,unc_cont_cube,dq_cont_cube = [fits.open(fn_cont)[i].data for i in [1,2,3]]
bkg = fits.open(fn_bkg)[1].data
cont_cube_sub = cont_cube.copy() - bkg

cont_cube[cont_cube < 5*unc_cont_cube] = np.nan
cont_cube_sub[cont_cube_sub < 5*unc_cont_cube] = np.nan

cont_map_nosub = np.nansum(cont_cube,axis=0)
cont_map_sub = np.nansum(cont_cube_sub,axis=0)
plt.figure(figsize=(2*aanda_col_width,aanda_col_width))
plt.subplot(121)
plt.imshow(np.log10(cont_map_nosub),origin='lower',cmap='magma')
plt.colorbar(location='top',label='log10(Flux Density (MJy sr$^{-1}$ $\mu$m))',pad=0)

plt.gca().annotate(
        'a)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='white', edgecolor='orange', pad=3.0))

plt.gca().annotate(
        'Before stripe correction',
        xy=(0, 0.9), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='white', edgecolor='orange', pad=3.0))

plt.subplot(122)
plt.imshow(np.log10(cont_map_sub),origin='lower',cmap='magma')
plt.colorbar(location='top',label='log10(Flux Density (MJy sr$^{-1}$ $\mu$m))',pad=0)

plt.gca().annotate(
        'b)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='white', edgecolor='orange', pad=3.0))

plt.gca().annotate(
        'After stripe correction',
        xy=(0, 0.9), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='white', edgecolor='orange', pad=3.0))


plt.savefig(output_foldername + 'stripe_correction_example.png',bbox_inches='tight',dpi=200)