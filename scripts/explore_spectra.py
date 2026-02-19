from ifu_analysis.jdspextract import load_spectra,merge_subcubes
from ifu_analysis.jdfitting import prepare_spectra_for_fit,convert_flux_wl
import matplotlib.pyplot as plt 
import numpy as np
from glob import glob

input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/pre20092025/HH211_Spectra/'
fn = input_foldername + 'HH211_aperA1.spectra'

# Grab spectra
spec_dict = load_spectra(fn)
spec_dict = merge_subcubes(spec_dict)
um, flux, flux_unc = [spec_dict[x] for x in ['um','flux','flux_unc']]
um_cut = 27.5

plt.plot(um,flux)
plt.xlabel('Wavelength (um)')
plt.ylabel('Flux Density (Jy)')
plt.show()