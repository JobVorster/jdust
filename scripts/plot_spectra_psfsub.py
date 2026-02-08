from ifu_analysis.jdspextract import load_spectra,merge_subcubes,read_aperture_ini
import numpy as np
import matplotlib.pyplot as plt
import os

input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/spectral_extraction/'

#['BASE','PSFMODEL','PSFSUB']



aperture = 'A5'
source_name = 'L1448MM1'

aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt'%(source_name)
output_foldername ='/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/spectral_extraction/spectra_plots/'

if os.path.isfile(aperture_filename):
	aper_names,aper_sizes,coord_list = read_aperture_ini(aperture_filename)

for aperture in aper_names:

	fn_base = input_foldername + 'BASE/L1448MM1_aper%s.spectra'%(aperture)
	fn_psfmodel = input_foldername + 'PSFMODEL/L1448MM1_aper%s.spectra'%(aperture)
	fn_psfsub = input_foldername + 'PSFSUB/L1448MM1_aper%s.spectra'%(aperture)



	sp_base,sp_psfmodel,sp_psfsub = merge_subcubes(load_spectra(fn_base)),merge_subcubes(load_spectra(fn_psfmodel)),merge_subcubes(load_spectra(fn_psfsub))



	um_base,flux_base,unc_base = [sp_base[x] for x in ['um', 'flux', 'flux_unc']]
	um_psfmodel,flux_psfmodel,unc_psfmodel = [sp_psfmodel[x] for x in ['um', 'flux', 'flux_unc']]
	um_psfsub,flux_psfsub,unc_psfsub = [sp_psfsub[x] for x in ['um', 'flux', 'flux_unc']]

	#inds_base = np.where(flux_base > 3*unc_base)
	#inds_psfmodel = np.where(flux_psfmodel > 3*unc_psfmodel)
	#inds_psfsub = np.where(flux_psfsub > 3*unc_psfsub)
	plt.close()
	plt.figure(figsize = (10,4))
	plt.scatter(um_base,flux_base,color='blue',s=0.7)
	plt.scatter(um_psfmodel,flux_psfmodel,label='PSF Model',color='red',s=0.7)
	plt.scatter(um_psfsub,flux_psfsub,label='PSF Sub',color='orange',s=0.7)
	plt.xlabel(r'Wavelength ($\mu$m)')
	plt.ylabel('Flux Density (Jy)')
	plt.title('L1448MM Aperture: %s'%(aperture))
	plt.yscale('log')
	plt.legend()
	plt.grid(which='major',alpha=0.2,linestyle='dotted')
	plt.savefig(output_foldername + '%s_%s_spectra.png'%(source_name,aperture),dpi=200,bbox_inches='tight')