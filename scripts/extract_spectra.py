from jdust_utils import *
from astropy.wcs import WCS
from glob import glob
import os.path
from scipy.ndimage import median_filter
from matplotlib.colors import LogNorm




#sources: BHR71, HH211, L1448MM1, SerpensSMM1
source_name = 'L1448MM1'

foldername = '/home/vorsteja/Documents/JOYS/JWST_cubes/%s/'%(source_name)
#foldername = '/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_PSFSub/'
cont_filename = '/home/vorsteja/Documents/JOYS/JDust/jwst-ifu-analysis/input-files/cont-mask-%s.txt'%(source_name)
aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/jwst-ifu-analysis/input-files/aperture-ini-%s.txt'%(source_name)


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


if os.path.isfile(aperture_filename):
	aper_names,aper_sizes,coord_list = read_aperture_ini(aperture_filename)
	plot_apertures(aper_names,aper_sizes,coord_list,ax1,wcs,color='White')

plt.savefig('./output-files/%s_continuum_SNR.jpg'%(source_name),dpi=300,bbox_inches='tight')

if os.path.isfile(aperture_filename):
	for aper_name,aper_size,(RA_centre,Dec_centre) in zip(aper_names,aper_sizes,coord_list):
		results = unstitched_spectrum_from_cube_list(filenames,RA_centre,Dec_centre,aper_size)

		if (0):
			plt.close()
			plt.figure(figsize = (16,4))
			for j in range(len(results['subcube_name'])):
				inds = np.where(results['um'][j] < 27)
				plt.plot(results['um'][j][inds],results['flux'][j][inds],color='black')
			plt.xlabel('Wavelength (um)')
			plt.suptitle('Source: %s, Aperture: %s'%(source_name,aper_name))
			plt.ylabel('Flux Density (Jy)')
			plt.minorticks_on()
			plt.grid(which ='both',alpha=0.3)
			plt.savefig('output-files/%s_aper%s_spectrum_unstitched.jpg'%(source_name,aper_name),dpi=300,bbox_inches='tight')

		results_stitched = stitch_subcubes(results)
		results_merged = merge_subcubes(results_stitched)
		save_spectra(results,'./output-files/%s_aper%s_unstitched.spectra'%(source_name,aper_name))


		#The merged spectra gives a problem with saving.
		#save_spectra(results_stitched,'./output-files/%s_aper%s.spectra'%(source_name,aper_name))

		
		if (1):
			plt.close()
			plt.figure(figsize = (16,4))
			for j in range(len(results_stitched['subcube_name'])):
				inds = np.where(results_stitched['um'][j] < 27)
				plt.plot(results_stitched['um'][j][inds],results_stitched['flux'][j][inds],color='black')
			plt.xlabel('Wavelength (um)')
			plt.suptitle('Source: %s, Aperture: %s'%(source_name,aper_name))
			plt.ylabel('Flux Density (Jy)')
			plt.minorticks_on()
			plt.grid(which ='both',alpha=0.3)
			plt.savefig('output-files/%s_aper%s_spectrum.jpg'%(source_name,aper_name),dpi=300,bbox_inches='tight')

exit()

#Tasks todo:

#Verify the MBB fitting function.
	#Write function to do saving of results and plotting.
	#Add residual plotting to the MBB fitting.

spectra_filename = '22'#L1448-mm_spectra_unstitched.json'

try: 
	results = read_spectra(spectra_filename)
except:
	for i,(RA_centre,Dec_centre) in enumerate(coord_list):
		#results = unstitched_spectrum_from_cube_list(filenames,RA_centre,Dec_centre,aper_size)


		for j in range(len(results['subcube_name'])):
			plt.plot(results['um'][j],results['flux'][j],color='black')
		plt.xlabel('Wavelength (um)')
		plt.ylabel('Flux Density (Jy)')

		plt.grid(which ='both',alpha=0.3)
		plt.show()
		exit()


		aper_name = 'A%d'%(i+1)
		source_name = results['source_name']
		#save_spectra(results,'%s_spectra_unstitched_%s.json'%(source_name,aper_name))
		#plot_unstitched_spectra(results)


		#pairs = find_neighbouring_subcubes(results)

		um = []
		flux = []
		flux_unc = []

		for i in range(len(results['subcube_name'])):
			um += list(results['um'][i])
			flux += list(results['flux'][i])
			flux_unc += list(results['flux_unc'][i])

		um = np.array(um)
		flux = np.array(flux)
		flux_unc = np.array(flux_unc)


		um_cont = []
		flux_cont = []
		flux_unc_cont = []

		for um0,um1 in um_ranges:
			inds_mask = np.where(np.logical_and(um > um0,um <um1))
			um_cont += list(um[inds_mask])
			flux_cont += list(flux[inds_mask])
			flux_unc_cont += list(flux_unc[inds_mask])

		um_cont = np.array(um_cont)
		flux_cont = np.array(flux_cont)
		flux_unc_cont = np.array(flux_unc_cont)

		result = fit_modified_blackbody(um_cont, flux_cont, flux_unc_cont, solid_angle=solid_angle, lambda0=850, 
		                            T_guess=50, beta_guess=0.5, 
		                            T_bounds=(10, 100), beta_bounds=(0, 3))
		fig, ax = plot_modified_blackbody_fit(um_cont, flux_cont, flux_unc_cont, result,solid_angle = solid_angle)
		plt.plot(um_cont,flux_cont,linestyle='None',marker='o')
		#plot_unstitched_spectra(results)
		plt.show()