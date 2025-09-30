import matplotlib.pyplot as plt 
from ifu_analysis.jdutils import get_subcube_name, get_JWST_IFU_um, get_wcs_arr, get_JWST_PSF
from ifu_analysis.jdlines import get_line_cube
from ifu_analysis.jdvis import align_axes,annotate_imshow
from astropy.io import fits 
from glob import glob
import numpy as np
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.stats import sigma_clipped_stats
from reproject import reproject_adaptive
import bettermoments.methods as bm
from astropy.wcs import WCS
from scipy.optimize import curve_fit

'''No Ne II at 12.8135 um is detected.
No Ne III at 15.5551 um is detected.

Fe II at 5.34 um is detected.
Fe II at 17.92 um is detected.
Fe II at 24.5019 um is not detected.
Fe II at 25.9884 um is detected and hidden under the PSF.

Fe I at 24.0423 um is detected and hidden under the PSF.

Ni II at 6.6360 um is detected. 

No Ar II is detected.

S I at 25.2490 um is detected.

H2 lines:
Near 5.05, 5.51, 6.11, 6.91, 8.03, 9.66, 12.28, 17.04 um

'''




unknownwavs = [4.9539, 6.635, 6.671, 6.86,8.5596,12.21,
13.7, 14.04,17.353,19.00, 19.4,20.6, 21.8445, 23.035, 23.8125, 23.095 ]
unknownnames = ['TBD']*len(unknownwavs)





 






h2wavs = [5.05,5.33042, 5.51,5.81, 6.11, 6.91,7.28070, 8.03, 9.66,10.1777, 12.28, 17.04]
h2names = ['H2']*len(h2wavs)

h2_unconf = [6.43710,  8.45367,  12.9275, 17.9320]
h2names = ['H2']*len(h2wavs)


line_list = {}
line_ums = [24.0423,5.34,17.92,24.5019,25.9884,6.6360,25.2490,14.9759,10.52, 11.3325,16.8933] + h2wavs + unknownwavs
line_names = ['Fe I','Fe II','Fe II','Fe II','Fe II','Ni II','S I','CO2','Co II','ClI','HD'] + h2names + unknownnames 

line_list['um'] = [7.342] #line_ums
line_list['names'] = ['SO2']#line_names


#To do for presentation:
	
	#INVESTIGATION
	#Identify remaining TBD lines (Lukasz TMC paper was useful, find another paper.)
	#The thing at 14.04 um is at the north end of the continuum, with wierd morphology.
	#Whatever is at 19.4 um terminates with the continuum.
	#Is it possible to distinguish dust that is cavity pushed, or jet launched and that fell?

	#BIG AIM: 
	#FIRST: If there is dust in outflows/winds/jets
	#SECOND: Physical properties of these dust (observational)?
		#METHODOLOGY: 
			#METHODS: No need to redo data reduction.
			#METHODS: 
		#RESULTS: Morphology/location.
			#RESULTS: Show PSF of source on each map.
			#RESULTS: Continuum (red), atomic jet (blue), H2 S1 (green) and/or CO (green).
			#RESULTS: Continuum vs H2 (warm + hot), atomic lines.
			#RESULTS: 22.6 um and 19.3 um continuum vs S I (blue-redshifted)
			#RESULTS: Continuum vs CO2, continuum vs 5 um CO, continuum vs water emission.
			#RESULTS: Map of SO2, CH4, Warm H2O, Cold H2O (van Gelder et al. 2024 paper) compare to continuum morphology.
				#RESULTS: Assess if there is contamination from the point source.
				#RESULTS: S I, Fe II, CO2, Water, and continuum.
			#RESULTS: NIRSpec CO maps.
		#RESULTS: Blackbody fitting+temperature.
			#RESULTS: Apertures for the interesting regions.
			#RESULTS: Distance + angle dependence.
		#DISCUSSION: Is the continuum scattering? One can look at the HI line maps. 
		#DISCUSSION: Mass with assumed geometry (spherical vs cylindrical) and kappa (opacity - different models)
		#DISCUSSION: Can you lift icy dust? Or does the wind sweeping necessarily sublimate the ice?
		#DISCUSSION: If the ice on the dust is *necessarily* sublimated by the wind --> comes from envelope.
		#DISCUSSION: Would stellar light necessarily sublimate ice on dust grains during wind lift?
		#DISCUSSION: If the ice is not neccesarily sublimated then it comes from outside the snowline of the molecules.
		#DISCUSSION: Outflow rate from velocities (which velocities - e.g. low velocity H2?)
			#Ratio of dust swept up compared to gas -- order of gas to dust ratio.
			#Comparison with dusty disk winds - MHD and photoevaporative (also EAS Talk about dust ejection)
		#DISCUSSION: What is the origin of this dust - disk, envelope, ISM (can it survive the jets?) 
		#DISCUSSION: Compare to HH211 and BHR71).
	#FINAL:Get dust masses per aperture, and get the dust to gas outflow rates.

	#CATEGORY 1: Maps and Morphology

	#PLANNING: Write out data reduction pipeline with products. 
	#ANALYSIS: Make stripe corrected cubes.
	#ANALYSIS: Do PSF subtraction with interpolation of non-line channels.
	#ANALYSIS: Try PSF subtraction with deconvolution.
	#WRITEUP: STRIPE CORRECTION AND PSF REMOVAL.


	#TODO: Scattering model --> Fit BB or MBB to emission and absorption flagged continuum.
		#Step 1: Set up flagging for emission lines automatically
		#Step 2: Set up flagging for absorption.
		#Step 3: Fit BB amplitude and temperature to get scattering component.
		#Step 4: Save amplitude and temperature maps.
		#Step 5: Make model cube of scattering component.
		#Step 6: Make scattering subtraction.
	#TODO: Scattered light continuum map to get shape of the cavity (MRS + NIRSPEC).
		#Step 1: 
	#TODO: Use extinction maps to make extinction cube/or do it spectra by spectra.
		#Step 1: Get extinction laws from Sam.
		#Step 2: Get extinction maps from Gabriella.
		#Step 3: Make A_lambda cubes.
		#Step 4: Correct for A with the formula below.
	#TODO: Continuum at position of Fe I, S I and Fe II blobs. 
	#TODO: One can make plots of continuum+some lines as a function of distance from source.
	#TODO: Alessio has a hypotheis that Fe I is the first stages of dust destruction, Fe II is when the dust is completely destroyed.

	#CATEGORY 2: Spectral analysis
	#TODO: Extract spectra from PSF-subtracted cubes.
	#TODO: Extinction correction to the spectra (keep track of where extinction correction is done).
	#TODO: Make scattered (star+disk) fit to short wavelength component + dust emission component.
		#Note caveat that we are filtering out hot dust (~1000 K) by virtue of fitting to the long wavelengths.
	#TODO: Get dust masses for the apertures, and make plots of dust temperature and 
	#TODO: Emission free apertures for foreground emission.


	#TODO: Dust maps at short wavelengths.
	#TODO: Spectrum of the companion dust blob. Make figure.
	#TODO: Look at water or oh or ice at the dust or not dust positions.
	#TODO: A_V = 1.086 tau_V
	#TODO: A_lambda = (A_V*(kappa_lambda/kappa (0.55 um)))/1.086
	#TODO: To fix F = 10^{-A_lambda/2.5}
	#TODO: Check units of flux density (Jy) to ergs s-1 cm-2 um-1
	#TODO: Gemini important lines
	#TODO: NIST Line identification.


	#CODING
	# Make map showing Fe II, H2, CO and continuum
	# Produces fits files of fitted continuum, and line subtraced for each line.
	# Add star position.
	# Add continuum contour.

	#LATER: https://bettermoments.readthedocs.io/en/latest/tutorials/Cookbook_1.html for moment maps.
	#LATER: Linear and gaussian fit instead of continuum estimation.

	#READING
	#Read up about the H2 lines.
	#Read up about the atomic lines.
	#Read up about infrared CO emission.
	#REMEMBER: Ne II and Ne III, and Ar II (all associated with photoionization) were NOT detected.

	#Furthest distance of dust from protostar per wavelength:
	#Distance to source 293 parsec
	#ch3-short: 3.00 - 4.00 arcsec (880 - 1170 au)
	#ch3-medium: 3.33 - 4.17 arcsec (975 - 1220 au)
	#ch3-long: 2.83 - 3.70 arcsec (830 - 1080 au)
	#ch4-short: 2.21 - 3.15 arcsec (650 - 920 au)
	#ch4-medium: 2.62 - 2.73 arcsec (770 - 800 au)
	#ch4-long: 0 arcsec

	#L1448MM has one of the highest outflow rates in the JOYS sample (Francis et al. in prep)
	#16.2 10-7 Msol yr-1 (warm h2), 75.9 10-9 Msol yr-1 (hot h2)
	#If I use the equation for dust sublimation radius from Bans and Konigl 2012, and a bolometric luminosity of 8.6 from Logan's paper,
	#The dust sublimation radius for 1 micron dust, with sublimation temperature of 1850 K L1448MM is 0.1 au.
	#Truncation radius for the magnetosphere is around 0.006 au for 
	#(Bstar = 1e3 G ADHOC, Rstar = 2 Rsol ADHOC, Mstar = 0.5 Msol ADHOC, Macc = 5.45e-6 Msol yr-1 LOGAN PAPER)
	#Equations from Bans and Konigl 2012
	#Sublimation radius:
	# ~0.2(L/40Lsol)**0.5 au (THIS IS AN ESTIMATE.)
	#Magnetospheric truncation radius: 
	# ~0.02 (Bstar/10**3 G)**(4/7) x (Rstar/2.4 Rsol)**(12/7) x (Mstar/ 2.4 Msol)**(-1/7) x (Macc/10**-7 Msol yr**-1)**(-2/7) au
	#Bans and Konigl argue that radiative outflow is unlikely beyond 2r_sub

	#Dust to gas ratio is not expected to be constant.
	#Larger grains closer to r_sub, because they are not sublimated. Further, there is a size sampling for the dust upliftment.
	#Bans and Konigl 2012, p 4, outflow is expected to be 1%-5% of accreted mass in their model.
	#Equation 5 of Bans and Konigl 2012 is a estimate of grain size that can be uplifted from the disk as a function of radius. 


def convolution_reprojection(map1,wcs1,um1, map2, wcs2,um2):
	if um1 > um2:
		ref_map = map1
		ref_wcs = wcs1
		ref_um = um1

		rep_map = map2
		rep_wcs = wcs2
		rep_um = um2
	else:
		ref_map = map2
		ref_wcs = wcs2
		ref_um = um2

		rep_map = map1
		rep_wcs = wcs1
		rep_um = um1

	ref_hdr = ref_wcs.to_header()
	rep_hdr = rep_wcs.to_header()

	#Convolution
	beam_size = get_JWST_PSF(ref_um)/3600 #FWHM in deg


	try:
		x_scale = ref_hdr['CDELT1'] #deg/pix
		y_scale = ref_hdr['CDELT2'] #deg/pix
	except:
		x_scale = ref_hdr['CD1_1'] #deg/pix
		y_scale = ref_hdr['CD2_2'] #deg/pix

	x_stddev = beam_size/x_scale#pixels 
	y_stddev = beam_size/y_scale
	kernel = Gaussian2DKernel(x_stddev,y_stddev)

	input_data = (rep_map,rep_hdr)
	output_projection = ref_hdr

	reprj_map, footpring = reproject_adaptive(input_data,output_projection,conserve_flux=True,shape_out = np.shape(ref_map))

	return ref_map, reprj_map, ref_wcs, ref_um

def choose_subchannel(lambda0,foldername,selecter,header_hdu):
	filenames = glob(foldername + selecter)
	filenames.sort()

	for fn in filenames:
		subcube = get_subcube_name(fn)
		um = get_JWST_IFU_um(fits.open(fn)[header_hdu].header)
		if (lambda0 > min(um)) and (lambda0 < max(um)):
			return fn,subcube

def line_figure(line_map,line_um,line_name,wcs):
	beam_size = get_JWST_PSF(line_um)
	fig = plt.figure(figsize = (4,4))
	ax = plt.subplot(projection=wcs)
	shp = np.shape(line_map)
	ax.imshow(line_map,origin='lower',cmap='gist_stern')
	#ax.text(0.1*shp[0],0.65*shp[1],'%.2f'%(line_um) + r' $\mu$m',fontsize = 12,color='white')
	#ax.text(0.1*shp[0],0.72*shp[1],'%s'%(line_name) ,fontsize = 12,color='white', fontweight='bold')

	ra,dec = ax.coords
	ra.set_axislabel('')
	dec.set_axislabel('')
	return fig,ax

##########################################################

#Dust slope estimation

##########################################################
if (0):
	def linear(x,a,b):
		return a*x+b

	#Import five continuum maps.
	cont_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/'
	cont_add = ['L1448-mm_%s_PSFsub_stripe_continuum.fits'%(i) for i in ['ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium']]

	ref_hdu = fits.open(cont_foldername + cont_add[-1])
	other_maps = [fits.open(cont_foldername + cont_add[i]) for i in range(len(cont_add)-1)]
	ums = [12.5,14.5,16.7,19.3,22.6]

	#Reproject all to 22.6 um map.
	all_maps = [ref_hdu[0].data]
	for i,hdu_rep in enumerate(other_maps):
		if i != 5:
			um1 = ums[i]
			um2 = ums[-1]

			map1 = hdu_rep[0].data
			wcs1 = WCS(hdu_rep[0].header)

			map2 = ref_hdu[0].data
			wcs2 = WCS(ref_hdu[0].header)

			ref_map, reprj_map, ref_wcs, ref_um = convolution_reprojection(map1,wcs1,um1, map2, wcs2,um2)
			all_maps.append(reprj_map)

	shp = np.shape(all_maps)
	all_maps = np.array(all_maps)
	slope_map = np.zeros(np.shape(all_maps[0]))
	intercept_map = np.zeros(np.shape(all_maps[0]))

	if (0):
		if (0):
			for img in all_maps:
				plt.imshow(img,origin='lower')
				plt.colorbar(location='top')
				plt.show()

		for i in range(shp[1]):
			for j in range(shp[2]):
				x = np.log10(np.array(ums))
				y = np.log10(all_maps[:,i,j])
				valid = np.isfinite(y)

				x = x[valid]
				y = y[valid]
				if sum(valid)> 4:
					popt,pcov = curve_fit(linear,x,y,p0 = [1,0])
					slope_map[i,j] = popt[0]
					intercept_map[i,j] = popt[1]

					plt.scatter(x,y)
					plt.show()

		slope_map[np.where(all_maps[1]< 20)] = np.nan
		plt.subplot(121)
		plt.imshow(slope_map,origin='lower',vmin=0,vmax=2)
		plt.colorbar(location='top')
		plt.subplot(122)
		plt.imshow(intercept_map,origin='lower')
		plt.colorbar(location='top')
		plt.show()

		exit()


		#Calculate pixel by pixel slope.

		#Make map of pixel by pixel slope.

		#CONCLUSION: This is doable computationally, but I am unsure what a MIR spectral index means.
		#This analysis MAY benefit from being done with the whole continuum cubes (giving more data points), however, the spectral index is not
		#a good tracer of the continuum over these wavelength ranges.











output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/line_maps/'
foldername = '/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_IFUAlign/'
line_selecter = '*s3d_LSRcorr.fits'
N_chans = 20
header_hdu = 1

wcs_arr = []
img_arr = []
ax_arr = []
fig_arr = []

cont_arr = []
for i,(lambda0,line_name) in enumerate(zip(line_list['um'],line_list['names'])):
	print('Line Cube %d of %d'%(i+1,len(line_list['um'])))
	
	

	fn, subcube = choose_subchannel(lambda0,foldername,line_selecter,header_hdu)
	
	

	wcs = get_wcs_arr([fn])[0]
	um, vlsr, data_cube, cont_cube, line_cube, unc_cube, dq_cube = get_line_cube(fn,lambda0,N_chans)
	#,curvature=1e4,scale=5,num_std=3,plot_index = plot_index)
	line_cube[line_cube < 5*unc_cube] = 0
	
	mom0 = np.nansum(line_cube,axis = 0)*np.diff(vlsr)[-1]




	cont_fn = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448-mm_%s_PSFsub_stripe_continuum.fits'%(subcube)
	try:
		cont_hdu = fits.open(cont_fn)
		cont_map = cont_hdu[0].data
	except:
		cont_map = np.zeros(np.shape(mom0))



	hdr = fits.open(fn)[1].header
	fwhm = get_JWST_PSF(lambda0)

	vmax = np.percentile(mom0,[99.9])[0]


	fig,ax = line_figure(mom0,lambda0,line_name,wcs)

	wcs_arr.append(wcs)
	imshow_cmap = 'gist_stern'

	ax.imshow(mom0,origin='lower',cmap=imshow_cmap,vmin=0,vmax=vmax)

	distance = 240
	linear_scale = 300
	source_name = 'L1448MM1'
	hide_ticks = False
	img_type = line_name
	colorbar_label = r'$\Sigma I_\nu$ (MJy sr$^{-1}$ km s$^{-1}$)'


	ax = annotate_imshow(ax,hdr,hide_ticks=hide_ticks,source_name=source_name,wavelength=r'%.3f $\mu$m'%(lambda0),img_type=img_type,
		beam_fwhm = fwhm,linear_scale = linear_scale, distance = distance,
		add_colorbar=False,colorbar_label=colorbar_label,dogrid=False)

	ax_arr.append(ax)
	img_arr.append(mom0)
	cont_arr.append(cont_map)



	fig = ax.get_figure()



	fig.savefig(output_foldername + 'L1448-mm_%s_%sum.png'%(line_name,lambda0),bbox_inches='tight',dpi=150)


	if (0):
		rms = 0
		M0, dM0 = bm.collapse_zeroth(vlsr,line_cube,rms)

		plt.figure(figsize = (16,8))
		plt.subplot(131)
		plt.imshow(mom0,origin='lower')
		plt.subplot(132)
		plt.imshow(M0,origin='lower')
		plt.subplot(133)
		plt.imshow(dM0,origin='lower')
		plt.show()

	if lambda0 in [14.9759,17.92,17.04]: #7 is CO2, 2 is a Fe line
		do_save = True
		if do_save:
			ratio_map = np.log10(mom0/cont_map)
			ratio_map[~np.isfinite(ratio_map)] = np.nan

			cont_hdu[0].data = ratio_map
			cont_hdu.writeto(output_foldername + 'contratio_%.3fum_%s.fits'%(lambda0,line_name),overwrite=True)

		ratio_map = np.log10(mom0/cont_map)
		ratio_map[~np.isfinite(ratio_map)] = np.nan

		fig,ax = line_figure(mom0*0,lambda0,line_name,wcs)

		wcs_arr.append(wcs)
		imshow_cmap = 'gist_stern'

		ax.imshow(ratio_map,origin='lower',cmap='turbo')
	

		distance = 240
		linear_scale = 300
		source_name = 'L1448MM1'
		hide_ticks = False
		img_type = 'log10(%s/Cont)'%(line_name)
		colorbar_label = r'log$_{10}$(%s/Cont)'%(line_name)


		ax = annotate_imshow(ax,hdr,hide_ticks=hide_ticks,source_name=source_name,wavelength=r'%.3f $\mu$m'%(lambda0),img_type=img_type,
			beam_fwhm = fwhm,linear_scale = linear_scale, distance = distance,
			add_colorbar=True,colorbar_label=colorbar_label,dogrid=False)

		fig = ax.get_figure()
		fig.savefig(output_foldername + 'L1448-mm_%s_vscont.png'%(line_name),bbox_inches='tight',dpi=150)
		print('Saved ratio fig!')


if (1):
#	ax_arr = align_axes(img_arr,ax_arr,wcs_arr,reference_ax = 4)
	for i,(lambda0,line_name,ax) in enumerate(zip(line_list['um'],line_list['names'],ax_arr)):
		print('Saving figure %d of %d'%(i+1,len(line_list['um'])))
		fig = ax.get_figure()
		fig.savefig(output_foldername + 'L1448-mm_%s_%sum.png'%(line_name,lambda0),bbox_inches='tight',dpi=150)


if (0):
	ind1 = 0
	ind2 = 1

	map1 = img_arr[ind1]
	wcs1 = wcs_arr[ind1]
	um1 = line_list['um'][ind1]

	map2 = img_arr[ind2]
	wcs2 = wcs_arr[ind2]
	um2 = line_list['um'][ind2]

	ref_map, reprj_map, ref_wcs, ref_um = convolution_reprojection(map1,wcs1,um1, map2, wcs2,um2)

	print('um1 = %.5f, um2 = %.5f'%(um1,um2))
	plt.figure(figsize = (9,9))
	plt.imshow(ref_map/reprj_map,vmin = 0,vmax = 1)
	plt.colorbar(location='top')
	plt.show()

	exit()



	#Line ratios will not work, because I need to convolve and reproject!!
	for lambda0at in [24.0423,5.34,17.92,24.5019,25.9884,6.6360,25.2490]:
		ind = np.where(np.array(line_list['um']) == lambda0at)[0]
		ind2 = np.where(np.array(line_list['um']) == 17.04)[0]
		print(ind,ind2)
		plt.imshow(img_arr[ind]/img_arr[ind2],origin='lower',cmap ='gist_stern')
		plt.colorbar()
		plt.show()


	print(np.shape(img_arr))


exit()






alma_co = '/home/vorsteja/Documents/JOYS/Anciliary Data/ALMA_Fits/member.uid___A001_X33e_Xb1.L1448-mm_calibrated.ms.CO_2_1.image.pbcor.fits'
alma_co_hdu = fits.open(alma_co)
co_chan = 147
alma_co_chmap = alma_co_hdu[0].data[0][co_chan]


alma_sio = '/home/vorsteja/Documents/JOYS/Anciliary Data/ALMA_Fits/member.uid___A001_X33e_Xb1.L1448-mm_calibrated.ms.SiO.image.pbcor.fits'
alma_sio_hdu = fits.open(alma_sio)
sio_chan = 104
alma_sio_chmap = alma_sio_hdu[0].data[0][sio_chan]






#Check if the median of a line is better PSF subtraction.
if (0):
	median_data = np.nanmedian(data_cube,axis=0)

	line_map = np.zeros(np.shape(median_data))
	for chan in data_cube:
		line_map += chan - median_data 


	plt.figure(figsize = (16,9))
	plt.subplot(131)
	plt.imshow(median_data,origin='lower')
	plt.title('Median')
	plt.subplot(132)
	plt.imshow(line_map,origin='lower')
	plt.title('Data - Median')
	plt.subplot(133)
	mom0 = np.nansum(line_cube,axis = 0)
	plt.imshow(mom0,origin='lower')
	plt.title('Baseline-Subbed')
	plt.show()





plt.figure(figsize = (16,9))
plt.subplot(131)
mom0 = np.nansum(cont_cube,axis = 0)
plt.imshow(mom0,origin='lower')
plt.title('Continuum')
plt.subplot(132)
mom0 = np.nansum(line_cube,axis = 0)
plt.imshow(mom0,origin='lower')
plt.title('Line')
plt.subplot(133)
plt.imshow(cont_map,origin='lower')
plt.title('PSF Subbed Continuum')
plt.contour(mom0,cmap = 'inferno',levels = 5)
#plt.contour(alma_co_chmap, cmap = 'jet',levels = 5)
plt.show()

exit()
for um0, chan in zip(um,line_cube):
	plt.imshow(chan, origin='lower')
	plt.title('%.4f um'%(um0))
	plt.show()


