from ifu_analysis.jdpsfsub import subtract_psf_cube,generate_single_miri_mrs_psf,get_offsets, Gauss2D_fit
from ifu_analysis.jdutils import unpack_hdu,get_JWST_PSF,define_circular_aperture
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import pandas as pd
from astropy.wcs import WCS
from scipy.ndimage import median_filter
from skimage import color, data, restoration
from sdeconv.api import SDeconvAPI
import torch
import sdeconv.deconv as deconv
from clij2fft.richardson_lucy import richardson_lucy_nc #https://github.com/MTLeist255/JWST_Deconvolution?tab=readme-ov-file

#Make this a loop through all subbands, with the right SNR_percentile.
#['3A','3B','3C','4A','4B','4C']
subband_arr = ['3A']#['1A','1B','1C','2A','2B',]
#['ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long']
fn_band_arr = ['ch3-short']#['ch1-short','ch1-medium','ch1-long','ch2-short','ch2-medium','ch2-long']
#fn_arr = ['/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_IFUAlign/L1448-mm_%s_s3d_LSRcorr.fits'%(x) for x in fn_band_arr]
fn_arr = ['/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/L1448-mm_%s_s3d_LSRcorr_stripecorr.fits'%(fn_band_arr[-1])]
	



SNR_percentile_arr = [99]*3 + [97.5]*3 
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/PSF_Subtraction/'

dosave = False
doplot = False

mask_method = 'APERTURE'
mask_par = None
aper_coords = ['03h25m38.8898s','+30d44m05.612s']


for filename,subband,SNR_percentile in zip(fn_arr,subband_arr,SNR_percentile_arr):

	#Fix this filename.
	if dosave:
		fn = filename.split('/')[-1].split('.fits')[0]
		saveto = output_foldername + fn + '_psf_options.csv'
	else:
		saveto = None

	print('Doing Subband %s'%(subband))

	data_cube,unc_cube,dq_cube,hdr,um,shp = unpack_hdu(filename)
	wcs = WCS(hdr)
	wcs = wcs.dropaxis(2)

	channel = 100
	chan_map = data_cube[channel]
	unc_map = data_cube[channel]



	shp = np.shape(chan_map)

	if mask_method == 'APERTURE':
		if aper_coords == None:
			raise ValueError('Please specify an aperture coordinate for aperture masking.')
		elif wcs == None:
			raise ValueError('Please specify a wcs for aperture masking.')
		else:
			#Aperture size is defined by the law et al relation.
			fwhm = 1*get_JWST_PSF(um[channel])
			RA, Dec = aper_coords
			aper = define_circular_aperture(RA,Dec,fwhm)
			mask_par = aper.to_pixel(wcs)

	x_offset_arcsec, y_offset_arcsec = get_offsets(chan_map,unc_map,subband,channel,mask_method,mask_par)
	dchan = 0
	psf_woffset, pix_scale = generate_single_miri_mrs_psf(subband,channel+dchan,
		x_offset_arcsec = x_offset_arcsec,y_offset_arcsec = y_offset_arcsec,shp=np.shape(chan_map))

	psf_woffset /= np.nanmax(psf_woffset)
	w, model_data = Gauss2D_fit(chan_map)
	scaling = w.amplitude
	psf_map = psf_woffset*scaling
	
	bfe_factor = 0.25 #estimate of brighter-fatter effect (MIRI MRS, Argyriou et al 2023)
	fwhm = (1+bfe_factor)*get_JWST_PSF(um[channel])
	RA, Dec = aper_coords
	aper = define_circular_aperture(RA,Dec,fwhm)
	mask_par = aper.to_pixel(wcs)
	aper_mask = np.mod(np.array(mask_par.to_mask(method='center').to_image(shp),dtype=int)+1,2)

	plt.figure(figsize = (16,9))
	plt.subplot(131)
	plt.imshow(np.log10(chan_map),origin='lower')
	plt.colorbar(location='top')
	plt.subplot(132)
	plt.imshow(np.log10(psf_map),origin='lower')
	plt.colorbar(location='top')
	plt.subplot(133)
	psf_sub = chan_map - psf_map

	psf_sub[aper_mask==0] = np.nan

	plt.imshow(np.log10(psf_sub),origin='lower')
	plt.contour(aper_mask)
	plt.colorbar(location='top')
	plt.show()

	exit()



	chan_map = np.nan_to_num(chan_map, nan=0.0, posinf=0.0, neginf=0.0)
	chan_map = median_filter(chan_map, size=3)

	psf_deconv = torch.tensor(psf_woffset[:,:37], dtype=torch.float32)
	chan_deconv = torch.tensor(chan_map[:,:37], dtype=torch.float32)
	
	print(np.shape(psf_deconv))
	print(np.shape(chan_deconv))
	
	if (1):
		for niter in range(10):
			#richlucy = deconv.SRichardsonLucy(psf_deconv, niter = niter,pad = 0)
			#out_image = richlucy(chan_deconv)

			out_image = richardson_lucy_nc(chan_map,psf_woffset,niter,0.0002)

			plt.figure(figsize = (16,9))
			plt.subplot(121)
			plt.imshow(np.log10(chan_map),origin='lower')
			plt.subplot(122)
			plt.imshow(np.log10(out_image),origin='lower')
			plt.title('niter = %d'%(niter))
			plt.show()


	if (0):
		for weight in np.arange(0.1,1,0.1):
			spfire = deconv.Spitfire(psf_deconv, weight = weight,reg = 0.995,gradient_step=0.01)
			out_image = spfire(chan_deconv).detach().numpy()

			plt.figure(figsize = (16,9))
			plt.subplot(121)
			plt.imshow(np.log10(chan_map),origin='lower')
			plt.subplot(122)
			plt.imshow(np.log10(out_image),origin='lower')
			plt.title('weight = %.2f'%(weight))
			plt.show()

	if (0):
		# Apply Wiener filter
		wiener = deconv.SWiener(psf_deconv, beta=0.005,pad=5)
		out_image = wiener(chan_deconv)

		plt.figure(figsize = (16,16))
		plt.subplot(331)
		plt.imshow(np.log10(chan_map),origin='lower')
		plt.subplot(332)
		plt.imshow(np.log10(out_image),origin='lower')
		plt.subplot(333)
		model = convolve(out_image.numpy(),psf_woffset)
		plt.imshow(chan_map - model,origin='lower')
		

		# Apply Wiener filter
		

		plt.subplot(334)
		plt.imshow(np.log10(chan_map),origin='lower')
		plt.subplot(335)
		plt.imshow(np.log10(out_image),origin='lower')
		plt.subplot(336)
		model = convolve(out_image.numpy(),psf_woffset)
		plt.imshow(chan_map - model,origin='lower')
		
		# Apply Wiener filter
		spfire = deconv.Spitfire(psf_deconv, weight=0.5,reg=0.995)
		out_image = spfire(chan_deconv)

		plt.subplot(337)
		plt.imshow(np.log10(chan_map),origin='lower')
		plt.subplot(338)
		plt.imshow(np.log10(out_image),origin='lower')
		plt.subplot(339)
		model = convolve(out_image.numpy(),psf_woffset)
		plt.imshow(chan_map - model,origin='lower')




		plt.show()

	exit()

	psf_sum = np.nansum(psf_woffset)	
	
	res_arr = []

	for N in range(20):

		beam_size = get_JWST_PSF(um[channel])/3600 #FWHM in deg


		try:
			x_scale = hdr['CDELT1'] #deg/pix
			y_scale = hdr['CDELT2'] #deg/pix
		except:
			x_scale = hdr['CD1_1'] #deg/pix
			y_scale = hdr['CD2_2'] #deg/pix

		x_stddev = beam_size/x_scale#pixels 
		y_stddev = beam_size/y_scale
		kernel = Gaussian2DKernel(x_stddev/(2*np.sqrt(2*np.log(2))),y_stddev/(2*np.sqrt(2*np.log(2))))

		deconvolved_RL = restoration.richardson_lucy(chan_map, psf_woffset, num_iter=N,clip=False)
		model = convolve(deconvolved_RL, psf_woffset)
		deconvolved_RL = convolve(deconvolved_RL,kernel)

		residuals = np.array(chan_map - model)/np.nanmax(chan_map)
		aper_mask = np.mod(np.array(mask_par.to_mask(method='center').to_image(shp),dtype=int)+1,2)
		all_residuals = np.nansum(np.abs(residuals[aper_mask]))
		res_arr.append(all_residuals)
		if (1):
			plt.figure(figsize = (16,9))
			plt.subplot(131)
			plt.imshow(np.log10(chan_map),origin='lower',vmin = 0,vmax = np.log10(1.2*np.nanmax(chan_map)))
			plt.title('%.4f um'%(um[channel]))
			plt.colorbar(location='top')
			plt.subplot(132)
			plt.imshow(np.log10(deconvolved_RL),origin='lower',vmin=0,vmax = np.log10(1.2*np.nanmax(deconvolved_RL)))
			plt.colorbar(location='top')
			plt.subplot(133)
			plt.imshow(chan_map-model,origin='lower',cmap='coolwarm')
			plt.colorbar(location='top')
			plt.show()
	plt.plot(np.abs(res_arr))


	#Try subtracting psf before deconvolution
	#Try subtracting psf after deconvolution
	plt.show()