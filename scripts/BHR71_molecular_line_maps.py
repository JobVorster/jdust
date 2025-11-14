
#Not all imports are necessarily used.
import matplotlib.pyplot as plt 
from ifu_analysis.jdutils import get_subcube_name, get_JWST_IFU_um, get_wcs_arr, get_JWST_PSF
from ifu_analysis.jdlines import get_line_cube,get_line_parameters,choose_subchannel,line_figure,gaussian
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
import sys

#Line parameters filename.
line_type = sys.argv[1]
store_type = sys.argv[2]

if store_type != 'MODEL':
	store_type = 'DATA'

available_lines = ['co2','h2o','co','hcn','c2h2','oh']

if line_type not in available_lines:
	print('Please specify a valid line,' + sum([x + ',' for x in available_lines]))

fn_line_pars = '/home/vorsteja/Documents/JOYS/JDust/line_parameter_files/%s_lines.csv'%(line_type)

#Input-output
source_name = 'BHR71-IRS1'
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/BHR71/line_maps/%s/'%(line_type)
foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/BHR71/ifualign/'
cont_inputfoldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/BHR71/continuum_maps/'

#Should not be necessary to change.
line_selecter = '*s3d_LSRcorr_stripecorr.fits'

N_chans = 40
header_hdu = 1
det_Nchan = 5
fit_Nchan = 19

do_fitting_plots = False

star_pos = None



waves, species, a_coeff, E_upper, g_upper,vtrans,jtrans = get_line_parameters(fn_line_pars)

line_list = {}
line_list['um'] = waves

if line_type == 'hcn':
	species = ['hcn'] * len(species)

line_list['names'] = species

total_mom0_dict = {}

wcs_arr = []
img_arr = []
ax_arr = []
fig_arr = []
cont_arr = []

for i,(lambda0,line_name) in enumerate(zip(line_list['um'],line_list['names'])):
	print('Line Cube %d of %d'%(i+1,len(line_list['um'])))
	
	

	fn, subcube = choose_subchannel(lambda0,foldername,line_selecter,header_hdu)

	

	print('Subcube: %s'%(subcube))
	wcs = get_wcs_arr([fn])[0]
	um, vlsr, data_cube, cont_cube, line_cube, unc_cube, dq_cube = get_line_cube(fn,lambda0,N_chans,store_type = store_type)
	dvlsr = np.abs(vlsr[1]-vlsr[0])
	shp = np.shape(line_cube)
	shp_2D = (shp[1],shp[2])	


	if subcube not in total_mom0_dict.keys():
		total_mom0_dict[subcube] = np.zeros(shp_2D)

	idx_fit = []
	idy_fit = []
	mom0_pixel_fit = []

	for idx in range(shp_2D[0]):
		for idy in range(shp_2D[1]):
			spec = line_cube[:,idx,idy]
			unc = unc_cube[:,idx,idy]
			line_ind = np.digitize([lambda0],um)[0]
			
			

			detection = spec[line_ind-det_Nchan//2-1:line_ind+det_Nchan//2 ] > 5*unc[line_ind-det_Nchan//2-1:line_ind+det_Nchan//2]

			um_fit = um[line_ind-fit_Nchan//2-1:line_ind+fit_Nchan//2 ]
			spec_fit = spec[line_ind-fit_Nchan//2-1:line_ind+fit_Nchan//2]
			if sum(detection) > 0:
				idx_fit.append(idx)
				idy_fit.append(idy)
				mom0_pixel = np.nansum(spec[line_ind-det_Nchan//2-1:line_ind+det_Nchan//2 ])*dvlsr
				mom0_pixel_fit.append(mom0_pixel)

				#This can check the spectrum of the detection.
				if (0):
					plt.plot(um,spec,label='data')
					plt.plot(um_fit,spec_fit)
					plt.vlines(lambda0,min(spec),max(spec),color='grey',linestyle='dashed')
					plt.xlabel('Wavelength (um)')
					plt.ylabel('Flux Density (Jy)')
					plt.title('Fitting failed.')
					plt.legend()
					plt.show()



				'''else:
					p0 = [max(spec_fit),um_fit[np.argmax(spec_fit)],0.5]
					try:
						popt,pcov = curve_fit(gaussian,um_fit,spec_fit,p0=p0,method='lm')
						pcov = np.sqrt(np.diag(pcov))

						idx_fit.append(idx)
						idy_fit.append(idy)
						mom0_pixel = popt[0]*popt[2]*np.sqrt(2*np.pi) #Area of a gaussian

						if do_fitting_plots:
							plt.close()
							plt.plot(um,spec,label='data')
							plt.plot(um_fit,spec_fit)
							plt.plot(um,gaussian(um,*popt),color='red',label='Gaussian')
							plt.vlines(lambda0,min(spec),max(spec),color='grey',linestyle='dashed')
							plt.xlabel('Wavelength (um)')
							plt.ylabel('Flux Density (Jy)')
							plt.legend()
							plt.show()
					except:
						print('Fitting failed.')
						plt.plot(um,spec,label='data')
						plt.plot(um_fit,spec_fit)
						plt.vlines(lambda0,min(spec),max(spec),color='grey',linestyle='dashed')
						plt.xlabel('Wavelength (um)')
						plt.ylabel('Flux Density (Jy)')
						plt.title('Fitting failed.')
						plt.legend()
						plt.show()'''

	single_line_map = np.zeros(shp_2D)
	for idx,idy,mom0_pixel in zip(idx_fit,idy_fit,mom0_pixel_fit):
		single_line_map[idx,idy] = mom0_pixel

	mom0 = single_line_map
	total_mom0_dict[subcube] += mom0
	cont_fn = cont_inputfoldername + '%s_%s_s3d_LSRcorr_cont2D.fits'%(source_name,subcube)
	try:
		cont_hdu = fits.open(cont_fn)
		cont_map = cont_hdu[0].data
		hdr_2D = cont_hdu[0].header
	except:
		cont_map = np.zeros(np.shape(mom0))

	hdr_2D['UNIT'] = 'MJy sr-1 km s-1'
	hdr_2D['WAVE'] = lambda0
	hdr_2D['ION'] = line_name

	hdu_line = fits.PrimaryHDU(data = mom0, header=hdr_2D)
	hdu_line.writeto(output_foldername + '%s_%s_%sum.fits'%(source_name,line_name,lambda0),overwrite=True)


	hdr = fits.open(fn)[1].header
	fwhm = get_JWST_PSF(lambda0)

	mom0[mom0==0] = np.nan

	vmin,vmax = np.nanpercentile(np.log10(mom0),[0.01,99.99])


	fig,ax = line_figure(np.log10(mom0),lambda0,line_name,wcs)

	wcs_arr.append(wcs)
	imshow_cmap = 'viridis'

	ax.imshow(np.log10(mom0),origin='lower',cmap=imshow_cmap,vmin=vmin,vmax=vmax)

	distance = 240
	linear_scale = 300
	plot_source_name = 'L1448MM1'
	hide_ticks = False
	img_type = line_name
	colorbar_label = r'log10($\Sigma I_\nu$ (MJy sr$^{-1}$ km s$^{-1}$))'


	ax = annotate_imshow(ax,hdr_2D,hide_ticks=hide_ticks,source_name=plot_source_name,wavelength=r'%.3f $\mu$m'%(lambda0),img_type=img_type,
		beam_fwhm = fwhm,linear_scale = linear_scale, distance = distance,
		add_colorbar=True,colorbar_label=colorbar_label,dogrid=False,star_pos=star_pos)

	ax_arr.append(ax)
	img_arr.append(mom0)
	cont_arr.append(cont_map)



	fig = ax.get_figure()


	fig.savefig(output_foldername + '%s_%s_%sum.png'%(source_name,line_name,lambda0),bbox_inches='tight',dpi=150)

#After loop print final map.
for subcube in total_mom0_dict.keys():
	total_mom0 = total_mom0_dict[subcube]
	total_mom0[total_mom0==0] = np.nan
	vmin,vmax = np.nanpercentile(np.log10(total_mom0),[0.01,99.99])


	fig,ax = line_figure(np.log10(total_mom0),lambda0,line_name,wcs)

	wcs_arr.append(wcs)
	imshow_cmap = 'viridis'

	ax.imshow(np.log10(total_mom0),origin='lower',cmap=imshow_cmap,vmin=vmin,vmax=vmax)

	distance = 240
	linear_scale = 300
	plot_source_name = 'L1448MM1'
	hide_ticks = False
	img_type = line_name
	colorbar_label = r'log10($\Sigma I_\nu$ (MJy sr$^{-1}$ km s$^{-1}$))'


	ax = annotate_imshow(ax,hdr_2D,hide_ticks=hide_ticks,source_name=plot_source_name,wavelength='Combined',img_type=img_type,
		beam_fwhm = fwhm,linear_scale = linear_scale, distance = distance,
		add_colorbar=True,colorbar_label=colorbar_label,dogrid=False,star_pos=star_pos)
	fig = ax.get_figure()


	fig.savefig(output_foldername + '%s_%s_%s.png'%(source_name,line_name,subcube),bbox_inches='tight',dpi=150)

