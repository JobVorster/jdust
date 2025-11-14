
#Not all imports are necessarily used.
import matplotlib.pyplot as plt 
from ifu_analysis.jdutils import get_subcube_name, get_JWST_IFU_um, get_wcs_arr, get_JWST_PSF
from ifu_analysis.jdlines import get_line_cube,get_line_parameters,choose_subchannel,line_figure
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

available_lines = ['h2o','co','co2','hcn','ions','h2','c2h2']

if line_type not in available_lines:
	print('Please specify a valid line,' + sum([x + ',' for x in available_lines]))

fn_line_pars = '/home/vorsteja/Documents/JOYS/JDust/line_parameter_files/%s_lines.csv'%(line_type)

#Input-output
source_name = 'L1448-mm'
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/line_cubes/%s/'%(line_type)
foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/'
cont_inputfoldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/PSF_Subtraction/continuum_maps/'

#Should not be necessary to change.
line_selecter = '*s3d_LSRcorr_stripecorr.fits'
if line_type == 'h2o':
	N_chans = 200
else:
	N_chans = 40
header_hdu = 1

star_pos = ['03h25m38.8898s','+30d44m05.612s']



waves, species, a_coeff, E_upper, g_upper,vtrans,jtrans = get_line_parameters(fn_line_pars)

line_list = {}
line_list['um'] = waves
line_list['names'] = species




wcs_arr = []
img_arr = []
ax_arr = []
fig_arr = []

cont_arr = []
for i,(lambda0,line_name) in enumerate(zip(line_list['um'],line_list['names'])):
	print('Line Cube %d of %d'%(i+1,len(line_list['um'])))
	
	

	fn, subcube = choose_subchannel(lambda0,foldername,line_selecter,header_hdu)
	
	

	wcs = get_wcs_arr([fn])[0]
	um, vlsr, data_cube, cont_cube, line_cube, unc_cube, dq_cube = get_line_cube(fn,lambda0,N_chans,store_type = store_type)
	#,curvature=1e4,scale=5,num_std=3,plot_index = plot_index)
	if store_type =='DATA':
		line_cube[line_cube < 5*unc_cube] = 0
	
	mom0 = np.nansum(line_cube,axis = 0)*np.diff(vlsr)[-1]




	cont_fn = cont_inputfoldername + '%s_%s_s3d_LSRcorr_stripecorr_cont2D.fits'%(source_name,subcube)
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

	vmax = np.percentile(mom0,[99.9])[0]


	fig,ax = line_figure(mom0,lambda0,line_name,wcs)

	wcs_arr.append(wcs)
	imshow_cmap = 'gist_stern'

	ax.imshow(mom0,origin='lower',cmap=imshow_cmap,vmin=0,vmax=vmax)

	distance = 240
	linear_scale = 300
	plot_source_name = 'L1448MM1'
	hide_ticks = False
	img_type = line_name
	colorbar_label = r'$\Sigma I_\nu$ (MJy sr$^{-1}$ km s$^{-1}$)'


	ax = annotate_imshow(ax,hdr_2D,hide_ticks=hide_ticks,source_name=plot_source_name,wavelength=r'%.3f $\mu$m'%(lambda0),img_type=img_type,
		beam_fwhm = fwhm,linear_scale = linear_scale, distance = distance,
		add_colorbar=False,colorbar_label=colorbar_label,dogrid=False,star_pos=star_pos)

	ax_arr.append(ax)
	img_arr.append(mom0)
	cont_arr.append(cont_map)



	fig = ax.get_figure()



	fig.savefig(output_foldername + '%s_%s_%sum.png'%(source_name,line_name,lambda0),bbox_inches='tight',dpi=150)

