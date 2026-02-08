import numpy as np
import matplotlib.pyplot as plt
from ifu_analysis.jdutils import get_JWST_PSF,define_circular_aperture,unpack_hdu,get_JWST_IFU_um,get_subcube_name
from pybaselines import Baseline

output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/publication_figures/'
input_foldername = '/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_IFUAlign/'
fn = input_foldername + 'L1448-mm_ch3-medium_s3d_LSRcorr.fits'
data_cube,unc_cube,dq_cube,hdr,um,shp = unpack_hdu(fn)

lam = 1e4
scale = 5
num_std = 3

pix = (21,31)
aanda_col_width = 3.46457 #inches
flux = data_cube[:,pix[1],pix[0]]


#TODO: Finalize formatting and naming for this figure.


plt.figure(figsize = (aanda_col_width,0.5*aanda_col_width))

plt.plot(um,flux,label='data',alpha=0.5,color='black')

baseline_fitter = Baseline(um, check_finite=True)
baseline, params = baseline_fitter.fabc(flux, lam=lam,scale=scale,num_std=num_std)
mask = params['mask']
flux_masked = flux.copy()[mask]
um_masked = um.copy()[mask]
plt.scatter(um_masked,flux_masked,label='Continuum',color='red',s=1)
plt.plot(um,baseline,label='Baseline',color='red',lw=1,alpha=0.7)

plt.legend(fontsize = 8,edgecolor='None')
plt.gca().tick_params(which ='both',direction='in')
plt.xlabel(r'',fontsize =9)
plt.xticks(fontsize=7)
plt.ylabel(r'',fontsize=9)
plt.yticks(fontsize=7)
plt.savefig(output_foldername + 'Continuum_masking_example.png',bbox_inches='tight',dpi=200)







colors = ['red','blue','orange']

lam_arr = [3,4,5]
scale_arr = [2,5,9]
num_std_arr = [2,3,5]

plt.figure(figsize = (aanda_col_width,2*aanda_col_width))

plt.subplot(311)

plt.plot(um,flux,label='data',alpha=0.5,color='black')

for col,lam in zip(colors,lam_arr):
	baseline_fitter = Baseline(um, check_finite=True)
	baseline, params = baseline_fitter.fabc(flux, lam=10**lam,scale=scale,num_std=num_std)
	mask = params['mask']
	flux_masked = flux.copy()[mask]
	um_masked = um.copy()[mask]
	plt.plot(um,baseline,label=r'log$_{10}$($\lambda_{\rm base}$) $=$' + '%d'%(lam),color=col,lw=1,alpha=0.7)

plt.legend(fontsize = 7)
plt.xlabel(r'',fontsize =9)

plt.gca().annotate(
        'a)',
        xy=(0.9, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='none', edgecolor='none', pad=3.0))

plt.gca().tick_params(which ='both',direction='in')
plt.xticks(fontsize=7)
plt.ylabel(r'Intensity (MJy sr$^{-1}$)',fontsize=9)
plt.yticks(fontsize=7)

lam = 1e4

plt.subplot(312)

plt.plot(um,flux,label='data',alpha=0.5,color='black')

for col,scale in zip(colors,scale_arr):
	baseline_fitter = Baseline(um, check_finite=True)
	baseline, params = baseline_fitter.fabc(flux, lam=lam,scale=scale,num_std=num_std)
	mask = params['mask']
	flux_masked = flux.copy()[mask]
	um_masked = um.copy()[mask]
	plt.plot(um,baseline,label=r'$n_{\rm base} =$' + '%d'%(scale),color=col,lw=1,alpha=0.7)



plt.gca().annotate(
        'b)',
        xy=(0.9, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='none', edgecolor='none', pad=3.0))

plt.legend(fontsize = 7)
plt.xlabel(r'',fontsize =9)
plt.gca().tick_params(which ='both',direction='in')
plt.xticks(fontsize=7)
plt.ylabel(r'Intensity (MJy sr$^{-1}$)',fontsize=9)
plt.yticks(fontsize=7)


plt.subplot(313)

plt.plot(um,flux,label='data',alpha=0.5,color='black')

for col,num_std in zip(colors,num_std_arr):
	baseline_fitter = Baseline(um, check_finite=True)
	baseline, params = baseline_fitter.fabc(flux, lam=lam,scale=scale,num_std=num_std)
	mask = params['mask']
	flux_masked = flux.copy()[mask]
	um_masked = um.copy()[mask]
	plt.plot(um,baseline,label=r'$\sigma_{\rm base} =$' + '%d'%(num_std),color=col,lw=1,alpha=0.7)

plt.legend(fontsize = 7)
plt.xlabel(r'Wavelength ($\mu$m)',fontsize =9)

plt.gca().annotate(
        'c)',
        xy=(0.9, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='none', edgecolor='none', pad=3.0))

plt.gca().tick_params(which ='both',direction='in')
plt.xticks(fontsize=7)
plt.ylabel(r'Intensity (MJy sr$^{-1}$)',fontsize=9)
plt.yticks(fontsize=7)






plt.savefig(output_foldername + 'continuum_masking.png',bbox_inches='tight',dpi=200)