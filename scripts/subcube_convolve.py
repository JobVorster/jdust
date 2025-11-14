from ifu_analysis.jdutils import get_JWST_PSF, unpack_hdu,get_JWST_IFU_um
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import matplotlib.pyplot as plt 
import numpy as np 
from astropy.io import fits
from reproject import reproject_adaptive


cont_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/PSF_Subtraction/continuum_maps/'

output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/PSF_Subtraction/convolution_reprojection/'


def convolve_cube(data_cube,hdr,beamsize):
    #Beamsize must be in degrees.

    try:
        x_scale = hdr['CDELT1'] #deg/pix
        y_scale = hdr['CDELT2'] #deg/pix
    except:
        x_scale = hdr['CD1_1'] #deg/pix
        y_scale = hdr['CD2_2'] #deg/pix

    x_stddev = beamsize/x_scale #pixels 
    y_stddev = beamsize/y_scale
    kernel = Gaussian2DKernel(x_stddev,y_stddev)
    conv_cube = np.zeros(np.shape(data_cube))
    for i,channelmap in enumerate(data_cube):
        if i % 10== 0:
            print('Convolving map %d of %d'%(i,len(data_cube)))
        conv_map = convolve(channelmap,kernel,normalize_kernel=True,preserve_nan=True)
        conv_cube[i] = conv_map
    return conv_cube



#What is the beamsize at 25 um?
fn_band_arr = ['ch1-short','ch1-medium','ch1-long','ch2-short','ch2-medium','ch2-long',
'ch3-short','ch3-medium','ch3-long','ch4-short','ch4-medium','ch4-long'] 

for subband in fn_band_arr[:-1]:

    long_subband = 'ch4-long'

    fn = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/PSF_Subtraction/L1448-mm_%s_s3d_LSRcorr_stripecorr_psfsub.fits'%(subband)
    fn_long = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/PSF_Subtraction/L1448-mm_%s_s3d_LSRcorr_stripecorr_psfsub.fits'%(long_subband)
    fn_cont_long = cont_foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_psfsub_cont2D.fits'%(long_subband)
    fn_cont_short = cont_foldername + 'L1448-mm_%s_s3d_LSRcorr_stripecorr_cont2D.fits'%(subband)


    hdr_short2D = fits.open(fn_cont_short)[0].header
    hdr_long2D = fits.open(fn_cont_long)[0].header

    data_long = fits.open(fn_cont_long)[0].data

    data_cube,unc_cube,dq_cube,hdr,um,shp = unpack_hdu(fn)

    data_long,hdr_long = fits.open(fn_long)[1].data,fits.open(fn_long)[1].header

    um_conv = np.nanmean(np.nanmean(get_JWST_IFU_um(hdr_long))) #micron.
    half_fwhm = get_JWST_PSF(um_conv)/(3600*np.sqrt(8*np.log(2)))

    conv_cube = convolve_cube(data_cube,hdr,half_fwhm)
    #hdu_conv = fits.PrimaryHDU(data=conv_cube,header=hdr)
    #hdu_conv.writeto('CONVTEST.fits',overwrite=True)

    shp_long = np.shape(data_long)

    conv_reprj_cube = np.zeros((len(um),shp_long[1],shp_long[2]))

    for i,chmap in enumerate(conv_cube):
        if i % 10== 0:
            print('Reprojecting map %d of %d'%(i,len(data_cube)))
        hdu_chmap = fits.PrimaryHDU(data=chmap,header=hdr_short2D)
        chmap_reprj, _ = reproject_adaptive(hdu_chmap,hdr_long2D)
            
        conv_reprj_cube[i] = chmap_reprj

    #Edit the hdrlong so that the wavelength information is the same as the short header.
    hdr_long['CDELT3'] = hdr['CDELT3']
    hdr_long['CRVAL3'] = hdr['CRVAL3']
    hdr_long['CRPIX3'] = hdr['CRPIX3']





    hdu_conv_reprj = fits.PrimaryHDU(data=conv_reprj_cube,header=hdr_long)
    hdu_conv_reprj.writeto(output_foldername + 'L1448MM_%s_convto_%s.fits'%(subband,long_subband),overwrite=True)



