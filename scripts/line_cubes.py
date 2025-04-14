from ifu_analysis.jdlines import get_line_cube


fn = '/home/vorsteja/Documents/JOYS/JWST_cubes/L1448MM1_IFUAlign/L1448-mm_ch1-short_s3d_LSRcorr.fits'
lambda0 = 5.0515
N_chans = 20

plot_index = [27,26]
get_line_cube(fn,lambda0,N_chans,curvature=1e4,scale=5,num_std=3,plot_index = plot_index)