# Perturbed PSF Subtraction Wrapper
# Produces PSF-subtracted cubes perturbed by the combined fitting and smoothing uncertainties.
# Requires the _options.csv files produced by the original PSF subtraction wrapper.
#
# Perturbation combinations use P=plus, M=minus notation for (scaling, xoffset, yoffset):
#   PPP = (+,+,+)   PMM = (+,-,-)   MPP = (-,+,+)   MMM = (-,-,-)
#   PPM = (+,+,-)   PMP = (+,-,+)   MPM = (-,+,-)   MMP = (-,-,+)
#
# Author : Job Vorster
# Version: 2025-07-03
# Email  : jobvorster8@gmail.com

import numpy as np
import pandas as pd
import configparser
import os
from astropy.io import fits
from astropy.wcs import WCS
from ifu_analysis.jdpsfsub import generate_single_miri_mrs_psf, get_aper_mask, get_options_csv_fn
from ifu_analysis.jdutils import unpack_hdu

# =============================================================================
# USER PARAMETERS — edit this block only
# =============================================================================

source_name  = 'L1448-mm'

# subband_arr  : 1A-style bands, used by stpsf / generate_single_miri_mrs_psf
# fn_band_arr  : ch-style bands, used in filenames and options CSV filenames
subband_arr = ['1A','1B','1C','2A','2B','2C','3A','3B','3C','4A','4B','4C']
fn_band_arr = ['ch1-short','ch1-medium','ch1-long',
               'ch2-short','ch2-medium','ch2-long',
               'ch3-short','ch3-medium','ch3-long',
               'ch4-short','ch4-medium','ch4-long']

input_foldername  = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/'
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/PSFSub_Docs/L1448MM1/'

aper_coords  = ['03h25m38.8898s', '+30d44m05.612s']
mask_psfsub  = True
bfe_factor   = 0.35
base_factor  = 1

# Combinations already computed — will be skipped
skip_list = ['PPP', 'MMM']

# =============================================================================
# END USER PARAMETERS
# =============================================================================

# All 8 perturbation combinations: (label, scaling_sign, xoffset_sign, yoffset_sign)
COMBINATIONS = [
    ('PPP', +1, +1, +1),
    ('PPM', +1, +1, -1),
    ('PMP', +1, -1, +1),
    ('PMM', +1, -1, -1),
    ('MPP', -1, +1, +1),
    ('MPM', -1, +1, -1),
    ('MMP', -1, -1, +1),
    ('MMM', -1, -1, -1),
]

PROGRESS_FILE = os.path.join(output_foldername, '%s_psfsub_progress.ini' % source_name)


# =============================================================================
# PROGRESS TRACKER
# =============================================================================

def init_progress(subband_arr, combinations, skip_list):
    '''
    Create a progress .ini file. Marks skipped combinations as done.
    Sections are keyed by the 1A-style subband.

    Parameters
    ----------
    subband_arr : list of str
        1A-style subband labels (e.g. '1A', '2B').
    combinations : list of tuples
        List of (label, s_sign, x_sign, y_sign) perturbation combinations.
    skip_list : list of str
        Labels to mark as already done.

    Returns
    -------
    config : configparser.ConfigParser
        Initialised progress config.
    '''
    config = configparser.ConfigParser()
    for subband in subband_arr:
        config[subband] = {}
        for label, *_ in combinations:
            config[subband][label] = 'done' if label in skip_list else 'pending'
    with open(PROGRESS_FILE, 'w') as f:
        config.write(f)
    print('Progress file created: %s' % PROGRESS_FILE)
    return config


def load_progress():
    '''
    Load the progress .ini file from disk.

    Returns
    -------
    config : configparser.ConfigParser
        Loaded progress config.
    '''
    config = configparser.ConfigParser()
    config.read(PROGRESS_FILE)
    return config


def mark_done(config, subband, label):
    '''
    Mark a (subband, label) combination as done and write to disk.

    Parameters
    ----------
    config : configparser.ConfigParser
    subband : str
        1A-style subband label.
    label : str
        Perturbation combination label (e.g. 'PPP').
    '''
    config[subband][label] = 'done'
    with open(PROGRESS_FILE, 'w') as f:
        config.write(f)


def is_done(config, subband, label):
    '''
    Check if a (subband, label) combination is marked as done.

    Parameters
    ----------
    config : configparser.ConfigParser
    subband : str
        1A-style subband label.
    label : str
        Perturbation combination label.

    Returns
    -------
    bool
    '''
    return config[subband].get(label, 'pending') == 'done'


def all_done(config):
    '''
    Check if every combination in the progress file is marked done.

    Parameters
    ----------
    config : configparser.ConfigParser

    Returns
    -------
    bool
    '''
    return all(
        config[section][key] == 'done'
        for section in config.sections()
        for key in config[section]
    )


# =============================================================================
# PSF SUBTRACTION
# =============================================================================

def subtract_perturbed(data_cube, um_cube, subband, scaling_arr, xoffset_arr, yoffset_arr,
                       aper_coords, mask_psfsub, base_factor, bfe_factor, wcs, shp):
    '''
    Subtract a perturbed PSF cube for a given set of parameter arrays.

    Parameters
    ----------
    data_cube : 3D array
        Science data cube.
    um_cube : 1D array
        Wavelength array in micron.
    subband : str
        1A-style MIRI MRS subband (e.g. '3A'). Used by stpsf.
    scaling_arr : 1D array
        Per-channel PSF scaling factors (perturbed).
    xoffset_arr : 1D array
        Per-channel x offsets in arcsec (perturbed).
    yoffset_arr : 1D array
        Per-channel y offsets in arcsec (perturbed).
    aper_coords : list of str
        [RA, Dec] in HHhMMmSS.Ss, DDdMMmSS.Ss format.
    mask_psfsub : bool
        If True, mask the PSF aperture region after subtraction.
    base_factor : float
        FWHM multiplier for the aperture mask.
    bfe_factor : float
        Additional FWHM multiplier for the BFE aperture.
    wcs : astropy WCS
        2D WCS of a channel map.
    shp : tuple
        Spatial shape (ny, nx) of a channel map.

    Returns
    -------
    psfsub : 3D array
        PSF-subtracted data cube.
    '''
    psfsub = data_cube.copy()

    for channel in range(len(um_cube)):
        if channel % 50 == 0:
            print('  Channel %d / %d' % (channel, len(um_cube)))

        chan_map = data_cube[channel]

        psf, _ = generate_single_miri_mrs_psf(
            subband, channel,
            x_offset_arcsec=xoffset_arr[channel],
            y_offset_arcsec=yoffset_arr[channel],
            shp=np.shape(chan_map)
        )
        psf /= np.nanmax(psf)
        psfsub[channel] = chan_map - psf * scaling_arr[channel]

        if mask_psfsub:
            aper_mask_bfe = get_aper_mask(
                um_cube[channel], aper_coords,
                base_factor + bfe_factor, wcs, shp
            )
            psfsub[channel][aper_mask_bfe == 1] = np.nan

    return psfsub


def save_cube(filename, psfsub_cube, label, fn_base):
    '''
    Save a perturbed PSF-subtracted cube to disk.

    Parameters
    ----------
    filename : str
        Path to the original input FITS file (used as the HDU template).
    psfsub_cube : 3D array
        PSF-subtracted data cube to write.
    label : str
        Perturbation label (e.g. 'PPP'), appended to the output filename.
    fn_base : str
        Base output filename path (without the _psfsub_<label>.fits suffix).
    '''
    with fits.open(filename) as hdu:
        hdu_out = hdu.copy()
        hdu_out[1].data = psfsub_cube
        outpath = fn_base + '_psfsub_%s.fits' % label
        hdu_out.writeto(outpath, overwrite=True)
    print('  Saved: %s' % outpath)


# =============================================================================
# MAIN
# =============================================================================

fn_arr          = [input_foldername + '%s_%s_s3d_LSRcorr_stripecorr.fits' % (source_name, fn_band)
                   for fn_band in fn_band_arr]
options_csv_arr = [get_options_csv_fn(output_foldername+'FIT_OPTIONS/', source_name, fn_band)
                   for fn_band in fn_band_arr]

# Initialise or resume progress (keyed by 1A-style subband)
if os.path.exists(PROGRESS_FILE):
    print('Resuming from existing progress file.')
    config = load_progress()
else:
    config = init_progress(subband_arr, COMBINATIONS, skip_list)

# Main loop — subband is 1A-style (for stpsf), fn_band is ch-style (for filenames/CSVs)
for filename, options_csv, subband, fn_band in zip(fn_arr, options_csv_arr, subband_arr, fn_band_arr):

    print('\n=== Subband %s (%s) ===' % (subband, fn_band))

    if all(is_done(config, subband, label) for label, *_ in COMBINATIONS):
        print('  All combinations already done, skipping.')
        continue

    df = pd.read_csv(options_csv)

    um             = df['um'].values
    smooth_scaling = df['smooth_scaling'].values
    smooth_xoffset = df['smooth_xoffset'].values
    smooth_yoffset = df['smooth_yoffset'].values

    dfit_scaling    = df['dfit_scaling'].values
    dfit_xoffset    = df['dfit_xoffset'].values
    dfit_yoffset    = df['dfit_yoffset'].values

    dsmooth_scaling = df['dsmooth_scaling'].values
    dsmooth_xoffset = df['dsmooth_xoffset'].values
    dsmooth_yoffset = df['dsmooth_yoffset'].values

    dtotal_scaling = np.sqrt(dfit_scaling**2 + dsmooth_scaling**2)
    dtotal_xoffset = np.sqrt(dfit_xoffset**2 + dsmooth_xoffset**2)
    dtotal_yoffset = np.sqrt(dfit_yoffset**2 + dsmooth_yoffset**2)

    data_cube, unc_cube, dq_cube, hdr, um_cube, shp = unpack_hdu(filename)
    wcs = WCS(hdr).dropaxis(2)

    # fn_base uses fn_band (ch-style) to match existing filename convention
    fn_base = output_foldername + source_name + '_%s_s3d_LSRcorr_stripecorr' % fn_band

    for label, s_sign, x_sign, y_sign in COMBINATIONS:

        if is_done(config, subband, label):
            print('  Skipping %s (already done).' % label)
            continue

        print('  Computing %s (%+d,%+d,%+d)...' % (label, s_sign, x_sign, y_sign))

        scaling = smooth_scaling + s_sign * dtotal_scaling
        xoffset = smooth_xoffset + x_sign * dtotal_xoffset
        yoffset = smooth_yoffset + y_sign * dtotal_yoffset

        psfsub = subtract_perturbed(
            data_cube, um_cube, subband,       # <-- 1A-style for stpsf
            scaling, xoffset, yoffset,
            aper_coords, mask_psfsub,
            base_factor, bfe_factor, wcs, shp
        )

        save_cube(filename, psfsub, label, fn_base)
        mark_done(config, subband, label)      # <-- 1A-style as progress key

# Clean up progress file if everything is done
if all_done(config):
    os.remove(PROGRESS_FILE)
    print('\nAll combinations complete. Progress file removed.')
else:
    print('\nSome combinations still pending. Progress file retained at: %s' % PROGRESS_FILE)