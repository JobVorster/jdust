from ifu_analysis.jdspextract import load_spectra,merge_subcubes
from ifu_analysis.jdfitting import prepare_spectra_for_fit,convert_flux_wl
import matplotlib.pyplot as plt 
import numpy as np
from glob import glob

input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/spectra/extracted_paper/'
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/publication_figures/'
# A&A two-column width is approximately 7.0 inches
fig_width = 3.5
fig_height = 7 * 0.75  # Adjust ratio as needed

fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height), sharex=True)
fig.subplots_adjust(hspace=0.02)  # Much tighter spacing between panels

# Define colors for apertures 1, 2, 3 (consistent across A, B, C)
colors = ['C0', 'C1', 'C2']  # Blue, orange, green
bs_color = 'C3'  # Red for BS

# Define scaling factors for each aperture number
scale_factors = {'1': 1, '2': 2, '3': 5, 'BS': 1}

# Storage for spectra
spectra_data = {}

# Grab spectra
filenames = glob(input_foldername + '*.spectra')
filenames.sort()
for fn in filenames:
    ap_name = fn.split('.')[0].split('aper')[-1]
    
    print(ap_name)
    spec_dict = load_spectra(fn)
    spec_dict = merge_subcubes(spec_dict)
    um, flux, flux_unc = [spec_dict[x] for x in ['um','flux','flux_unc']]
    um_cut = 27.5
    fit_wavelengths = [[0,um_cut]]
    prepared_spectra = prepare_spectra_for_fit(um,flux,flux_unc,fit_wavelengths=fit_wavelengths)
    spectra_cols = ['um','flux','unc']
    fit_um, fit_flux, fit_unc = [prepared_spectra['fitdata:%s'%(x)] for x in spectra_cols]
    
    # Store the data
    spectra_data[ap_name] = {
        'fit_um': fit_um,
        'fit_flux': fit_flux,
        'fit_unc': fit_unc
    }

# Plot the data
for row, letter in enumerate(['A', 'B', 'C']):
    ax = axes[row]
    
    # For panel A, include BS aperture
    if letter == 'A':
        apertures = ['BS', '1', '2', '3']
        aperture_colors = [bs_color] + colors
    else:
        apertures = ['1', '2', '3']
        aperture_colors = colors
    
    # Determine offset based on max flux values in this panel (with scaling and 1e-19 division applied)
    max_flux = 0
    for number in apertures:
        ap_name = 'BS' if number == 'BS' else f'{letter}{number}'
        if ap_name in spectra_data:
            scaled_flux = (spectra_data[ap_name]['fit_flux'] * scale_factors[number]) / 1e-19
            max_flux = max(max_flux, np.max(scaled_flux))
    
    offset_step = max_flux * 0.9
    
    for idx, number in enumerate(apertures):
        ap_name = 'BS' if number == 'BS' else f'{letter}{number}'
        color = aperture_colors[idx]
        
        if ap_name in spectra_data:
            data = spectra_data[ap_name]
            fit_um = data['fit_um']
            fit_flux = data['fit_flux']
            fit_unc = data['fit_unc']
            
            # Apply scaling and divide by 1e-19
            scale = scale_factors[number]
            fit_flux_scaled = (fit_flux * scale) / 1e-19
            fit_unc_scaled = (fit_unc * scale) / 1e-19
            
            # Apply offset
            offset = idx * offset_step
            
            # Plot zero line (dashed, same color)
            ax.axhline(y=offset, color=color, linestyle='--', 
                      lw=0.8, alpha=0.7, zorder=1)
            
            # Detect where flux >= 5*unc (detections)
            detection_mask = (fit_flux * scale) >= 5 * (fit_unc * scale)
            
            # Fill between zero line and spectrum for detections
            if np.any(detection_mask):
                ax.fill_between(fit_um, offset, fit_flux_scaled + offset,
                               where=detection_mask, color=color, 
                               alpha=0.3, zorder=1)
            
            # Plot all data as line
            label = f'{ap_name}' if scale == 1 else f'{ap_name} (×{scale})'
            ax.plot(fit_um, fit_flux_scaled + offset, 
                   color=color, lw=1, label=label, zorder=2)
    
    # Formatting
    ax.set_xlim(4.7, 27.5)
    ax.legend(loc='upper right', fontsize=6, frameon=False)
    ax.set_ylabel(r'$\lambda F_\lambda$ (W cm$^{-2}$)', fontsize=10)
    ax.tick_params(labelsize=9)
    
    # Only show x-label on bottom row
    if row == 2:
        ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=10)

plt.tight_layout()
plt.savefig(output_foldername + 'aperture_spectra.png', dpi=300, bbox_inches='tight')


#Merge the subcubes.

#Line mask.

#Regrid.

#Plot.