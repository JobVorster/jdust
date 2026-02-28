from ifu_analysis.jdfitting import read_optool
import numpy as np 
import matplotlib.pyplot as plt 

# Setup
opacity_foldername = '/home/vorsteja/Documents/JOYS/JDust/optool_opacities/'
output_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/publication_figures/'
grain_species = ['olmg50', 'pyrmg70', 'for', 'ens']
grain_names = ['Olivine', 'Pyroxene', 'Forsterite', 'Enstatite']
panel_labels = ['a', 'b', 'c', 'd']
Nspecies = len(grain_species)
grain_sizes = [0.1, 1.5]
Nsizes = len(grain_sizes)

# Read first file to get wavelength array size
op_fn = opacity_foldername + '%s_%.1fum.dat' % (grain_species[0], grain_sizes[0])
header, wave, _, _, _ = read_optool(op_fn)

# Initialize arrays
kabs_arr = np.zeros((len(grain_sizes), len(grain_species), len(wave)))
ksca_arr = np.zeros((len(grain_sizes), len(grain_species), len(wave)))

# Read all opacity files
for i, gsize in enumerate(grain_sizes):
    for j, gspecies in enumerate(grain_species):
        op_fn = opacity_foldername + '%s_%.1fum.dat' % (gspecies, gsize)
        header, wave, kappa_abs, kappa_scat, g = read_optool(op_fn)
        kabs_arr[i, j, :] = kappa_abs
        ksca_arr[i, j, :] = kappa_scat

# Set up publication-quality figure
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7
})


fig, axes = plt.subplots(4, 1, figsize=(3.5, 7), sharex=True, sharey=True)

# Colors for grain sizes
size_colors = {0.1: '#E63946', 1.5: '#1D3557'}  # Red for 0.1 μm, Navy for 1.5 μm

# Line styles for absorption vs scattering
linestyles_opacity = {'abs': '-', 'sca': '--'}
linewidths = 1.2

# Create each panel
for j, (gspecies, gname) in enumerate(zip(grain_species, grain_names)):
    ax = axes[j]
    
    for i, gsize in enumerate(grain_sizes):
        # Plot absorption opacity
        ax.semilogy(wave, kabs_arr[i, j, :],
                   color=size_colors[gsize],
                   linestyle=linestyles_opacity['abs'],
                   linewidth=linewidths,
                   label=f'{gsize} μm abs' if j == 0 else '')
        
        # Plot scattering opacity
        ax.semilogy(wave, ksca_arr[i, j, :],
                   color=size_colors[gsize],
                   linestyle=linestyles_opacity['sca'],
                   linewidth=linewidths,
                   label=f'{gsize} μm sca' if j == 0 else '')
    
    ax.set_xlim(4.5, 28)
    ax.set_ylim(1e-1, 1e5)
    ax.grid(True, alpha=0.2, linewidth=0.5, which='both')
    
    # Add panel label (a, b, c, d)
    ax.text(0.05, 0.95, f'({panel_labels[j]})', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='none', alpha=0.8))
    
    # Add mineral name inside panel
    ax.text(0.95, 0.95, gname, transform=ax.transAxes,
            fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='none', alpha=0.8))
    
    # Add y-label to each panel
    ax.set_ylabel(r'$\kappa_\nu$ (cm$^2$ g$^{-1}$)', fontsize=9)
    
    # Add legend only to first panel - positioned in lower right
    if j == 0:
        ax.legend(loc='lower right', frameon=False, 
                 ncol=2, bbox_to_anchor=(0.98, 0.02))

# Set x-label only on bottom panel
axes[-1].set_xlabel(r'Wavelength ($\mu$m)', fontsize=9)

plt.tight_layout()
plt.savefig(output_foldername + 'dust_opacities.png', dpi=300, bbox_inches='tight')
#plt.show()
print("Figure saved to %s"%(output_foldername))