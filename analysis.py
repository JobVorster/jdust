from rt_utils import *
import numpy as np 
import matplotlib.pyplot as plt 

R_solar = 6.96e10 #cm
au = 1.496e13 #cm


N = 300
n = np.zeros((N,N,N))

R_c = 1
dMenv = 1.4e-8
Mstar = 1

pars = [R_c,dMenv,Mstar]

n = make_envelope(n,pars,'RFIE')*1e3
pars = [10]
# Add disk
n0 = 1e7  # Density at stellar radius
Rstar = 2.09*R_solar/au  # Stellar radius in au
h0 = 0.01*Rstar  # Scale height at inner radius
alpha = 2.25  # Density power law
beta = 1.25  # Scale height power law
Rout = 30.0  # Outer disk radius in au
disk_pars = [n0, Rstar, h0, alpha, beta, Rout]
n = make_disk(n, disk_pars, 'SFAD')


n = make_cavity(n,pars,'CONE',fill_value = 1e-7)

test_plot_3D(np.log10(n),150,150,150)
save_non_hierarchical_cloud_file(n,'testcloud.cloud')


#ASOC_driver.py sample_emit.ini
#ASOCS.py sample_sca.ini

#without SHG
#ASOC.py sample_emit.ini