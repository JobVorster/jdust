from numpy import *
import matplotlib.pyplot as plt


#############################################
#An aside about units:

#



#############################################

G = 4.3009172706e-3 #pc Msol-1 (km s-1)2

def check_make_input(make_type,pars,valid_types,valid_shapes):
    '''
	Check if the envelope/cavity/disk type and parameter list shape is correct.

	Parameters
	----------

	make_type : string
		Subtype of envelope/disk/cavity.

	pars : list
		Array of parameters for the density distribution.

	valid_types : list of string
		Valid subtypes of envelope/disk/cavity.

	valid_shapes : list 
		Containing the shape of a valid pars list. E.g. [(1,)]

	Returns
	-------

	None 
		Raises a value error if the make_type is not in valid_types, or the pars is not of the correct shape.
    '''
    if make_type not in valid_types:
        raise ValueError('Please specify a valid type. Options: %s'%(str(valid_types)))

    #Check if the parameters corresponding to the envelope type is valid.
    for str_type, shp in zip(valid_types,valid_shapes):
        if make_type == str_type:
            if shape(pars) != shp:
                raise ValueError('Pars array wrong shape. Please specify pars with a array of shape %s.'%(str(shp)))

def return_coords(n):
    '''
    Return meshgrid Cartesian coordinates for a square density distribution.

    Parameters
    ----------

    n : 3D array
        Density distribution (cm-3).

    Returns
    -------

    X : 3D array
        X coordinate (in pixels).

    Y : 3D array
        Y coordinate (in pixels).

    Z : 3D array
        Z coordinate (in pixels).
    '''
    Nz, Ny, Nx = shape(n)
    iz, iy, ix = arange(0,Nz), arange(0,Ny), arange(0,Nx)
    centre_z, centre_y, centre_x = [round(xi/2) for xi in [Nz, Ny, Nx]] #Middle of the envelope (0,0,0)
    Z, Y, X = meshgrid(iz - centre_z, iy - centre_y, ix - centre_x) #Makes the indices 3D arrays centred in the middle. 
    return X, Y, Z    

def make_envelope(n,pars,env_type):
    '''
    Make a envelope density distribution.

    Parameters
    ----------

    n : 3D array
        Cartesian density cube in units of cm-3.
    
    pars : 1D array
         Length defined by envelope type.

    env_type : string
        Envelope type. This defines the parameters required for the envelope creation.
        
        Supported types:
        ----------------

        PLSP: Power law sphere (n0 * np.exp(-alpha*r) with r the radius)

        Parameters:

        n0 : Density at the centre (cm-3)
        alpha : Power law exponent (unitless)
        ----------------
        RFIE: Rotationally flattened infalling envelope 
        From Equation 1 of Whitney, B.A. et. al. 2003, ApJ 591, 1049-1063.
        
        Parameters:
        
        R_c : float
            Centrifugal radius of the disk (in au).

        dMenv : float
            Envelope infall rate (Msol yr-1)

        Mstar : float
            Stellar mass (Msol)
        ----------------

    Returns
    -------

    n : 3D array
        Cartesian density cube in units of cm-3, with the envelope ADDED.
    
    '''
    valid_types = ['PLSP','RFIE']
    valid_shapes = [(2,),(3,)]

    check_make_input(env_type,pars,valid_types,valid_shapes)

    X, Y, Z = return_coords(n)    
    r = sqrt(X**2 + Y**2 + Z**2)

    Rdisk = sqrt(X**2 + Y**2) #Radius along the disk.
    mu = Z/r #Cosine angle.

    if env_type == 'PLSP':
        n0, alpha = pars
        n += n0*exp(-alpha*r) 
        return n

    if env_type == 'RFIE':
        R_c, dMenv, Mstar = pars
        mu0 = zeros(shape(mu))
        inds_inner = where(r > 0.01*R_c) #logical_and(,abs(Z) >= 1)
        mu0[inds_inner] += solve_mu0(mu[inds_inner],r[inds_inner],R_c)
        n[inds_inner] += RFIE_density(r[inds_inner],mu[inds_inner],mu0[inds_inner],R_c,dMenv,Mstar)
        return n

def calc_mu_mu0(mu,mu0,r,R_c):
    '''
    When mu --> 0, then mu/mu0 --> 1-R_c/r from the streamline equation.
    This function implements it to correct for numerical errors with small mu,mu0.

    Parameters
    ----------

    mu : 3D array
        Cosine angles.

    mu0 : 3D array
        Cosine angles of the streamlines.

    r : 3D array
        Radius from the centre (in au).

    R_c : float
        Centrifugal radius (in au).

    Returns
    -------

    mu_mu0 : 3D array
        mu/mu0 with the appropriate limiting values.

    Notes
    -----

    This implementation is very simple, just looking for values where mu==0.
    For complex grids, this would require updates to do more rigourous limiting.
    '''
    mu_mu0 = zeros(shape(mu))
    mu_mu0[mu==0] = 1-R_c/r[mu==0] #This comes out of the equation for a streamline if mu --> 0.
    mu_mu0[mu!=0] = mu[mu!=0]/mu0[mu!=0]
    return mu_mu0

def test_plot_3D(cube,slicex,slicey,slicez,contour_cmap = 'Greys_r',img_cmap = 'gist_stern',figsize = (10,4)):
    '''
    Plots the X, Y, Z slices of a cube.

    Parameters
    ----------

    cube : 3D array
        Array to plot.

    slicex : float
        X index to cut.

    slicey : float
        Y index to cut.

    slicez : float
        Z index to cut.

    contour_cmap : string
        Colourmap for the contour plot.
    
    img_cmap : string
        Colourmap for the image plot.

    figsize : 2x1 tuple
        Figure size.

    Returns
    -------

    None
        Makes a 3x1 plot of the different axes.
    '''
    plt.figure(figsize = figsize)
    plt.subplot(131)
    toplot = cube[slicex]
    plt.imshow(toplot,origin='lower',cmap=img_cmap)
    plt.colorbar(location='top',fraction=0.046, pad=0)
    plt.contour(toplot,cmap=contour_cmap,levels=8)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.subplot(132)
    toplot = cube[:,slicey]
    plt.imshow(toplot,origin='lower',cmap=img_cmap)
    plt.colorbar(location='top',fraction=0.046, pad=0)
    plt.contour(toplot,cmap=contour_cmap,levels=8)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.subplot(133)
    toplot = cube[:,:,slicez].T
    plt.imshow(toplot,origin='lower',cmap=img_cmap)
    plt.colorbar(location='top',fraction=0.046, pad=0)
    plt.contour(toplot,cmap=contour_cmap,levels=8)
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.tight_layout()
    plt.show()

def make_cavity(n, pars, cav_type='CONE',fill_value = 0):
    '''
    Cuts, i.e. removes, a cavity from a density distribution.

    Parameters
    ----------

    n : 3D array
        Cartesian density cube in units of cm-3: 

    pars : 1D array
         Length defined by envelope type.
    
    fill_value : float
        Constant density to fill the cavity (in cm-3).

    cav_type : string
        Envelope type. This defines the parameters required for the envelope creation.
        
        Supported types:
        ----------------

        CONE: Removes all density in a cone shape. The cavity is bipolar cones in the z-direction.

        Parameters:

        theta : Opening angle in degrees
        
        ----------------
        IN DEVELOPMENT
        Curved: Rotationally flattened infalling envelope from Whitney, B.A. et. al. 2003, ApJ 591, 1049-1063.

    Returns
    -------

    n : 3D array
        Cartesian density cube in units of cm-3, with the cavity SUBTRACTED.
    '''
    
    valid_types = ['CONE']
    valid_shapes = [(1,)]

    check_make_input(cav_type,pars,valid_types,valid_shapes)

    X, Y, Z = return_coords(n)
    Rdisk = sqrt(X**2 + Y**2)
    if cav_type == 'CONE':
        theta = pars[0]
        # Create cavity by setting density to zero in the cone region
        inds_cavity = where(Rdisk < abs(Z) * tan(deg2rad(theta)))
        n[inds_cavity] = fill_value
        
        return n

def fstream(mu,r,mu0,R_c):
    '''
    Stream-line equation for a rotationally flattening envelope.
    From Equation 2 of Whitney, B.A. et. al. 2003, ApJ 591, 1049-1063.
    
    Parameters
    ----------

    mu : float or array-like
        Cosine of the angle relative to the disk.
        
    r : float or array-like
        Radius from the centre (in au).
        
    mu0 : float or array-like
        Cosine polar angle of a streamline of infalling particles as r--> inf.

    R_c : float
        Centrifugal radius (in au).

    Returns
    -------

    obj : float or array like
        Objective function that should be minimized to calculate mu0.
    '''
    obj = mu0**3 + mu0*(r/R_c -1) - mu*(r/R_c)
    return obj

def dfstream(mu,r,mu0,R_c):
    '''
    Derivative of stream-line equation for a rotationally flattening envelope.
    Derived from Equation 2 of Whitney, B.A. et. al. 2003, ApJ 591, 1049-1063.
    
    Parameters
    ----------

    mu : float or array-like
        Cosine of the angle relative to the disk.
        
    r : float or array-like
        Radius from the centre (in au).
        
    mu0 : float or array-like
        Cosine polar angle of a streamline of infalling particles as r--> inf.

    R_c : float
        Centrifugal radius (in au).

    Returns
    -------

    der : float or array like
        Derivative of the streamline function.
    '''

    test_zero_div(R_c,'R_c')
    der = 3*mu0**2+r/R_c-1
    return der

def solve_mu0(mu,r,R_c,mu0_guess=0.5,epsilon=1e-6,N_iter=10000,trunc=1,verbose=False):
    '''
    Solve for mu0 for mu, r, and R_c specified, with the Newton Method. 

    Parameters
    ----------

    mu : float or array-like
        Cosine of the angle relative to the disk.
        
    r : float or array-like
        Radius from the centre (in au).
        
    R_c : float
        Centrifugal radius (in au).

    mu0_guess : float
        Initial guess for the mu0 cube.

    epsilon : float
        Tolerance for the Newton method.

    N_iter : integer
        Number of maximum iterations for optimization.

    trunc : float
        Value between 0 and 1 to adapt the numerical update. 
        A value of 1 is the full Newton's Method, while values lower
        than one is slower, but more stable.

    Returns
    -------

    mu0 : float or array like
        mu0 for a cube of same shape as mu and r.
    '''
    shp = shape(mu)
    mu0 = full(shp,mu0_guess)
    if verbose:
        print('Calculating Streamlines...')
    for i in range(N_iter):
        inds = where(abs(fstream(mu,r,mu0,R_c)) > epsilon) #only solve for indices where the objective function is larger than tolerance.
        mu0[inds] = mu0[inds] - trunc*fstream(mu[inds],r[inds],mu0[inds],R_c)/dfstream(mu[inds],r[inds],mu0[inds],R_c)
        if verbose:
            if i%10 == 0:
                print('Number of Voxels to Optimize = %d'%(len(inds[0])))
        if len(inds[0]) ==0: #if no more indices are relevant, stop calculating.
            break
    disk_plane_mask = abs(mu) < 1e-7
    if any(disk_plane_mask):
        mu0[disk_plane_mask] = 0
    if verbose:
        print('Done.')
    return mu0

def RFIE_density(r,mu,mu0,R_c,dMenv,Mstar,do_tests = True):
    '''
    The density distribution of a rotationally flattened infalling envelope 
    From Whitney, B.A. et. al. 2003, ApJ 591, 1049-1063.

    Parameters
    ----------

    r : array
        Radius from source (in au)

    mu : array
        Cosine angle in terms of the disk plane.

    mu0 : array
        Cosine angle in terms of the disk plane, of infalling streamlines.

    R_c : float
        Centrifugal radius (in au).

    dMenv : float
        Accretion rate of the envelope (in solar mass per year).

    Mstar : float
        Stellar mass (in solar masses).

    do_tests : boolean
        Whether to do tests for division by zero. 
        This option makes it easier to find the specific cause of division errors.

    Returns
    -------

    n : array
        Density distribution. 
    '''
    mu_mu0 = calc_mu_mu0(mu,mu0,r,R_c)
    if do_tests:
        test_zero_div(R_c,'R_c')
        test_zero_div(G*Mstar/R_c**3,'G*Mstar/R_c**3')
        test_zero_div(r/R_c,'r/R_c')
        test_zero_div(1+mu_mu0,'1+mu/mu0')
        #test_zero_div(mu_mu0+(2*mu0**2*R_c)/r,'mu/mu0+(2*mu0**2*R_c)/r')
    
    n = (dMenv/(4*pi))*(G*Mstar/R_c**3)**-0.5*(r/R_c)**-1.5*(1+mu_mu0)**-0.5*(mu_mu0+(2*mu0**2*R_c)/r)**-1
    return n

def test_zero_div(cube,array_str,zero_val=1e-10):
    '''
    Tests for division by zero in an array.

    Parameters
    ----------

    cube : float or array
        Any array or float.

    Returns
    -------

    None
        Raises value error if there is zero division.
        The program would break in any way, but this shows where the problem is.
    '''
    if hasattr(cube, "__len__"): #Tests if is array.
        if sum(abs(cube.flatten()) < zero_val)>0:
            shp = shape(cube)
            raise ValueError('Division by zero. abs(%s) = %.2E'%(array_str,min(abs(cube.flatten()))))
    else:
        if abs(cube) < zero_val:
            raise ValueError('Division by zero. abs(%s) = %.2E'%(cube,array))

def SFAD_disk(Rdisk,z,n0, Rstar,h0,alpha,beta):
    '''
    Standard flared accretion disk density distribution.
    From Whitney, B.A. et. al. 2003, ApJ 591, 1049-1063.

    Parameters
    ----------

    Rdisk : array
        Radial coordinate in disk midplane (in au).

    z : array
        Height from disk midplane (in au).

    n0 : float
        Density at stellar radius (in cm-3).

    Rstar : float
        Stellar radius (in au).

    h0 : float
        Scale height at disk inner radius (in au).

    alpha : float
        Spectral index for density distribution.

    beta : float
        Spectral index for scale height.

    Returns
    -------

    n : array
        Density of disk (in au).
    '''
    n = n0*(1-sqrt(Rstar/Rdisk))*(Rstar/Rdisk)**alpha*exp(-0.5*(z/scaleheight(h0,Rdisk/R_star,beta))**2)
    return n

def scaleheight(h0,R,beta):
    '''
    Scale height of a standard flared accretion disk.

    Parameters
    ----------

    h0 : float
        Scale height at inner radius (in au).

    R : array
        Radius (as fraction of stellar radius).

    beta : float
        Spectral index for scale height.

    Returns
    -------

    h : array
        Scale height (in au).
    '''
    h = h0*R**beta
    return h


def save_non_hierarchical_cloud_file(n,filename):
	'''
	Saves a square density distribution into a SOC cloud file.

	Parameters
	----------

	n : 3D array
		Square density distribution in cm-3

	filename : string
		Filename for cloud file.

	Returns
	-------

	None
		Saves the cloud file.
	'''
	shp = list(shape(n))
	if shp.count(shp[0]) != len(shp):
		raise ValueError('Density cube n is not square.')

	N = shp[0]                                     # model with N^3 cells
	fp  =  open(filename, 'w')
	asarray([N, N, N, 1, N*N*N], int32).tofile(fp)  #  NX, NY, NZ, LEVELS, CELLS
	asarray([N*N*N,], int32).tofile(fp)             #  cells on the first (and only) level
	n.tofile(fp)
	fp.close()

def make_disk(n, pars, disk_type='SFAD'):
    '''
    Make a disk density distribution.

    Parameters
    ----------

    n : 3D array
        Cartesian density cube in units of cm-3.
    
    pars : 1D array
         Length defined by disk type.

    disk_type : string
        Disk type. This defines the parameters required for the disk creation.
        
        Supported types:
        ----------------

        SFAD: Standard Flared Accretion Disk
        From Whitney, B.A. et. al. 2003, ApJ 591, 1049-1063.
        
        Parameters:
        
        n0 : float
            Density at stellar radius (in cm-3).
            
        Rstar : float
            Stellar radius (in au).
            
        h0 : float
            Scale height at disk inner radius (in au).
            
        alpha : float
            Spectral index for density distribution.
            
        beta : float
            Spectral index for scale height.
        
        Rout : float
            Outer radius of the disk (in au).
        ----------------

    Returns
    -------

    n : 3D array
        Cartesian density cube in units of cm-3, with the disk ADDED.
    '''
    valid_types = ['SFAD']
    valid_shapes = [(6,)]

    check_make_input(disk_type, pars, valid_types, valid_shapes)

    X, Y, Z = return_coords(n)
    Rdisk = sqrt(X**2 + Y**2)  # Radius along the disk plane
    
    if disk_type == 'SFAD':
        n0, Rstar, h0, alpha, beta, Rout = pars
        
        # Only add density within the disk outer radius
        disk_region = where(logical_and(Rdisk >= Rstar, Rdisk <= Rout))
        
        if len(disk_region[0]) > 0:
            # Apply the SFAD disk model to the disk region
            disk_density = zeros(shape(n))
            valid_R = Rdisk[disk_region]
            valid_z = Z[disk_region]
            
            # Calculate scale height
            h = scaleheight(h0, valid_R/Rstar, beta)
            
            # Calculate disk density where Rdisk > Rstar
            disk_density[disk_region] = n0 * (1-sqrt(Rstar/valid_R)) * (Rstar/valid_R)**alpha * exp(-0.5*(valid_z/h)**2)
            
            # Add disk density to the cube
            n += disk_density
            
    return n
