import h5py as h5
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt

DX = 0.03125 # Grid spacing in FlashX simulations

def heater_init(xmin, xmax, num_sites):
    r"""
    Initialize the nucleation sites on the 1-D heater. Returns a 1D line of sites
    that is dependant on wall temperature.

    Args:
        xmin (float): The x-coordinate where heater begins.
        xmax (float): The x-coordinate where heater ends.
        num_sites (int): The number of nucleation sites in the domain.

    Returns:
        numpy.ndarray: The coordinates of the heater nucleation sites.
    """
    x_sites = np.ndarray(num_sites, dtype=float)
    y_sites = np.ndarray(num_sites, dtype=float)

    halton_sequence = qmc.Halton(d=2, seed=1)
    halton_sample = halton_sequence.random(num_sites)

    x_sites[:] = xmin + halton_sample[:, 0] * (xmax - xmin) 
    y_sites[:] = 1e-13

    return x_sites, y_sites

def dfun_init(x_grid, y_grid, x_sites, y_sites, seed_radius):
    r"""
    Initialize the distance function for given nucleation sites.

    Args:
        x_grid (numpy.ndarray): The x-coordinates of the grid.
        y_grid (numpy.ndarray): The y-coordinates of the grid.
        x_sites (numpy.ndarray): The x-coordinates of the nucleation sites.
        y_sites (numpy.ndarray): The y-coordinates of the nucleation sites.
        seed_radius (float): The radius of the nucleation site.

    Returns:
        numpy.ndarray: The initial distance function with nucleated bubbles.
    """
    dfun = np.zeros_like(x_grid) - np.inf
    seed_height = seed_radius * np.cos(np.pi/4)
    for htr_points_xy in zip(x_sites, y_sites):
        seed_x = htr_points_xy[0]
        seed_y = htr_points_xy[1] + seed_height

        interim_dfun = seed_radius - np.sqrt((x_grid - seed_x)**2 + (y_grid - seed_y)**2)
        dfun = np.maximum(dfun, interim_dfun)
    return dfun

def tag_renucleation(x_sites, y_sites, dfun, coordx, coordy, seed_radius, curr_iter, nuc_wait_time=0.4):
    r"""
    Tag the nucleation sites for renucleation after a certain time.

    Args:
        x_sites (numpy.ndarray): The x-coordinates of the nucleation sites.
        y_sites (numpy.ndarray): The y-coordinates of the nucleation sites.
        dfun (numpy.ndarray): The distance function at the current time.
        coordx (numpy.ndarray): The x-coordinates of the first row of the grid.
        coordy (numpy.ndarray): The y-coordinates of the first column of the grid.
        curr_iter (int): The current iteration of the model.
        nuc_wait_time (float): The minimum time after which renucleation happens. Multiple of 0.1

    Returns:
        numpy.ndarray: The tagged nucleation sites.
    """
    tagged_sites = np.zeros_like(x_sites, dtype=bool)
    nuc_plot_interval = nuc_wait_time * 10
    seed_height = seed_radius * np.cos(np.pi/4)
    for i, htr_points_xy in enumerate(zip(x_sites, y_sites)):
        seed_x = htr_points_xy[0]
        seed_y = htr_points_xy[1] + seed_height
        x_i = np.searchsorted(coordx, seed_x)
        y_i = np.searchsorted(coordy, seed_y)

        dfun_site = (dfun[y_i, x_i] + dfun[y_i+1, x_i] + dfun[y_i, x_i+1] + dfun[y_i+1, x_i+1])/4.0  # Average of the 4 cells surrounding the nucleation site

        if dfun_site < 0 and curr_iter % nuc_plot_interval == 0:
            tagged_sites[i] = True

    return tagged_sites
    
def renucleate(x_grid, y_grid, x_sites, y_sites, tagged_sites, curr_dfun, seed_radius):
    r"""
    Renucleate the sites that are tagged for renucleation.

    Args:
        x_grid (numpy.ndarray): The x-coordinates of the grid.
        y_grid (numpy.ndarray): The y-coordinates of the grid.
        x_sites (numpy.ndarray): The x-coordinates of the nucleation sites.
        y_sites (numpy.ndarray): The y-coordinates of the nucleation sites.
        curr_dfun (numpy.ndarray): The distance function at the current time.
        seed_radius (float): The radius of the nucleation site.

    Returns:
        numpy.ndarray: The updated nucleation sites.
    """
    seed_height = seed_radius * np.cos(np.pi/4)
    for i, htr_points_xy in enumerate(zip(x_sites, y_sites)):
        if tagged_sites[i]:
            seed_x = htr_points_xy[0]
            seed_y = htr_points_xy[1] + seed_height

            interim_dfun = seed_radius - np.sqrt((x_grid - seed_x)**2 + (y_grid - seed_y)**2)
            curr_dfun = np.maximum(curr_dfun, interim_dfun)
    
    return curr_dfun


if __name__ == '__main__':
    sim = h5.File('/Users/shakeel/bubbleml_data/PoolBoiling-WallSuperheat-FC72-2D/Twall-100.hdf5', 'r')
    dfun_0 = sim['dfun'][...][0]
    x_0, y_0 = sim['x'][...][0], sim['y'][...][0]
    coordx, coordy = x_0[0], np.transpose(y_0)[0]
    

    init_nucl_coordx, init_nucl_coordy = heater_init(-5.0, 5.0, 40) # Coordinates of 40 nucleation sites

    my_dfun = dfun_init(x_0, y_0, init_nucl_coordx, init_nucl_coordy, seed_radius=0.1) # Initialize the distance function with nucleated bubbles at t-0
    
    # Plot the initial distance function for testing
    my_dfun[my_dfun>0] *= (255/my_dfun.max())
    my_dfun[my_dfun<0] = 255
    my_dfun = my_dfun.astype(np.uint8)
    plt.imsave(f'my_dfun.png', np.flipud(my_dfun), cmap='GnBu')
    plt.close()

    dfun_40 = sim['dfun'][...][40]

    # Renucleation algorithm
    tagged_nucl_sites = tag_renucleation(init_nucl_coordx, init_nucl_coordy, dfun_40, coordx, coordy, seed_radius=0.1, curr_iter=40, nuc_wait_time=0.4) 
    dfun_40 = renucleate(x_0, y_0, init_nucl_coordx, init_nucl_coordy, tagged_nucl_sites, dfun_40, seed_radius=0.1) 

    # Plot the distance function at t=40 after renucleation for testing 
    dfun_40[dfun_40>0] *= (255/dfun_40.max())
    dfun_40[dfun_40<0] = 255
    dfun_40 = dfun_40.astype(np.uint8)
    plt.imsave(f'dfun_40_renucleated.png', np.flipud(dfun_40), cmap='GnBu')
    plt.close()

