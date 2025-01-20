# Visualization of the BDM model

import matplotlib.pyplot as plt

def plot_lattice(lattice):
    plt.imshow(lattice, cmap='gray')
    plt.colorbar()
    plt.show()