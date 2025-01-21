# Visualization of the BDM model

import matplotlib.pyplot as plt
from grid import Grid
import os

def plot_grid(grid: Grid, time_point: float):
    """Visualize the grid with blue squares for cells and white squares for empty locations
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    lattice_size = grid.lattice_size
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, lattice_size - 0.5)
    ax.set_ylim(-0.5, lattice_size - 0.5)
    
    for i in range(lattice_size):
        for j in range(lattice_size):
            location = grid.lattice[i,j]
            color = 'blue' if location.has_cell() else 'white'
            square = plt.Rectangle((i-0.5, j-0.5), 1, 1, 
                                 facecolor=color, 
                                 edgecolor='black')
            ax.add_patch(square)
    cell_density = grid.num_cells / (lattice_size ** 2) * 100
    ax.set_title(f'Time: {time_point}, Cell density: {cell_density:.2f}%')
    
    # Create 'plots' directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/grid_{time_point}.png')
    plt.close()
    