# Visualization of the BDM model

import matplotlib.pyplot as plt
from grid import Grid
import os
from typing import List

def plot_grid(grid: Grid, time_point: float):
    """Visualize the grid with blue squares for cells and white squares for empty locations"""
    fig, ax = plt.subplots(figsize=(10, 10))
    lattice_size = grid.lattice_size

    # Set equal aspect ratio and limits
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, lattice_size - 0.5)
    ax.set_ylim(-0.5, lattice_size - 0.5)

    for i in range(lattice_size):
        for j in range(lattice_size):
            location = grid.lattice[i, j]
            color = "blue" if location.has_cell() else "white"
            square = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=color, edgecolor="black")
            ax.add_patch(square)
    cell_density = grid.num_cells / (lattice_size**2) * 100
    ax.set_title(f"Time: {time_point}, Cell density: {cell_density:.2f}%")

    # Create 'plots' directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/grid_{int(time_point)}.png")
    plt.close()

def plot_cell_density(time_points: List[float],
                      cell_densities: List[float],
                      red_dot_time: float,
                      red_dot_density: float):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(time_points, cell_densities, 'ko-')
    ax.plot(red_dot_time, red_dot_density, 'ro')
    ax.set_xlabel("Time")
    ax.set_ylabel("Cell density") 
    ax.set_title("Cell density over time")
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/cell_density_{int(red_dot_time)}.png")
    plt.close()
