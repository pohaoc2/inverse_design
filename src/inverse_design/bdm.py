# Implementation of a birth-death-migration model using the Gillespie algorithm

import numpy as np
from cell import Cell
from grid import Grid

class BDM:
    def __init__(self, lattice_size, initial_density=0.05):
        """
        Initialize BDM model
        Args:
            lattice_size: Size of the square lattice
            initial_density: Initial fraction of sites occupied by cells (default 0.05)
        """
        self.lattice_size = lattice_size
        self.initial_density = initial_density
        self.grid = self.initialize()
    
    def initialize(self):
        """
        Initialize lattice with agents randomly placed according to initial_density
        Returns: Grid object with agents placed
        """
        # Create grid with empty cells
        grid = Grid(self.lattice_size)
        
        # Calculate number of agents to place
        total_sites = self.lattice_size * self.lattice_size
        num_agents = int(self.initial_density * total_sites)
        
        # Randomly select positions for agents
        flat_indices = np.random.choice(total_sites, size=num_agents, replace=False)
        row_indices = flat_indices // self.lattice_size
        col_indices = flat_indices % self.lattice_size
        
        # Place agents at selected positions
        for i, j in zip(row_indices, col_indices):
            location = grid.get_location(i, j)
            cell = Cell(status="cell")
            location.set_cell(cell)
        
        return grid
    
    def get_grid(self):
        """Return the current grid state"""
        return self.grid

if __name__ == "__main__":
    lattice_size = 100
    initial_density = 0.05
    bdm = BDM(lattice_size, initial_density)
    grid = bdm.get_grid()
    print(grid.lattice)
    