# Implementation of a birth-death-migration model using the Gillespie algorithm

import numpy as np
from location import Location

class Grid:
    def __init__(self, lattice_size):
        self.lattice_size = lattice_size
        self.initialize(lattice_size)

    def initialize(self, lattice_size):
        self.lattice = np.full((lattice_size, lattice_size), None, dtype=object)
        for i in range(lattice_size):
            for j in range(lattice_size):
                self.lattice[i,j] = Location(i, j)
        
        for i in range(lattice_size):
            for j in range(lattice_size):
                location = self.lattice[i,j]
                location.calculate_valid_neighbors(lattice_size)

    def get_lattice(self):
        return self.lattice
    
    def get_lattice_size(self):
        return self.lattice_size

    def get_location(self, i, j):
        return self.lattice[i,j]

    def get_empty_von_neumann_neighbors(self, i, j):
        location = self.get_location(i, j)
        return location.get_empty_von_neumann_neighbors()

