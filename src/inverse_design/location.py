from cell import Cell

class Location:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.cell = None
        self.valid_neighbors = []
    
    @property    
    def coordinates(self):
        """Return coordinates as a tuple"""
        return (self.i, self.j)
    
    def get_neighbor_coordinates(self):
        """Get potential neighbor coordinates without bounds checking"""
        return [
            (self.i-1, self.j),
            (self.i+1, self.j), 
            (self.i, self.j-1),
            (self.i, self.j+1)
        ]
    
    def calculate_valid_neighbors(self, lattice_size):
        """Calculate and store valid neighbor coordinates"""
        potential_neighbors = self.get_neighbor_coordinates()
        self.valid_neighbors = [
            n for n in potential_neighbors 
            if (0 <= n[0] < lattice_size and 0 <= n[1] < lattice_size)
        ]

    def get_cell(self):
        return self.cell

    def set_cell(self, cell):
        self.cell = cell

    def get_von_neumann_neighbors(self):
        return self.valid_neighbors

    def has_cell(self):
        return self.cell is not None

    def remove_cell(self):
        self.cell = None
