# Implementation of a cell class

import numpy as np


class Cell:
    def __init__(self, status="cell"):
        self.status = status
    
    def get_status(self):
        return self.status
    
    def set_status(self, status):
        """Set the status of the cell
        Available statuses: "cell"
        """
        self.status = status
    
        
    def proliferate(self, prob_proliferate):
        """
        Attempt to place daughter cell in random neighboring site
        Returns True if successful, False if aborted
        """
        if np.random.random() > prob_proliferate:
            return False
        return False