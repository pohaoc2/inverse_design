import numpy as np
from typing import Tuple, List
from config import BDMConfig
from grid import Grid
from cell import Cell
from location import Location
class Gillespie:
    def __init__(self, config: BDMConfig):
        """
        Initialize Gillespie algorithm for BDM model
        Args:
            config: Configuration object containing rates
        """
        self.config = config
        self.proliferate_rate = config.rates.proliferate
        self.death_rate = config.rates.death
        self.migrate_rate = config.rates.migrate
        
    def calculate_propensities(self, grid: Grid) -> Tuple[float, List[Tuple[str, int, int]]]:
        """
        Calculate propensities for all possible events
        Args:
            grid: Current grid state
        Returns:
            total_propensity: Sum of all event propensities
            events: List of possible events with their locations
        """
        events = []
        total_propensity = 0
        
        # Iterate through all grid positions
        for i in range(grid.lattice_size):
            for j in range(grid.lattice_size):
                location = grid.get_location(i, j)
                if location.has_cell():
                    # Death event
                    total_propensity += self.death_rate
                    events.append(("death", i, j))
                    
                    # Migration events - check neighboring sites
                    total_propensity += self.migrate_rate
                    events.append(("migrate", i, j))

                    # Proliferation events - only add once if there are empty neighbors
                    total_propensity += self.proliferate_rate
                    events.append(("proliferate", i, j))
                        
        return total_propensity, events
    
    def _move_cell(self, from_location, to_location):
        """
        Move a cell from one location to another
        Args:
            from_location: Source location containing the cell
            to_location: Target location to move the cell to
        """
        cell = from_location.get_cell()
        from_location.remove_cell()
        to_location.set_cell(cell)
    
    def _get_random_empty_neighbor(self, grid: Grid, i: int, j: int) -> tuple[Location, int, int] | None:
        """
        Get a random empty neighbor from all von Neumann neighbors
        Args:
            grid: Current grid state
            i, j: Location coordinates
        Returns:
            Tuple of (target_location, new_i, new_j) if found, None otherwise
        """
        location = grid.get_location(i, j)
        all_neighbors = location.get_von_neumann_neighbors()
        if all_neighbors:
            new_i, new_j = all_neighbors[np.random.randint(len(all_neighbors))]
            target_location = grid.get_location(new_i, new_j)
            if not target_location.has_cell():
                return target_location, new_i, new_j
        return None

    def _migrate_cell(self, grid: Grid, i: int, j: int):
        """
        Attempt to migrate a cell to a random neighboring site
        Args:
            grid: Current grid state
            i, j: Location coordinates of the cell to migrate
        """
        location = grid.get_location(i, j)
        neighbor_result = self._get_random_empty_neighbor(grid, i, j)
        if neighbor_result is not None:
            target_location, _, _ = neighbor_result
            self._move_cell(location, target_location)

    def _proliferate_cell(self, grid: Grid, i: int, j: int):
        """
        Attempt to create a new cell in a random neighboring site
        Args:
            grid: Current grid state
            i, j: Location coordinates of the parent cell
        """
        neighbor_result = self._get_random_empty_neighbor(grid, i, j)
        if neighbor_result is not None:
            target_location, _, _ = neighbor_result
            new_cell = Cell(status="cell")
            target_location.set_cell(new_cell)

    def execute_event(self, grid: Grid, event_type: str, i: int, j: int):
        """
        Execute the chosen event
        Args:
            grid: Current grid state
            event_type: Type of event to execute
            i, j: Location of the event
        """
        if event_type == "death":
            location = grid.get_location(i, j)
            location.remove_cell()
            
        elif event_type == "migrate":
            self._migrate_cell(grid, i, j)
                
        elif event_type == "proliferate":
            self._proliferate_cell(grid, i, j)
        else:
            raise ValueError(f"Invalid event type: {event_type}")
    
    def step(self, grid: Grid) -> float:
        """
        Execute one step of the Gillespie algorithm
        Args:
            grid: Current grid state
        Returns:
            time_increment: Time increment for this step
        """
        # Calculate propensities
        total_propensity, events = self.calculate_propensities(grid)
        
        if total_propensity == 0 or not events:
            return 0
            
        # Generate random numbers
        r1, r2 = np.random.random(2)
        
        # Calculate time increment
        time_increment = -np.log(r1) / total_propensity
        
        # Choose event
        event_index = int(r2 * len(events))
        event_type, i, j = events[event_index]
        
        # Execute event
        self.execute_event(grid, event_type, i, j)
        
        return time_increment
    
    def run(self, grid: Grid, max_time: float = 1.0) -> List[float]:
        """
        Run the Gillespie algorithm until max_time
        Args:
            grid: Initial grid state
            max_time: Maximum simulation time
        Returns:
            time_points: List of time points when events occurred
        """
        current_time = 0
        time_points = [current_time]
        
        while current_time < max_time:
            time_increment = self.step(grid)
            if time_increment == 0:
                break
            current_time += time_increment
            time_points.append(current_time)
            
        return time_points