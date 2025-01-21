# Implementation of a birth-death-migration model using the Gillespie algorithm

import hydra
import copy
from omegaconf import DictConfig, OmegaConf
import numpy as np
from cell import Cell
from grid import Grid
from config import BDMConfig
import os
import logging
from typing import List
from gillespie import Gillespie
from vis import plot_grid


class BDM:
    def __init__(self, config: BDMConfig):
        """
        Initialize BDM model using Hydra config
        Args:
            config: Hydra configuration object
        """
        self.lattice_size = config.lattice.size
        self.initial_density = config.lattice.initial_density
        self.proliferate_rate = config.rates.proliferate
        self.death_rate = config.rates.death
        self.migrate_rate = config.rates.migrate
        self.current_time = 0.0
        self.grid = self.initialize()
        self.gillespie = Gillespie(config)
        self.log_config(config)
        
    def log_config(self, config: BDMConfig):
        """
        Log model parameters and save configuration file
        Args:
            config: Hydra configuration object
        """
        log = logging.getLogger(__name__)
        
        # Log model parameters only
        log.info("Model Parameters:")
        log.info(f"- Lattice size: {self.lattice_size}")
        log.info(f"- Initial density: {self.initial_density}")
        log.info(f"- Proliferation rate: {self.proliferate_rate}")
        log.info(f"- Death rate: {self.death_rate}")
        log.info(f"- Migration rate: {self.migrate_rate}")
        
        # Save config to YAML file in Hydra's output directory
        if hydra.core.hydra_config.HydraConfig.initialized():
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            config_path = os.path.join(output_dir, "config_log.yaml")
            with open(config_path, 'w') as f:
                OmegaConf.save(config=config, f=f)
            log.info(f"Configuration saved to {config_path}")
    
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
            cell = Cell(status="cell", location=location)
            location.set_cell(cell)

        self.original_grid = copy.deepcopy(grid)
        return grid
    
    def get_grid(self):
        """Return the current grid state"""
        return self.grid

    def step(self, max_time: float = 1.0) -> tuple[List[float], List[tuple[str, int, int]], List[Grid]]:
        """
        Step the simulation forward by a specified time duration using Gillespie algorithm
        Args:
            max_time: Duration to simulate (default: 1.0)
        Returns:
            time_points: List of time points when events occurred
            events: List of (event_type, i, j) tuples describing what happened
            grid_states: List of grids at each time point
        """
        time_points, events, grid_states = self.gillespie.run(self.grid, max_time)
        self.current_time += max(time_points)
        return time_points, events, grid_states

    def analyze_events_by_timeunit(self, time_points: List[float], events: List[tuple[str, int, int]], 
                                 time_unit: float = 1.0) -> None:
        """
        Analyze and output events grouped by time unit
        Args:
            time_points: List of event times
            events: List of (event_type, i, j) tuples
            time_unit: Size of time interval for analysis
        """
        log = logging.getLogger(__name__)
        current_unit = time_unit
        
        # Group events by time unit
        while True:
            # Find events in current time interval
            events_in_interval = [(t, e) for t, e in zip(time_points, events) 
                                if t <= current_unit]
            
            if not events_in_interval:
                break
                
            # Count event types
            event_counts = {}
            for _, event_type in events_in_interval:
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Log summary
            log.info(f"Time unit {current_unit}:")
            log.info(f"  Total events: {len(events_in_interval)}")
            for event_type, count in event_counts.items():
                log.info(f"  - {event_type}: {count}")
            
            # Remove processed events
            remaining_indices = [i for i, t in enumerate(time_points) 
                               if t > current_unit]
            time_points = [time_points[i] for i in remaining_indices]
            events = [events[i] for i in remaining_indices]
            
            current_unit += time_unit

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'bdm.log')),
            logging.StreamHandler()
        ]
    )
    
    bdm = BDM(cfg)
    
    max_time = 700.0
    time_unit = 5.0
    log = logging.getLogger(__name__)
    
    # Run complete simulation
    log.info("Starting simulation...")
    time_points, events, grid_states = bdm.step(max_time)
    time_to_grids ={time_point: grid_state for time_point, grid_state in zip(time_points, grid_states)}

    # Plot grid states closest to each time unit
    max_units = int(np.ceil(max_time))
    plot_grid(bdm.original_grid, 0)
    for unit in range(1, max_units+1):
        target_time = unit * time_unit
        # Find closest time point to the target time
        closest_time = min(time_points, key=lambda x: abs(x - target_time))
        plot_grid(time_to_grids[closest_time], np.round(closest_time, 2))
    
    log.info(f"Simulation completed with {len(events)} total events")
    # Analyze events by time unit
    # log.info("\nAnalyzing events by time unit:")
    # bdm.analyze_events_by_timeunit(time_points.copy(), events.copy(), time_unit)

if __name__ == "__main__":
    """
    Example usage:
    python bdm.py
    python bdm.py --config=experiments/exp1_config.yaml
    python bdm.py lattice.size=200 rates.proliferate=2.0
    """
    main()
    