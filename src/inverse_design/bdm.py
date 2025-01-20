# Implementation of a birth-death-migration model using the Gillespie algorithm

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from cell import Cell
from grid import Grid
from config import BDMConfig
import os
import logging
from typing import List
from gillespie import Gillespie


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
            cell = Cell(status="cell")
            location.set_cell(cell)
        
        return grid
    
    def get_grid(self):
        """Return the current grid state"""
        return self.grid

    def step(self, max_time: float = 1.0) -> List[float]:
        """
        Step the simulation forward using Gillespie algorithm
        Args:
            max_time: Maximum simulation time
        Returns:
            time_points: List of time points when events occurred
        """
        return self.gillespie.run(self.grid, max_time)

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
    # Run simulation for 10 time units
    time_points = bdm.step(max_time=10.0)
    print(f"Simulation completed with {len(time_points)} events")
    # print(bdm.grid)

if __name__ == "__main__":
    """
    Example usage:
    python bdm.py
    python bdm.py --config=experiments/exp1_config.yaml
    python bdm.py lattice.size=200 rates.proliferate=2.0
    """
    main()
    