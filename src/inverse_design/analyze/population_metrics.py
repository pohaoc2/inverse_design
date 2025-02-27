from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import re
import logging
import warnings
import numpy as np
from .file_utils import FileParser
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy import stats


class PopulationMetrics:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def calculate_doub_time(n1: float, n2: float, time_difference: float) -> float:
        """Calculate cell population doubling time based on initial and final cell counts

        Args:
            n1: Initial cell count
            n2: Final cell count
            time_difference: Time elapsed between counts (minutes)

        Returns:
            Doubling time in hours. Returns float('inf') if no growth or negative growth.
        """
        if n2 <= n1 or n1 <= 0:
            return float("inf")
        doubling_time = time_difference * np.log(2) / np.log(n2 / n1)
        return doubling_time / 60

    @staticmethod
    def calculate_colony_diameter(cells: List[Dict[str, Any]], locations: List[Dict[str, Any]], C: float = 30.0) -> float:
        """Calculate the colony diameter based on cell coordinates.

        Args:
            cells: no used
            locations: List of location dictionaries containing cell coordinates
            C: Scaling factor (default = 30)

        Returns:
            Colony diameter
        """

        if not locations:
            return 0.0

        u_values = [location["coordinate"][0] for location in locations]
        v_values = [location["coordinate"][1] for location in locations]
        w_values = [location["coordinate"][2] for location in locations]

        umax, umin = max(u_values), min(u_values)
        vmax, vmin = max(v_values), min(v_values)
        wmax, wmin = max(w_values), min(w_values)

        diameter = (
            C * (max(umax - umin + 1, 0) + max(vmax - vmin + 1, 0) + max(wmax - wmin + 1, 0)) / 3
        )
        return diameter

    @staticmethod
    def calculate_symmetry(cells: List[Dict[str, Any]], locations: List[Dict[str, Any]]) -> float:
        """Calculate colony symmetry based on cell distribution.

        Args:
            cells: no used
            locations: List of location dictionaries containing cell coordinates

        Returns:
            Symmetry score between 0 and 1
        """
        return 0.0
        if not cells:
            return 0.0

        # Calculate center of mass
        com_x = np.mean([cell["coordinate"][0] for cell in cells])
        com_y = np.mean([cell["coordinate"][1] for cell in cells])
        com_z = np.mean([cell["coordinate"][2] for cell in cells])

        # Calculate radial distribution
        distances = [
            np.sqrt(
                (cell["coordinate"][0] - com_x)**2 +
                (cell["coordinate"][1] - com_y)**2 +
                (cell["coordinate"][2] - com_z)**2
            )
            for cell in cells
        ]

        # Calculate symmetry score based on distance distribution
        std_dist = np.std(distances)
        mean_dist = np.mean(distances)
        cv = std_dist / mean_dist if mean_dist > 0 else float('inf')
        
        # Convert to score between 0 and 1 (lower CV means higher symmetry)
        symmetry_score = 1 / (1 + cv)
        return symmetry_score

    @staticmethod
    def calculate_shannon(cells: List[Dict[str, Any]]) -> float:
        """Calculate Shannon diversity index based on cell states.

        Args:
            cells: List of cell dictionaries

        Returns:
            Shannon diversity index
        """
        return 0.0
        if not cells:
            return 0.0

        # Count states
        state_counts = {}
        for cell in cells:
            state = cell["state"]
            state_counts[state] = state_counts.get(state, 0) + 1

        # Calculate Shannon index
        total_cells = len(cells)
        shannon_index = 0.0
        for count in state_counts.values():
            p = count / total_cells
            shannon_index -= p * np.log(p)

        return shannon_index

    @staticmethod
    def calculate_density(cells: List[Dict[str, Any]]) -> float:
        """Calculate colony density (cells per unit area) for 2D colonies.

        Args:
            cells: List of cell dictionaries
            C: Scaling factor (default = 30)

        Returns:
            Cell density (cells per unit area)
        """
        return 0.0
        if not cells:
            return 0.0

        # Calculate approximate colony area using diameter
        diameter = PopulationMetrics.calculate_colony_diameter(cells)
        area = np.pi * (diameter/2)**2

        return len(cells) / area if area > 0 else 0.0

    @staticmethod
    def calculate_act(cells: List[Dict[str, Any]]) -> float:
        """Calculate the number of active cells over total number of cells"""
        if not cells:
            return 0.0

        return len([cell for cell in cells if cell["state"] in ["PROLIFERATIVE", "MIGRATORY"]]) / len(cells)

    @staticmethod
    def calculate_n_cells(cells: List[Dict[str, Any]]) -> float:
        """Calculate the number of cells in the colony"""
        if not cells:
            return 0.0

        return len(cells)

    def parse_location_file(self, filename: str) -> Dict[str, str]:
        """Parse location filename to extract experiment info"""
        return FileParser.parse_simulation_file(filename, "LOCATIONS")

    def load_locations_data(
        self, folder_path: Path, timestamp: str
    ) -> List[Tuple[Dict[str, str], List[Dict[str, Any]]]]:
        """Load location data from a specific timestamp for all seeds"""
        results = []
        try:
            file_pattern = f"*_{timestamp}.LOCATIONS.json"
            location_files = folder_path.glob(file_pattern)

            for location_file in location_files:
                file_info = self.parse_location_file(location_file.name)
                if file_info:
                    with open(location_file, "r") as f:
                        location_data = json.load(f)
                        results.append((file_info, location_data))

        except Exception as e:
            self.logger.error(f"Error loading locations data from {folder_path}: {str(e)}")

        return results

    @staticmethod
    def calculate_colony_growth(
        colony_diameters: Dict[str, Dict[str, List[float]]],
        timestamps: List[float],
    ) -> Dict[str, Dict[str, float]]:
        """Fit colony diameter growth to linear function (y = mx + c)

        Args:
            colony_diameters: Dictionary of diameters structured as {(exp_group, exp_name): {seed: [diameters]}}
            timestamps: List of timepoints in hours

        Returns:
            Dictionary with structure:
            {

            }
        """
        results = {}
        slopes = []
        r_values = []

        for diameters in colony_diameters.values():
            if len(diameters) != len(timestamps):
                warnings.warn(f"Length of diameters ({len(diameters)}) does not match length of timestamps ({len(timestamps)})")
                shorter_length = min(len(diameters), len(timestamps))
                diameters = diameters[:shorter_length]
                timestamps = timestamps[:shorter_length]

            slope, intercept, r_value, p_value, slope_std = stats.linregress(
                timestamps, diameters
            )
            slopes.append(slope)
            r_values.append(r_value)

        results = {
            "slope": np.median(slopes),
            "slope_std": np.std(slopes),
            "r_value": np.median(r_values)
        }

        return results
