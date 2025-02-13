from typing import List, Dict, Any, Tuple
import numpy as np
import json
import re
import logging
from pathlib import Path
from .file_utils import FileParser

class CellMetrics:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def calculate_growth_rate(
        cells_t1: List[Dict[str, Any]], cells_t2: List[Dict[str, Any]], time_difference: float
    ) -> float:
        """Calculate growth rate between two timepoints"""
        if not cells_t1 or not cells_t2:
            return 0.0

        n1 = len(cells_t1)
        n2 = len(cells_t2)
        return (n2 - n1) / time_difference

    @staticmethod
    def calculate_average_volume(cells: List[Dict[str, Any]]) -> float:
        """Calculate average cell volume"""
        if not cells:
            return 0.0

        volumes = [cell["volume"] for cell in cells]
        return sum(volumes) / len(volumes)

    @staticmethod
    def calculate_activity(cells: List[Dict[str, Any]]) -> float:
        """Calculate cell activity (proliferative + migratory cells fraction)

        Args:
            cells: List of cell dictionaries

        Returns:
            Activity ratio between 0 and 1
        """
        if not cells:
            return 0.0

        active_cells = sum(1 for cell in cells if cell["state"] in ["PROLIFERATIVE", "MIGRATORY"])
        return active_cells / len(cells)

    @staticmethod
    def calculate_doubling_time(n1: float, n2: float, time_difference: float) -> float:
        """Calculate cell population doubling time based on initial and final cell counts

        Args:
            n1: Initial cell count
            n2: Final cell count
            time_difference: Time elapsed between counts (minutes)

        Returns:
            Doubling time in hours. Returns float('inf') if no growth or negative growth.
        """
        if n2 <= n1 or n1 <= 0:
            return float('inf')
        doubling_time = time_difference * np.log(2) / np.log(n2/n1)
        return doubling_time / 60 

    def parse_cell_file(self, filename: str) -> Dict[str, str]:
        """Parse cell filename to extract experiment info"""
        return FileParser.parse_simulation_file(filename, "CELLS")

    def load_cells_data(
        self, folder_path: Path, timestamp: str
    ) -> List[Tuple[Dict[str, str], List[Dict[str, Any]]]]:
        """Load cell data from a specific timestamp for all seeds

        Args:
            folder_path: Path to simulation folder
            timestamp: Timestamp string (e.g., '000000' or '000720')

        Returns:
            List of tuples (file_info, cell_data)
        """
        results = []
        try:
            file_pattern = f"*_{timestamp}.CELLS.json"
            cell_files = folder_path.glob(file_pattern)

            for cell_file in cell_files:
                file_info = self.parse_cell_file(cell_file.name)
                if file_info:
                    with open(cell_file, "r") as f:
                        cell_data = json.load(f)
                        results.append((file_info, cell_data))

        except Exception as e:
            self.logger.error(f"Error loading cells data from {folder_path}: {str(e)}")

        return results 