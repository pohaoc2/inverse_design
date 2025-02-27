import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from inverse_design.analyze.cell_metrics import CellMetrics
from inverse_design.analyze.population_metrics import PopulationMetrics
from inverse_design.analyze.metrics_config import CELLULAR_METRICS, POPULATION_METRICS

class TemporalAnalyzer:
    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir)
        self.input_folder = Path(base_output_dir + "/inputs")
        self.cell_metrics = CellMetrics()
        self.population_metrics = PopulationMetrics()

    def calculate_temporal_metrics(
        self, 
        folder_path: Path, 
        timestamps: List[str]
    ) -> Dict[str, Dict[str, List[float]]]:
        """Calculate metrics over time for a single simulation.

        Args:
            folder_path: Path to simulation folder
            timestamps: List of timestamps to analyze

        Returns:
            Dictionary containing cellular and population metrics over time
        """
        temporal_data = {
            "cellular": {
                "volumes": [],
                "activity": [],
                "states": [],
                "age": []
            },
            "population": {
                "n_cells": [],
                "symmetry": [],
                "shannon": [],
                "colony_diameter": []
            },
            "timestamps": timestamps
        }

        # Load data for each timestamp
        for t in timestamps:
            cells_data = self.cell_metrics.load_cells_data(folder_path, t)
            
            # We expect only one experiment group/name per folder
            info, cells = cells_data[0]  
            
            # Calculate cellular metrics
            temporal_data["cellular"]["volumes"].append(
                self.cell_metrics.calculate_average_volume(cells)
            )
            temporal_data["cellular"]["activity"].append(
                self.cell_metrics.calculate_activity(cells)
            )
            temporal_data["cellular"]["states"].append(
                self.cell_metrics.calculate_cell_states(cells)
            )
            temporal_data["cellular"]["age"].append(
                self.cell_metrics.calculate_average_age(cells)
            )

            # Calculate population metrics
            temporal_data["population"]["n_cells"].append(len(cells))
            temporal_data["population"]["symmetry"].append(
                self.population_metrics.calculate_symmetry(cells)
            )
            temporal_data["population"]["shannon"].append(
                self.population_metrics.calculate_shannon_diversity(cells)
            )
            temporal_data["population"]["colony_diameter"].append(
                self.population_metrics.calculate_colony_diameter(cells)
            )

        return temporal_data

    def process_folder_set(
        self, 
        folders: List[str], 
        timestamps: List[str]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Process multiple folders and analyze their temporal metrics."""
        all_folder_metrics = {}

        for folder_name in folders:
            folder_path = self.input_folder / folder_name
            
            # Calculate temporal metrics
            temporal_data = self.calculate_temporal_metrics(folder_path, timestamps)
            
            # Analyze trends
            trends = self.analyze_trends(temporal_data)
            
            all_folder_metrics[folder_name] = {
                "temporal_data": temporal_data,
                "trends": trends
            }

        return all_folder_metrics 