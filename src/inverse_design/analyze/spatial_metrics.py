from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import re
import logging
import numpy as np
from .file_utils import FileParser
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy import stats


class SpatialMetrics:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def calculate_colony_diameter(locations: List[Dict[str, Any]], C: float = 30.0) -> float:
        """Calculate the colony diameter based on coordinates from .LOCATIONS.json

        Args:
            locations: List of dictionaries containing location data
            C: Scaling factor (default = 30)

        Returns:
            Colony diameter
        """
        if not locations:
            return 0.0

        u_values = [loc["coordinate"][0] for loc in locations]
        v_values = [loc["coordinate"][1] for loc in locations]
        w_values = [loc["coordinate"][2] for loc in locations]

        umax, umin = max(u_values), min(u_values)
        vmax, vmin = max(v_values), min(v_values)
        wmax, wmin = max(w_values), min(w_values)

        diameter = (
            C * (max(umax - umin + 1, 0) + max(vmax - vmin + 1, 0) + max(wmax - wmin + 1, 0)) / 3
        )
        return diameter

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

    def calculate_colony_diameter_over_time(
        self, folder_path: Path, timestamps: List[str], C: float = 30.0
    ) -> Dict[str, Dict[str, float]]:
        """Calculate colony diameter for multiple timepoints

        Args:
            folder_path: Path to simulation folder
            timestamps: List of timestamp strings (e.g., ['000000', '000720', ...])
            C: Scaling factor (default = 30)

        Returns:
            Dictionary with structure:
            {
                "exp_group_name": {
                    "mean_diameter": List[float],  # Average diameter across seeds
                    "std_diameter": List[float],   # Standard deviation across seeds
                    "timestamps": List[str],       # Corresponding timestamps
                }
            }
        """
        results = {}

        for timestamp in timestamps:
            locations_data = self.load_locations_data(folder_path, timestamp)

            exp_diameters = {}
            for file_info, locations in locations_data:
                exp_key = (file_info["exp_group"], file_info["exp_name"])

                if exp_key not in exp_diameters:
                    exp_diameters[exp_key] = {"diameters": [], "timestamp": timestamp}

                diameter = self.calculate_colony_diameter(locations, C)
                exp_diameters[exp_key]["diameters"].append(diameter)

            for exp_key, data in exp_diameters.items():
                if exp_key not in results:
                    results[exp_key] = {"mean_diameter": [], "std_diameter": [], "timestamps": []}

                diameters = data["diameters"]
                results[exp_key]["mean_diameter"].append(np.mean(diameters))
                results[exp_key]["std_diameter"].append(np.std(diameters))
                results[exp_key]["timestamps"].append(timestamp)

        return results

    def fit_colony_growth(
        self,
        folder_path: Path,
        timestamps: List[str],
        C: float = 30.0,
        time_conversion: float = 1 / 60,
    ) -> Dict[str, Dict[str, float]]:
        """Fit colony diameter growth to linear function (y = mx + c)

        Args:
            folder_path: Path to simulation folder
            timestamps: List of timestamp strings
            C: Scaling factor for diameter calculation
            time_conversion: Factor to convert timestamp to hours

        Returns:
            Dictionary with structure:
            {
                (exp_group, exp_name): {
                    "slope": float,  # Growth rate (m)
                    "intercept": float,  # Initial diameter (c)
                    "r_squared": float,  # R-squared value
                    "slope_std": float,  # Standard error of slope
                    "intercept_std": float  # Standard error of intercept
                }
            }
        """
        results = {}

        exp_data = {}  # {(exp_group, exp_name): {seed: [(time, diameter), ...], ...}}

        for timestamp in timestamps:
            locations_data = self.load_locations_data(folder_path, timestamp)
            time_hr = float(timestamp) * time_conversion

            for file_info, locations in locations_data:
                exp_key = (file_info["exp_group"], file_info["exp_name"])
                seed = file_info["seed"]

                if exp_key not in exp_data:
                    exp_data[exp_key] = {}
                if seed not in exp_data[exp_key]:
                    exp_data[exp_key][seed] = []

                diameter = self.calculate_colony_diameter(locations, C)
                exp_data[exp_key][seed].append((time_hr, diameter))

        for exp_key, seed_data in exp_data.items():
            all_times = []
            all_diameters = []

            for seed, points in seed_data.items():
                times, diameters = zip(*sorted(points))
                all_times.extend(times)
                all_diameters.extend(diameters)

            slope, intercept, r_value, p_value, slope_std = stats.linregress(
                all_times, all_diameters
            )

            results[exp_key] = {
                "slope": slope,  # Growth rate (diameter units per hour)
                "intercept": intercept,  # Initial diameter
                "r_squared": r_value**2,
                "slope_std": slope_std,
                "intercept_std": stats.sem(
                    [d - (slope * t + intercept) for t, d in zip(all_times, all_diameters)]
                ),
            }

        return results

    def plot_multiple_colony_growth(
        self,
        base_folder_path: Path,
        n_inputs: int,
        timestamps: List[str],
        output_file: Path = None,
        C: float = 30.0,
        time_conversion: float = 1 / 60,
        random_seed: int = None,
        show_fit: bool = True,
    ) -> None:
        """Plot colony diameter growth over time for multiple simulations in one figure"""
        if random_seed is not None:
            np.random.seed(random_seed)

        input_folders = list(base_folder_path.glob("input_*"))
        if len(input_folders) < n_inputs:
            n_inputs = len(input_folders)
            self.logger.warning(f"Only {n_inputs} input folders available")

        selected_folders = np.random.choice(input_folders, size=n_inputs, replace=False)

        plt.figure(figsize=(12, 8))
        colors = list(TABLEAU_COLORS.values())

        for i, folder in enumerate(selected_folders):
            color = colors[i % len(colors)]

            seed_data = {}  # {seed: [(time, diameter), ...]}

            for timestamp in timestamps:
                locations_data = self.load_locations_data(folder, timestamp)
                time_hr = float(timestamp) * time_conversion

                for file_info, locations in locations_data:
                    seed = file_info["seed"]
                    if seed not in seed_data:
                        seed_data[seed] = []

                    diameter = self.calculate_colony_diameter(locations, C)
                    seed_data[seed].append((time_hr, diameter))

            for seed, points in seed_data.items():
                times, diameters = zip(*sorted(points))
                if seed == list(seed_data.keys())[0]:  # First seed gets the label
                    plt.plot(times, diameters, color=color, alpha=0.3, label=f"Input {folder.name}")
                else:
                    plt.plot(times, diameters, color=color, alpha=0.3)

            all_times = sorted(set(t for points in seed_data.values() for t, _ in points))
            mean_diameters = []
            std_diameters = []

            for t in all_times:
                diameters = [d for s in seed_data.values() for t2, d in s if t2 == t]
                mean_diameters.append(np.mean(diameters))
                std_diameters.append(np.std(diameters))

            # plt.plot(all_times, mean_diameters, color=color, linewidth=2)

            if show_fit:
                fit_results = self.fit_colony_growth(folder, timestamps, C, time_conversion)
                for exp_key, fit_data in fit_results.items():
                    slope = fit_data["slope"]
                    intercept = fit_data["intercept"]
                    r_squared = fit_data["r_squared"]

                    x_fit = np.array([min(all_times), max(all_times)])
                    y_fit = slope * x_fit + intercept
                    plt.plot(x_fit, y_fit, "--", color=color, linewidth=1.5)

        plt.xlabel("Time (hours)")
        plt.ylabel("Colony Diameter")
        plt.title(f"Colony Growth Over Time ({n_inputs} Random Inputs)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()
