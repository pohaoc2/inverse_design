from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
import re
import numpy as np


class SimulationMetrics:
    def __init__(self, base_output_dir: str):
        """Initialize the metrics calculator

        Args:
            base_output_dir: Base directory containing all simulation output folders
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = logging.getLogger(__name__)

    def parse_filename(self, filename: str) -> Dict[str, str]:
        """Parse filename to extract experiment info

        Args:
            filename: Format <exp_group>_<exp_name>_<seed>_<timestamp>.CELLS.json

        Returns:
            Dictionary containing parsed components
        """
        pattern = r"(.+?)_(.+?)_(\d+)_(\d+)\.CELLS\.json"
        match = re.match(pattern, filename)
        if match:
            return {
                "exp_group": match.group(1),
                "exp_name": match.group(2),
                "seed": match.group(3),
                "timestamp": match.group(4),
            }
        return {}

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
            # Find all files matching the timestamp
            file_pattern = f"*_{timestamp}.CELLS.json"
            cell_files = folder_path.glob(file_pattern)

            for cell_file in cell_files:
                file_info = self.parse_filename(cell_file.name)
                if file_info:  # Only process if filename matches expected pattern
                    with open(cell_file, "r") as f:
                        cell_data = json.load(f)
                        results.append((file_info, cell_data))

        except Exception as e:
            self.logger.error(f"Error loading cells data from {folder_path}: {str(e)}")

        return results

    def calculate_growth_rate(
        self, cells_t1: List[Dict[str, Any]], cells_t2: List[Dict[str, Any]], time_difference: float
    ) -> float:
        """Calculate growth rate between two timepoints"""
        if not cells_t1 or not cells_t2:
            return 0.0

        n1 = len(cells_t1)
        n2 = len(cells_t2)
        return (n2 - n1) / time_difference

    def calculate_average_volume(self, cells: List[Dict[str, Any]]) -> float:
        """Calculate average cell volume"""
        if not cells:
            return 0.0

        volumes = [cell["volume"] for cell in cells]
        return sum(volumes) / len(volumes)

    def calculate_activity(self, cells: List[Dict[str, Any]]) -> float:
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

    def calculate_doubling_time(self, n1: float, n2: float, time_difference: float) -> float:
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

    def analyze_simulation(
        self, folder_path: Path, t1: str, t2: str, time_difference: float
    ) -> Dict[str, float]:
        """Analyze a single simulation folder with multiple seeds"""
        # Load data for all seeds at both timepoints
        cells_data_t1 = self.load_cells_data(folder_path, t1)
        cells_data_t2 = self.load_cells_data(folder_path, t2)

        # Group metrics by experiment group and name
        metrics_by_exp = {}

        # Process each seed's data
        for (info_t1, cells_t1), (info_t2, cells_t2) in zip(cells_data_t1, cells_data_t2):
            if (
                info_t1["exp_group"] != info_t2["exp_group"]
                or info_t1["exp_name"] != info_t2["exp_name"]
            ):
                continue

            exp_key = (info_t1["exp_group"], info_t1["exp_name"])
            if exp_key not in metrics_by_exp:
                metrics_by_exp[exp_key] = {
                    "growth_rates": [],
                    "avg_volumes_t1": [],
                    "avg_volumes_t2": [],
                    "num_cells_t1": [],
                    "num_cells_t2": [],
                    "activity_t1": [],
                    "activity_t2": [],
                    "seed_count": 0,
                    "doubling_time": [],
                }

            # Calculate metrics for this seed
            metrics_by_exp[exp_key]["growth_rates"].append(
                self.calculate_growth_rate(cells_t1, cells_t2, time_difference)
            )
            metrics_by_exp[exp_key]["avg_volumes_t1"].append(
                self.calculate_average_volume(cells_t1)
            )
            metrics_by_exp[exp_key]["avg_volumes_t2"].append(
                self.calculate_average_volume(cells_t2)
            )
            metrics_by_exp[exp_key]["num_cells_t1"].append(len(cells_t1))
            metrics_by_exp[exp_key]["num_cells_t2"].append(len(cells_t2))
            metrics_by_exp[exp_key]["activity_t1"].append(self.calculate_activity(cells_t1))
            metrics_by_exp[exp_key]["activity_t2"].append(self.calculate_activity(cells_t2))
            metrics_by_exp[exp_key]["seed_count"] += 1
            metrics_by_exp[exp_key]["doubling_time"].append(self.calculate_doubling_time(len(cells_t1), len(cells_t2), time_difference))
            if metrics_by_exp[exp_key]["seed_count"] == 20:
                break
        # Average metrics across seeds
        averaged_metrics = {}
        def filter_valid_metrics(metrics_list):
            """Filter out seeds with NaN or inf values from a list of metrics"""
            valid_indices = []
            for i in range(len(metrics_list[0])):
                has_invalid = any(
                    np.isnan(metric[i]) or np.isinf(metric[i])
                    for metric in metrics_list
                )
                if not has_invalid:
                    valid_indices.append(i)
            return valid_indices, [
                [metric[i] for i in valid_indices]
                for metric in metrics_list
            ]

        for (exp_group, exp_name), metrics in metrics_by_exp.items():
            # Filter out seeds with NaN or inf values
            metrics_to_check = [
                metrics["growth_rates"],
                metrics["avg_volumes_t1"],
                metrics["avg_volumes_t2"],
                metrics["activity_t1"],
                metrics["activity_t2"],
                metrics["doubling_time"]
            ]
            
            valid_indices, filtered_metrics = filter_valid_metrics(metrics_to_check)
            
            # Update metrics with filtered values
            metrics["growth_rates"] = filtered_metrics[0]
            metrics["avg_volumes_t1"] = filtered_metrics[1]
            metrics["avg_volumes_t2"] = filtered_metrics[2]
            metrics["num_cells_t1"] = [metrics["num_cells_t1"][i] for i in valid_indices]
            metrics["num_cells_t2"] = [metrics["num_cells_t2"][i] for i in valid_indices]
            metrics["activity_t1"] = filtered_metrics[3]
            metrics["activity_t2"] = filtered_metrics[4]
            metrics["doubling_time"] = filtered_metrics[5]
            metrics["seed_count"] = len(valid_indices)
            print("="*15)
            
            avg_cells_t1 = sum(metrics["num_cells_t1"]) / len(metrics["num_cells_t1"])
            avg_cells_t2 = sum(metrics["num_cells_t2"]) / len(metrics["num_cells_t2"])
            averaged_metrics.update(
                {
                    "exp_group": exp_group,
                    "exp_name": exp_name,
                    "growth_rate": sum(metrics["growth_rates"]) / len(metrics["growth_rates"]),
                    "growth_rate_std": np.std(metrics["growth_rates"]),
                    "doubling_time": sum(metrics["doubling_time"]) / len(metrics["doubling_time"]),
                    "doubling_time_std": np.std(metrics["doubling_time"]),
                    "avg_volume_t1": sum(metrics["avg_volumes_t1"]) / len(metrics["avg_volumes_t1"]),
                    "avg_volume_t1_std": np.std(metrics["avg_volumes_t1"]),
                    "avg_volume_t2": sum(metrics["avg_volumes_t2"]) / len(metrics["avg_volumes_t2"]),
                    "avg_volume_t2_std": np.std(metrics["avg_volumes_t2"]),
                    "num_cells_t1": sum(metrics["num_cells_t1"]) / len(metrics["num_cells_t1"]),
                    "num_cells_t2": sum(metrics["num_cells_t2"]) / len(metrics["num_cells_t2"]),
                    "activity_t1": sum(metrics["activity_t1"]) / len(metrics["activity_t1"]),
                    "activity_t1_std": np.std(metrics["activity_t1"]),
                    "activity_t2": sum(metrics["activity_t2"]) / len(metrics["activity_t2"]),
                    "activity_t2_std": np.std(metrics["activity_t2"]),
                    "seed_count": metrics["seed_count"],
                }
            )

        return averaged_metrics

    def analyze_all_simulations(
        self, t1: str = "000000", t2: str = "000720", time_difference: float = 720.0
    ) -> pd.DataFrame:
        """Analyze all simulation folders and compile results"""
        results = []

        # Find all simulation folders
        sim_folders = [f for f in self.base_output_dir.glob("in_*")]
        for folder in sim_folders:
            try:
                metrics = self.analyze_simulation(folder, t1, t2, time_difference)
                metrics["simulation"] = folder.name
                results.append(metrics)
            except Exception as e:
                self.logger.error(f"Error processing {folder}: {str(e)}")

        # Create DataFrame and save to CSV
        df = pd.DataFrame(results)
        output_file = self.base_output_dir / "simulation_metrics.csv"
        df.to_csv(output_file, index=False)

        self.logger.info(f"Saved metrics for {len(results)} simulations to {output_file}")
        return df


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Example usage
    metrics_calculator = SimulationMetrics("ARCADE_OUTPUT/")
    #metrics_calculator = SimulationMetrics("~/ARCADE/outputs/")
    df = metrics_calculator.analyze_all_simulations(t1="000000", t2="010080", time_difference=10080)

    # Print summary statistics
    print("\nSummary Statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()
