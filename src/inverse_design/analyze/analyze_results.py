from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
import re
import numpy as np
from inverse_design.analyze.cell_metrics import CellMetrics
from inverse_design.analyze.spatial_metrics import SpatialMetrics


class SimulationMetrics:
    def __init__(self, base_output_dir: str):
        """Initialize the metrics calculator

        Args:
            base_output_dir: Base directory containing all simulation output folders
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = logging.getLogger(__name__)
        self.cell_metrics = CellMetrics()
        self.spatial_metrics = SpatialMetrics()

    def _round_metrics(self, metrics_dict: Dict[str, Any], decimals: int = 3) -> Dict[str, Any]:
        """Round all numeric values in a dictionary to specified decimals.
        
        Args:
            metrics_dict: Dictionary containing metrics
            decimals: Number of decimal places to round to
            
        Returns:
            Dictionary with rounded numeric values
        """
        rounded_dict = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (float, np.floating)):
                rounded_dict[key] = round(value, decimals)
            else:
                rounded_dict[key] = value
        return rounded_dict

    def analyze_simulation(
        self, folder_path: Path, t1: str, t2: str
    ) -> Dict[str, float]:
        """Analyze a single simulation folder with multiple seeds"""
        cells_data_t1 = self.cell_metrics.load_cells_data(folder_path, t1)
        cells_data_t2 = self.cell_metrics.load_cells_data(folder_path, t2)
        time_difference = int(t2) - int(t1)
        timestamps = ["000000", "000720", "001440", "002160", "002880", "003600", "004320",
                     "005040", "005760", "006480", "007200", "007920", "008640", "009360", "010080"]
        fit_data = self.spatial_metrics.fit_colony_growth(folder_path, timestamps)

        metrics_by_exp = {}

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

            metrics_by_exp[exp_key]["growth_rates"].append(
                self.cell_metrics.calculate_growth_rate(cells_t1, cells_t2, time_difference)
            )
            metrics_by_exp[exp_key]["avg_volumes_t1"].append(
                self.cell_metrics.calculate_average_volume(cells_t1)
            )
            metrics_by_exp[exp_key]["avg_volumes_t2"].append(
                self.cell_metrics.calculate_average_volume(cells_t2)
            )
            metrics_by_exp[exp_key]["num_cells_t1"].append(len(cells_t1))
            metrics_by_exp[exp_key]["num_cells_t2"].append(len(cells_t2))
            metrics_by_exp[exp_key]["activity_t1"].append(self.cell_metrics.calculate_activity(cells_t1))
            metrics_by_exp[exp_key]["activity_t2"].append(self.cell_metrics.calculate_activity(cells_t2))
            metrics_by_exp[exp_key]["seed_count"] += 1
            metrics_by_exp[exp_key]["doubling_time"].append(self.cell_metrics.calculate_doubling_time(len(cells_t1), len(cells_t2), time_difference))

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
            metrics_to_check = [
                metrics["growth_rates"],
                metrics["avg_volumes_t1"],
                metrics["avg_volumes_t2"],
                metrics["activity_t1"],
                metrics["activity_t2"],
                metrics["doubling_time"]
            ]
            
            valid_indices, filtered_metrics = filter_valid_metrics(metrics_to_check)
            
            metrics["growth_rates"] = filtered_metrics[0]
            metrics["avg_volumes_t1"] = filtered_metrics[1]
            metrics["avg_volumes_t2"] = filtered_metrics[2]
            metrics["num_cells_t1"] = [metrics["num_cells_t1"][i] for i in valid_indices]
            metrics["num_cells_t2"] = [metrics["num_cells_t2"][i] for i in valid_indices]
            metrics["activity_t1"] = filtered_metrics[3]
            metrics["activity_t2"] = filtered_metrics[4]
            metrics["doubling_time"] = filtered_metrics[5]
            metrics["seed_count"] = len(valid_indices)
            
            colony_metrics = fit_data.get((exp_group, exp_name), {})
            
            avg_cells_t1 = sum(metrics["num_cells_t1"]) / len(metrics["num_cells_t1"])
            avg_cells_t2 = sum(metrics["num_cells_t2"]) / len(metrics["num_cells_t2"])
            
            metrics_dict = {
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
                "num_cells_t1": avg_cells_t1,
                "num_cells_t2": avg_cells_t2,
                "activity_t1": sum(metrics["activity_t1"]) / len(metrics["activity_t1"]),
                "activity_t1_std": np.std(metrics["activity_t1"]),
                "activity_t2": sum(metrics["activity_t2"]) / len(metrics["activity_t2"]),
                "activity_t2_std": np.std(metrics["activity_t2"]),
                "seed_count": metrics["seed_count"],
                "colony_growth_rate": colony_metrics.get("slope", 0.0),
                "colony_growth_rate_std": colony_metrics.get("slope_std", 0.0),
                "initial_colony_diameter": colony_metrics.get("intercept", 0.0),
                "initial_colony_diameter_std": colony_metrics.get("intercept_std", 0.0),
                "colony_growth_r_squared": colony_metrics.get("r_squared", 0.0)
            }
            
            averaged_metrics.update(self._round_metrics(metrics_dict))

        return averaged_metrics

    def analyze_all_simulations(
        self, t1: str = "000000", t2: str = "000720"
    ) -> pd.DataFrame:
        """Analyze all simulation folders and compile results"""
        results = []

        sim_folders = [f for f in self.base_output_dir.glob("input_*")]
        for folder in sim_folders[:20]:
            try:
                metrics = self.analyze_simulation(folder, t1, t2)
                metrics["simulation"] = folder.name
                results.append(metrics)
            except Exception as e:
                self.logger.error(f"Error processing {folder}: {str(e)}")

        df = pd.DataFrame(results)
        output_file = self.base_output_dir / "simulation_metrics.csv"
        df.to_csv(output_file, index=False)

        self.logger.info(f"Saved metrics for {len(results)} simulations to {output_file}")
        return df

    def analyze_colony_growth(
        self, timestamps: List[str] = ["000000", "000720", "001440"]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze colony diameter growth over time for all simulations

        Returns:
            Tuple of (growth_metrics_df, fit_metrics_df)
        """
        growth_results = []
        fit_results = []
        sim_folders = [f for f in self.base_output_dir.glob("input_*")]

        for folder in sim_folders:
            try:
                # Get growth data
                diameter_data = self.spatial_metrics.calculate_colony_diameter_over_time(
                    folder, timestamps
                )
                
                # Get fit data
                fit_data = self.spatial_metrics.fit_colony_growth(folder, timestamps)
                
                # Store fit results
                for (exp_group, exp_name), fit_metrics in fit_data.items():
                    fit_results.append({
                        "simulation": folder.name,
                        "exp_group": exp_group,
                        "exp_name": exp_name,
                        "growth_rate": fit_metrics["slope"],
                        "growth_rate_std": fit_metrics["slope_std"],
                        "initial_diameter": fit_metrics["intercept"],
                        "initial_diameter_std": fit_metrics["intercept_std"],
                        "r_squared": fit_metrics["r_squared"]
                    })
                
                # Store growth data (existing code)
                for (exp_group, exp_name), data in diameter_data.items():
                    for i, timestamp in enumerate(data["timestamps"]):
                        growth_results.append({
                            "simulation": folder.name,
                            "exp_group": exp_group,
                            "exp_name": exp_name,
                            "timestamp": timestamp,
                            "mean_diameter": data["mean_diameter"][i],
                            "std_diameter": data["std_diameter"][i]
                        })

            except Exception as e:
                self.logger.error(f"Error processing {folder}: {str(e)}")

        # Create DataFrames and save to CSV
        growth_df = pd.DataFrame(growth_results)
        fit_df = pd.DataFrame(fit_results)
        
        growth_df.to_csv(self.base_output_dir / "colony_growth_metrics.csv", index=False)
        fit_df.to_csv(self.base_output_dir / "colony_growth_fit_metrics.csv", index=False)

        self.logger.info("Saved colony growth and fit metrics")
        return growth_df, fit_df

    def visualize_colony_growth(
        self,
        n_inputs: int = 5,
        timestamps: List[str] = ["000000", "002520", "005040", "007560", "010080"],
        random_seed: int = None
    ) -> None:
        """Create visualization of colony growth for randomly selected simulations

        Args:
            n_inputs: Number of input folders to randomly select and plot
            timestamps: List of timestamps to analyze
            random_seed: Random seed for reproducibility
        """
        try:
            output_file = self.base_output_dir / "colony_growth_comparison.png"
            output_file = "colony_growth_comparison.png"
            self.spatial_metrics.plot_multiple_colony_growth(
                self.base_output_dir,
                n_inputs,
                timestamps,
                output_file=output_file,
                random_seed=random_seed
            )
            self.logger.info(f"Saved colony growth comparison plot to {output_file}")
        except Exception as e:
            self.logger.error(f"Error creating comparison plot: {str(e)}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    metrics_calculator = SimulationMetrics("ARCADE_OUTPUT_completed/")
    
    timestamps = ["000000", "000720", "001440", "002160", "002880", "003600", "004320",
                 "005040", "005760", "006480", "007200", "007920", "008640", "009360", "010080"]
    metrics_calculator.analyze_all_simulations(t1="000000", t2="010080")

    if 0:
        metrics_calculator.visualize_colony_growth(
            n_inputs=10,
            timestamps=timestamps,
            random_seed=42
        )


if __name__ == "__main__":
    main()
