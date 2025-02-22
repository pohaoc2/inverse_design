from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
import re
import numpy as np
from inverse_design.analyze.cell_metrics import CellMetrics
from inverse_design.analyze.spatial_metrics import SpatialMetrics
from inverse_design.analyze.analyze_seed_results import SeedAnalyzer
from inverse_design.analyze.analyze_aggregated_results import collect_parameter_data


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
        self.seed_analyzer = SeedAnalyzer(base_output_dir)
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
        self, folder_path: Path, t1: str, t2: str, timestamps: List[str]
    ) -> Dict[str, float]:
        """Analyze a single simulation folder with multiple seeds"""
        cells_data_t1 = self.cell_metrics.load_cells_data(folder_path, t1)
        cells_data_t2 = self.cell_metrics.load_cells_data(folder_path, t2)
        time_difference = int(t2) - int(t1)
        fit_data = self.spatial_metrics.fit_colony_growth(folder_path, timestamps)

        metrics_by_exp = self.seed_analyzer.process_parameter_seeds(cells_data_t1, cells_data_t2, time_difference)

        averaged_metrics = {}

        for (exp_group, exp_name), metrics in metrics_by_exp.items():
            colony_metrics = fit_data.get((exp_group, exp_name), {})

            avg_cells_t1 = np.median(metrics["n_cells_t1"])
            avg_cells_t2 = np.median(metrics["n_cells_t2"])

            metrics_dict = {
                # "exp_group": exp_group,
                # "exp_name": exp_name,
                # "growth_rate": np.median(metrics["growth_rates"]),
                # "growth_rate_std": np.std(metrics["growth_rates"]),
                "doub_time": np.median(metrics["doub_time"]),
                "doub_time_std": np.std(metrics["doub_time"]),
                # "avg_volume_t1": np.median(metrics["avg_volumes_t1"]),
                # "avg_volume_t1_std": np.std(metrics["avg_volumes_t1"]),
                "vol_t2": np.median(metrics["avg_volumes_t2"]),
                "vol_t2_std": np.std(metrics["avg_volumes_t2"]),
                # "n_cells_t1": avg_cells_t1,
                # "n_cells_t1_std": np.std(metrics["num_cells_t1"]),
                "n_cells_t2": avg_cells_t2,
                "n_cells_t2_std": np.std(metrics["n_cells_t2"]),
                # "activity_t1": np.median(metrics["act_t1"]),
                # "activity_t1_std": np.std(metrics["act_t1"]),
                "act_t2": np.median(metrics["act_t2"]),
                "act_t2_std": np.std(metrics["act_t2"]),
                # "seed_count": metrics["seed_count"],
                "colony_g_rate": colony_metrics.get("slope", 0.0),
                "colony_g_rate_std": colony_metrics.get("slope_std", 0.0),
                # "initial_colony_diameter": colony_metrics.get("intercept", 0.0),
                # "initial_colony_diameter_std": colony_metrics.get("intercept_std", 0.0),
                "colony_g_r_squared": colony_metrics.get("r_squared", 0.0),
                # "states_t1": metrics["states_t1"],
                "states_t2": metrics["states_t2"],
            }

            averaged_metrics.update(self._round_metrics(metrics_dict))

        return averaged_metrics

    def analyze_all_simulations(
        self, timestamps: List[str], t1: str = "000000", t2: str = "000720"
    ) -> pd.DataFrame:
        """Analyze all simulation folders and compile results"""
        results = []

        sim_folders = sorted(
            [f for f in self.base_output_dir.glob("input_*")],
            key=lambda x: int(re.search(r"input_(\d+)", x.name).group(1)),
        )
        for folder in sim_folders:
            try:
                metrics = self.analyze_simulation(folder, t1, t2, timestamps)
                metrics["input_folder"] = folder.name
                results.append(metrics)
            except Exception as e:
                self.logger.error(f"Error processing {folder}: {str(e)}")
        df = pd.DataFrame(results)
        # Reorder columns to move input_folder to second-to-last position
        cols = df.columns.tolist()
        cols.remove("input_folder")
        cols.remove("states_t2")
        cols.append("input_folder")
        cols.append("states_t2")
        df = df[cols]

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
                    fit_results.append(
                        {
                            "input_folder": folder.name,
                            "exp_group": exp_group,
                            "exp_name": exp_name,
                            "growth_rate": fit_metrics["slope"],
                            "growth_rate_std": fit_metrics["slope_std"],
                            "initial_diameter": fit_metrics["intercept"],
                            "initial_diameter_std": fit_metrics["intercept_std"],
                            "r_squared": fit_metrics["r_squared"],
                        }
                    )

                # Store growth data (existing code)
                for (exp_group, exp_name), data in diameter_data.items():
                    for i, timestamp in enumerate(data["timestamps"]):
                        growth_results.append(
                            {
                                "input_folder": folder.name,
                                "exp_group": exp_group,
                                "exp_name": exp_name,
                                "timestamp": timestamp,
                                "mean_diameter": data["mean_diameter"][i],
                                "std_diameter": data["std_diameter"][i],
                            }
                        )

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
        random_seed: int = None,
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
                random_seed=random_seed,
            )
            self.logger.info(f"Saved colony growth comparison plot to {output_file}")
        except Exception as e:
            self.logger.error(f"Error creating comparison plot: {str(e)}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parameter_base_folder = "ARCADE_OUTPUT/MANUAL_VOLUME_APOTOSIS/"
    metrics_calculator = SimulationMetrics(parameter_base_folder)

    timestamps = [
        "000000",
        "000720",
        "001440",
        "002160",
        "002880",
        "003600",
        "004320",
        "005040",
        "005760",
        "006480",
        "007200",
        "007920",
        "008640",
        "009360",
        "010080",
    ]
    metrics_calculator.analyze_all_simulations(timestamps=timestamps, t1="000000", t2="010080")

    input_files = list(Path(parameter_base_folder).glob("input_*"))
    input_files = [f.name for f in input_files]
    parameter_list = [
        "CELL_VOLUME_MU",
        "CELL_VOLUME_SIGMA",
        "NECROTIC_FRACTION",
        "APOPTOSIS_AGE_SIGMA",
        "ACCURACY",
        "AFFINITY",
        "COMPRESSION_TOLERANCE",
    ]
    all_param_df = collect_parameter_data(input_files, parameter_base_folder, parameter_list)
    all_param_df.to_csv(f"{parameter_base_folder}/all_param_df.csv", index=False)

    if 0:
        metrics_calculator.visualize_colony_growth(
            n_inputs=10, timestamps=timestamps, random_seed=42
        )


if __name__ == "__main__":
    main()
