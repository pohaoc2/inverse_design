import hydra
import os
import json
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import numpy as np

from .abc import ABC
from .abc_with_model import ABCWithModel
from .config import BDMConfig, ABCConfig
from .metrics import Metric, Target
from . import evaluate
from .vis import plot_abc_results
from .utils import get_samples_data


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Example script demonstrating how to use the ABC inference"""

    # Load configurations
    bdm_config = BDMConfig.from_dictconfig(cfg.bdm)
    abc_config = ABCConfig.from_dictconfig(cfg.abc)

    # Define targets
    targets = [Target(Metric.DENSITY, 70.0), Target(Metric.TIME_TO_EQUILIBRIUM, 1400.0)]

    # Initialize ABC
    # abc = ABC(bdm_config, abc_config, targets)
    abc = ABCWithModel(bdm_config, abc_config, targets)
    try:
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"../results/abc_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Run inference
        param_metrics_distances_results = abc.run_inference()

        # Save all samples data
        all_samples_df = get_samples_data(
            param_metrics_distances_results=param_metrics_distances_results,
            model_type=abc_config.model_type,
            save_path=os.path.join(output_dir, "all_samples_metrics.csv"),
        )
        accepted_params = [
            {
                "proliferate": sample["proliferate"],
                "death": sample["death"],
                "migrate": sample["migratory"],
            }
            for sample in param_metrics_distances_results
            if sample["accepted"]
        ]
        # Get best parameters
        best_sample = min(param_metrics_distances_results, key=lambda x: x["distance"])

        best_params = {
            "proliferate": best_sample["proliferate"],
            "death": best_sample["death"],
            "migrate": best_sample["migratory"],
        }
        best_metrics = {
            "cell_density": best_sample["cell_density"],
            "time_to_eq": best_sample["time_to_eq"],
        }
        # Print best parameters
        for param_name, value in best_params.items():
            print(f"{param_name}: {value:.3f}")

        # Save configuration and results
        config_dict = {
            "timestamp": timestamp,
            "targets": [
                {"metric": t.metric.value, "value": t.value, "weight": t.weight} for t in targets
            ],
            "best_parameters": best_params,
            "best_metrics": best_metrics,
            "bdm_config": OmegaConf.to_container(cfg.bdm, resolve=True),
            "abc_config": OmegaConf.to_container(cfg.abc, resolve=True),
        }

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

        # Calculate PDFs and compare sampling methods
        parameter_pdfs = evaluate.estimate_pdfs(accepted_params)
        sampling_results = evaluate.compare_sampling_methods(abc, parameter_pdfs, num_samples=1)

        # Save sampling results
        with open(os.path.join(output_dir, "sampling_results.json"), "w") as f:
            # Convert any numpy values to float and Metric enum to string for JSON serialization
            json_results = {}

            # Add prior information from config
            json_results["prior_info"] = {
                param_name: {"min": float(ranges.min), "max": float(ranges.max)}
                for param_name, ranges in abc_config.parameter_ranges.items()
            }

            # Add posterior information
            json_results["posterior_info"] = {
                "independent": {
                    param_name: {
                        "mean": float(kde.resample(1000).mean()),
                        "std": float(kde.resample(1000).std()),
                    }
                    for param_name, kde in parameter_pdfs["independent"].items()
                },
                "joint": {
                    "mean": [float(x) for x in parameter_pdfs["joint_mean"]],
                    "covariance": [[float(x) for x in row] for row in parameter_pdfs["joint_cov"]],
                    "parameter_names": parameter_pdfs["param_names"],
                },
            }

            # Add sampling method results
            for method, results in sampling_results.items():
                json_results[method] = {}
                for k, v in results.items():
                    if k == "metrics":
                        # Handle metrics dictionary separately to convert Metric enum keys
                        json_results[method][k] = {
                            metric.value if hasattr(metric, "value") else str(metric): float(value)
                            for metric, value in v.items()
                        }
                    else:
                        # Handle other values
                        json_results[method][k] = (
                            float(v) if isinstance(v, (np.float32, np.float64)) else v
                        )

            json.dump(json_results, f, indent=4)

        print("\nSampling Method Comparison:")
        for method, results in sampling_results.items():
            print(f"\n{method}:")
            if method == "independent_means":
                print(f"  Distance: {results['distance']:.4f}")
                print("  Metrics:")
                for metric_name, value in results["metrics"].items():
                    print(f"    {metric_name.value}: {value:.4f}")
            else:  # joint_sampling
                print(f"  Mean Distance: {results['distance_mean']:.4f}")
                print(f"  Distance Std: {results['distance_std']:.4f}")
                print("  Mean Metrics:")
                for metric_name, value in results["metrics"].items():
                    print(f"    {metric_name.value}: {value:.4f}")

        # Plot and save results
        all_metrics = [sample["cell_density"] for sample in param_metrics_distances_results]
        all_metrics += [sample["time_to_eq"] for sample in param_metrics_distances_results]
        accepted_metrics = [
            sample["cell_density"]
            for sample in param_metrics_distances_results
            if sample["accepted"]
        ]
        accepted_metrics += [
            sample["time_to_eq"] for sample in param_metrics_distances_results if sample["accepted"]
        ]

        plot_abc_results(
            accepted_params=accepted_params,
            pdf_results=parameter_pdfs,
            all_metrics=all_metrics,
            accepted_metrics=accepted_metrics,
            targets=targets,
            abc_config=abc_config,
            model_config=bdm_config,
            save_dir=output_dir,
        )

        print(f"\nResults saved to: {output_dir}")

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
