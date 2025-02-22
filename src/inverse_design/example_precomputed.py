import hydra
import os
import json
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import numpy as np
import pandas as pd
from inverse_design.abc.abc_precomputed import ABCPrecomputed
from inverse_design.conf.config import ABCConfig
from inverse_design.models.model_base import ModelRegistry
from inverse_design.common.enum import Target, Metric
from inverse_design.vis.vis import plot_parameter_kde, plot_joint_distribution, plot_pca_visualization
from inverse_design.analyze import evaluate
from inverse_design.utils.create_input_files import generate_parameters_from_kde
import matplotlib.pyplot as plt
from scipy import stats


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_abc_precomputed(cfg: DictConfig):
    """Example script demonstrating how to use the ABC inference with precomputed results

    The precomputed results file should contain:
    - All parameter variations
    - All calculated metrics for each parameter set
    """
    # Get model implementation
    model = ModelRegistry.get_model(cfg.abc.model_type)
    abc_config = ABCConfig.from_dictconfig(cfg.abc)

    # Get model config using the appropriate config class
    config_class = model.get_config_class()
    model_config_key = cfg.abc.model_type.lower()
    if not hasattr(cfg, model_config_key):
        raise ValueError(f"Configuration for model type {cfg.abc.model_type} not found in config")

    # Convert to model-specific config class
    model_config = config_class.from_dictconfig(cfg[model_config_key])

    # Get targets from config or use model defaults
    if hasattr(cfg.abc, "targets") and cfg.abc.targets:
        targets = [
            Target(
                metric=Metric(target.metric), value=float(target.value), weight=float(target.weight)
            )
            for target in cfg.abc.targets
        ]
    else:
        # Use model defaults if no targets in config
        targets = model.get_default_targets()
    param_file = "inputs/STEM_CELL/stem_cell_vary_volume_prior/parameter_log.csv"
    metrics_file = "ARCADE_OUTPUT/STEM_CELL/STEM_CELL_VARY_VOLUME/simulation_metrics.csv"
    output_dir = "inputs/STEM_CELL/stem_cell_vary_volume_posterior"
    param_df = pd.read_csv(param_file)
    constant_columns = param_df.columns[param_df.nunique() == 1]
    param_df = param_df.drop(columns=constant_columns)
    param_names = param_df.columns.tolist()
    targets = [
        Target(metric=Metric.get("doub_time"), value=35.0, weight=1.0),
        Target(metric=Metric.get("doub_time_std"), value=10.0, weight=1.0),
        Target(metric=Metric.get("act_t2"), value=0.6, weight=1.0),
        #Target(metric=Metric.get("colony_growth_rate"), value=0.8, weight=1.0),
    ]
    
    # Save targets to CSV
    targets_df = pd.DataFrame([
        {
            'metric': target.metric.value,
            'value': target.value,
            'weight': target.weight
        } for target in targets
    ])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    targets_df.to_csv(os.path.join(output_dir, 'targets.csv'), index=False)
    
    abc = ABCPrecomputed(
        model_config, abc_config, targets, param_file=param_file, metrics_file=metrics_file
    )

    targets_list = [targets] # [targets, targets_2]
    accepted_params_list = []
    for i, targets in enumerate(targets_list):
        abc.update_targets(targets)
        param_metrics_distances_results = abc.run_inference()
        param_keys = list(param_metrics_distances_results[0].keys())
        params = [
            {key: sample[key] for key in param_keys}
            for sample in param_metrics_distances_results
        ]

        accepted_params = [
            {key: sample[key] for key in param_keys if key in param_names}
            for sample in param_metrics_distances_results
            if sample["accepted"]
        ]
        if len(accepted_params) == 0:
            raise ValueError("No accepted parameters")
        accepted_params_list.append(accepted_params)
        parameter_pdfs = evaluate.estimate_pdfs(accepted_params)
        generate_parameters_from_kde(parameter_pdfs, 64, output_dir=output_dir)
        if 1:
            save_path = f"{output_dir}/prior_posterior_pdfs_{i}.png"
            plot_parameter_kde(parameter_pdfs, abc_config, save_path)

        if 0:
            save_path = f"{output_dir}/joint_distribution_{i}.png"
            plot_joint_distribution(accepted_params, save_path)

    if 0:
        save_path = f"{output_dir}/pca_visualization.png"
        plot_pca_visualization(accepted_params_list, save_path)


if __name__ == "__main__":
    run_abc_precomputed()