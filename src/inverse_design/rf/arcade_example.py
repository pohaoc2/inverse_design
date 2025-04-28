import time
import subprocess
from pathlib import Path
import logging
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.stats import gaussian_kde, qmc
from inverse_design.common.enum import Target, Metric
from inverse_design.rf.abc_smc_rf_arcade import ABCSMCRF
from inverse_design.analyze.parameter_config import PARAM_RANGES, SOURCE_PARAM_RANGES
from inverse_design.analyze.source_metrics import calculate_capillary_density, calculate_distance_between_points

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42)
PARAM_RANGES = PARAM_RANGES.copy()
TARGET_RANGES = {
    "symmetry": (0.6, 1.2),
    "cycle_length": (10.0, 70.0),
    "act_ratio": (0.0, 1.0),
    "doub_time": (20.0, 100.0),
    "symmetry_std": (0.0, 0.01),
    "cycle_length_std": (0.0, 0.5),
    "act_ratio_std": (0.0, 0.01),
    "doub_time_std": (0.0, 0.5),
}

def prior_pdf(params, param_columns, param_ranges=PARAM_RANGES, config_params=None):
    """
    Evaluate the uniform prior density at the given parameters.
    
    Parameters:
    -----------
    params : array-like
        Array of parameter values for parameters being inferred,
        in the same order as they appear in INFER_PARAMS
        
    Returns:
    --------
    density : float
        Prior probability density at the given parameters (constant if valid, 0 if invalid)
    """
    if len(params) != len(param_ranges):
        raise ValueError(f"Expected {len(param_ranges)} parameters, got {len(params)}")
    if config_params["perturbed_config"] == "cellular":   
        # Check if the number of parameters matches
        for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
            param_index = param_columns.index(param_name)
            if params[param_index] < min_val or params[param_index] > max_val:
                return 0.0  # Parameter out of range, zero density
            else:
                return 1.0 / (max_val - min_val)
    else:
        for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
            param_index = param_columns.index(param_name)
            if param_name in ["X_SPACING", "Y_SPACING"]:
                if config_params["point_based"] and param_name == "Y_SPACING":
                    original_y_spacing = int(params[param_index].split(":")[0])
                    if original_y_spacing < min_val or original_y_spacing > max_val:
                        return 0.0  # Parameter out of range, zero density
                else:
                    spacing = int(params[param_index].split(":")[-1])
                    if spacing < min_val or spacing > max_val:
                        return 0.0  # Parameter out of range, zero density
            elif param_name in ["GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]:
                if params[param_index] < min_val or params[param_index] > max_val:
                    return 0.0  # Parameter out of range, zero density
    return 1.0 / (max_val - min_val)

def perturbation_kernel(params, param_columns, iteration=1, max_iterations=5, param_ranges=PARAM_RANGES, seed=42, config_params=None):
    perturbed_params = params.copy()
    scale_factor = max(0.01, 0.1 * (1 - iteration/max_iterations))
    # Match the param_ranges with the parameter columns
    param_ranges = {col_name: param_ranges[col_name] for col_name in param_columns}
    if len(params) != len(param_ranges):
        raise ValueError(f"Expected {len(param_ranges)} parameters, got {len(params)}")
    if config_params["perturbed_config"] == "cellular":
        for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
            param_index = param_columns.index(param_name)
            param_range = max_val - min_val
            scale = param_range * scale_factor
            perturbed_params[param_index] += np.random.normal(0, scale)
    elif config_params["perturbed_config"] == "source":
        for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
            param_index = param_columns.index(param_name)
            if param_name in ["X_SPACING", "Y_SPACING"]:
                if config_params["point_based"] and param_name == "Y_SPACING":
                    original_y_spacing = int(perturbed_params[param_index].split(":")[0])
                    perturb_value = config_params["y_interval"] * np.random.randint(-2, 3)
                    final_y_spacing = original_y_spacing + perturb_value
                    perturbed_params[param_index] = f"{final_y_spacing}:{final_y_spacing+1}"
                else:
                    if param_name == "X_SPACING":
                        x_spacing_value = float(perturbed_params[param_index].split(":")[-1]) + np.random.randint(-4, 5)
                    else:
                        y_spacing_value = float(perturbed_params[param_index].split(":")[-1]) + np.random.randint(-4, 5)
                        # swap x and y spacing if y spacing is smaller than x spacing
                        x_final = min(x_spacing_value, y_spacing_value)
                        y_final = max(x_spacing_value, y_spacing_value)
                        perturbed_params[param_index-1] = "*:" + str(int(x_final))
                        perturbed_params[param_index] = "*:" + str(int(y_final))
            elif param_name in ["GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]:
                param_range = max_val - min_val
                scale = param_range * scale_factor
                perturbed_params[param_index] += np.random.normal(0, scale)
    elif config_params["perturbed_config"] == "combined":
        # First handle cellular parameters
        cellular_params = {k: v for k, v in param_ranges.items() if k not in ["X_SPACING", "Y_SPACING", "GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]}
        for i, (param_name, (min_val, max_val)) in enumerate(cellular_params.items()):
            param_index = param_columns.index(param_name)
            param_range = max_val - min_val
            scale = param_range * scale_factor
            perturbed_params[param_index] += np.random.normal(0, scale)
            
        # Then handle source parameters
        source_params = {k: v for k, v in param_ranges.items() if k in ["X_SPACING", "Y_SPACING", "GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]}
        for i, (param_name, (min_val, max_val)) in enumerate(source_params.items()):
            param_index = param_columns.index(param_name)
            if param_name in ["X_SPACING", "Y_SPACING"]:
                if config_params["point_based"] and param_name == "Y_SPACING":
                    original_y_spacing = int(perturbed_params[param_index].split(":")[0])
                    perturb_value = config_params["y_interval"] * np.random.randint(-2, 3)
                    final_y_spacing = original_y_spacing + perturb_value
                    perturbed_params[param_index] = f"{final_y_spacing}:{final_y_spacing+1}"
                else:
                    if param_name == "X_SPACING":
                        x_spacing_value = float(perturbed_params[param_index].split(":")[-1]) + np.random.randint(-4, 5)
                    else:
                        y_spacing_value = float(perturbed_params[param_index].split(":")[-1]) + np.random.randint(-4, 5)
                        # swap x and y spacing if y spacing is smaller than x spacing
                        x_final = min(x_spacing_value, y_spacing_value)
                        y_final = max(x_spacing_value, y_spacing_value)
                        perturbed_params[param_index-1] = "*:" + str(int(x_final))
                        perturbed_params[param_index] = "*:" + str(int(y_final))
            elif param_name in ["GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]:
                param_range = max_val - min_val
                scale = param_range * scale_factor
                perturbed_params[param_index] += np.random.normal(0, scale)
    return perturbed_params

def plot_variable_importance(smc_rf, statistic_names, n_statistics, save_path=None):
    """Plot variable importance from the final iteration"""
    importance = smc_rf.get_variable_importance(t=None, n_statistics=n_statistics)
    # Ensure we have the right number of names
    if len(importance) != len(statistic_names):
        raise ValueError(f"Number of statistics ({len(importance)}) doesn't match number of names ({len(statistic_names)})")
    
    plt.figure(figsize=(12, 6))
    plt.bar(statistic_names, importance)
    plt.xticks(rotation=45, ha='right')
    plt.title('Summary Statistic Importance')
    plt.ylabel('Importance')
    plt.xlabel('Summary Statistics')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_parameter_iterations(smc_rf, param_names, sobol_power=8, plot_kde=False, save_path=None):
    """Plot parameter distributions across iterations.
    
    Parameters:
    -----------
    smc_rf : ABCSMCRF
        The fitted ABC-SMC-RF object
    param_names : list
        List of parameter names
    plot_kde : bool, optional
        Whether to plot KDE (default: True)
    save_path : str, optional
        Path to save the figure
    """
    n_iterations = len(smc_rf.parameter_samples)
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_iterations, figsize=(12, 5 * n_params), squeeze=False)
    # Generate Sobol samples for prior
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=len(param_names), seed=42)
    n_samples = 2**sobol_power
    
    for idx, param_name in enumerate(param_names):
        param_idx = smc_rf.parameter_columns.index(param_name)
        min_val, max_val = smc_rf.param_ranges[param_name]
        
        for t in range(n_iterations):
            params, _, weights = smc_rf.get_iteration_results(t)
            param_values = params[:, param_idx]
            ax = axes[idx, t]
            ax.set_xlim(min_val, max_val)
            
            # Plot histogram
            ax.hist(param_values, bins=10, weights=weights, alpha=0.5, 
                   density=True, color='blue', label='Posterior', edgecolor='black')
            
            # Calculate and plot KDE
            if plot_kde and len(param_values) > 1:
                try:
                    kde = gaussian_kde(param_values, weights=weights)
                    x_range = np.linspace(min_val, max_val, 200)
                    ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
                except np.linalg.LinAlgError:
                    print(f"Error computing KDE for {param_name} at iteration {t+1}")
            

            
            if idx == 0:
                ax.set_title(f'Iteration {t+1}')
            
            if t == 0:
                ax.set_ylabel(param_name)
                prior_samples = np.random.uniform(min_val, max_val, n_samples)
                ax.hist(prior_samples, bins=20, alpha=0.5, 
                    density=True, color='gray', label='Prior')

            if idx == n_params - 1:
                ax.set_xlabel('Value')
            
            if idx == 0 and t == 0:
                ax.legend()
    
    plt.suptitle('Parameter Distributions Across Iterations')
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_parameter_iterations.png")
    else:
        plt.show()

def plot_statistic_iterations(smc_rf, target_names, target_values, plot_kde=True, save_path=None):
    """Plot statistic distributions across iterations.
    
    Parameters:
    -----------
    smc_rf : ABCSMCRF
        The fitted ABC-SMC-RF object
    target_names : list
        List of target statistic names
    target_values : list
        List of target values
    plot_kde : bool, optional
        Whether to plot KDE (default: True)
    save_path : str, optional
        Path to save the figure
    """
    n_iterations = len(smc_rf.parameter_samples)
    n_stats = len(target_names)
    fig, axes = plt.subplots(n_stats, n_iterations, figsize=(12, 5 * n_stats), squeeze=False)
    
    for idx, (stat_name, target_val) in enumerate(zip(target_names, target_values)):
        min_val, max_val = TARGET_RANGES[stat_name]
        
        for t in range(n_iterations):
            _, stats, weights = smc_rf.get_iteration_results(t)
            stat_values = stats[:, idx]
            # remove NaN or -inf, inf values
            weights = weights[~np.isnan(stat_values) & ~np.isinf(stat_values)]
            stat_values = stat_values[~np.isnan(stat_values) & ~np.isinf(stat_values)]
            
            ax = axes[idx, t]
            ax.set_xlim(min_val, max_val)
            
            # Plot histogram
            if t == 0:
                ax.hist(stat_values, bins=10, weights=weights, alpha=0.5, color='gray', label='Prior', edgecolor='black')
            else:
                ax.hist(stat_values, bins=10, weights=weights, alpha=0.5, color='blue', label='Posterior', edgecolor='black')
            
            # Add target line
            ax.axvline(target_val, color='g', linestyle='--', label='Target')
            ax.annotate(f'{target_val:.2f}', xy=(target_val, 0), 
                        xytext=(10, 10), textcoords='offset points', color='g',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))
            # Calculate and plot KDE
            if plot_kde and len(stat_values) > 1:
                try:
                    kde = gaussian_kde(stat_values, weights=weights)
                    x_range = np.linspace(min_val, max_val, 200)
                    kde_values = kde(x_range)
                    ax.plot(x_range, kde_values, 'r-', lw=2, label='KDE')
                    
                    # Calculate and plot the mode of KDE
                    mode_idx = np.argmax(kde_values)
                    mode_x = x_range[mode_idx]
                    mode_y = kde_values[mode_idx]
                    ax.plot(mode_x, mode_y, 'r^', markersize=10, label='KDE Mode')
                    ax.annotate(f'{mode_x:.2f}', xy=(mode_x, mode_y), 
                                xytext=(10, 10), textcoords='offset points',
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
                except np.linalg.LinAlgError and ValueError:
                    print(f"Error computing KDE for {stat_name} at iteration {t+1}")
            
            if idx == 0:
                ax.set_title(f'Iteration {t+1}')
            
            if t == 0:
                ax.set_ylabel(stat_name)
            
            if idx == n_stats - 1:
                ax.set_xlabel('Value')
            
            if idx == 0 and (t == 0 or t == 1):
                ax.legend()
    
    plt.suptitle('Statistics Distributions Across Iterations')
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_statistic_iterations.png")
    else:
        plt.show()

def save_targets_to_json(target_names, target_values, output_file="targets.json"):
    """
    Save target names and their corresponding values to a JSON file.
    
    Parameters:
    -----------
    target_names : list
        List of target statistic names
    target_values : list
        List of target values
    output_file : str, optional
        Path to the output JSON file (default: "targets.json")
    """
    targets_dict = dict(zip(target_names, target_values))
    with open(output_file, 'w') as f:
        json.dump(targets_dict, f, indent=4)

def run_example():
    """Run the ABC-SMC-DRF example on the ARCADE model"""
    target_names = ["symmetry", "doub_time", "act_ratio"]#, "cycle_length"]
    target_values = [0.75, 32, 0.7]#, 30]
    #target_names = target_names + [name+"_std" for name in target_names]
    #target_values = target_values + [value*0.05 for value in target_values]    
    targets = []
    for name, value in zip(target_names, target_values):
        targets.append(Target(metric=Metric.get(name), value=value, weight=1.0))
    n_statistics = len(target_names)
    print("\nRunning ABC-SMC-DRF...")
    start_time = time.time()
    sobol_power = 10
    radius = 10
    margin = 2
    hex_size = 30
    side_length = hex_size / np.sqrt(3)
    input_configs = [
        {
            "perturbed_config": "cellular",
            "template_path": "sample_cellular_v3.xml",
            "point_based": None,
            "y_interval": None,
            "radius_bound": None,
            "side_length": None
        },
        {
            "perturbed_config": "source",
            "template_path": "sample_source_v3.xml",
            "point_based": True,
            "y_interval": 4,
            "radius_bound": radius+margin,
            "side_length": side_length
        },
        {
            "perturbed_config": "combined",
            "template_path": "sample_combined_v3.xml",  # Template with both cellular and source parameters
            "point_based": False,
            "y_interval": 4,
            "radius_bound": radius+margin,
            "side_length": side_length
        }
    ]
    n_samples = 2**sobol_power
    config_params = input_configs[2]
    if config_params["perturbed_config"] == "cellular":
        param_ranges = PARAM_RANGES.copy()
    elif config_params["perturbed_config"] == "source":
        param_ranges = SOURCE_PARAM_RANGES.copy()
    elif config_params["perturbed_config"] == "combined":
        param_ranges = {**PARAM_RANGES, **SOURCE_PARAM_RANGES}
    if config_params["point_based"]:
        param_ranges.pop("X_SPACING")
    param_ranges = {k: v for k, v in param_ranges.items() if v[0] != v[1]}
    smc_rf = ABCSMCRF(
        n_iterations=2,           
        sobol_power=sobol_power,            
        rf_type='DRF',
        n_trees=2,
        min_samples_leaf=5,
        param_ranges=param_ranges,
        random_state=42, 
        criterion='CART',
        subsample_ratio=0.5,
        perturbation_kernel=perturbation_kernel,
        prior_pdf=prior_pdf,
        config_params=config_params
    )
    timestamps = [
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
    #timestamps = timestamps[:2]
    source_type = "point" if config_params["point_based"] else "grid"
    input_dir = f"inputs/abc_smc_rf_n{n_samples}_{config_params['perturbed_config']}_{source_type}/"
    output_dir = f"ARCADE_OUTPUT/ABC_SMC_RF_N{n_samples}_{config_params['perturbed_config']}_{source_type}/"
    jar_path = "models/arcade-logging-necrotic.jar"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_targets_to_json(target_names, target_values, f"{output_dir}targets.json")
    smc_rf.fit(target_names, target_values, input_dir, output_dir, jar_path, timestamps)
    asd()
    plot_dir = f"{output_dir}/PLOTS/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    smc_rf.plot_tree(
        iteration=-1,
        feature_names=target_names,
        max_depth=10,
        target_values=target_values,
        save_path=f"{plot_dir}/arcade_tree"
    )
    print(f"ABC-SMC-DRF completed in {time.time() - start_time:.2f} seconds")

    posterior_samples = smc_rf.posterior_sample(n_samples * 2)
    n_samples = posterior_samples.shape[0]
    print("\nParameter estimation results:")
    valid_parameters = pd.read_csv(f"{output_dir}/iter_0/all_param_df.csv")
    valid_parameters.drop(columns=["input_folder"], inplace=True)
    if len(valid_parameters.columns) != len(smc_rf.param_ranges.keys()):
        valid_parameters = valid_parameters[smc_rf.param_ranges.keys()]
    if config_params["point_based"]:
        valid_parameters.drop(columns=["CAPILLARY_DENSITY"], inplace=True)
    else:
        valid_parameters.drop(columns=["DISTANCE_TO_CENTER"], inplace=True)
    parameter_columns = list(valid_parameters.columns)
    for i, param_name in enumerate(parameter_columns):
        if config_params["perturbed_config"] == "cellular":
            print(f"{param_name}: {np.mean(posterior_samples[:, i]):.4f} ± {np.std(posterior_samples[:, i]):.4f}")
        else:
            if param_name == "X_SPACING":
                continue
            elif param_name == "Y_SPACING":
                if config_params["point_based"]:
                    x_center = int((6 * config_params["radius_bound"] - 3) // 2)
                    y_center = int((6 * config_params["radius_bound"] - 3) // 2)
                    point_center = (x_center, y_center, 0)
                    source_sites = []
                    for j in range(n_samples):
                        source_sites.append(np.array([x_center, posterior_samples[j][i].split(":")[1], 0]).astype(int))
                    distance_to_center = [calculate_distance_between_points(
                        point_center,
                        source_site,
                        config_params["side_length"],
                        config_params["radius_bound"],
                    ) for source_site in source_sites]
                    print(f"DISTANCE_TO_CENTER: {np.mean(distance_to_center):.4f} ± {np.std(distance_to_center):.4f}")
                else:
                    capillary_densities = [calculate_capillary_density(
                        radius + margin,
                        int(posterior_samples[j][i-1].split(":")[1]),
                        int(posterior_samples[j][i].split(":")[1]),
                        config_params["side_length"],
                    ) for j in range(n_samples)]
                    print(f"CAPILLARY_DENSITY: {np.mean(capillary_densities):.4f} ± {np.std(capillary_densities):.4f}")
                
            else:
                print(f"{param_name}: {np.mean(posterior_samples[:, i]):.4f} ± {np.std(posterior_samples[:, i]):.4f}")
    # Plot results
    if config_params["perturbed_config"] == "cellular":
        param_names = ["AFFINITY", "COMPRESSION_TOLERANCE", "CELL_VOLUME_MU"]
    elif config_params["perturbed_config"] == "source":
        if config_params["point_based"]:
            param_names = ["Y_SPACING", "GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]
        else:
            param_names = ["X_SPACING", "Y_SPACING", "GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]
    elif config_params["perturbed_config"] == "combined":
        param_names = ["COMPRESSION_TOLERANCE", "CELL_VOLUME_MU", "X_SPACING", "Y_SPACING", "GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]

    plot_parameter_iterations(smc_rf, param_names, save_path=f"{plot_dir}/arcade_params_iterations")
    plot_statistic_iterations(smc_rf, target_names, target_values, save_path=f"{plot_dir}/arcade_stats_iterations", plot_kde=False)
    plot_variable_importance(smc_rf, target_names, n_statistics, save_path=f"{plot_dir}/arcade_variable_importance.png")

if __name__ == "__main__":
    run_example()
