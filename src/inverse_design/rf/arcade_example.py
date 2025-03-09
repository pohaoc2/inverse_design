import time
import subprocess
from pathlib import Path
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import gaussian_kde
from inverse_design.common.enum import Target, Metric
from inverse_design.rf.abc_smc_rf_arcade import ABCSMCRF
from inverse_design.analyze.parameter_config import PARAM_RANGES
from scipy.stats import qmc
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42)
PARAM_RANGES = PARAM_RANGES.copy()
TARGET_RANGES = {
    "symmetry": (0.0, 1.0),
    "cycle_length": (10.0, 70.0),
    "act": (0.0, 1.0),
    "doub_time": (10.0, 70.0),
    "symmetry_std": (0.0, 0.01),
    "cycle_length_std": (0.0, 0.5),
    "act_std": (0.0, 0.01),
    "doub_time_std": (0.0, 0.5),
}

def prior_pdf(params, param_ranges=PARAM_RANGES):
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
    
    # Check if the number of parameters matches
    if len(params) != len(param_ranges):
        raise ValueError(f"Expected {len(param_ranges)} parameters, got {len(params)}")

    for i, (min_val, max_val) in enumerate(param_ranges.values()):
        if params[i] < min_val or params[i] > max_val:
            return 0.0  # Parameter out of range, zero density
    
    return 1.0 / (max_val - min_val)

def perturbation_kernel(params, iteration=1, max_iterations=5, param_ranges=PARAM_RANGES, seed=42):
    np.random.seed(seed)
    # Check if the number of parameters matches
    if len(params) != len(param_ranges):
        raise ValueError(f"Expected {len(param_ranges)} parameters, got {len(params)}")
    
    perturbed_params = params.copy()
    
    # Decrease perturbation scale as iterations progress
    scale_factor = max(0.01, 0.1 * (1 - iteration/max_iterations))
    for i, (min_val, max_val) in enumerate(param_ranges.values()):
        param_range = max_val - min_val
        scale = param_range * scale_factor
        perturbed_params[i] += np.random.normal(0, scale)
    

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
    fig, axes = plt.subplots(n_params, n_iterations, figsize=(15, 4 * n_params), squeeze=False)
    # Generate Sobol samples for prior
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=len(param_names), seed=42)
    n_samples = 2**sobol_power
    
    for idx, param_name in enumerate(param_names):
        param_idx = list(smc_rf.param_ranges.keys()).index(param_name)
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
    fig, axes = plt.subplots(n_stats, n_iterations, figsize=(15, 4 * n_stats), squeeze=False)
    
    for idx, (stat_name, target_val) in enumerate(zip(target_names, target_values)):
        min_val, max_val = TARGET_RANGES[stat_name]
        
        for t in range(n_iterations):
            _, stats, weights = smc_rf.get_iteration_results(t)
            stat_values = stats[:, idx]
            ax = axes[idx, t]
            ax.set_xlim(min_val, max_val)
            
            # Plot histogram
            if t == 0:
                ax.hist(stat_values, bins=10, weights=weights, alpha=0.5, 
                        density=True, color='gray', label='Prior', edgecolor='black')
            else:
                ax.hist(stat_values, bins=10, weights=weights, alpha=0.5, 
                        density=True, color='blue', label='Posterior', edgecolor='black')
            
            # Add target line
            ax.axvline(target_val, color='g', linestyle='--', label='Target')
            
            # Calculate and plot KDE
            if plot_kde and len(stat_values) > 1:
                try:
                    kde = gaussian_kde(stat_values, weights=weights)
                    x_range = np.linspace(min_val, max_val, 200)
                    ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
                except np.linalg.LinAlgError:
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
    target_names = ["symmetry", "cycle_length", "act", "doub_time"]
    target_names = target_names + [name+"_std" for name in target_names]
    target_values = [0.8, 30, 0.6, 35]
    target_values = target_values + [value*0.05 for value in target_values]    
    targets = []
    for name, value in zip(target_names, target_values):
        targets.append(Target(metric=Metric.get(name), value=value, weight=1.0))
    n_statistics = len(target_names)
    print("\nRunning ABC-SMC-DRF...")
    start_time = time.time()
    param_ranges = PARAM_RANGES.copy()
    param_ranges = {k: v for k, v in param_ranges.items() if v[0] != v[1]}
    sobol_power = 8
    n_samples = 2**sobol_power
    smc_rf = ABCSMCRF(
        n_iterations=2,           
        sobol_power=sobol_power,            
        rf_type='DRF',
        n_trees=100,
        min_samples_leaf=5,
        param_ranges=param_ranges,
        random_state=42, 
        criterion='CART',
        subsample_ratio=0.5,
        perturbation_kernel=perturbation_kernel,
        prior_pdf=prior_pdf
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
    input_dir = f"inputs/abc_smc_rf_n{n_samples}/sym_cyc_act_doub_std/"
    output_dir = f"ARCADE_OUTPUT/ABC_SMC_RF_N{n_samples}/sym_cyc_act_doub_std/"
    jar_path = "models/arcade-test-cycle-fix-affinity.jar"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_targets_to_json(target_names, target_values, f"{output_dir}targets.json")
    smc_rf.fit(target_names, target_values, input_dir, output_dir, jar_path, timestamps)
    smc_rf.plot_tree(
        iteration=-1,  # last iteration
        feature_names=target_names,  # your statistics names
        max_depth=10,  # adjust for more or less detail
        target_values=target_values,
        save_path="rf/plots/arcade_tree"
    )
    print(f"ABC-SMC-DRF completed in {time.time() - start_time:.2f} seconds")

    posterior_samples = smc_rf.posterior_sample(1000)
    print("\nParameter estimation results:")
    for i, param_name in enumerate(smc_rf.param_ranges.keys()):
        print(f"{param_name}: {np.mean(posterior_samples[:, i]):.4f} ± {np.std(posterior_samples[:, i]):.4f}")
    
    # Plot results
    param_names = ["AFFINITY", "COMPRESSION_TOLERANCE", "CELL_VOLUME_MU"]
    plot_parameter_iterations(smc_rf, param_names, save_path="rf/plots/arcade_params_iterations")
    plot_statistic_iterations(smc_rf, target_names, target_values, save_path="rf/plots/arcade_stats_iterations")
    plot_variable_importance(smc_rf, target_names, n_statistics, save_path="rf/plots/arcade_variable_importance.png")

if __name__ == "__main__":
    run_example()
