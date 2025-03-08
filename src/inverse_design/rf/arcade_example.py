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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42)
PARAM_RANGES = PARAM_RANGES.copy()
TARGET_RANGES = {
    "symmetry": (0.0, 1.0),
    "cycle_length": (10.0, 100.0),
    "act": (0.0, 1.0),
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
    
    return 1.0

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

def plot_parameter_iterations(smc_rf, param_names):
    """Plot parameter distributions across iterations"""
    n_iterations = len(smc_rf.parameter_samples)
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_iterations, figsize=(15, 6), squeeze=False)
    
    for idx, param_name in enumerate(param_names):
        param_idx = list(smc_rf.param_ranges.keys()).index(param_name)
        min_val, max_val = smc_rf.param_ranges[param_name]
        for t in range(n_iterations):
            params, _, weights = smc_rf.get_iteration_results(t)
            param_values = params[:, param_idx]
            ax = axes[idx, t]
            ax.hist(param_values, bins=10, weights=weights, alpha=0.5, density=True, color='skyblue', label='Histogram')
            if len(param_values) > 1:
                try:
                    kde = gaussian_kde(param_values, weights=weights)
                    x_range = np.linspace(min_val, max_val, 200)
                    ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
                except np.linalg.LinAlgError:
                    print(f"Error computing KDE for {param_name} at iteration {t+1}")
            ax.set_xlim(min_val, max_val)
            if idx == 0:
                ax.set_title(f'Iteration {t+1}')
                
            if t == 0:
                ax.set_ylabel(param_name)
                
            if idx == n_params - 1:
                ax.set_xlabel('Value')
                
            if idx == 0 and t == 0:
                ax.legend()
            
    plt.suptitle('Parameter Distributions Across Iterations')
    plt.tight_layout()
    plt.show()

def plot_variable_importance(smc_rf, statistic_names):
    """Plot variable importance from the final iteration"""
    importance = smc_rf.get_variable_importance()
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
    plt.show()

def plot_statistics_iterations(smc_rf, target_names, target_values):
    """Plot statistics distributions across iterations"""
    n_iterations = len(smc_rf.parameter_samples)
    n_stats = len(target_names)
    fig, axes = plt.subplots(n_stats, n_iterations, figsize=(15, 6), squeeze=False)
    
    for idx, (stat_name, target_val) in enumerate(zip(target_names, target_values)):
        min_val, max_val = TARGET_RANGES[stat_name]
        for t in range(n_iterations):
            _, stats, weights = smc_rf.get_iteration_results(t)
            stat_values = stats[:, idx]
            ax = axes[idx, t]
            ax.set_xlim(min_val, max_val)
            # Plot histogram
            ax.hist(stat_values, bins=10, weights=weights, alpha=0.5, density=True, color='skyblue', label='Histogram')
            
            # Calculate and plot KDE
            """
            if len(stat_values) > 1:
                try:
                    kde = gaussian_kde(stat_values, weights=weights)
                    x_range = np.linspace(min(stat_values), max(stat_values), 200)
                    ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
                except np.linalg.LinAlgError:
                    print(f"Error computing KDE for {stat_name} at iteration {t+1}")
            """
            # Plot target value
            ax.axvline(target_val, color='g', linestyle='--', label='Target' if t == 0 else None)
            
            if idx == 0:
                ax.set_title(f'Iteration {t+1}')
            
            if t == 0:
                ax.set_ylabel(stat_name)
            
            if idx == n_stats - 1:
                ax.set_xlabel('Value')
            
            if idx == 0 and t == 0:
                ax.legend()
    
    plt.suptitle('Statistics Distributions Across Iterations')
    plt.tight_layout()
    #plt.show()

def run_example():
    """Run the ABC-SMC-DRF example on the ARCADE model"""
    targets = [
        Target(metric=Metric.get("symmetry"), value=0.8, weight=1.0),
        Target(metric=Metric.get("cycle_length"), value=30, weight=1.0),
        Target(metric=Metric.get("act"), value=0.6, weight=1.0),
    ]
    target_names = [target.metric.value for target in targets]
    target_values = [target.value for target in targets]
    print("\nRunning ABC-SMC-DRF...")
    start_time = time.time()
    param_ranges = PARAM_RANGES.copy()
    param_ranges = {k: v for k, v in param_ranges.items() if v[0] != v[1]}
    smc_rf = ABCSMCRF(
        n_iterations=2,           
        sobol_power=8,            
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
    input_dir = "inputs/abc_smc_rf/"
    output_dir = "ARCADE_OUTPUT/ABC_SMC_RF/"
    jar_path = "models/arcade-test-cycle-fix-affinity.jar"
    smc_rf.fit(target_names, target_values, input_dir, output_dir, jar_path, timestamps)

    print(f"ABC-SMC-DRF completed in {time.time() - start_time:.2f} seconds")
    smc_rf.plot_tree(
        iteration=-1,  # last iteration
        feature_names=['symmetry', 'cycle_length', 'act'],  # your statistics names
        max_depth=10  # adjust for more or less detail
    )
    asd()
    posterior_samples = smc_rf.posterior_sample(1000)
    print("\nParameter estimation results:")
    for i, param_name in enumerate(smc_rf.param_ranges.keys()):
        print(f"{param_name}: {np.mean(posterior_samples[:, i]):.4f} ± {np.std(posterior_samples[:, i]):.4f}")
    
    # Plot results
    param_names = ["AFFINITY", "COMPRESSION_TOLERANCE", "CELL_VOLUME_MU"]
    #plot_parameter_iterations(smc_rf, param_names)
    #plot_statistics_iterations(smc_rf, target_names, target_values)
    #plot_variable_importance(smc_rf, target_names)

if __name__ == "__main__":
    run_example()
