import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
from abc_smc_rf import ABCSMCRF
import logging
import os
from inverse_design.common.enum import Target, Metric
import subprocess
from pathlib import Path
from inverse_design.analyze.parameter_config import PARAM_RANGES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42)
PARAM_RANGES = PARAM_RANGES.copy()


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

def perturbation_kernel(params, iteration=1, max_iterations=5, param_ranges=PARAM_RANGES):

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

def run_example():
    """Run the ABC-SMC-DRF example on the ARCADE model"""
    targets = [
        Target(metric=Metric.get("symmetry"), value=0.8, weight=1.0),
        Target(metric=Metric.get("cycle_length"), value=30, weight=1.0),
        Target(metric=Metric.get("act"), value=0.6, weight=1.0),
    ]
    # Initialize and run ABC-SMC-DRF
    print("\nRunning ABC-SMC-DRF...")
    start_time = time.time()
    param_ranges = PARAM_RANGES.copy()
    param_ranges = {k: v for k, v in param_ranges.items() if v[0] != v[1]}


    smc_rf = ABCSMCRF(
        n_iterations=2,           # Number of SMC iterations
        sobol_power=1,            # Number of particles per iteration
        rf_type='DRF',            # Use Distributional Random Forest for multivariate inference
        n_trees=100,              # Number of trees in the random forest
        min_samples_leaf=5,       # Minimum samples per leaf
        param_ranges=param_ranges,
        random_state=42,          # Random seed
        criterion='CART',         # Splitting criterion (CART or MMD)
        perturbation_kernel=perturbation_kernel,
        prior_pdf=prior_pdf
    )
    timestamps = [
        "000000",
        "000100",
        "000200",
        "000300",
        "000400",
        "000500",
    ]
    input_dir = "inputs/abc_smc_rf/"
    output_dir = "ARCADE_OUTPUT/ABC_SMC_RF/"
    jar_path = "models/arcade-test-cycle-fix-affinity.jar"

    smc_rf.fit(targets, input_dir, output_dir, jar_path, timestamps)
    print(f"ABC-SMC-DRF completed in {time.time() - start_time:.2f} seconds")    
    # Generate posterior samples
    posterior_samples = smc_rf.posterior_sample(1000)
    
    # Analyze results
    print("\nParameter estimation results:")
    for i, param_name in enumerate(smc_rf.param_ranges.keys()):
        print(f"{param_name}: {np.mean(posterior_samples[:, i]):.4f} ± {np.std(posterior_samples[:, i]):.4f}")
    # Plot variable importance
    stat_names = [target.metric.value for target in targets]
    plot_variable_importance(smc_rf, stat_names)
    

def plot_iterations(smc_rf, true_params):
    """Plot parameter distributions across iterations"""
    n_iterations = len(smc_rf.parameter_samples)
    
    fig, axes = plt.subplots(2, n_iterations, figsize=(15, 6), squeeze=False)
    
    for p in range(2):
        param_name = 'α' if p == 0 else 'γ'
        true_value = true_params[p]
        
        for t in range(n_iterations):
            params, weights = smc_rf.get_iteration_results(t)
            
            ax = axes[p, t]
            ax.hist(params[:, p], bins=20, weights=weights, alpha=0.7, density=True)
            ax.axvline(true_value, color='red', linestyle='--')
            
            if p == 0:
                ax.set_title(f'Iteration {t+1}')
                
            if t == 0:
                ax.set_ylabel(param_name)
                
            if p == 1:
                ax.set_xlabel('Value')
                
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

if __name__ == "__main__":
    run_example()
