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
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(42)


def simulate_data(input_file: Path, output_base_dir: Path, jar_path: str) -> bool:
    """Run a single ARCADE simulation

    Args:
        input_file: Path to input XML file
        output_base_dir: Base directory for outputs
        jar_path: Path to arcade_v3.jar

    Returns:
        bool: True if simulation completed successfully
    """
    output_dir = f"{output_base_dir}/inputs/{input_file.stem}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        result = subprocess.run(
            ["java", "-jar", str(jar_path), "patch", str(input_file), str(output_dir)],
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running simulation for {input_file.name}")
        logging.error(f"Error message: {e.stderr}")
        return False



def compute_statistics(data):
    """
    Compute summary statistics from the simulated time series data.
    
    Parameters:
    -----------
    data : array
        Time series data with shape (time_points, 2)
    
    Returns:
    --------
    stats : array
        Summary statistics
    """
    # Extract prey and predator populations
    prey = data[:, 0]
    predator = data[:, 1]
    
    # Basic statistics
    prey_mean = np.mean(prey)
    prey_std = np.std(prey)
    prey_min = np.min(prey)
    prey_max = np.max(prey)
    
    predator_mean = np.mean(predator)
    predator_std = np.std(predator)
    predator_min = np.min(predator)
    predator_max = np.max(predator)
    
    # Correlation between prey and predator
    correlation = np.corrcoef(prey, predator)[0, 1]
    
    # Calculate oscillation properties
    # Number of peaks in prey population
    prey_peaks = len([i for i in range(1, len(prey)-1) if prey[i-1] < prey[i] and prey[i] > prey[i+1]])
    predator_peaks = len([i for i in range(1, len(predator)-1) if predator[i-1] < predator[i] and predator[i] > predator[i+1]])
    
    # Time lag between prey and predator peaks
    # (This is a simplified measure of phase difference)
    cross_correlation = np.correlate(prey - np.mean(prey), predator - np.mean(predator), mode='full')
    time_lag = np.argmax(cross_correlation) - len(prey) + 1
    
    # Combine all statistics
    stats = np.array([
        prey_mean, prey_std, prey_min, prey_max,
        predator_mean, predator_std, predator_min, predator_max,
        correlation, prey_peaks, predator_peaks, time_lag
    ])
    
    return stats

# Define the simulator function for ABC-SMC-DRF
def abc_simulator(params):
    """Wrapper around simulation for ABC-SMC-DRF"""
    data = simulate_data(params)
    return compute_statistics(data)

# Define prior functions
def prior_sampler(n_samples):
    """Sample from the prior distribution"""
    alpha = np.random.uniform(0.5, 2.0, n_samples)
    gamma = np.random.uniform(0.5, 2.0, n_samples)
    return np.column_stack([alpha, gamma])

def prior_pdf(params):
    """Evaluate the prior density at the given parameters"""
    alpha, gamma = params
    if 0.5 <= alpha <= 2.0 and 0.5 <= gamma <= 2.0:
        return 1.0 / ((2.0 - 0.5) * (2.0 - 0.5))  # Uniform density
    else:
        return 0.0

def perturbation_kernel(params):
    """Perturb parameters for the next iteration"""
    alpha, gamma = params
    # Add Gaussian noise with adaptive scale
    alpha_new = alpha + np.random.normal(0, 0.1)
    gamma_new = gamma + np.random.normal(0, 0.1)
    return np.array([alpha_new, gamma_new])

def run_example():
    """Run the ABC-SMC-DRF example on the ARCADE model"""
    prior_samples = prior_sampler(1000)
    targets = [
        Target(metric=Metric.get("symmetry"), value=0.8, weight=1.0),
        Target(metric=Metric.get("cycle_length"), value=30, weight=1.0),
        Target(metric=Metric.get("act"), value=0.6, weight=1.0),
    ]
    # Initialize and run ABC-SMC-DRF
    print("\nRunning ABC-SMC-DRF...")
    start_time = time.time()
    
    smc_rf = ABCSMCRF(
        n_iterations=5,           # Number of SMC iterations
        sobol_power=1,            # Number of particles per iteration
        rf_type='DRF',            # Use Distributional Random Forest for multivariate inference
        n_trees=100,              # Number of trees in the random forest
        min_samples_leaf=5,       # Minimum samples per leaf
        param_ranges=PARAM_RANGES,
        random_state=42,          # Random seed
        criterion='CART',         # Splitting criterion (CART or MMD)
        perturbation_kernel=perturbation_kernel,
        prior_pdf=prior_pdf
    )
    timestamps = [
        "000000",
        "000720",
    ]
    input_dir = "inputs/abc_smc_rf/"
    output_dir = "ARCADE_OUTPUT/ABC_SMC_RF/"
    jar_path = "models/arcade-test-cycle-fix-affinity.jar"

    smc_rf.fit(targets, input_dir, output_dir, jar_path, timestamps)

    asd()
    print(f"ABC-SMC-DRF completed in {time.time() - start_time:.2f} seconds")    
    # Generate posterior samples
    posterior_samples = smc_rf.posterior_sample(1000)
    
    # Analyze results
    print("\nParameter estimation results:")
    print(f"True alpha: {TRUE_ALPHA:.4f}, Estimated: {np.mean(posterior_samples[:, 0]):.4f} ± {np.std(posterior_samples[:, 0]):.4f}")
    print(f"True gamma: {TRUE_GAMMA:.4f}, Estimated: {np.mean(posterior_samples[:, 1]):.4f} ± {np.std(posterior_samples[:, 1]):.4f}")
    
    # Plot parameter distributions
    plt.figure(figsize=(12, 10))
    
    # Alpha marginal distribution
    plt.subplot(2, 2, 1)
    plt.hist(prior_samples[:, 0], bins=20, alpha=0.5, label='Prior', density=True)
    plt.hist(posterior_samples[:, 0], bins=20, alpha=0.5, label='Posterior', density=True)
    plt.axvline(TRUE_ALPHA, color='red', linestyle='--', label='True')
    plt.xlabel('α (prey growth rate)')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Marginal Distribution for α')
    
    # Gamma marginal distribution
    plt.subplot(2, 2, 4)
    plt.hist(prior_samples[:, 1], bins=20, alpha=0.5, label='Prior', density=True, orientation='horizontal')
    plt.hist(posterior_samples[:, 1], bins=20, alpha=0.5, label='Posterior', density=True, orientation='horizontal')
    plt.axhline(TRUE_GAMMA, color='red', linestyle='--', label='True')
    plt.ylabel('γ (predator death rate)')
    plt.xlabel('Density')
    plt.legend()
    plt.title('Marginal Distribution for γ')
    
    # Joint distribution
    plt.subplot(2, 2, 2)
    plt.scatter(prior_samples[:, 0], prior_samples[:, 1], alpha=0.3, label='Prior', s=10)
    plt.scatter(posterior_samples[:, 0], posterior_samples[:, 1], alpha=0.3, label='Posterior', s=10)
    plt.scatter(TRUE_ALPHA, TRUE_GAMMA, color='red', marker='*', s=200, label='True')
    plt.xlabel('α (prey growth rate)')
    plt.ylabel('γ (predator death rate)')
    plt.legend()
    plt.title('Joint Parameter Distribution')
    
    # KDE of posterior
    plt.subplot(2, 2, 3)
    from scipy.stats import gaussian_kde
    x, y = np.mgrid[0.5:1.5:100j, 0.5:1.5:100j]
    positions = np.vstack([x.ravel(), y.ravel()])
    values = np.vstack([posterior_samples[:, 0], posterior_samples[:, 1]])
    kernel = gaussian_kde(values)
    z = np.reshape(kernel(positions).T, x.shape)
    plt.contourf(x, y, z, cmap='viridis')
    plt.scatter(TRUE_ALPHA, TRUE_GAMMA, color='red', marker='*', s=200, label='True')
    plt.xlabel('α (prey growth rate)')
    plt.ylabel('γ (predator death rate)')
    plt.title('Posterior Density')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot the evolution of parameter estimates across iterations
    plot_iterations(smc_rf, true_params=[TRUE_ALPHA, TRUE_GAMMA])
    
    # Plot variable importance
    plot_variable_importance(smc_rf)
    
    # Simulate from the posterior for model checking
    posterior_check(smc_rf, observed_data)

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

def plot_variable_importance(smc_rf):
    """Plot variable importance from the final iteration"""
    importance = smc_rf.get_variable_importance()
    
    # Statistic names
    stat_names = [
        'Prey Mean', 'Prey Std', 'Prey Min', 'Prey Max',
        'Predator Mean', 'Predator Std', 'Predator Min', 'Predator Max',
        'Correlation', 'Prey Peaks', 'Predator Peaks', 'Time Lag'
    ]
    
    plt.figure(figsize=(12, 6))
    plt.bar(stat_names, importance)
    plt.xticks(rotation=45, ha='right')
    plt.title('Variable Importance')
    plt.ylabel('Importance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def posterior_check(smc_rf, observed_data):
    """Perform posterior predictive check"""
    # Get 20 parameter samples from the posterior
    posterior_params = smc_rf.posterior_sample(20)
    
    plt.figure(figsize=(12, 8))
    
    # Plot observed data
    plt.subplot(2, 1, 1)
    plt.plot(TIME_POINTS, observed_data[:, 0], 'b-', linewidth=2, label='Observed Prey')
    plt.plot(TIME_POINTS, observed_data[:, 1], 'r-', linewidth=2, label='Observed Predator')
    plt.title('Observed Data')
    plt.ylabel('Population')
    plt.grid(True)
    plt.legend()
    
    # Plot simulations from posterior
    plt.subplot(2, 1, 2)
    for i, params in enumerate(posterior_params):
        # Simulate without noise for clearer visualization
        sim_data = simulate_data(params, add_noise=False)
        if i == 0:
            plt.plot(TIME_POINTS, sim_data[:, 0], 'b-', alpha=0.3, label='Posterior Prey')
            plt.plot(TIME_POINTS, sim_data[:, 1], 'r-', alpha=0.3, label='Posterior Predator')
        else:
            plt.plot(TIME_POINTS, sim_data[:, 0], 'b-', alpha=0.3)
            plt.plot(TIME_POINTS, sim_data[:, 1], 'r-', alpha=0.3)
    
    plt.title('Posterior Predictive Check')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_example()
