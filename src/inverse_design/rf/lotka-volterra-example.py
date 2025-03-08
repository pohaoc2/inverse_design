import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from inverse_design.rf.abc_smc_rf_lotka_volterra import ABCSMCRF


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(42)

# True parameter values we want to infer
TRUE_ALPHA = 1.0  # Prey growth rate
TRUE_GAMMA = 1.0  # Predator death rate

# Fixed parameters
BETA = 0.1  # Predation rate
DELTA = 0.075  # Predator growth from predation

# Initial conditions
INITIAL_PREY = 10
INITIAL_PREDATOR = 5

# Simulation settings
TIME_POINTS = np.linspace(0, 30, 31)  # 0 to 30 in steps of 1
NOISE_LEVEL = 0.05  # Observation noise (5%)

def lotka_volterra(z, t, alpha, beta, delta, gamma):
    """
    Lotka-Volterra model differential equations.
    
    Parameters:
    -----------
    z : list
        Current state [prey, predator]
    t : float
        Current time (not used, but required by odeint)
    alpha : float
        Prey growth rate
    beta : float
        Predation rate
    delta : float
        Predator growth from predation
    gamma : float
        Predator death rate
    
    Returns:
    --------
    dz : list
        [d(prey)/dt, d(predator)/dt]
    """
    x, y = z
    dx_dt = alpha * x - beta * x * y
    dy_dt = delta * x * y - gamma * y
    return [dx_dt, dy_dt]

def simulate_data(params, time_points=TIME_POINTS, initial_conditions=[INITIAL_PREY, INITIAL_PREDATOR], 
                  add_noise=True, noise_level=NOISE_LEVEL):
    """
    Simulate data from the Lotka-Volterra model.
    
    Parameters:
    -----------
    params : list
        [alpha, gamma] - Parameters to infer
    time_points : array
        Time points to simulate
    initial_conditions : list
        [initial_prey, initial_predator]
    add_noise : bool
        Whether to add observation noise
    noise_level : float
        Standard deviation of noise relative to signal
    
    Returns:
    --------
    result : array
        Simulated data with shape (len(time_points), 2)
    """
    alpha, gamma = params
    
    # Solve the ODE
    try:
        result = odeint(lotka_volterra, initial_conditions, time_points, 
                        args=(alpha, BETA, DELTA, gamma))
    except:
        # If integration fails, return very different values to ensure rejection
        return np.ones((len(time_points), 2)) * 1000
    
    # Add observation noise if requested
    if add_noise:
        noise = np.random.normal(0, noise_level, result.shape) * result
        result = result + noise
        
    return result

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
    """Run the ABC-SMC-DRF example on the Lotka-Volterra model"""
    # Generate observed data with true parameters
    true_params = [TRUE_ALPHA, TRUE_GAMMA]
    observed_data = simulate_data(true_params, add_noise=True)
    observed_stats = compute_statistics(observed_data)
    if 0:
        print("True parameters: alpha =", TRUE_ALPHA, ", gamma =", TRUE_GAMMA)
        print("Generated observed data with", len(TIME_POINTS), "time points")
        
        # Plot the observed data
        plt.figure(figsize=(10, 6))
        plt.plot(TIME_POINTS, observed_data[:, 0], 'b-', label='Prey (observed)')
        plt.plot(TIME_POINTS, observed_data[:, 1], 'r-', label='Predator (observed)')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title('Observed Lotka-Volterra Dynamics')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Initialize and run ABC-SMC-DRF
    print("\nRunning ABC-SMC-DRF...")
    start_time = time.time()
    
    smc_rf = ABCSMCRF(
        n_iterations=2,           # Number of SMC iterations
        n_particles=500,          # Number of particles per iteration
        rf_type='DRF',            # Use Distributional Random Forest for multivariate inference
        n_trees=100,              # Number of trees in the random forest
        min_samples_leaf=5,       # Minimum samples per leaf
        random_state=42,          # Random seed
        criterion='CART',         # Splitting criterion (CART or MMD)
        prior_sampler=prior_sampler,
        perturbation_kernel=perturbation_kernel,
        prior_pdf=prior_pdf
    )
    
    smc_rf.fit(observed_stats, abc_simulator)

    print(f"ABC-SMC-DRF completed in {time.time() - start_time:.2f} seconds")
    feature_names = ['prey_mean', 'prey_std', 'prey_min', 'prey_max', 'predator_mean', 'predator_std', 'predator_min', 'predator_max', 'correlation', 'prey_peaks', 'predator_peaks', 'time_lag']
    smc_rf.plot_tree(
        iteration=-1,  # last iteration
        feature_names=feature_names,  # your statistics names
        max_depth=10  # adjust for more or less detail
    )
    # Generate posterior samples
    posterior_samples = smc_rf.posterior_sample(1000)
    prior_samples = prior_sampler(1000)
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
    #plt.show()
    #plot_iterations(smc_rf, true_params=[TRUE_ALPHA, TRUE_GAMMA])
    plot_variable_importance(smc_rf)
    #posterior_check(smc_rf, observed_data)

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
