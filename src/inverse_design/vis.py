# Visualization of the BDM model

import matplotlib.pyplot as plt
from .grid import Grid
import os
from typing import List
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
from .abc import Metric, Target

def plot_grid(grid: Grid, time_point: float):
    """Visualize the grid with blue squares for cells and white squares for empty locations"""
    fig, ax = plt.subplots(figsize=(10, 10))
    lattice_size = grid.lattice_size

    # Set equal aspect ratio and limits
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, lattice_size - 0.5)
    ax.set_ylim(-0.5, lattice_size - 0.5)

    for i in range(lattice_size):
        for j in range(lattice_size):
            location = grid.lattice[i, j]
            color = "blue" if location.has_cell() else "white"
            square = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=color, edgecolor="black")
            ax.add_patch(square)
    cell_density = grid.num_cells / (lattice_size**2) * 100
    ax.set_title(f"Time: {time_point}, Cell density: {cell_density:.2f}%")

    # Create 'plots' directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/grid_{int(time_point)}.png")
    plt.close()

def plot_cell_density(time_points: List[float],
                      cell_densities: List[float],
                      red_dot_time: float,
                      red_dot_density: float):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(time_points, cell_densities, 'ko-')
    ax.plot(red_dot_time, red_dot_density, 'ro')
    ax.set_xlabel("Time")
    ax.set_ylabel("Cell density") 
    ax.set_title("Cell density over time")
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/cell_density_{int(red_dot_time)}.png")
    plt.close()

def plot_metric_distribution(all_metrics: Dict[str, List[Dict]], metric: Metric, 
                           targets: List[Target], save_path: str, 
                           x_range: Optional[Tuple[float, float]] = None):
    """Plot distribution of a metric for all and accepted samples
    Args:
        all_metrics: Dictionary containing 'all' and 'accepted' metrics
        metric: The metric to plot
        targets: List of targets to find the corresponding target value
        save_path: Path to save the plot
        x_range: Optional tuple of (min, max) for x-axis limits
    """
    # Get metric values for all and accepted samples
    all_values = [metrics[metric] for metrics in all_metrics['all']]
    accepted_values = [metrics[metric] for metrics in all_metrics['accepted']]
    
    # Find target value for this metric
    target_value = next(t.value for t in targets if t.metric == metric)
    
    plt.figure(figsize=(8, 6))
    
    # Calculate common bins
    bins = np.histogram_bin_edges(all_values, bins=10)
    
    # Plot histograms with common bins
    plt.hist(all_values, bins=bins, alpha=0.3, density=True, 
            label='All samples', color='blue')
    plt.hist(accepted_values, bins=bins, alpha=0.3, density=True, 
            label='Accepted samples', color='orange')
    
    # Add KDE curves if enough samples
    if x_range is None:
        x_min, x_max = min(all_values), max(all_values)
    else:
        x_min, x_max = x_range
    x_plot = np.linspace(x_min, x_max, 200)
    
    if len(all_values) > 1:
        kde_all = stats.gaussian_kde(all_values)
        plt.plot(x_plot, kde_all(x_plot), color='blue', 
                linewidth=2, label='All samples (KDE)')
    
    if len(accepted_values) > 1:
        kde_accepted = stats.gaussian_kde(accepted_values)
        plt.plot(x_plot, kde_accepted(x_plot), color='orange', 
                linewidth=2, label='Accepted samples (KDE)')
    
    plt.axvline(target_value, color='r', linestyle='--', 
               label=f'Target {metric.value}')
    
    plt.xlabel(f'{metric.value}')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {metric.value}')
    if x_range:
        plt.xlim(x_range)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_parameter_distributions(accepted_params: List[Dict], pdf_results: Dict, save_path: str):
    """Plot both independent and joint parameter distributions
    Args:
        accepted_params: List of accepted parameter dictionaries
        pdf_results: Results from _estimate_pdfs
        save_path: Path to save the plot
    """
    param_names = pdf_results['param_names']
    n_params = len(param_names)
    
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    fig.suptitle('Parameter Distributions and Correlations')
    
    # Convert accepted params to array for easier plotting
    X = np.array([list(p.values()) for p in accepted_params])
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:  # Diagonal: show marginal distributions
                ax.hist(X[:, i], bins=20, density=True, alpha=0.6)
                kde = pdf_results['independent'][param_names[i]]
                x = np.linspace(X[:, i].min(), X[:, i].max(), 100)
                ax.plot(x, kde(x))
                if i == n_params - 1:
                    ax.set_xlabel(param_names[i])
                if j == 0:
                    ax.set_ylabel(param_names[i])
            else:  # Off-diagonal: show scatter plots
                ax.scatter(X[:, j], X[:, i], alpha=0.5, s=10)
                if i == n_params - 1:
                    ax.set_xlabel(param_names[j])
                if j == 0:
                    ax.set_ylabel(param_names[i])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
