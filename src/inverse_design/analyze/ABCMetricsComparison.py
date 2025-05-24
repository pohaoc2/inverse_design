import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils.utils import remove_outliers


class ABCMetricsComparison:
    """
    Class to quantify differences between target metrics and simulation metrics
    from ABC posterior distribution sampling.
    """
    
    def __init__(self, target_metrics, simulation_metrics, metric_names=None):
        """
        Initialize with target and simulation metrics.
        
        Parameters:
        -----------
        target_metrics : dict or array-like
            Target metric values. If dict, keys should be metric names.
            If array-like, should be [doubling_time, std_doubling, symmetry, std_symmetry]
        simulation_metrics : array-like of shape (n_samples, n_metrics)
            Simulation metrics from posterior sampling
        metric_names : list, optional
            Names of metrics. Default: ['doubling_time', 'std_doubling', 'symmetry', 'std_symmetry']
        """
        
        if metric_names is None:
            self.metric_names = ['doubling_time', 'std_doubling', 'symmetry', 'std_symmetry']
        else:
            self.metric_names = metric_names
            
        # Convert target_metrics to array if it's a dict
        if isinstance(target_metrics, dict):
            self.target_metrics = np.array([target_metrics[name] for name in self.metric_names])
        else:
            self.target_metrics = np.array(target_metrics)
            
        self.simulation_metrics = np.array(simulation_metrics)
        
        # Ensure shapes are compatible
        if len(self.target_metrics) != self.simulation_metrics.shape[1]:
            raise ValueError("Number of target metrics must match number of simulation metric columns")
    
    def compute_distance_metrics(self):
        """
        Compute various distance metrics between target and simulation metrics.
        """
        n_samples = self.simulation_metrics.shape[0]
        results = {}
        
        # 1. Euclidean distance (raw)
        euclidean_distances = np.sqrt(np.sum((self.simulation_metrics - self.target_metrics)**2, axis=1))
        results['euclidean_raw'] = {
            'distances': euclidean_distances,
            'mean': np.mean(euclidean_distances),
            'std': np.std(euclidean_distances),
            'median': np.median(euclidean_distances)
        }
        
        # 2. Normalized Euclidean distance (by target values)
        target_normalized = self.target_metrics.copy()
        target_normalized[target_normalized == 0] = 1e-10  # Avoid division by zero
        normalized_diff = (self.simulation_metrics - self.target_metrics) / np.abs(target_normalized)
        euclidean_normalized = np.sqrt(np.sum(normalized_diff**2, axis=1))
        results['euclidean_normalized'] = {
            'distances': euclidean_normalized,
            'mean': np.mean(euclidean_normalized),
            'std': np.std(euclidean_normalized),
            'median': np.median(euclidean_normalized)
        }
        
        # 3. Standardized distance (z-score based)
        scaler = StandardScaler()
        # Fit on combined data to get proper scaling
        combined_data = np.vstack([self.target_metrics.reshape(1, -1), self.simulation_metrics])
        combined_scaled = scaler.fit_transform(combined_data)
        target_scaled = combined_scaled[0]
        sim_scaled = combined_scaled[1:]
        
        standardized_distances = np.sqrt(np.sum((sim_scaled - target_scaled)**2, axis=1))
        results['euclidean_standardized'] = {
            'distances': standardized_distances,
            'mean': np.mean(standardized_distances),
            'std': np.std(standardized_distances),
            'median': np.median(standardized_distances)
        }
        
        # 4. Manhattan distance (L1 norm)
        manhattan_distances = np.sum(np.abs(self.simulation_metrics - self.target_metrics), axis=1)
        results['manhattan_raw'] = {
            'distances': manhattan_distances,
            'mean': np.mean(manhattan_distances),
            'std': np.std(manhattan_distances),
            'median': np.median(manhattan_distances)
        }
        
        # 5. Relative percentage error for each metric
        relative_errors = np.abs((self.simulation_metrics - self.target_metrics) / 
                                np.maximum(np.abs(self.target_metrics), 1e-10)) * 100
        results['relative_errors'] = {
            'errors': relative_errors,
            'mean_per_metric': np.mean(relative_errors, axis=0),
            'std_per_metric': np.std(relative_errors, axis=0),
            'overall_mean': np.mean(relative_errors)
        }
        
        # 6. Mahalanobis-like distance (using simulation covariance)
        sim_cov = np.cov(self.simulation_metrics.T)
        try:
            sim_cov_inv = np.linalg.inv(sim_cov)
            mahal_distances = []
            for sim in self.simulation_metrics:
                diff = sim - self.target_metrics
                mahal_dist = np.sqrt(diff.T @ sim_cov_inv @ diff)
                mahal_distances.append(mahal_dist)
            mahal_distances = np.array(mahal_distances)
            results['mahalanobis'] = {
                'distances': mahal_distances,
                'mean': np.mean(mahal_distances),
                'std': np.std(mahal_distances),
                'median': np.median(mahal_distances)
            }
        except np.linalg.LinAlgError:
            results['mahalanobis'] = None
            print("Warning: Could not compute Mahalanobis distance (singular covariance matrix)")
        
        return results
    
    def compute_metric_specific_statistics(self):
        """
        Compute statistics for each individual metric.
        """
        stats_per_metric = {}
        
        for i, metric_name in enumerate(self.metric_names):
            target_val = self.target_metrics[i]
            sim_vals = self.simulation_metrics[:, i]
            stats_per_metric[metric_name] = {
                'target_value': target_val,
                'simulation_mean': np.mean(sim_vals),
                'simulation_std': np.std(sim_vals),
                'simulation_median': np.median(sim_vals),
                'bias': np.mean(sim_vals) - target_val,
                'mse': mean_squared_error([target_val] * len(sim_vals), sim_vals),
                'mae': mean_absolute_error([target_val] * len(sim_vals), sim_vals),
                'relative_bias_percent': ((np.mean(sim_vals) - target_val) / max(abs(target_val), 1e-10)) * 100,
                'coverage_95': np.percentile(sim_vals, [2.5, 97.5]),
                'target_in_95_interval': (np.percentile(sim_vals, 2.5) <= target_val <= np.percentile(sim_vals, 97.5))
            }
        
        return stats_per_metric
    
    def plot_comparison(self, figsize=(15, 10)):
        """
        Create comprehensive comparison plots.
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('ABC Posterior vs Target Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Individual metric comparisons
        for i, metric_name in enumerate(self.metric_names):
            row = i // 2
            col = i % 2
            ax = axes[row, col] if i < 4 else None
            
            if ax is not None:
                target_val = self.target_metrics[i]
                sim_vals = self.simulation_metrics[:, i]
                
                # Histogram of simulation values
                ax.hist(sim_vals, bins=30, alpha=0.7, density=True, 
                       color='lightblue', edgecolor='black', linewidth=0.5)
                
                # Target value line
                ax.axvline(target_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Target: {target_val:.3f}')
                
                # Simulation mean line
                sim_mean = np.mean(sim_vals)
                ax.axvline(sim_mean, color='blue', linestyle='-', linewidth=2, 
                          label=f'Sim Mean: {sim_mean:.3f}')
                
                ax.set_title(f'{metric_name.replace("_", " ").title()}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Distance distribution plot
        distances = self.compute_distance_metrics()
        ax = axes[1, 2]
        
        # Plot multiple distance metrics
        for dist_name, dist_data in distances.items():
            if dist_data is not None and 'distances' in dist_data:
                if dist_name in ['euclidean_normalized', 'euclidean_standardized']:
                    ax.hist(dist_data['distances'], bins=20, alpha=0.5, 
                           label=dist_name.replace('_', ' ').title(), density=True)
        
        ax.set_title('Distance Distributions')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report.
        """
        distance_results = self.compute_distance_metrics()
        metric_stats = self.compute_metric_specific_statistics()
        
        print("="*80)
        print("ABC POSTERIOR METRICS COMPARISON REPORT")
        print("="*80)
        
        print("\n1. OVERALL DISTANCE METRICS:")
        print("-" * 40)
        for dist_name, dist_data in distance_results.items():
            if dist_data is not None:
                if 'mean' in dist_data:
                    print(f"{dist_name.replace('_', ' ').title():.<25} "
                          f"Mean: {dist_data['mean']:.4f}, "
                          f"Std: {dist_data['std']:.4f}, "
                          f"Median: {dist_data['median']:.4f}")
        
        print("\n2. INDIVIDUAL METRIC ANALYSIS:")
        print("-" * 40)
        for metric_name, stats in metric_stats.items():
            print(f"\n{metric_name.replace('_', ' ').title()}:")
            print(f"  Target Value:           {stats['target_value']:.4f}")
            print(f"  Simulation Mean:        {stats['simulation_mean']:.4f}")
            print(f"  Simulation Std:         {stats['simulation_std']:.4f}")
            print(f"  Bias:                   {stats['bias']:.4f}")
            print(f"  Relative Bias (%):      {stats['relative_bias_percent']:.2f}%")
            print(f"  MSE:                    {stats['mse']:.6f}")
            print(f"  MAE:                    {stats['mae']:.4f}")
            print(f"  95% Coverage:           [{stats['coverage_95'][0]:.4f}, {stats['coverage_95'][1]:.4f}]")
            print(f"  Target in 95% Interval: {stats['target_in_95_interval']}")
        
        print("\n3. RELATIVE ERROR SUMMARY:")
        print("-" * 40)
        rel_errors = distance_results['relative_errors']
        for i, metric_name in enumerate(self.metric_names):
            print(f"{metric_name.replace('_', ' ').title():.<25} "
                  f"Mean Error: {rel_errors['mean_per_metric'][i]:.2f}%, "
                  f"Std Error: {rel_errors['std_per_metric'][i]:.2f}%")
        print(f"{'Overall Mean Relative Error':.<25} {rel_errors['overall_mean']:.2f}%")
        
        return distance_results, metric_stats

# Example usage:
if __name__ == "__main__":
    # Example data - replace with your actual data
    np.random.seed(42)
    
    # Target metrics: [doubling_time, std_doubling, symmetry, std_symmetry]
    metric_names = ["symmetry", "symmetry_std", "doub_time", "doub_time_std", "colony_growth"]
    target_metrics = {
        "symmetry": 0.806,
        "symmetry_std": 0.067,
        #"cycle_length": 30.0,
        "doub_time": 45.5,
        "doub_time_std": 13.79,
        #"n_cells": 100,
        #"act_ratio": 0.7,
        #"act_ratio_std": 0.067,
        #"activity": 0.5,
        "colony_growth": 18.3,
        #"vol": 5203.72
    }
    target_metrics = np.array([target_metrics[name] for name in metric_names])
    # Load posterior metrics from csv files
    parameter_base_folder = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N512_combined_grid_breast"
    for i in range(9, 10):
        posterior_metrics_file = f"{parameter_base_folder}/iter_{i}/final_metrics.csv"
        posterior_metrics_df = pd.read_csv(posterior_metrics_file)
        n_samples = posterior_metrics_df.shape[0]
        posterior_metrics_df = posterior_metrics_df[metric_names]
        posterior_metrics_df = posterior_metrics_df.replace([np.inf, -np.inf], np.nan).dropna()
        posterior_metrics_df = remove_outliers(posterior_metrics_df, 1.5)
        posterior_metrics_df = posterior_metrics_df.dropna()
        print(f"Removed {n_samples - posterior_metrics_df.shape[0]} outliers for {metric_names} in df")


        n_samples = posterior_metrics_df.shape[0]
        simulation_metrics = np.array([
            posterior_metrics_df[metric_names[0]],  # symmetry
            posterior_metrics_df[metric_names[1]],   # symmetry_std
            posterior_metrics_df[metric_names[2]], # doub_time
            posterior_metrics_df[metric_names[3]],  # doub_time_std
            posterior_metrics_df[metric_names[4]]  # colony_growth
        ]).T
    
        # Create comparison object
        comparator = ABCMetricsComparison(target_metrics, simulation_metrics, metric_names)
        
        # Generate report
        distance_results, metric_stats = comparator.generate_summary_report()
        
        # Create plots
        fig = comparator.plot_comparison()
        plt.show()
        
        # Access specific results
        print(f"\nBest fit sample index: {np.argmin(distance_results['euclidean_normalized']['distances'])}")
        print(f"Best fit distance: {np.min(distance_results['euclidean_normalized']['distances']):.4f}")