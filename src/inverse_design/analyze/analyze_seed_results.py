import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from inverse_design.analyze.save_aggregated_results import SimulationMetrics
from inverse_design.analyze.analyze_aggregated_results import analyze_metric_percentiles
from typing import List
class SeedAnalyzer:
    def __init__(self, base_output_dir: str):
        """Initialize the seed analyzer.
        
        Args:
            base_output_dir: Base directory containing all simulation output folders
        """
        self.base_output_dir = Path(base_output_dir)
        self.sim_metrics = SimulationMetrics(base_output_dir)
    
    def analyze_seeds_for_folders(self, input_folders, t1: str, t2: str, timestamps: List[str]):
        """Analyze metrics across seeds for given input folders.
        
        Args:
            input_folders: List of input folder names to analyze
            t1: First timepoint
            t2: Second timepoint
            timestamps: List of all timestamps
            
        Returns:
            Dictionary containing metrics for each folder's seeds
        """
        all_seed_metrics = {}
        
        for folder_name in input_folders:
            folder_path = self.base_output_dir / folder_name
            
            # Load cell data
            cells_data_t1 = self.sim_metrics.cell_metrics.load_cells_data(folder_path, t1)
            cells_data_t2 = self.sim_metrics.cell_metrics.load_cells_data(folder_path, t2)
            time_difference = int(t2) - int(t1)
            
            # Process seeds for this parameter set
            metrics_by_exp = self.sim_metrics.process_parameter_seeds(
                cells_data_t1, cells_data_t2, time_difference
            )
            
            all_seed_metrics[folder_name] = metrics_by_exp
        return all_seed_metrics
    
    def plot_seed_comparisons(self, csv_file: str, metric_name: str, t1: str = "000000", 
                            t2: str = "010080", timestamps: List[str] = None):
        """Plot violin plots comparing seed distributions for top and bottom performers.
        
        Args:
            csv_file: Path to aggregated metrics CSV file
            metric_name: Name of metric to analyze
            t1: First timepoint
            t2: Second timepoint
            timestamps: List of timestamps for analysis
        """
        # Get top and bottom 10% folders
        top_10_folders, bottom_10_folders = analyze_metric_percentiles(
            csv_file, metric_name, verbose=False
        )
        
        # Analyze seeds for both groups
        if timestamps is None:
            timestamps = [t1, t2]
            
        top_metrics = self.analyze_seeds_for_folders(top_10_folders, t1, t2, timestamps)
        bottom_metrics = self.analyze_seeds_for_folders(bottom_10_folders, t1, t2, timestamps)
        
        print(f"number of top 10% folders: {len(top_metrics)}")
        print(f"number of bottom 10% folders: {len(bottom_metrics)}")
        if 0:
            for folder, metrics in top_metrics.items():
                print(f"folder: {folder}")
                for exp_key, exp_metrics in metrics.items():
                    for metric, values in exp_metrics.items():
                        print(metric, values)
                asd()
        # Prepare data for plotting
        plot_data = []
        metrics_to_plot = ['doubling_time', 'num_cells_t2']
        
        # Process top performers
        for folder, metrics in top_metrics.items():
            for exp_key, exp_metrics in metrics.items():
                for metric in metrics_to_plot:
                    for value in exp_metrics[metric]:
                        plot_data.append({
                            'Metric': metric,
                            'Value': value,
                            'Group': 'Top 10%',
                            'Folder': folder
                        })
        
        # Process bottom performers
        for folder, metrics in bottom_metrics.items():
            for exp_key, exp_metrics in metrics.items():
                for metric in metrics_to_plot:
                    for value in exp_metrics[metric]:
                        plot_data.append({
                            'Metric': metric,
                            'Value': value,
                            'Group': 'Bottom 10%',
                            'Folder': folder
                        })
        
        # Create plot
        df = pd.DataFrame(plot_data)
        print(df)
        plt.figure(figsize=(10, 10))
        
        for idx, metric in enumerate(metrics_to_plot):
            plt.subplot(1, len(metrics_to_plot), idx + 1)
            metric_data = df[df['Metric'] == metric]
            
            sns.violinplot(data=metric_data, x='Group', y='Value')
            sns.stripplot(data=metric_data, x='Group', y='Value', 
                         color='k', alpha=0.9, size=4, jitter=0.2)
            
            plt.title(f'{metric} Distribution')
            plt.xticks(rotation=45)
            
            # Add statistics
            top_vals = metric_data[metric_data['Group'] == 'Top 10%']['Value']
            bottom_vals = metric_data[metric_data['Group'] == 'Bottom 10%']['Value']
            
            stats_text = f"Top 10%:\nMean: {top_vals.mean():.2f}\nStd: {top_vals.std():.2f}\n\n"
            stats_text += f"Bottom 10%:\nMean: {bottom_vals.mean():.2f}\nStd: {bottom_vals.std():.2f}"
            
            plt.text(0.95, 0.95, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        plt.savefig(f'{self.base_output_dir}/seed_comparisons.png')
        plt.show()

if __name__ == "__main__":
    # Example usage
    base_dir = "ARCADE_OUTPUT/SMALL_STD_ONLY_VOLUME"
    base_dir = "ARCADE_OUTPUT/MANUAL_VOLUME_APOTOSIS"
    csv_file = f"{base_dir}/simulation_metrics.csv"
    
    analyzer = SeedAnalyzer(base_dir)
    analyzer.plot_seed_comparisons(
        csv_file=csv_file,
        metric_name="doub_time_std",
        t1="000000",
        t2="010080"
    )
