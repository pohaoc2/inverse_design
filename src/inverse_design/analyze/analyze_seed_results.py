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
            exp_name = list(metrics_by_exp.keys())[0]
            all_seed_metrics[folder_name] = metrics_by_exp[exp_name]
        return all_seed_metrics
    
    def plot_seed_comparisons(self, csv_file: str, metric_name: str, percentile: float = 10, t1: str = "000000", 
                            t2: str = "010080", timestamps: List[str] = None, metrics_to_plot: List[str] = None):
        """Plot violin plots comparing seed distributions for top and bottom performers.
        
        Args:
            csv_file: Path to aggregated metrics CSV file
            metric_name: Name of metric to analyze
            t1: First timepoint
            t2: Second timepoint
            timestamps: List of timestamps for analysis
        """
        # Get top and bottom folders
        top_n_input_file, bottom_n_input_file, _ = analyze_metric_percentiles(
            csv_file, metric_name, percentile, verbose=False
        )
        
        # Get seed metrics for each group
        top_seed_metrics = self.analyze_seeds_for_folders(top_n_input_file, t1, t2, timestamps)
        bottom_seed_metrics = self.analyze_seeds_for_folders(bottom_n_input_file, t1, t2, timestamps)
        
        if timestamps is None:
            timestamps = [t1, t2]
            
        # Create plot data from seed metrics
        plot_data = []
        for group, seed_metrics in [('top', top_seed_metrics), ('bottom', bottom_seed_metrics)]:
            for folder, folder_metrics in seed_metrics.items():
                for metric in metrics_to_plot:
                    values = folder_metrics[metric]
                    for value in values:
                        plot_data.append({
                            'metric': metric,
                            'value': value,
                            'group': group,
                            'folder': folder
                        })
        
        # Convert to DataFrame
        plot_df = pd.DataFrame(plot_data)
        
        print(f"number of top {percentile}% folders: {len(top_n_input_file)}")
        print(f"number of bottom {percentile}% folders: {len(bottom_n_input_file)}")
        
        # Create plots
        plt.figure(figsize=(4*len(metrics_to_plot), 8))
        
        for idx, metric in enumerate(metrics_to_plot):
            plt.subplot(1, len(metrics_to_plot), idx + 1)
            metric_data = plot_df[plot_df['metric'] == metric]
            
            # Create violin plot
            sns.violinplot(data=metric_data, 
                         x='group', 
                         y='value',
                         order=['top', 'bottom'])
            
            # Add individual points
            sns.stripplot(data=metric_data, 
                         x='group', 
                         y='value',
                         order=['top', 'bottom'],
                         color='black',
                         alpha=0.8,
                         size=3,
                         jitter=0.2)
            
            plt.title(f'{metric} Distribution')
            plt.xticks(rotation=45)
            
            # Add statistics
            top_vals = metric_data[metric_data['group'] == 'top']['value']
            bottom_vals = metric_data[metric_data['group'] == 'bottom']['value']
            
            stats_text = f"Top {percentile}%:\nMean: {top_vals.mean():.2f}\nStd: {top_vals.std():.2f}\n\n"
            stats_text += f"Bottom {percentile}%:\nMean: {bottom_vals.mean():.2f}\nStd: {bottom_vals.std():.2f}"
            
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
    base_dir = "ARCADE_OUTPUT/STEM_CELL"
    csv_file = f"{base_dir}/simulation_metrics.csv"
    percentile = 10
    metrics_to_plot = ['doub_time', 'act_t2', 'n_cells_t2']
    analyzer = SeedAnalyzer(base_dir)
    analyzer.plot_seed_comparisons(
        csv_file=csv_file,
        metric_name="doub_time_std",
        percentile=percentile,
        metrics_to_plot=metrics_to_plot,
        t1="000000",
        t2="010080"
    )
