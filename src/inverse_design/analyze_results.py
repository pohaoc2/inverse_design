from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
import re
import numpy as np

class SimulationMetrics:
    def __init__(self, base_output_dir: str):
        """Initialize the metrics calculator
        
        Args:
            base_output_dir: Base directory containing all simulation output folders
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = logging.getLogger(__name__)
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """Parse filename to extract experiment info
        
        Args:
            filename: Format <exp_group>_<exp_name>_<seed>_<timestamp>.CELLS.json
            
        Returns:
            Dictionary containing parsed components
        """
        pattern = r"(.+?)_(.+?)_(\d+)_(\d+)\.CELLS\.json"
        match = re.match(pattern, filename)
        if match:
            return {
                'exp_group': match.group(1),
                'exp_name': match.group(2),
                'seed': match.group(3),
                'timestamp': match.group(4)
            }
        return {}
        
    def load_cells_data(self, folder_path: Path, timestamp: str) -> List[Tuple[Dict[str, str], List[Dict[str, Any]]]]:
        """Load cell data from a specific timestamp for all seeds
        
        Args:
            folder_path: Path to simulation folder
            timestamp: Timestamp string (e.g., '000000' or '000720')
            
        Returns:
            List of tuples (file_info, cell_data)
        """
        results = []
        try:
            # Find all files matching the timestamp
            file_pattern = f"*_{timestamp}.CELLS.json"
            cell_files = folder_path.glob(file_pattern)
            
            for cell_file in cell_files:
                file_info = self.parse_filename(cell_file.name)
                if file_info:  # Only process if filename matches expected pattern
                    with open(cell_file, 'r') as f:
                        cell_data = json.load(f)
                        results.append((file_info, cell_data))
                        
        except Exception as e:
            self.logger.error(f"Error loading cells data from {folder_path}: {str(e)}")
        
        return results

    def calculate_growth_rate(self, 
                            cells_t1: List[Dict[str, Any]], 
                            cells_t2: List[Dict[str, Any]], 
                            time_difference: float) -> float:
        """Calculate growth rate between two timepoints"""
        if not cells_t1 or not cells_t2:
            return 0.0
        
        n1 = len(cells_t1)
        n2 = len(cells_t2)
        return (n2 - n1) / time_difference

    def calculate_average_volume(self, cells: List[Dict[str, Any]]) -> float:
        """Calculate average cell volume"""
        if not cells:
            return 0.0
        
        volumes = [cell['volume'] for cell in cells]
        return sum(volumes) / len(volumes)

    def analyze_simulation(self, 
                         folder_path: Path, 
                         t1: str, 
                         t2: str, 
                         time_difference: float) -> Dict[str, float]:
        """Analyze a single simulation folder with multiple seeds"""
        # Load data for all seeds at both timepoints
        cells_data_t1 = self.load_cells_data(folder_path, t1)
        cells_data_t2 = self.load_cells_data(folder_path, t2)
        
        # Group metrics by experiment group and name
        metrics_by_exp = {}
        
        # Process each seed's data
        for (info_t1, cells_t1), (info_t2, cells_t2) in zip(cells_data_t1, cells_data_t2):
            if info_t1['exp_group'] != info_t2['exp_group'] or info_t1['exp_name'] != info_t2['exp_name']:
                continue
                
            exp_key = (info_t1['exp_group'], info_t1['exp_name'])
            if exp_key not in metrics_by_exp:
                metrics_by_exp[exp_key] = {
                    'growth_rates': [],
                    'avg_volumes_t1': [],
                    'avg_volumes_t2': [],
                    'num_cells_t1': [],
                    'num_cells_t2': [],
                    'seed_count': 0
                }
            
            # Calculate metrics for this seed
            metrics_by_exp[exp_key]['growth_rates'].append(
                self.calculate_growth_rate(cells_t1, cells_t2, time_difference)
            )
            metrics_by_exp[exp_key]['avg_volumes_t1'].append(
                self.calculate_average_volume(cells_t1)
            )
            metrics_by_exp[exp_key]['avg_volumes_t2'].append(
                self.calculate_average_volume(cells_t2)
            )
            metrics_by_exp[exp_key]['num_cells_t1'].append(len(cells_t1))
            metrics_by_exp[exp_key]['num_cells_t2'].append(len(cells_t2))
            metrics_by_exp[exp_key]['seed_count'] += 1
        
        # Average metrics across seeds
        averaged_metrics = {}
        for (exp_group, exp_name), metrics in metrics_by_exp.items():
            averaged_metrics.update({
                'exp_group': exp_group,
                'exp_name': exp_name,
                'growth_rate': sum(metrics['growth_rates']) / len(metrics['growth_rates']),
                'growth_rate_std': np.std(metrics['growth_rates']),
                'avg_volume_t1': sum(metrics['avg_volumes_t1']) / len(metrics['avg_volumes_t1']),
                'avg_volume_t1_std': np.std(metrics['avg_volumes_t1']),
                'avg_volume_t2': sum(metrics['avg_volumes_t2']) / len(metrics['avg_volumes_t2']),
                'avg_volume_t2_std': np.std(metrics['avg_volumes_t2']),
                'num_cells_t1': sum(metrics['num_cells_t1']) / len(metrics['num_cells_t1']),
                'num_cells_t2': sum(metrics['num_cells_t2']) / len(metrics['num_cells_t2']),
                'seed_count': metrics['seed_count']
            })
        
        return averaged_metrics

    def analyze_all_simulations(self, 
                              t1: str = '000000', 
                              t2: str = '000720', 
                              time_difference: float = 720.0) -> pd.DataFrame:
        """Analyze all simulation folders and compile results"""
        results = []
        
        # Find all simulation folders
        sim_folders = [f for f in self.base_output_dir.glob("input_*")]
        
        for folder in sim_folders:
            try:
                metrics = self.analyze_simulation(folder, t1, t2, time_difference)
                metrics['simulation'] = folder.name
                results.append(metrics)
            except Exception as e:
                self.logger.error(f"Error processing {folder}: {str(e)}")
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(results)
        output_file = self.base_output_dir / "simulation_metrics.csv"
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Saved metrics for {len(results)} simulations to {output_file}")
        return df

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    metrics_calculator = SimulationMetrics("ARCADE_OUTPUT/")
    df = metrics_calculator.analyze_all_simulations(
        t1='000000',
        t2='000720',
        time_difference=720.0
    )
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

if __name__ == "__main__":
    main() 