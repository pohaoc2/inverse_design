from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Any
import logging

class SimulationMetrics:
    def __init__(self, base_output_dir: str):
        """Initialize the metrics calculator
        
        Args:
            base_output_dir: Base directory containing all simulation output folders
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = logging.getLogger(__name__)
        
    def load_cells_data(self, folder_path: Path, timestamp: str) -> List[Dict[str, Any]]:
        """Load cell data from a specific timestamp
        
        Args:
            folder_path: Path to simulation folder
            timestamp: Timestamp string (e.g., '000000' or '000720')
            
        Returns:
            List of cell dictionaries
        """
        try:
            file_pattern = f"test_uniform_0000_{timestamp}.CELLS.json"
            cell_file = next(folder_path.glob(file_pattern))
            with open(cell_file, 'r') as f:
                return json.load(f)
        except (StopIteration, FileNotFoundError):
            self.logger.error(f"Could not find cell data for timestamp {timestamp} in {folder_path}")
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Error parsing JSON for timestamp {timestamp} in {folder_path}")
            return []

    def calculate_growth_rate(self, 
                            cells_t1: List[Dict[str, Any]], 
                            cells_t2: List[Dict[str, Any]], 
                            time_difference: float) -> float:
        """Calculate growth rate between two timepoints
        
        Args:
            cells_t1: Cell data at first timepoint
            cells_t2: Cell data at second timepoint
            time_difference: Time difference between points
            
        Returns:
            Growth rate (cells/time)
        """
        if not cells_t1 or not cells_t2:
            return 0.0
        
        n1 = len(cells_t1)
        n2 = len(cells_t2)
        return (n2 - n1) / time_difference

    def calculate_average_volume(self, cells: List[Dict[str, Any]]) -> float:
        """Calculate average cell volume
        
        Args:
            cells: Cell data
            
        Returns:
            Average volume
        """
        if not cells:
            return 0.0
        
        volumes = [cell['volume'] for cell in cells]
        return sum(volumes) / len(volumes)

    def analyze_simulation(self, 
                         folder_path: Path, 
                         t1: str, 
                         t2: str, 
                         time_difference: float) -> Dict[str, float]:
        """Analyze a single simulation folder
        
        Args:
            folder_path: Path to simulation folder
            t1: First timestamp
            t2: Second timestamp
            time_difference: Time difference between points
            
        Returns:
            Dictionary of metrics
        """
        cells_t1 = self.load_cells_data(folder_path, t1)
        cells_t2 = self.load_cells_data(folder_path, t2)
        
        metrics = {
            'growth_rate': self.calculate_growth_rate(cells_t1, cells_t2, time_difference),
            'avg_volume_t1': self.calculate_average_volume(cells_t1),
            'avg_volume_t2': self.calculate_average_volume(cells_t2),
            'num_cells_t1': len(cells_t1),
            'num_cells_t2': len(cells_t2)
        }
        
        return metrics

    def analyze_all_simulations(self, 
                              t1: str = '000000', 
                              t2: str = '000720', 
                              time_difference: float = 720.0) -> pd.DataFrame:
        """Analyze all simulation folders and compile results
        
        Args:
            t1: First timestamp
            t2: Second timestamp
            time_difference: Time difference between points
            
        Returns:
            DataFrame containing all metrics for all simulations
        """
        results = []
        
        # Find all simulation folders
        sim_folders = [f for f in self.base_output_dir.glob("input_*")]
        
        for folder in sim_folders:
            print(folder)
            try:
                metrics = self.analyze_simulation(folder, t1, t2, time_difference)
                metrics = {'simulation': folder.name, **metrics}
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