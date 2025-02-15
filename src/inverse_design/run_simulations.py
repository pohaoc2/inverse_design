import subprocess
from pathlib import Path
import concurrent.futures
import logging
from tqdm import tqdm
import sys
import os


def run_single_simulation(input_file: Path, output_base_dir: Path, jar_path: str) -> bool:
    """Run a single ARCADE simulation

    Args:
        input_file: Path to input XML file
        output_base_dir: Base directory for outputs
        jar_path: Path to arcade_v3.jar

    Returns:
        bool: True if simulation completed successfully
    """
    output_dir = output_base_dir / input_file.stem
    output_dir.mkdir(exist_ok=True, parents=True)

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


def run_simulations(
    input_dir: str = "perturbed_inputs",
    output_dir: str = "simulation_results",
    jar_path: str = "arcade_v3.jar",
    max_workers: int = 4,
    start_index: int = 1,
) -> None:
    """Run ARCADE simulations for all input files in parallel

    Args:
        input_dir: Directory containing input XML files
        output_dir: Directory for simulation outputs
        jar_path: Path to arcade_v3.jar
        max_workers: Maximum number of parallel simulations
        start_index: Starting index for input files (default: 1, runs all files)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("simulation_run.log"), logging.StreamHandler(sys.stdout)],
    )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    jar_path = Path(jar_path)

    if not jar_path.exists():
        raise FileNotFoundError(f"ARCADE jar file not found at {jar_path}")

    input_files = sorted(input_dir.glob("*.xml"))
    if not input_files:
        raise FileNotFoundError(f"No XML files found in {input_dir}")

    start_file = f"input_{start_index}"
    input_files = [f for f in input_files if f.stem >= start_file]
    
    if not input_files:
        raise FileNotFoundError(f"No XML files found with index >= {start_index}")

    logging.info(f"Found {len(input_files)} input files starting from index {start_index}")
    logging.info(f"Running simulations with {max_workers} parallel workers")

    output_dir.mkdir(exist_ok=True, parents=True)

    successful = 0
    failed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(run_single_simulation, input_file, output_dir, jar_path): input_file
            for input_file in input_files
        }

        with tqdm(total=len(input_files), desc="Running simulations") as pbar:
            for future in concurrent.futures.as_completed(future_to_file):
                input_file = future_to_file[future]
                try:
                    if future.result():
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    logging.error(f"Simulation failed for {input_file.name}: {str(e)}")
                    failed += 1
                pbar.update(1)

    logging.info(f"Completed simulations: {successful} successful, {failed} failed")


if __name__ == "__main__":
    run_simulations(
        input_dir="inputs/perturbed_inputs",
        output_dir="ARCADE_OUTPUT/",
        jar_path="models/arcade-test-cycle.jar",
        max_workers=4,
        start_index=1,
    )
