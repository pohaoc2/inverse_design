import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import scipy.stats.qmc
from typing import Dict
import os


def save_parameter_ranges(param_ranges: dict, output_dir: str):
    """Save the parameter ranges used for generating inputs to a CSV file."""
    # Create DataFrame with min and max values
    ranges_df = pd.DataFrame(
        {
            "parameter": param_ranges.keys(),
            "min_value": [r[0] for r in param_ranges.values()],
            "max_value": [r[1] for r in param_ranges.values()],
        }
    )

    # Save to CSV
    ranges_df.to_csv(f"{output_dir}/parameter_ranges.csv", index=False)


def generate_perturbed_parameters(
    sobol_power: int,
    param_ranges: dict,
    template_path: str = "sample_input_v3.xml",
    output_dir: str = "perturbed_inputs",
    seed: int = 42,
):
    """Generate perturbed parameter sets using Sobol sampling

    Args:
        sobol_power: Power of 2 for number of samples (n_samples = 2^sobol_power)
        param_ranges: Dictionary of parameter ranges {param_name: (min, max)}
        template_path: Path to template XML file
        output_dir: Directory to save generated XML files
        seed: Random seed for reproducibility
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save parameter ranges to CSV
    save_parameter_ranges(param_ranges, output_dir)

    # Define parameter ranges
    param_ranges_list = list(param_ranges.values())

    # Initialize Sobol sequence generator with random seed
    sobol_engine = scipy.stats.qmc.Sobol(d=len(param_ranges_list), scramble=True, seed=seed)

    # Generate samples in [0, 1] space
    samples = sobol_engine.random_base2(m=sobol_power)
    n_samples = len(samples)

    print(f"Generating {n_samples} samples...")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Read template XML
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Prepare DataFrame for parameter logging
    param_log = []

    # Generate XML files for each parameter set
    for i, sample in enumerate(samples):
        # Scale sample to parameter ranges
        params = []
        for value, (min_val, max_val) in zip(sample, param_ranges_list):
            scaled_value = min_val + (max_val - min_val) * value
            params.append(scaled_value)

        (
            volume_mu,
            volume_sigma,
            apop_age_mu,
            apop_age_sigma,
            necrotic_fraction,
            accuracy,
            affinity,
            compression_tolerance,
        ) = params

        # Find cancerous population element
        cancerous_pop = root.find(".//population[@id='cancerous']")

        # Update parameters
        for param in cancerous_pop.findall("population.parameter"):
            if param.get("id") == "CELL_VOLUME":
                param.set("value", f"NORMAL(MU={volume_mu:.1f},SIGMA={volume_sigma:.1f})")
            elif param.get("id") == "APOPTOSIS_AGE":
                param.set("value", f"NORMAL(MU={apop_age_mu:.1f},SIGMA={apop_age_sigma:.1f})")
            elif param.get("id") == "NECROTIC_FRACTION":
                param.set("value", f"{necrotic_fraction:.3f}")
            elif param.get("id") == "ACCURACY":
                param.set("value", f"{accuracy:.3f}")
            elif param.get("id") == "AFFINITY":
                param.set("value", f"{affinity:.3f}")
            elif param.get("id") == "COMPRESSION_TOLERANCE":
                param.set("value", f"{compression_tolerance:.3f}")

        # Save modified XML
        if not os.path.exists(f"{output_dir}/inputs"):
            os.makedirs(f"{output_dir}/inputs")
        output_file = f"{output_dir}/inputs/input_{i+1}.xml"
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

        # Log parameters
        param_log.append(
            {
                "file_name": f"input_{i+1}.xml",
                "CELL_VOLUME_MU": volume_mu,
                "CELL_VOLUME_SIGMA": volume_sigma,
                "APOPTOSIS_AGE_MU": apop_age_mu,
                "APOPTOSIS_AGE_SIGMA": apop_age_sigma,
                "NECROTIC_FRACTION": necrotic_fraction,
                "ACCURACY": accuracy,
                "AFFINITY": affinity,
                "COMPRESSION_TOLERANCE": compression_tolerance,
            }
        )

    # Save parameters to Excel
    df = pd.DataFrame(param_log)
    df.to_csv(f"{output_dir}/parameter_log.csv", index=False)

    print(f"Generated {n_samples} XML files and parameter log in {output_dir}/")


def generate_parameters_from_kde(
    parameter_pdfs: Dict,
    n_samples: int,
    output_dir: str = "kde_sampled_inputs",
    template_path: str = "sample_input_v3.xml",
):
    """Generate parameter sets by sampling from kernel density estimates (KDE)

    Args:
        parameter_pdfs: Dictionary containing KDE objects for each parameter
        n_samples: Number of parameter sets to generate
        output_dir: Directory to save generated XML files
        template_path: Path to template XML file
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read template XML
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Prepare DataFrame for parameter logging
    param_log = []

    # Sample from each parameter's KDE
    kde_dict = parameter_pdfs["independent"]
    param_names = parameter_pdfs["param_names"]

    # Generate samples for each parameter
    sampled_params = {param: kde.resample(n_samples)[0] for param, kde in kde_dict.items()}

    # Generate XML files for each parameter set
    for i in range(n_samples):
        params = {param: sampled_params[param][i] for param in param_names}

        # Find cancerous population element
        cancerous_pop = root.find(".//population[@id='cancerous']")

        # Update parameters
        for param in cancerous_pop.findall("population.parameter"):
            param_id = param.get("id")
            if param_id == "CELL_VOLUME" and "CELL_VOLUME_SIGMA" in params:
                param.set(
                    "value", f"NORMAL(MU=2250,SIGMA={max(0, params['CELL_VOLUME_SIGMA']):.1f})"
                )
            elif param_id == "APOPTOSIS_AGE" and "APOPTOSIS_AGE_SIGMA" in params:
                param.set(
                    "value", f"NORMAL(MU=120960,SIGMA={max(0, params['APOPTOSIS_AGE_SIGMA']):.1f})"
                )
            elif param_id == "NECROTIC_FRACTION" and "NECROTIC_FRACTION" in params:
                param.set("value", f"{min(1, max(0, params['NECROTIC_FRACTION'])):.3f}")
            elif param_id == "ACCURACY" and "ACCURACY" in params:
                param.set("value", f"{min(1, max(0, params['ACCURACY'])):.3f}")
            elif param_id == "AFFINITY" and "AFFINITY" in params:
                param.set("value", f"{min(1, max(0, params['AFFINITY'])):.3f}")
            elif param_id == "COMPRESSION_TOLERANCE" and "COMPRESSION_TOLERANCE" in params:
                param.set("value", f"{params['COMPRESSION_TOLERANCE']:.3f}")

        # Save modified XML
        if not os.path.exists(f"{output_dir}/inputs"):
            os.makedirs(f"{output_dir}/inputs")
        output_file = f"{output_dir}/inputs/input_{i+1}.xml"
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

        # Log parameters
        params["file_name"] = f"input_{i+1}.xml"
        param_log.append(params)

    # Save parameters to CSV
    df = pd.DataFrame(param_log)
    df.to_csv(f"{output_dir}/kde_sampled_parameters_log.csv", index=False)

    print(f"Generated {n_samples} XML files and parameter log in {output_dir}/")


# Example usage


def main():
    output_dir = "inputs/STEM_CELL/stem_cell_longer"
    param_ranges = {
        "CELL_VOLUME_MU": (1750, 2650),
        "CELL_VOLUME_SIGMA": (50, 250),
        "APOPTOSIS_AGE_MU": (120960, 120960),
        "APOPTOSIS_AGE_SIGMA": (6000, 6000),
        "NECROTIC_FRACTION": (1.0, 1.0),
        "ACCURACY": (0.0, 1.0),
        "AFFINITY": (0.0, 1.0),
        "COMPRESSION_TOLERANCE": (4, 8),
    }
    generate_perturbed_parameters(
        sobol_power=8,
        param_ranges=param_ranges,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
