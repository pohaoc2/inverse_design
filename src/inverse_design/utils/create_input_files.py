import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import scipy.stats.qmc
from typing import Dict


def generate_perturbed_parameters(
    sobol_power,
    volume_mu_range=(1500, 3000),
    volume_sigma_range=(50, 150),
    apop_age_mu_range=(100000, 140000),
    apop_age_sigma_range=(5000, 15000),
    necrotic_fraction_range=(0.3, 0.7),
    accuracy_range=(0.6, 1.0),
    affinity_range=(0.3, 0.7),
    compression_tolerance_range=(0.0, 0.2),
    template_path="sample_input_v3.xml",
    output_dir="perturbed_inputs",
):
    # Define parameter ranges
    param_ranges = [
        volume_mu_range,
        volume_sigma_range,
        apop_age_mu_range,
        apop_age_sigma_range,
        necrotic_fraction_range,
        accuracy_range,
        affinity_range,
        compression_tolerance_range,
    ]

    # Initialize Sobol sequence generator
    sobol_engine = scipy.stats.qmc.Sobol(
        d=len(param_ranges),
        scramble=True,
    )

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
        for value, (min_val, max_val) in zip(sample, param_ranges):
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
        output_file = f"{output_dir}/input_{i+1}.xml"
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

        # Log parameters
        param_log.append(
            {
                "file_name": f"input_{i+1}.xml",
                "volume_mu": volume_mu,
                "volume_sigma": volume_sigma,
                "apop_age_mu": apop_age_mu,
                "apop_age_sigma": apop_age_sigma,
                "necrotic_fraction": necrotic_fraction,
                "accuracy": accuracy,
                "affinity": affinity,
                "compression_tolerance": compression_tolerance,
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
    Path(output_dir).mkdir(exist_ok=True)

    # Read template XML
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Prepare DataFrame for parameter logging
    param_log = []

    # Sample from each parameter's KDE
    kde_dict = parameter_pdfs["independent"]
    param_names = parameter_pdfs["param_names"]
    
    # Generate samples for each parameter
    sampled_params = {
        param: kde.resample(n_samples)[0]
        for param, kde in kde_dict.items()
    }

    # Generate XML files for each parameter set
    for i in range(n_samples):
        params = {
            param: sampled_params[param][i]
            for param in param_names
        }

        # Find cancerous population element
        cancerous_pop = root.find(".//population[@id='cancerous']")

        # Update parameters
        for param in cancerous_pop.findall("population.parameter"):
            param_id = param.get("id")
            if param_id == "CELL_VOLUME" and "volume_sigma" in params:
                param.set("value", f"NORMAL(MU=2250,SIGMA={max(0, params['volume_sigma']):.1f})")
            elif param_id == "APOPTOSIS_AGE" and "apop_age_sigma" in params:
                param.set("value", f"NORMAL(MU=120960,SIGMA={max(0, params['apop_age_sigma']):.1f})")
            elif param_id == "NECROTIC_FRACTION" and "necrotic_fraction" in params:
                param.set("value", f"{min(1, max(0, params['necrotic_fraction'])):.3f}")
            elif param_id == "ACCURACY" and "accuracy" in params:
                param.set("value", f"{min(1, max(0, params['accuracy'])):.3f}")
            elif param_id == "AFFINITY" and "affinity" in params:
                param.set("value", f"{min(1, max(0, params['affinity'])):.3f}")
            elif param_id == "COMPRESSION_TOLERANCE":
                param.set("value", "4.35")

        # Save modified XML
        output_file = f"{output_dir}/input_{i+1}.xml"
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
    output_dir = "inputs/stem_cell"
    generate_perturbed_parameters(
        sobol_power=8,
        volume_mu_range=(2250, 2250),
        volume_sigma_range=(50, 150),
        apop_age_mu_range=(120960, 120960),
        apop_age_sigma_range=(6000, 6000),
        necrotic_fraction_range=(0.0, 1.0),
        accuracy_range=(0.0, 1.0),
        affinity_range=(0.0, 1.0),
        compression_tolerance_range=(2.5, 6.5),
        output_dir=output_dir,
    )

if __name__ == "__main__":
    main()
