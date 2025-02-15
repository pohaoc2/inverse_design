import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import scipy.stats.qmc


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


# Example usage
if __name__ == "__main__":
    generate_perturbed_parameters(
        sobol_power=8,
        volume_mu_range=(2250, 2250),
        volume_sigma_range=(50, 500),
        apop_age_mu_range=(120960, 120960),
        apop_age_sigma_range=(5040, 20160),
        necrotic_fraction_range=(0.2, 0.8),
        accuracy_range=(0.5, 1.0),
        affinity_range=(0.3, 0.7),
        compression_tolerance_range=(4.35, 4.35),
    )
