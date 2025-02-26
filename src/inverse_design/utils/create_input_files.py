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
            CELL_VOLUME_MU,
            CELL_VOLUME_SIGMA,
            APOPTOSIS_AGE_MU,
            APOPTOSIS_AGE_SIGMA,
            NECROTIC_FRACTION,
            ACCURACY,
            AFFINITY,
            COMPRESSION_TOLERANCE,
            SYNTHESIS_DURATION_MU,
            SYNTHESIS_DURATION_SIGMA,
            BASAL_ENERGY_MU,
            BASAL_ENERGY_SIGMA,
            PROLIFERATION_ENERGY_MU,
            PROLIFERATION_ENERGY_SIGMA,
            MIGRATION_ENERGY_MU,
            MIGRATION_ENERGY_SIGMA,
            METABOLIC_PREFERENCE_MU,
            METABOLIC_PREFERENCE_SIGMA,
            CONVERSION_FRACTION_MU,
            CONVERSION_FRACTION_SIGMA,
            RATIO_GLUCOSE_PYRUVATE_MU,
            RATIO_GLUCOSE_PYRUVATE_SIGMA,
            LACTATE_RATE_MU,
            LACTATE_RATE_SIGMA,
            AUTOPHAGY_RATE_MU,
            AUTOPHAGY_RATE_SIGMA,
            GLUCOSE_UPTAKE_RATE_MU,
            GLUCOSE_UPTAKE_RATE_SIGMA,
            ATP_PRODUCTION_RATE_MU,
            ATP_PRODUCTION_RATE_SIGMA,
            MIGRATORY_THRESHOLD_MU,
            MIGRATORY_THRESHOLD_SIGMA,
        ) = params

        # Create a dictionary for easier parameter access
        params = {
            'CELL_VOLUME_MU': CELL_VOLUME_MU,
            'CELL_VOLUME_SIGMA': CELL_VOLUME_SIGMA,
            'APOPTOSIS_AGE_MU': APOPTOSIS_AGE_MU,
            'APOPTOSIS_AGE_SIGMA': APOPTOSIS_AGE_SIGMA,
            'NECROTIC_FRACTION': NECROTIC_FRACTION,
            'ACCURACY': ACCURACY,
            'AFFINITY': AFFINITY,
            'COMPRESSION_TOLERANCE': COMPRESSION_TOLERANCE,
            'SYNTHESIS_DURATION_MU': SYNTHESIS_DURATION_MU,
            'SYNTHESIS_DURATION_SIGMA': SYNTHESIS_DURATION_SIGMA,
            'BASAL_ENERGY_MU': BASAL_ENERGY_MU,
            'BASAL_ENERGY_SIGMA': BASAL_ENERGY_SIGMA,
            'PROLIFERATION_ENERGY_MU': PROLIFERATION_ENERGY_MU,
            'PROLIFERATION_ENERGY_SIGMA': PROLIFERATION_ENERGY_SIGMA,
            'MIGRATION_ENERGY_MU': MIGRATION_ENERGY_MU,
            'MIGRATION_ENERGY_SIGMA': MIGRATION_ENERGY_SIGMA,
            'METABOLIC_PREFERENCE_MU': METABOLIC_PREFERENCE_MU,
            'METABOLIC_PREFERENCE_SIGMA': METABOLIC_PREFERENCE_SIGMA,
            'CONVERSION_FRACTION_MU': CONVERSION_FRACTION_MU,
            'CONVERSION_FRACTION_SIGMA': CONVERSION_FRACTION_SIGMA,
            'RATIO_GLUCOSE_PYRUVATE_MU': RATIO_GLUCOSE_PYRUVATE_MU,
            'RATIO_GLUCOSE_PYRUVATE_SIGMA': RATIO_GLUCOSE_PYRUVATE_SIGMA,
            'LACTATE_RATE_MU': LACTATE_RATE_MU,
            'LACTATE_RATE_SIGMA': LACTATE_RATE_SIGMA,
            'AUTOPHAGY_RATE_MU': AUTOPHAGY_RATE_MU,
            'AUTOPHAGY_RATE_SIGMA': AUTOPHAGY_RATE_SIGMA,
            'GLUCOSE_UPTAKE_RATE_MU': GLUCOSE_UPTAKE_RATE_MU,
            'GLUCOSE_UPTAKE_RATE_SIGMA': GLUCOSE_UPTAKE_RATE_SIGMA,
            'ATP_PRODUCTION_RATE_MU': ATP_PRODUCTION_RATE_MU,
            'ATP_PRODUCTION_RATE_SIGMA': ATP_PRODUCTION_RATE_SIGMA,
            'MIGRATORY_THRESHOLD_MU': MIGRATORY_THRESHOLD_MU,
            'MIGRATORY_THRESHOLD_SIGMA': MIGRATORY_THRESHOLD_SIGMA,
        }

        # Find cancerous population element
        cancerous_pop = root.find(".//population[@id='cancerous']")

        # Update parameters
        for param in cancerous_pop.findall("population.parameter"):
            param_id = param.get("id")
            
            # Handle existing parameters
            if param_id == "CELL_VOLUME":
                param.set("value", f"NORMAL(MU={params['CELL_VOLUME_MU']:.1f},SIGMA={max(0, params['CELL_VOLUME_SIGMA']):.1f})")
            elif param_id == "APOPTOSIS_AGE":
                param.set("value", f"NORMAL(MU={params['APOPTOSIS_AGE_MU']:.1f},SIGMA={max(0, params['APOPTOSIS_AGE_SIGMA']):.1f})")
            elif param_id == "NECROTIC_FRACTION":
                param.set("value", f"{NECROTIC_FRACTION:.3f}")
            elif param_id == "ACCURACY":
                param.set("value", f"{ACCURACY:.3f}")
            elif param_id == "AFFINITY":
                param.set("value", f"{AFFINITY:.3f}")
            elif param_id == "COMPRESSION_TOLERANCE":
                param.set("value", f"{COMPRESSION_TOLERANCE:.3f}")
            
            # Handle metabolism parameters
            elif param_id == "metabolism/BASAL_ENERGY":
                param.set("value", f"NORMAL(MU={params['BASAL_ENERGY_MU']:.6f},SIGMA={params['BASAL_ENERGY_SIGMA']:.6f})")
            elif param_id == "metabolism/PROLIFERATION_ENERGY":
                param.set("value", f"NORMAL(MU={params['PROLIFERATION_ENERGY_MU']:.6f},SIGMA={params['PROLIFERATION_ENERGY_SIGMA']:.6f})")
            elif param_id == "metabolism/MIGRATION_ENERGY":
                param.set("value", f"NORMAL(MU={params['MIGRATION_ENERGY_MU']:.6f},SIGMA={params['MIGRATION_ENERGY_SIGMA']:.6f})")
            elif param_id == "metabolism/METABOLIC_PREFERENCE":
                param.set("value", f"NORMAL(MU={params['METABOLIC_PREFERENCE_MU']:.3f},SIGMA={params['METABOLIC_PREFERENCE_SIGMA']:.3f})")
            elif param_id == "metabolism/CONVERSION_FRACTION":
                param.set("value", f"NORMAL(MU={params['CONVERSION_FRACTION_MU']:.3f},SIGMA={params['CONVERSION_FRACTION_SIGMA']:.3f})")
            elif param_id == "metabolism/RATIO_GLUCOSE_PYRUVATE":
                param.set("value", f"NORMAL(MU={params['RATIO_GLUCOSE_PYRUVATE_MU']:.3f},SIGMA={params['RATIO_GLUCOSE_PYRUVATE_SIGMA']:.3f})")
            elif param_id == "metabolism/LACTATE_RATE":
                param.set("value", f"NORMAL(MU={params['LACTATE_RATE_MU']:.3f},SIGMA={params['LACTATE_RATE_SIGMA']:.3f})")
            elif param_id == "metabolism/AUTOPHAGY_RATE":
                param.set("value", f"NORMAL(MU={params['AUTOPHAGY_RATE_MU']:.6f},SIGMA={params['AUTOPHAGY_RATE_SIGMA']:.6f})")
            elif param_id == "metabolism/GLUCOSE_UPTAKE_RATE":
                param.set("value", f"NORMAL(MU={params['GLUCOSE_UPTAKE_RATE_MU']:.3f},SIGMA={params['GLUCOSE_UPTAKE_RATE_SIGMA']:.3f})")
            elif param_id == "metabolism/ATP_PRODUCTION_RATE":
                param.set("value", f"NORMAL(MU={params['ATP_PRODUCTION_RATE_MU']:.3f},SIGMA={params['ATP_PRODUCTION_RATE_SIGMA']:.3f})")
            elif param_id == "signaling/MIGRATORY_THRESHOLD":
                param.set("value", f"NORMAL(MU={params['MIGRATORY_THRESHOLD_MU']:.3f},SIGMA={params['MIGRATORY_THRESHOLD_SIGMA']:.3f})")
            elif param_id == "proliferation/SYNTHESIS_DURATION":
                param.set("value", f"NORMAL(MU={params['SYNTHESIS_DURATION_MU']:.1f},SIGMA={params['SYNTHESIS_DURATION_SIGMA']:.1f})")

        # Save modified XML
        if not os.path.exists(f"{output_dir}/inputs"):
            os.makedirs(f"{output_dir}/inputs")
        output_file = f"{output_dir}/inputs/input_{i+1}.xml"
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

        # Log parameters
        param_log.append(
            {
                "file_name": f"input_{i+1}.xml",
                "CELL_VOLUME_MU": CELL_VOLUME_MU,
                "CELL_VOLUME_SIGMA": CELL_VOLUME_SIGMA,
                "APOPTOSIS_AGE_MU": APOPTOSIS_AGE_MU,
                "APOPTOSIS_AGE_SIGMA": APOPTOSIS_AGE_SIGMA,
                "NECROTIC_FRACTION": NECROTIC_FRACTION,
                "ACCURACY": ACCURACY,
                "AFFINITY": AFFINITY,
                "COMPRESSION_TOLERANCE": COMPRESSION_TOLERANCE,
                "SYNTHESIS_DURATION_MU": SYNTHESIS_DURATION_MU,
                "SYNTHESIS_DURATION_SIGMA": SYNTHESIS_DURATION_SIGMA,
                "BASAL_ENERGY_MU": BASAL_ENERGY_MU,
                "BASAL_ENERGY_SIGMA": BASAL_ENERGY_SIGMA,
                "PROLIFERATION_ENERGY_MU": PROLIFERATION_ENERGY_MU,
                "PROLIFERATION_ENERGY_SIGMA": PROLIFERATION_ENERGY_SIGMA,
                "MIGRATION_ENERGY_MU": MIGRATION_ENERGY_MU,
                "MIGRATION_ENERGY_SIGMA": MIGRATION_ENERGY_SIGMA,
                "METABOLIC_PREFERENCE_MU": METABOLIC_PREFERENCE_MU,
                "METABOLIC_PREFERENCE_SIGMA": METABOLIC_PREFERENCE_SIGMA,
                "CONVERSION_FRACTION_MU": CONVERSION_FRACTION_MU,
                "CONVERSION_FRACTION_SIGMA": CONVERSION_FRACTION_SIGMA,
                "RATIO_GLUCOSE_PYRUVATE_MU": RATIO_GLUCOSE_PYRUVATE_MU,
                "RATIO_GLUCOSE_PYRUVATE_SIGMA": RATIO_GLUCOSE_PYRUVATE_SIGMA,
                "LACTATE_RATE_MU": LACTATE_RATE_MU,
                "LACTATE_RATE_SIGMA": LACTATE_RATE_SIGMA,
                "AUTOPHAGY_RATE_MU": AUTOPHAGY_RATE_MU,
                "AUTOPHAGY_RATE_SIGMA": AUTOPHAGY_RATE_SIGMA,
                "GLUCOSE_UPTAKE_RATE_MU": GLUCOSE_UPTAKE_RATE_MU,
                "GLUCOSE_UPTAKE_RATE_SIGMA": GLUCOSE_UPTAKE_RATE_SIGMA,
                "ATP_PRODUCTION_RATE_MU": ATP_PRODUCTION_RATE_MU,
                "ATP_PRODUCTION_RATE_SIGMA": ATP_PRODUCTION_RATE_SIGMA,
                "MIGRATORY_THRESHOLD_MU": MIGRATORY_THRESHOLD_MU,
                "MIGRATORY_THRESHOLD_SIGMA": MIGRATORY_THRESHOLD_SIGMA,
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
            
            # Handle cellular properties
            if param_id == "CELL_VOLUME" and "CELL_VOLUME_MU" in params and "CELL_VOLUME_SIGMA" in params:
                param.set("value", f"NORMAL(MU={params['CELL_VOLUME_MU']:.1f},SIGMA={max(0, params['CELL_VOLUME_SIGMA']):.1f})")
            elif param_id == "APOPTOSIS_AGE" and "APOPTOSIS_AGE_MU" in params and "APOPTOSIS_AGE_SIGMA" in params:
                param.set("value", f"NORMAL(MU={params['APOPTOSIS_AGE_MU']:.1f},SIGMA={max(0, params['APOPTOSIS_AGE_SIGMA']):.1f})")
            elif param_id == "NECROTIC_FRACTION" and "NECROTIC_FRACTION" in params:
                param.set("value", f"{min(1, max(0, params['NECROTIC_FRACTION'])):.3f}")
            elif param_id == "ACCURACY" and "ACCURACY" in params:
                param.set("value", f"{min(1, max(0, params['ACCURACY'])):.3f}")
            elif param_id == "AFFINITY" and "AFFINITY" in params:
                param.set("value", f"{min(1, max(0, params['AFFINITY'])):.3f}")
            elif param_id == "COMPRESSION_TOLERANCE" and "COMPRESSION_TOLERANCE" in params:
                param.set("value", f"{params['COMPRESSION_TOLERANCE']:.3f}")
            
            # Handle proliferation parameters
            elif param_id == "proliferation/SYNTHESIS_DURATION":
                if "SYNTHESIS_DURATION_MU" in params and "SYNTHESIS_DURATION_SIGMA" in params:
                    param.set("value", f"NORMAL(MU={params['SYNTHESIS_DURATION_MU']:.1f},SIGMA={params['SYNTHESIS_DURATION_SIGMA']:.1f})")
            
            # Handle metabolism parameters
            elif param_id == "metabolism/BASAL_ENERGY":
                if "BASAL_ENERGY_MU" in params and "BASAL_ENERGY_SIGMA" in params:
                    param.set("value", f"NORMAL(MU={params['BASAL_ENERGY_MU']:.6f},SIGMA={params['BASAL_ENERGY_SIGMA']:.6f})")
            elif param_id == "metabolism/PROLIFERATION_ENERGY":
                if "PROLIFERATION_ENERGY_MU" in params and "PROLIFERATION_ENERGY_SIGMA" in params:
                    param.set("value", f"NORMAL(MU={params['PROLIFERATION_ENERGY_MU']:.6f},SIGMA={params['PROLIFERATION_ENERGY_SIGMA']:.6f})")
            elif param_id == "metabolism/MIGRATION_ENERGY":
                if "MIGRATION_ENERGY_MU" in params and "MIGRATION_ENERGY_SIGMA" in params:
                    param.set("value", f"NORMAL(MU={params['MIGRATION_ENERGY_MU']:.6f},SIGMA={params['MIGRATION_ENERGY_SIGMA']:.6f})")
            elif param_id == "metabolism/METABOLIC_PREFERENCE":
                if "METABOLIC_PREFERENCE_MU" in params and "METABOLIC_PREFERENCE_SIGMA" in params:
                    param.set("value", f"NORMAL(MU={params['METABOLIC_PREFERENCE_MU']:.3f},SIGMA={params['METABOLIC_PREFERENCE_SIGMA']:.3f})")
            elif param_id == "metabolism/CONVERSION_FRACTION":
                if "CONVERSION_FRACTION_MU" in params and "CONVERSION_FRACTION_SIGMA" in params:
                    param.set("value", f"NORMAL(MU={params['CONVERSION_FRACTION_MU']:.3f},SIGMA={params['CONVERSION_FRACTION_SIGMA']:.3f})")
            elif param_id == "metabolism/RATIO_GLUCOSE_PYRUVATE":
                if "RATIO_GLUCOSE_PYRUVATE_MU" in params and "RATIO_GLUCOSE_PYRUVATE_SIGMA" in params:
                    param.set("value", f"NORMAL(MU={params['RATIO_GLUCOSE_PYRUVATE_MU']:.3f},SIGMA={params['RATIO_GLUCOSE_PYRUVATE_SIGMA']:.3f})")
            elif param_id == "metabolism/LACTATE_RATE":
                if "LACTATE_RATE_MU" in params and "LACTATE_RATE_SIGMA" in params:
                    param.set("value", f"NORMAL(MU={params['LACTATE_RATE_MU']:.3f},SIGMA={params['LACTATE_RATE_SIGMA']:.3f})")
            elif param_id == "metabolism/AUTOPHAGY_RATE":
                if "AUTOPHAGY_RATE_MU" in params and "AUTOPHAGY_RATE_SIGMA" in params:
                    param.set("value", f"NORMAL(MU={params['AUTOPHAGY_RATE_MU']:.6f},SIGMA={params['AUTOPHAGY_RATE_SIGMA']:.6f})")
            elif param_id == "metabolism/GLUCOSE_UPTAKE_RATE":
                if "GLUCOSE_UPTAKE_RATE_MU" in params and "GLUCOSE_UPTAKE_RATE_SIGMA" in params:
                    param.set("value", f"NORMAL(MU={params['GLUCOSE_UPTAKE_RATE_MU']:.3f},SIGMA={params['GLUCOSE_UPTAKE_RATE_SIGMA']:.3f})")
            elif param_id == "metabolism/ATP_PRODUCTION_RATE":
                if "ATP_PRODUCTION_RATE_MU" in params and "ATP_PRODUCTION_RATE_SIGMA" in params:
                    param.set("value", f"NORMAL(MU={params['ATP_PRODUCTION_RATE_MU']:.3f},SIGMA={params['ATP_PRODUCTION_RATE_SIGMA']:.3f})")
            
            # Handle signaling parameters
            elif param_id == "signaling/MIGRATORY_THRESHOLD":
                if "MIGRATORY_THRESHOLD_MU" in params and "MIGRATORY_THRESHOLD_SIGMA" in params:
                    param.set("value", f"NORMAL(MU={params['MIGRATORY_THRESHOLD_MU']:.3f},SIGMA={params['MIGRATORY_THRESHOLD_SIGMA']:.3f})")

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


def main():
    output_dir = "inputs/STEM_CELL/meta_signal_heterogeneity"
    param_ranges = {
        "CELL_VOLUME_MU": (2000, 2500),
        "CELL_VOLUME_SIGMA": (50, 250),
        "APOPTOSIS_AGE_MU": (120960, 120960),
        "APOPTOSIS_AGE_SIGMA": (6000, 6000),
        "NECROTIC_FRACTION": (1.0, 1.0),
        "ACCURACY": (0.3, 1.0),
        "AFFINITY": (0.0, 1.0),
        "COMPRESSION_TOLERANCE": (3, 10),
        
        "SYNTHESIS_DURATION_MU": (580, 680),
        "SYNTHESIS_DURATION_SIGMA": (20, 70),
        "BASAL_ENERGY_MU": (0.0008, 0.0012),
        "BASAL_ENERGY_SIGMA": (0.00006, 0.0001),
        "PROLIFERATION_ENERGY_MU": (0.0008, 0.0012),
        "PROLIFERATION_ENERGY_SIGMA": (0.00006, 0.0001),
        "MIGRATION_ENERGY_MU": (0.00016, 0.00024),
        "MIGRATION_ENERGY_SIGMA": (0.000012, 0.00002),
        "METABOLIC_PREFERENCE_MU": (0.24, 0.36),
        "METABOLIC_PREFERENCE_SIGMA": (0.019, 0.029),
        "CONVERSION_FRACTION_MU": (0.2, 0.3),
        "CONVERSION_FRACTION_SIGMA": (0.016, 0.024),
        "RATIO_GLUCOSE_PYRUVATE_MU": (0.4, 0.6),
        "RATIO_GLUCOSE_PYRUVATE_SIGMA": (0.032, 0.048),
        "LACTATE_RATE_MU": (0.08, 0.12),
        "LACTATE_RATE_SIGMA": (0.006, 0.01),
        "AUTOPHAGY_RATE_MU": (0.00008, 0.00012),
        "AUTOPHAGY_RATE_SIGMA": (0.000006, 0.00001),
        "GLUCOSE_UPTAKE_RATE_MU": (0.9, 1.34),
        "GLUCOSE_UPTAKE_RATE_SIGMA": (0.072, 0.107),
        "ATP_PRODUCTION_RATE_MU": (7.14, 10.71),
        "ATP_PRODUCTION_RATE_SIGMA": (0.57, 0.86),
        #"MIGRATORY_THRESHOLD_MU": (8, 12),
        #"MIGRATORY_THRESHOLD_SIGMA": (0.64, 0.96),
    }
    
    generate_perturbed_parameters(
        sobol_power=9,
        param_ranges=param_ranges,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
