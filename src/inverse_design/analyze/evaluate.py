import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from inverse_design.metrics.metrics import MetricsFactory


def estimate_pdfs(params: List[Dict]) -> Dict:
    """
    Estimate PDFs for each parameter independently and jointly using multivariate normal
    Args:
        params: List of parameter dictionaries
    Returns:
        Dictionary containing:
            'independent': Dict mapping parameter names to their KDE objects
            'joint_mean': Mean vector of parameters
            'joint_cov': Covariance matrix of parameters
            'param_names': List of parameter names
    """
    if not params:
        return {}

    # Extract parameter values into arrays
    param_names = list(params[0].keys())
    param_arrays = {param: np.array([p[param] for p in params]) for param in param_names}
    # Compute independent KDEs for each parameter
    param_pdfs = {}
    for param_name, values in param_arrays.items():
        try:
            kde = stats.gaussian_kde(values)
            param_pdfs[param_name] = kde
        except Exception as e:
            print(f"Error computing KDE for {param_name}: {e}")
            param_pdfs[param_name] = None

    # Compute joint distribution parameters
    X = np.array([list(p.values()) for p in params])
    joint_mean = np.mean(X, axis=0)
    joint_cov = np.cov(X, rowvar=False)

    return {
        "independent": param_pdfs,
        "joint_mean": joint_mean,
        "joint_cov": joint_cov,
        "param_names": param_names,
    }


def evaluate_parameters(abc_instance, parameters: Dict[str, float]) -> Tuple[Dict, float]:
    """Helper function to evaluate a single parameter set
    Args:
        abc_instance: ABC instance for accessing config and metric calculations
        parameters: Dictionary of parameter names and values
    Returns:
        Tuple of (metrics, distance)
    """
    from inverse_design.models.bdm.bdm import BDM

    config = abc_instance.model_config.copy()
    for param_name, value in parameters.items():
        setattr(config.rates, param_name, value)
    model = BDM(config)
    model_output = model.step()
    metrics_calculator = MetricsFactory.create_metrics(
        abc_instance.model_type,
        model_output,
    )
    metrics = abc_instance.calculate_all_metrics(metrics_calculator=metrics_calculator)
    distance = abc_instance.calculate_distance(metrics)
    return {"metrics": metrics, "distance": distance}

