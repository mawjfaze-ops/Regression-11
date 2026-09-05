"""
Utility functions for regression analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def generate_noisy_data(n: int = 50, 
                        noise_std: float = 0.1,
                        random_seed: int = 42) -> Dict:
    """
    Generate synthetic data for testing regression models.
    
    Args:
        n: Number of data points
        noise_std: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with x and y arrays
    """
    np.random.seed(random_seed)
    x = np.linspace(0, 10, n)
    noise = np.random.normal(0, noise_std, n)
    
    # Linear data
    y_linear = 2.5 * x + 1.0 + noise
    
    # Exponential data
    y_exp = 2.0 * np.exp(0.3 * x) + noise * 0.5
    
    # Power law data
    y_power = 3.0 * x ** 1.5 + noise
    
    # Quadratic data
    y_quad = 0.5 * x**2 - 2.0 * x + 3.0 + noise
    
    return {
        'x': x,
        'y_linear': y_linear,
        'y_exp': y_exp,
        'y_power': y_power,
        'y_quad': y_quad,
        'noise': noise
    }


def create_reciprocal_data(n: int = 20, random_seed: int = 42) -> Dict:
    """Generate data for reciprocal regression examples."""
    np.random.seed(random_seed)
    x = np.linspace(1, 10, n)
    noise = np.random.normal(0, 0.05, n)
    
    # y = 2 + 3/x
    y_rec = 2 + 3 / x + noise
    
    # y = x / (2 + 0.5*x)
    y_mm = x / (2 + 0.5 * x) + noise * 0.1
    
    return {
        'x': x,
        'y_rec': y_rec,
        'y_mm': y_mm
    }


def print_comparison_table(results: Dict, 
                          model_names: List[str]) -> None:
    """
    Print a comparison table of different regression models.
    
    Args:
        results: Dictionary of results from different models
        model_names: List of model names
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    metrics = ['sse', 'mle_variance', 'r2']
    metric_labels = ['SSE', 'σ²_ML', 'R²']
    
    # Create comparison table
    data = {}
    for metric, label in zip(metrics, metric_labels):
        data[label] = []
        for name in model_names:
            if name in results and metric in results[name]:
                value = results[name][metric]
                if metric == 'r2' and value is not None:
                    data[label].append(f"{value:.4f}")
                else:
                    data[label].append(f"{value:.4f}")
            else:
                data[label].append("N/A")
    
    df = pd.DataFrame(data, index=model_names)
    print(df.to_string(float_format="%.4f"))
    