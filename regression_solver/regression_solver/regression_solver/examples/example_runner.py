"""
Example runner demonstrating all regression types
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from regression_solver import RegressionSolver
from regression_solver.utils import generate_noisy_data, create_reciprocal_data


def run_all_examples():
    """Run all regression examples."""
    
    solver = RegressionSolver(verbose=True)
    
    print("\n" + "█"*70)
    print(" REGRESSION SOLVER - COMPLETE EXAMPLE RUN")
    print("█"*70)
    
    # Generate synthetic data
    data = generate_noisy_data(n=30, noise_std=0.2, random_seed=42)
    recip_data = create_reciprocal_data(n=15, random_seed=42)
    
    # ============================================================
    # Example 1: Linear Regression
    # ============================================================
    print("\n" + "█"*70)
    print("EXAMPLE 1: LINEAR REGRESSION")
    print("█"*70)
    
    x_lin = data['x']
    y_lin = data['y_linear']
    
    results_linear = solver.linear_regression(x_lin, y_lin)
    quality_linear = solver.model_quality(y_lin, results_linear['predictions'])
    
    # Prediction example
    new_x = np.array([11, 12, 13])
    pred_linear = solver.predict(new_x, 'linear')
    print(f"\nPredictions for x = {new_x}: {pred_linear}")
    
    solver.final_summary('linear')
    
    # ============================================================
    # Example 2: Polynomial Regression
    # ============================================================
    print("\n" + "█"*70)
    print("EXAMPLE 2: POLYNOMIAL REGRESSION")
    print("█"*70)
    
    x_poly = data['x']
    y_poly = data['y_quad']
    
    for degree in [2, 3]:
        print(f"\n--- Degree {degree} ---")
        results_poly = solver.polynomial_regression(x_poly, y_poly, degree=degree)
        quality_poly = solver.model_quality(y_poly, results_poly['predictions'])
        solver.final_summary('polynomial')
    
    # ============================================================
    # Example 3: Exponential Regression
    # ============================================================
    print("\n" + "█"*70)
    print("EXAMPLE 3: EXPONENTIAL REGRESSION")
    print("█"*70)
    
    x_exp = data['x']
    y_exp = data['y_exp']
    
    results_exp = solver.exponential_regression(x_exp, y_exp)
    quality_exp = solver.model_quality(y_exp, results_exp['predictions'])
    
    # Prediction example
    pred_exp = solver.predict([11, 12], 'exponential')
    print(f"\nPredictions for x = [11, 12]: {pred_exp}")
    
    solver.final_summary('exponential')
    
    # ============================================================
    # Example 4: Power Law Regression
    # ============================================================
    print("\n" + "█"*70)
    print("EXAMPLE 4: POWER LAW REGRESSION")
    print("█"*70)
    
    x_power = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_power = 3 * x_power ** 1.5 + np.random.normal(0, 0.5, len(x_power))
    
    results_power = solver.power_law_regression(x_power, y_power)
    quality_power = solver.model_quality(y_power, results_power['predictions'])
    
    solver.final_summary('power_law')
    
    # ============================================================
    # Example 5: Reciprocal Regression
    # ============================================================
    print("\n" + "█"*70)
    print("EXAMPLE 5: RECIPROCAL REGRESSION")
    print("█"*70)
    
    x_rec = recip_data['x']
    y_rec = recip_data['y_rec']
    
    results_rec = solver.reciprocal_regression(x_rec, y_rec, 'y_a_b_over_x')
    quality_rec = solver.model_quality(y_rec, results_rec['predictions'])
    
    solver.final_summary('reciprocal')
    
    # ============================================================
    # Example 6: MAP Estimation (Ridge Regression)
    # ============================================================
    print("\n" + "█"*70)
    print("EXAMPLE 6: MAP ESTIMATION (RIDGE REGRESSION)")
    print("█"*70)
    
    x_ridge = np.linspace(0, 10, 20)
    y_ridge = 2 * x_ridge + 3 + np.random.normal(0, 0.5, 20)
    
    results_map = solver.map_estimation(x_ridge, y_ridge, lambda_reg=0.5)
    
    # ============================================================
    # Example 7: Basis Function Regression
    # ============================================================
    print("\n" + "█"*70)
    print("EXAMPLE 7: BASIS FUNCTION REGRESSION")
    print("█"*70)
    
    x_basis = np.linspace(-5, 5, 25)
    y_basis = np.sin(x_basis) + 0.5 * np.cos(2 * x_basis) + np.random.normal(0, 0.1, 25)
    
    # Define basis functions
    basis_funcs = [
        lambda x: np.ones_like(x),      # φ₀(x) = 1
        lambda x: x,                    # φ₁(x) = x
        lambda x: x**2,                 # φ₂(x) = x²
        lambda x: np.sin(x),            # φ₃(x) = sin(x)
        lambda x: np.cos(2*x),          # φ₄(x) = cos(2x)
    ]
    
    results_basis = solver.basis_function_regression(x_basis, y_basis, basis_funcs)
    quality_basis = solver.model_quality(y_basis, results_basis['predictions'])
    
    # ============================================================
    # Summary Comparison
    # ============================================================
    print("\n" + "█"*70)
    print(" SUMMARY COMPARISON")
    print("█"*70)
    
    print("\nModel Quality Comparison:")
    print("-" * 50)
    print(f"{'Model':<20} {'SSE':<12} {'R²':<10} {'RMSE':<10}")
    print("-" * 50)
    
    models = [
        ('Linear', results_linear, quality_linear),
        ('Polynomial (deg 2)', results_poly, quality_poly),
        ('Exponential', results_exp, quality_exp),
        ('Power Law', results_power, quality_power),
        ('Reciprocal', results_rec, quality_rec),
        ('Basis Function', results_basis, quality_basis)
    ]
    
    for name, res, qual in models:
        sse = res.get('sse', 0)
        r2 = qual.get('r2', 0)
        rmse = qual.get('rmse', 0)
        print(f"{name:<20} {sse:<12.4f} {r2:<10.4f} {rmse:<10.4f}")
    
    print("\n" + "="*70)
    print(" ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    run_all_examples()
    