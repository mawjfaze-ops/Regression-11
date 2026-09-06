# test_regression.py - Quick test for your regression solver

import numpy as np
from regression_solver import RegressionSolver

print("🧪 Testing Regression Solver...")
print("=" * 50)

# Create solver instance
solver = RegressionSolver(verbose=True)

# Test data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2.1, 3.8, 5.2, 7.0, 9.1, 10.8, 12.5, 14.2])

print("\n📊 Test 1: Linear Regression")
print("-" * 40)
results = solver.linear_regression(x, y)
solver.model_quality(y, results['predictions'])

print("\n📊 Test 2: Exponential Regression")
print("-" * 40)
x_exp = np.array([0, 1, 2, 3, 4])
y_exp = np.array([1.0, 2.718, 7.389, 20.085, 54.598])
results_exp = solver.exponential_regression(x_exp, y_exp)

print("\n📊 Test 3: Polynomial Regression")
print("-" * 40)
x_poly = np.array([-3, -2, -1, 0, 1, 2, 3])
y_poly = np.array([9, 4, 1, 0, 1, 4, 9])
results_poly = solver.polynomial_regression(x_poly, y_poly, degree=2)

print("\n" + "=" * 50)
print("✅ All tests completed successfully!")
