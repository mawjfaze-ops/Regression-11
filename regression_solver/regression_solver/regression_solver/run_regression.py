# run_regression.py
# Place this in the main Regression-11 folder (same level as setup.py)

import numpy as np
from regression_solver import RegressionSolver

# Create solver instance
solver = RegressionSolver(verbose=True)

# Your data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2.1, 3.8, 5.2, 7.0, 9.1, 10.8, 12.5, 14.2])

# Run linear regression
results = solver.linear_regression(x, y)

# Check quality
solver.model_quality(y, results['predictions'])

print("\n" + "="*50)
print("✅ Analysis complete!")
print("="*50)
print(f"📐 Equation: y = {results['intercept']:.4f} + {results['slope']:.4f}x")
