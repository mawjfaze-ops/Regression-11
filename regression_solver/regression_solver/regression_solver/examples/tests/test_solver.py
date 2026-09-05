"""
Unit tests for RegressionSolver
"""

import unittest
import numpy as np
from regression_solver import RegressionSolver


class TestRegressionSolver(unittest.TestCase):
    """Test cases for RegressionSolver class."""
    
    def setUp(self):
        """Set up test data."""
        self.solver = RegressionSolver(verbose=False)
        np.random.seed(42)
        
        # Simple linear data
        self.x_lin = np.array([1, 2, 3, 4, 5])
        self.y_lin = np.array([2.1, 4.0, 5.8, 8.1, 10.2])
        
        # Exponential data
        self.x_exp = np.array([0, 1, 2, 3, 4])
        self.y_exp = np.array([1.0, 2.718, 7.389, 20.085, 54.598])
        
        # Power law data
        self.x_power = np.array([1, 2, 3, 4, 5])
        self.y_power = np.array([2, 8, 18, 32, 50])
        
        # Reciprocal data
        self.x_rec = np.array([1, 2, 3, 4, 5])
        self.y_rec = np.array([3.0, 2.5, 2.333, 2.25, 2.2])
    
    def test_linear_regression(self):
        """Test linear regression."""
        results = self.solver.linear_regression(self.x_lin, self.y_lin)
        
        # Check coefficients
        self.assertAlmostEqual(results['slope'], 2.02, places=2)
        self.assertAlmostEqual(results['intercept'], 0.06, places=2)
        
        # Check predictions
        pred = results['predictions']
        self.assertEqual(len(pred), len(self.y_lin))
        
        # Check residuals sum (should be approximately zero)
        self.assertAlmostEqual(np.sum(results['residuals']), 0, places=5)
    
    def test_polynomial_regression(self):
        """Test polynomial regression."""
        # Test quadratic fit
        results = self.solver.polynomial_regression(self.x_power, self.y_power, degree=2)
        
        # For y = 2*x², coefficients should be [0, 0, 2]
        self.assertAlmostEqual(results['coefficients'][2], 2.0, places=1)
        
        # Test predictions match data
        pred = results['predictions']
        mse = np.mean((self.y_power - pred) ** 2)
        self.assertLess(mse, 0.1)
    
    def test_exponential_regression(self):
        """Test exponential regression."""
        results = self.solver.exponential_regression(self.x_exp, self.y_exp)
        
        # For y = e^x, a should be ~1, b ~1
        self.assertAlmostEqual(results['a'], 1.0, places=1)
        self.assertAlmostEqual(results['b'], 1.0, places=1)
        
        # Test predictions
        pred = results['predictions']
        mse = np.mean((self.y_exp - pred) ** 2)
        self.assertLess(mse, 0.5)
    
    def test_power_law_regression(self):
        """Test power law regression."""
        results = self.solver.power_law_regression(self.x_power, self.y_power)
        
        # For y = 2*x², a should be ~2, b ~2
        self.assertAlmostEqual(results['a'], 2.0, places=1)
        self.assertAlmostEqual(results['b'], 2.0, places=1)
    
    def test_reciprocal_regression(self):
        """Test reciprocal regression."""
        results = self.solver.reciprocal_regression(self.x_rec, self.y_rec, 'y_a_b_over_x')
        
        # For y = 2 + 1/x, a ~ 2, b ~ 1
        self.assertAlmostEqual(results['a'], 2.0, places=1)
        self.assertAlmostEqual(results['b'], 1.0, places=1)
    
    def test_map_estimation(self):
        """Test MAP estimation."""
        results = self.solver.map_estimation(self.x_lin, self.y_lin, lambda_reg=0.1)
        
        self.assertEqual(len(results['w_map']), 2)
        self.assertEqual(len(results['w_mle']), 2)
    
    def test_model_quality(self):
        """Test model quality calculations."""
        y = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        quality = self.solver.model_quality(y, y_pred)
        
        self.assertIn('sse', quality)
        self.assertIn('mse', quality)
        self.assertIn('rmse', quality)
        self.assertIn('r2', quality)
    
    def test_predict(self):
        """Test prediction functionality."""
        # Fit linear model
        results = self.solver.linear_regression(self.x_lin, self.y_lin)
        
        # Make predictions
        x_new = np.array([6, 7, 8])
        pred = self.solver.predict(x_new, 'linear')
        
        self.assertEqual(len(pred), len(x_new))
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test negative y for exponential
        with self.assertRaises(ValueError):
            self.solver.exponential_regression(np.array([1, 2]), np.array([-1, 2]))
        
        # Test zero for power law
        with self.assertRaises(ValueError):
            self.solver.power_law_regression(np.array([0, 1]), np.array([1, 2]))
    
    def test_basis_function_regression(self):
        """Test basis function regression."""
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1 + np.random.normal(0, 0.1, 20)
        
        basis_funcs = [
            lambda x: np.ones_like(x),
            lambda x: x,
            lambda x: x**2
        ]
        
        results = self.solver.basis_function_regression(x, y, basis_funcs)
        
        self.assertEqual(len(results['coefficients']), 3)
        self.assertIn('predictions', results)
        self.assertIn('residuals', results)


if __name__ == '__main__':
    unittest.main()
    