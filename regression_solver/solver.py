"""
Comprehensive Regression Solver
Supports linear, polynomial, exponential, power law, and reciprocal regression
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Union
import warnings

warnings.filterwarnings('ignore')


class RegressionSolver:
    """
    Comprehensive Regression Solver supporting multiple regression types.
    
    Topics covered:
    1. Linear Regression with Intercept
    2. Polynomial Regression
    3. MLE of Noise Variance
    4. MAP Estimation (Ridge Regression)
    5. Exponential Regression (y = a*e^(bx))
    6. Power Law Regression (y = a*x^b)
    7. Reciprocal Regression (multiple forms)
    8. Basis Function Regression
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the regression solver.
        
        Args:
            verbose: If True, print detailed step-by-step output
        """
        self.verbose = verbose
        self.X = None
        self.y = None
        self.N = None
        self.results = {}
        
    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and format input data."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")
            
        self.X = X
        self.y = y
        self.N = X.shape[0]
        return X, y
        
    def _print_step(self, step_num: int, title: str, content: str = ""):
        """Format and print step information."""
        if not self.verbose:
            return
        print(f"\n{'='*60}")
        print(f"Step {step_num}: {title}")
        print(f"{'='*60}")
        if content:
            print(content)
    
    def _print_table(self, df: pd.DataFrame, title: str = "Data Table"):
        """Print a formatted table."""
        if not self.verbose:
            return
        print(f"\n{title}:")
        print(df.to_string(float_format="%.4f"))
    
    def _least_squares(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve least squares problem using normal equations or pseudoinverse."""
        try:
            XTX = X.T @ X
            XTy = X.T @ y
            w = np.linalg.solve(XTX, XTy)
            return w
        except np.linalg.LinAlgError:
            # Use pseudoinverse if matrix is singular
            w = np.linalg.pinv(X) @ y
            return w

    # ==============================
    # 1. LINEAR REGRESSION
    # ==============================
    
    def linear_regression(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform linear regression with intercept.
        
        Model: y = w0 + w1*x + epsilon
        
        Returns:
            dict: Contains coefficients, predictions, residuals, and statistics
        """
        self._validate_data(X, y)
        
        self._print_step(1, "Linear Regression with Intercept")
        print(f"Number of observations: N = {self.N}")
        print(f"Model: y_n = w0 + w1*x_n + epsilon_n")
        
        # Compute using formulas
        x = self.X.flatten()
        y = self.y.flatten()
        
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        
        # Create table
        df = pd.DataFrame({
            'x': x,
            'y': y,
            'x²': x**2,
            'xy': x*y
        })
        self._print_table(df)
        
        if self.verbose:
            print(f"\nSums:")
            print(f"Σx = {sum_x:.4f}")
            print(f"Σy = {sum_y:.4f}")
            print(f"Σx² = {sum_x2:.4f}")
            print(f"Σxy = {sum_xy:.4f}")
        
        # Calculate slope and intercept
        w1 = (self.N * sum_xy - sum_x * sum_y) / (self.N * sum_x2 - sum_x ** 2)
        x_bar = np.mean(x)
        y_bar = np.mean(y)
        w0 = y_bar - w1 * x_bar
        
        self._print_step(3, "Parameter Estimates")
        print(f"Slope (w1) = {w1:.6f}")
        print(f"Intercept (w0) = {w0:.6f}")
        print(f"\nFitted model: ŷ = {w0:.4f} + {w1:.4f}x")
        
        # Predictions and residuals
        y_pred = w0 + w1 * x
        residuals = y - y_pred
        sse = np.sum(residuals ** 2)
        
        # Store results
        self.results = {
            'type': 'linear',
            'coefficients': np.array([w0, w1]),
            'intercept': w0,
            'slope': w1,
            'predictions': y_pred,
            'residuals': residuals,
            'sse': sse,
            'mle_variance': sse / self.N,
            'unbiased_variance': sse / (self.N - 2) if self.N > 2 else np.nan
        }
        
        return self.results
    
    # ==============================
    # 2. POLYNOMIAL REGRESSION
    # ==============================
    
    def polynomial_regression(self, X: np.ndarray, y: np.ndarray, 
                             degree: int = 2) -> Dict:
        """
        Perform polynomial regression of specified degree.
        
        Model: y = w0 + w1*x + w2*x² + ... + wM*x^M + epsilon
        
        Args:
            degree: Degree of polynomial (M)
        """
        self._validate_data(X, y)
        
        self._print_step(1, f"Polynomial Regression (Degree {degree})")
        print(f"Number of observations: N = {self.N}")
        print(f"Model: y = w0 + w1*x + w2*x² + ... + w{degree}*x^{degree} + epsilon")
        print(f"Number of parameters: p = {degree + 1}")
        
        # Create design matrix
        X_poly = self._create_design_matrix(degree)
        
        # Show design matrix
        if self.verbose and self.N <= 10:
            df = pd.DataFrame(X_poly, columns=[f'x^{i}' for i in range(degree + 1)])
            self._print_table(df, "Design Matrix (X)")
        
        # Solve using least squares
        w = self._least_squares(X_poly, self.y)
        
        # Display coefficients
        self._print_step(2, "Coefficient Estimates")
        for i, coef in enumerate(w.flatten()):
            if i == 0:
                print(f"w{i} (intercept) = {coef:.6f}")
            else:
                print(f"w{i} (x^{i}) = {coef:.6f}")
        
        # Create fitted equation string
        eq_parts = []
        for i, coef in enumerate(w.flatten()):
            if i == 0:
                eq_parts.append(f"{coef:.4f}")
            elif i == 1:
                eq_parts.append(f"{coef:+.4f}x")
            else:
                eq_parts.append(f"{coef:+.4f}x^{i}")
        eq_str = "".join(eq_parts)
        
        print(f"\nFitted model: ŷ = {eq_str}")
        
        # Predictions and residuals
        y_pred = X_poly @ w
        residuals = self.y.flatten() - y_pred.flatten()
        sse = np.sum(residuals ** 2)
        
        # Store results
        self.results = {
            'type': 'polynomial',
            'degree': degree,
            'coefficients': w.flatten(),
            'predictions': y_pred.flatten(),
            'residuals': residuals,
            'sse': sse,
            'mle_variance': sse / self.N,
            'unbiased_variance': sse / (self.N - (degree + 1)) if self.N > degree + 1 else np.nan
        }
        
        return self.results
    
    def _create_design_matrix(self, degree: int) -> np.ndarray:
        """Create polynomial design matrix."""
        X_poly = np.ones((self.N, degree + 1))
        for i in range(1, degree + 1):
            X_poly[:, i] = self.X.flatten() ** i
        return X_poly
    
    # ==============================
    # 3. MLE OF NOISE VARIANCE
    # ==============================
    
    def estimate_noise_variance(self, y: np.ndarray, y_pred: np.ndarray,
                               p: Optional[int] = None) -> Dict:
        """
        Estimate noise variance using MLE and unbiased estimator.
        
        Args:
            y: Original observations
            y_pred: Predicted values
            p: Number of estimated parameters (for unbiased estimate)
        """
        residuals = y.flatten() - y_pred.flatten()
        sse = np.sum(residuals ** 2)
        N = len(y)
        
        # MLE estimate (biased)
        sigma2_ml = sse / N
        
        # Unbiased estimate
        if p is None:
            sigma2_unbiased = sse / (N - 2) if N > 2 else np.nan
        else:
            sigma2_unbiased = sse / (N - p) if N > p else np.nan
        
        self._print_step(1, "MLE of Noise Variance")
        print(f"Number of observations: N = {N}")
        print(f"SSE = {sse:.4f}")
        print(f"σ²_ML = SSE/N = {sse:.4f}/{N} = {sigma2_ml:.6f}")
        
        if not np.isnan(sigma2_unbiased):
            print(f"σ²_unbiased = SSE/(N-p) = {sse:.4f}/({N} - {p if p else 2}) = {sigma2_unbiased:.6f}")
        
        if self.verbose:
            print("\nResidual Analysis:")
            df = pd.DataFrame({
                'y': y.flatten(),
                'ŷ': y_pred.flatten(),
                'e = y - ŷ': residuals,
                'e²': residuals**2
            })
            print(df.to_string(float_format="%.4f"))
        
        results = {
            'sse': sse,
            'sigma2_ml': sigma2_ml,
            'sigma2_unbiased': sigma2_unbiased,
            'residuals': residuals
        }
        
        return results
    
    # ==============================
    # 4. MAP ESTIMATION (RIDGE)
    # ==============================
    
    def map_estimation(self, X: np.ndarray, y: np.ndarray,
                      lambda_reg: float = 1.0,
                      regularize_intercept: bool = False) -> Dict:
        """
        Perform MAP estimation with Gaussian prior (Ridge Regression).
        
        Equivalent to Ridge Regression with regularization parameter lambda.
        
        Args:
            lambda_reg: Regularization parameter (λ = σ²/τ²)
            regularize_intercept: Whether to penalize intercept
        """
        self._validate_data(X, y)
        
        self._print_step(1, "MAP Estimation (Ridge Regression)")
        print(f"Regularization parameter λ = {lambda_reg}")
        print(f"Regularize intercept: {regularize_intercept}")
        
        X_design = np.column_stack([np.ones(self.N), self.X])
        
        # Regularization matrix
        if regularize_intercept:
            reg_matrix = np.eye(X_design.shape[1]) * lambda_reg
        else:
            reg_matrix = np.eye(X_design.shape[1]) * lambda_reg
            reg_matrix[0, 0] = 0  # Don't regularize intercept
        
        # Solve MAP estimate
        XTX_reg = X_design.T @ X_design + reg_matrix
        XTy = X_design.T @ self.y
        w_map = np.linalg.solve(XTX_reg, XTy)
        
        # Compare with MLE
        w_mle = self._least_squares(X_design, self.y)
        
        self._print_step(2, "Parameter Estimates")
        print(f"MLE:  w0 = {w_mle[0, 0]:.6f}, w1 = {w_mle[1, 0]:.6f}")
        print(f"MAP:  w0 = {w_map[0, 0]:.6f}, w1 = {w_map[1, 0]:.6f}")
        
        return {
            'w_map': w_map.flatten(), 
            'w_mle': w_mle.flatten(),
            'lambda': lambda_reg
        }
    
    # ==============================
    # 5. EXPONENTIAL REGRESSION
    # ==============================
    
    def exponential_regression(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Fit exponential model: y = a * exp(b*x)
        
        Linearization: ln(y) = ln(a) + b*x
        """
        self._validate_data(X, y)
        
        # Check for positive y values
        if np.any(y <= 0):
            raise ValueError("All y values must be positive for exponential regression")
        
        self._print_step(1, "Exponential Regression via Linearization")
        print(f"Original model: y = a*exp(b*x)")
        print(f"Linearization: ln(y) = ln(a) + b*x")
        
        x = self.X.flatten()
        y_orig = self.y.flatten()
        
        # Transform variables
        Y = np.log(y_orig)  # ln(y)
        X_trans = x          # X = x
        
        # Prepare data table
        df = pd.DataFrame({
            'x': x,
            'y': y_orig,
            'X = x': X_trans,
            'Y = ln(y)': Y,
            'X²': X_trans**2,
            'XY': X_trans * Y
        })
        self._print_table(df, "Transformed Data")
        
        # Compute sums
        sum_X = np.sum(X_trans)
        sum_Y = np.sum(Y)
        sum_X2 = np.sum(X_trans**2)
        sum_XY = np.sum(X_trans * Y)
        
        if self.verbose:
            print(f"\nSums in transformed space:")
            print(f"ΣX = {sum_X:.4f}")
            print(f"ΣY = {sum_Y:.4f}")
            print(f"ΣX² = {sum_X2:.4f}")
            print(f"ΣXY = {sum_XY:.4f}")
        
        # Linear regression in transformed space
        B = (self.N * sum_XY - sum_X * sum_Y) / (self.N * sum_X2 - sum_X ** 2)
        A = (sum_Y - B * sum_X) / self.N
        
        # Convert back to original parameters
        a = np.exp(A)
        b = B
        
        self._print_step(3, "Parameter Estimates")
        print(f"In transformed space: A = ln(a) = {A:.6f}, B = b = {B:.6f}")
        print(f"In original space: a = exp(A) = {a:.6f}, b = {b:.6f}")
        print(f"\nFitted model: ŷ = {a:.4f}*exp({b:.4f}*x)")
        
        # Predictions in original scale
        y_pred = a * np.exp(b * x)
        residuals = y_orig - y_pred
        sse = np.sum(residuals ** 2)
        
        self.results = {
            'type': 'exponential',
            'a': a,
            'b': b,
            'A': A,
            'B': B,
            'predictions': y_pred,
            'residuals': residuals,
            'sse': sse,
            'mle_variance': sse / self.N
        }
        
        return self.results
    
    # ==============================
    # 6. POWER LAW REGRESSION
    # ==============================
    
    def power_law_regression(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Fit power law model: y = a * x^b
        
        Linearization: ln(y) = ln(a) + b*ln(x)
        """
        self._validate_data(X, y)
        
        # Check for positive values
        if np.any(X <= 0) or np.any(y <= 0):
            raise ValueError("x and y must be positive for power law regression")
        
        self._print_step(1, "Power Law Regression via Linearization")
        print(f"Original model: y = a*x^b")
        print(f"Linearization: ln(y) = ln(a) + b*ln(x)")
        
        x = self.X.flatten()
        y_orig = self.y.flatten()
        
        # Transform variables
        X_trans = np.log(x)  # ln(x)
        Y_trans = np.log(y_orig)  # ln(y)
        
        # Prepare data table
        df = pd.DataFrame({
            'x': x,
            'y': y_orig,
            'X = ln(x)': X_trans,
            'Y = ln(y)': Y_trans,
            'X²': X_trans**2,
            'XY': X_trans * Y_trans
        })
        self._print_table(df, "Transformed Data (Log-Log)")
        
        # Compute sums
        sum_X = np.sum(X_trans)
        sum_Y = np.sum(Y_trans)
        sum_X2 = np.sum(X_trans**2)
        sum_XY = np.sum(X_trans * Y_trans)
        
        if self.verbose:
            print(f"\nSums in transformed space:")
            print(f"ΣX = {sum_X:.4f}")
            print(f"ΣY = {sum_Y:.4f}")
            print(f"ΣX² = {sum_X2:.4f}")
            print(f"ΣXY = {sum_XY:.4f}")
        
        # Linear regression in transformed space
        B = (self.N * sum_XY - sum_X * sum_Y) / (self.N * sum_X2 - sum_X ** 2)
        A = (sum_Y - B * sum_X) / self.N
        
        # Convert back to original parameters
        a = np.exp(A)
        b = B
        
        self._print_step(3, "Parameter Estimates")
        print(f"In transformed space: A = ln(a) = {A:.6f}, B = b = {B:.6f}")
        print(f"In original space: a = exp(A) = {a:.6f}, b = {b:.6f}")
        print(f"\nFitted model: ŷ = {a:.4f}*x^{b:.4f}")
        
        # Predictions in original scale
        y_pred = a * (x ** b)
        residuals = y_orig - y_pred
        sse = np.sum(residuals ** 2)
        
        self.results = {
            'type': 'power_law',
            'a': a,
            'b': b,
            'A': A,
            'B': B,
            'predictions': y_pred,
            'residuals': residuals,
            'sse': sse,
            'mle_variance': sse / self.N
        }
        
        return self.results
    
    # ==============================
    # 7. RECIPROCAL REGRESSION
    # ==============================
    
    def reciprocal_regression(self, X: np.ndarray, y: np.ndarray,
                             model_type: str = 'y_a_b_over_x') -> Dict:
        """
        Fit reciprocal models.
        
        model_type options:
        - 'y_a_b_over_x': y = a + b/x
        - 'one_over_y_a_plus_bx': 1/y = a + b*x
        - 'y_one_over_a_plus_bx': y = 1/(a + b*x)
        - 'y_ax_over_b_plus_x': y = a*x/(b + x)
        """
        self._validate_data(X, y)
        
        # Check for zeros
        if np.any(X == 0) and model_type in ['y_a_b_over_x', 'y_ax_over_b_plus_x']:
            raise ValueError("x cannot be zero for this reciprocal model")
        if np.any(y == 0) and model_type in ['one_over_y_a_plus_bx', 'y_one_over_a_plus_bx']:
            raise ValueError("y cannot be zero for this reciprocal model")
        
        model_names = {
            'y_a_b_over_x': "y = a + b/x",
            'one_over_y_a_plus_bx': "1/y = a + b*x",
            'y_one_over_a_plus_bx': "y = 1/(a + b*x)",
            'y_ax_over_b_plus_x': "y = a*x/(b + x)"
        }
        
        self._print_step(1, "Reciprocal Regression")
        print(f"Original model: {model_names[model_type]}")
        
        x = self.X.flatten()
        y_orig = self.y.flatten()
        
        if model_type == 'y_a_b_over_x':
            # y = a + b/x  ->  Y = y, X = 1/x
            X_trans = 1 / x
            Y_trans = y_orig
            param_map = {'A': 'a', 'B': 'b'}
            
        elif model_type == 'one_over_y_a_plus_bx':
            # 1/y = a + b*x  ->  Y = 1/y, X = x
            X_trans = x
            Y_trans = 1 / y_orig
            param_map = {'A': 'a', 'B': 'b'}
            
        elif model_type == 'y_one_over_a_plus_bx':
            # y = 1/(a + b*x)  ->  Y = 1/y, X = x
            X_trans = x
            Y_trans = 1 / y_orig
            param_map = {'A': 'a', 'B': 'b'}
            
        elif model_type == 'y_ax_over_b_plus_x':
            # y = a*x/(b+x)  ->  1/y = (b/a)*(1/x) + 1/a
            X_trans = 1 / x
            Y_trans = 1 / y_orig
            param_map = {'A': '1/a', 'B': 'b/a'}
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Prepare data table
        df = pd.DataFrame({
            'x': x,
            'y': y_orig,
            'X': X_trans,
            'Y': Y_trans,
            'X²': X_trans**2,
            'XY': X_trans * Y_trans
        })
        self._print_table(df, "Transformed Data")
        
        # Linear regression in transformed space
        sum_X = np.sum(X_trans)
        sum_Y = np.sum(Y_trans)
        sum_X2 = np.sum(X_trans**2)
        sum_XY = np.sum(X_trans * Y_trans)
        
        B = (self.N * sum_XY - sum_X * sum_Y) / (self.N * sum_X2 - sum_X ** 2)
        A = (sum_Y - B * sum_X) / self.N
        
        self._print_step(3, "Parameter Estimates (Transformed Space)")
        print(f"Y = A + B*X")
        print(f"A = {A:.6f}, B = {B:.6f}")
        
        # Convert back to original parameters
        if model_type == 'y_a_b_over_x':
            a, b = A, B
            
        elif model_type == 'one_over_y_a_plus_bx':
            a, b = A, B
            print(f"y = 1/({a:.4f} + {b:.4f}x)")
            
        elif model_type == 'y_one_over_a_plus_bx':
            a, b = A, B
            
        elif model_type == 'y_ax_over_b_plus_x':
            a = 1 / A
            b = B / A
        
        print(f"\nIn original space:")
        print(f"a = {a:.6f}, b = {b:.6f}")
        
        # Predictions
        if model_type == 'y_a_b_over_x':
            y_pred = a + b / x
            model_str = f"{a:.4f} + {b:.4f}/x"
        elif model_type == 'one_over_y_a_plus_bx':
            y_pred = 1 / (a + b * x)
            model_str = f"1/({a:.4f} + {b:.4f}x)"
        elif model_type == 'y_one_over_a_plus_bx':
            y_pred = 1 / (a + b * x)
            model_str = f"1/({a:.4f} + {b:.4f}x)"
        elif model_type == 'y_ax_over_b_plus_x':
            y_pred = a * x / (b + x)
            model_str = f"{a:.4f}x/({b:.4f} + x)"
        
        print(f"\nFitted model: ŷ = {model_str}")
        
        residuals = y_orig - y_pred
        sse = np.sum(residuals ** 2)
        
        self.results = {
            'type': 'reciprocal',
            'model_type': model_type,
            'a': a,
            'b': b,
            'A': A,
            'B': B,
            'predictions': y_pred,
            'residuals': residuals,
            'sse': sse,
            'mle_variance': sse / self.N
        }
        
        return self.results
    
    # ==============================
    # 8. BASIS FUNCTION REGRESSION
    # ==============================
    
    def basis_function_regression(self, X: np.ndarray, y: np.ndarray,
                                 basis_funcs: List[Callable]) -> Dict:
        """
        Perform regression with custom basis functions.
        
        Model: y = w0*phi0(x) + w1*phi1(x) + ... + wM*phiM(x)
        
        Args:
            basis_funcs: List of basis functions phi_j(x)
        """
        self._validate_data(X, y)
        
        self._print_step(1, "Basis Function Regression")
        print(f"Number of observations: N = {self.N}")
        print(f"Number of basis functions: M = {len(basis_funcs)}")
        
        # Create design matrix
        X_basis = np.zeros((self.N, len(basis_funcs)))
        for j, func in enumerate(basis_funcs):
            X_basis[:, j] = func(self.X.flatten())
        
        # Show design matrix
        if self.verbose and self.N <= 10:
            df = pd.DataFrame(X_basis, columns=[f'φ{j}(x)' for j in range(len(basis_funcs))])
            self._print_table(df, "Design Matrix (Φ)")
        
        # Solve using least squares
        w = self._least_squares(X_basis, self.y)
        
        self._print_step(2, "Coefficient Estimates")
        for j, coef in enumerate(w.flatten()):
            print(f"w{j} = {coef:.6f}")
        
        # Predictions
        y_pred = X_basis @ w
        residuals = self.y.flatten() - y_pred.flatten()
        sse = np.sum(residuals ** 2)
        
        self.results = {
            'type': 'basis_function',
            'coefficients': w.flatten(),
            'predictions': y_pred.flatten(),
            'residuals': residuals,
            'sse': sse,
            'mle_variance': sse / self.N
        }
        
        return self.results
    
    # ==============================
    # MODEL QUALITY CALCULATIONS
    # ==============================
    
    def model_quality(self, y: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate model quality metrics including SSE, MSE, RMSE, R².
        """
        y = y.flatten()
        y_pred = y_pred.flatten()
        
        residuals = y - y_pred
        sse = np.sum(residuals ** 2)
        mse = sse / len(y)
        rmse = np.sqrt(mse)
        
        y_bar = np.mean(y)
        sst = np.sum((y - y_bar) ** 2)
        r2 = 1 - sse / sst if sst > 0 else np.nan
        
        if self.verbose:
            print("\n" + "="*60)
            print("Model Quality Metrics")
            print("="*60)
            print(f"SSE (Sum of Squared Errors) = {sse:.4f}")
            print(f"MSE (Mean Squared Error) = {mse:.4f}")
            print(f"RMSE (Root MSE) = {rmse:.4f}")
            print(f"R² (Coefficient of Determination) = {r2:.4f}")
            
            if not np.isnan(r2):
                if r2 < 0:
                    print("Warning: R² is negative (model is worse than mean prediction)")
                elif r2 > 0.9:
                    print("R² > 0.9: Excellent fit")
                elif r2 > 0.7:
                    print("R² > 0.7: Good fit")
                elif r2 > 0.5:
                    print("R² > 0.5: Moderate fit")
                else:
                    print("R² < 0.5: Poor fit")
        
        return {
            'sse': sse,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'residuals': residuals
        }
    
    # ==============================
    # PREDICTION
    # ==============================
    
    def predict(self, X_new: np.ndarray, model_type: str = 'linear',
               **kwargs) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X_new: New input values
            model_type: Type of model ('linear', 'polynomial', 'exponential', 
                       'power_law', 'reciprocal')
        """
        X_new = np.asarray(X_new).flatten()
        
        if model_type == 'linear':
            w0, w1 = self.results['coefficients']
            return w0 + w1 * X_new
        
        elif model_type == 'polynomial':
            degree = self.results['degree']
            coefficients = self.results['coefficients']
            X_poly = np.ones((len(X_new), degree + 1))
            for i in range(1, degree + 1):
                X_poly[:, i] = X_new ** i
            return X_poly @ coefficients
        
        elif model_type == 'exponential':
            a = self.results['a']
            b = self.results['b']
            return a * np.exp(b * X_new)
        
        elif model_type == 'power_law':
            a = self.results['a']
            b = self.results['b']
            return a * (X_new ** b)
        
        elif model_type == 'reciprocal':
            model_type_specific = self.results['model_type']
            a, b = self.results['a'], self.results['b']
            if model_type_specific == 'y_a_b_over_x':
                return a + b / X_new
            elif model_type_specific in ['one_over_y_a_plus_bx', 'y_one_over_a_plus_bx']:
                return 1 / (a + b * X_new)
            elif model_type_specific == 'y_ax_over_b_plus_x':
                return a * X_new / (b + X_new)
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    # ==============================
    # FINAL SUMMARY
    # ==============================
    
    def final_summary(self, model_type: str):
        """Print final summary of results."""
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        
        if model_type == 'linear':
            print(f"Model: y = {self.results['intercept']:.4f} + {self.results['slope']:.4f}x")
            print(f"SSE = {self.results['sse']:.4f}")
            print(f"σ²_ML = {self.results['mle_variance']:.4f}")
            if not np.isnan(self.results['unbiased_variance']):
                print(f"σ²_unbiased = {self.results['unbiased_variance']:.4f}")
        
        elif model_type == 'polynomial':
            eq_parts = []
            for i, coef in enumerate(self.results['coefficients']):
                if i == 0:
                    eq_parts.append(f"{coef:.4f}")
                elif i == 1:
                    eq_parts.append(f"{coef:+.4f}x")
                else:
                    eq_parts.append(f"{coef:+.4f}x^{i}")
            print(f"Model: y = {''.join(eq_parts)}")
            print(f"SSE = {self.results['sse']:.4f}")
            print(f"σ²_ML = {self.results['mle_variance']:.4f}")
            if not np.isnan(self.results['unbiased_variance']):
                print(f"σ²_unbiased = {self.results['unbiased_variance']:.4f}")
        
        elif model_type == 'exponential':
            print(f"Model: y = {self.results['a']:.4f} * exp({self.results['b']:.4f}x)")
            print(f"SSE (original scale) = {self.results['sse']:.4f}")
            print(f"σ²_ML = {self.results['mle_variance']:.4f}")
        
        elif model_type == 'power_law':
            print(f"Model: y = {self.results['a']:.4f} * x^{self.results['b']:.4f}")
            print(f"SSE (original scale) = {self.results['sse']:.4f}")
            print(f"σ²_ML = {self.results['mle_variance']:.4f}")
        
        elif model_type == 'reciprocal':
            model_str = {
                'y_a_b_over_x': f"y = {self.results['a']:.4f} + {self.results['b']:.4f}/x",
                'one_over_y_a_plus_bx': f"y = 1/({self.results['a']:.4f} + {self.results['b']:.4f}x)",
                'y_one_over_a_plus_bx': f"y = 1/({self.results['a']:.4f} + {self.results['b']:.4f}x)",
                'y_ax_over_b_plus_x': f"y = {self.results['a']:.4f}x/({self.results['b']:.4f} + x)"
            }[self.results['model_type']]
            print(f"Model: {model_str}")
            print(f"SSE = {self.results['sse']:.4f}")
            print(f"σ²_ML = {self.results['mle_variance']:.4f}")


# run_solver.py - Place this in your Regression-11 folder
import sys
import os

# Add the current folder to Python path so it can find your code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Now import your solver directly from the file
from regression_solver.solver import RegressionSolver

# Create solver instance
solver = RegressionSolver(verbose=True)

# Your data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2.1, 3.8, 5.2, 7.0, 9.1, 10.8, 12.5, 14.2])

# Run linear regression
results = solver.linear_regression(x, y)

# Check quality
solver.model_quality(y, results['predictions'])

print("\n✅ Analysis complete!")
print(f"📐 Equation: y = {results['intercept']:.4f} + {results['slope']:.4f}x")
