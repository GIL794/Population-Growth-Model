"""
Population Growth Models

This module implements various population growth models including:
- Exponential growth
- Logistic growth
- Predator-prey dynamics (Lotka-Volterra equations)

Each model uses numerical methods to solve differential equations.
"""

import numpy as np
from typing import Callable, Tuple, List, Union


class NumericalSolver:
    """Numerical methods for solving ordinary differential equations (ODEs)."""
    
    @staticmethod
    def euler_method(f: Callable, y0: float, t: np.ndarray) -> np.ndarray:
        """
        Solve ODE using Euler's method.
        
        Args:
            f: Function representing dy/dt = f(y, t)
            y0: Initial condition
            t: Time array
            
        Returns:
            Array of solution values
        """
        y = np.zeros(len(t))
        y[0] = y0
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            y[i] = y[i-1] + dt * f(y[i-1], t[i-1])
            
        return y
    
    @staticmethod
    def rk4_method(f: Callable, y0: float, t: np.ndarray) -> np.ndarray:
        """
        Solve ODE using 4th order Runge-Kutta method.
        
        Args:
            f: Function representing dy/dt = f(y, t)
            y0: Initial condition
            t: Time array
            
        Returns:
            Array of solution values
        """
        y = np.zeros(len(t))
        y[0] = y0
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            k1 = f(y[i-1], t[i-1])
            k2 = f(y[i-1] + dt * k1 / 2, t[i-1] + dt / 2)
            k3 = f(y[i-1] + dt * k2 / 2, t[i-1] + dt / 2)
            k4 = f(y[i-1] + dt * k3, t[i])
            
            y[i] = y[i-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            
        return y
    
    @staticmethod
    def rk4_system(f: Callable, y0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Solve system of ODEs using 4th order Runge-Kutta method.
        
        Args:
            f: Function representing dy/dt = f(y, t), returns array
            y0: Initial conditions array
            t: Time array
            
        Returns:
            2D array of solution values (time x variables)
        """
        n = len(y0)
        y = np.zeros((len(t), n))
        y[0] = y0
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            k1 = f(y[i-1], t[i-1])
            k2 = f(y[i-1] + dt * k1 / 2, t[i-1] + dt / 2)
            k3 = f(y[i-1] + dt * k2 / 2, t[i-1] + dt / 2)
            k4 = f(y[i-1] + dt * k3, t[i])
            
            y[i] = y[i-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            
        return y


class ExponentialGrowth:
    """
    Exponential growth model: dN/dt = r * N
    
    Where:
    - N is the population size
    - r is the intrinsic growth rate
    - t is time
    """
    
    def __init__(self, growth_rate: float):
        """
        Initialize exponential growth model.
        
        Args:
            growth_rate: Intrinsic growth rate (r)
        """
        self.r = growth_rate
    
    def differential_equation(self, N: float, t: float) -> float:
        """
        Calculate dN/dt for exponential growth.
        
        Args:
            N: Current population size
            t: Current time
            
        Returns:
            Rate of change dN/dt
        """
        return self.r * N
    
    def analytical_solution(self, N0: float, t: np.ndarray) -> np.ndarray:
        """
        Analytical solution: N(t) = N0 * e^(rt)
        
        Args:
            N0: Initial population size
            t: Time array
            
        Returns:
            Population sizes at each time point
        """
        return N0 * np.exp(self.r * t)
    
    def solve(self, N0: float, t: np.ndarray, method: str = 'rk4') -> np.ndarray:
        """
        Solve exponential growth model numerically.
        
        Args:
            N0: Initial population size
            t: Time array
            method: Numerical method ('euler' or 'rk4')
            
        Returns:
            Population sizes at each time point
        """
        solver = NumericalSolver()
        
        if method == 'euler':
            return solver.euler_method(self.differential_equation, N0, t)
        elif method == 'rk4':
            return solver.rk4_method(self.differential_equation, N0, t)
        else:
            raise ValueError(f"Unknown method: {method}")


class LogisticGrowth:
    """
    Logistic growth model: dN/dt = r * N * (1 - N/K)
    
    Where:
    - N is the population size
    - r is the intrinsic growth rate
    - K is the carrying capacity
    - t is time
    """
    
    def __init__(self, growth_rate: float, carrying_capacity: float):
        """
        Initialize logistic growth model.
        
        Args:
            growth_rate: Intrinsic growth rate (r)
            carrying_capacity: Maximum sustainable population (K)
        """
        self.r = growth_rate
        self.K = carrying_capacity
    
    def differential_equation(self, N: float, t: float) -> float:
        """
        Calculate dN/dt for logistic growth.
        
        Args:
            N: Current population size
            t: Current time
            
        Returns:
            Rate of change dN/dt
        """
        return self.r * N * (1 - N / self.K)
    
    def analytical_solution(self, N0: float, t: np.ndarray) -> np.ndarray:
        """
        Analytical solution: N(t) = K / (1 + ((K - N0) / N0) * e^(-rt))
        
        Args:
            N0: Initial population size
            t: Time array
            
        Returns:
            Population sizes at each time point
        """
        return self.K / (1 + ((self.K - N0) / N0) * np.exp(-self.r * t))
    
    def solve(self, N0: float, t: np.ndarray, method: str = 'rk4') -> np.ndarray:
        """
        Solve logistic growth model numerically.
        
        Args:
            N0: Initial population size
            t: Time array
            method: Numerical method ('euler' or 'rk4')
            
        Returns:
            Population sizes at each time point
        """
        solver = NumericalSolver()
        
        if method == 'euler':
            return solver.euler_method(self.differential_equation, N0, t)
        elif method == 'rk4':
            return solver.rk4_method(self.differential_equation, N0, t)
        else:
            raise ValueError(f"Unknown method: {method}")


class PredatorPreyModel:
    """
    Lotka-Volterra predator-prey model:
    
    dN/dt = α * N - β * N * P  (Prey equation)
    dP/dt = δ * N * P - γ * P  (Predator equation)
    
    Where:
    - N is the prey population
    - P is the predator population
    - α is the prey growth rate
    - β is the predation rate
    - δ is the predator efficiency (conversion of prey to predators)
    - γ is the predator death rate
    """
    
    def __init__(self, alpha: float, beta: float, delta: float, gamma: float):
        """
        Initialize predator-prey model.
        
        Args:
            alpha: Prey growth rate
            beta: Predation rate
            delta: Predator efficiency
            gamma: Predator death rate
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
    
    def differential_equations(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        Calculate derivatives for predator-prey system.
        
        Args:
            y: Array [N, P] where N is prey, P is predator
            t: Current time
            
        Returns:
            Array [dN/dt, dP/dt]
        """
        N, P = y
        
        dN_dt = self.alpha * N - self.beta * N * P
        dP_dt = self.delta * N * P - self.gamma * P
        
        return np.array([dN_dt, dP_dt])
    
    def solve(self, N0: float, P0: float, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve predator-prey model numerically.
        
        Args:
            N0: Initial prey population
            P0: Initial predator population
            t: Time array
            
        Returns:
            Tuple of (prey_population, predator_population) arrays
        """
        solver = NumericalSolver()
        y0 = np.array([N0, P0])
        
        solution = solver.rk4_system(self.differential_equations, y0, t)
        
        return solution[:, 0], solution[:, 1]


def compare_numerical_methods(model: Union[ExponentialGrowth, LogisticGrowth], N0: float, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare Euler and RK4 methods with analytical solution (if available).
    
    Args:
        model: Population growth model (ExponentialGrowth or LogisticGrowth)
        N0: Initial population
        t: Time array
        
    Returns:
        Tuple of (analytical, euler, rk4) solutions
    """
    analytical = model.analytical_solution(N0, t)
    euler = model.solve(N0, t, method='euler')
    rk4 = model.solve(N0, t, method='rk4')
    
    return analytical, euler, rk4
