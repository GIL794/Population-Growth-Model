"""
Tests for population growth models.

This module contains basic tests to verify the correctness of the
implemented population models and numerical methods.
"""

import numpy as np
from population_models import (
    ExponentialGrowth,
    LogisticGrowth,
    PredatorPreyModel,
    NumericalSolver
)


def test_exponential_growth():
    """Test exponential growth model."""
    print("Testing Exponential Growth Model...")
    
    r = 0.1
    N0 = 100
    t = np.linspace(0, 10, 100)
    
    model = ExponentialGrowth(growth_rate=r)
    
    # Test analytical solution
    N_analytical = model.analytical_solution(N0, t)
    assert N_analytical[0] == N0, "Initial population should match N0"
    assert N_analytical[-1] > N0, "Population should grow"
    
    # Test numerical solution
    N_numerical = model.solve(N0, t, method='rk4')
    assert len(N_numerical) == len(t), "Solution length should match time array"
    assert N_numerical[0] == N0, "Initial population should match N0"
    
    # Compare analytical and numerical (should be very close)
    error = np.abs(N_numerical - N_analytical).max()
    assert error < 0.01, f"RK4 error too large: {error}"
    
    print("  ✓ Exponential growth model passed all tests")


def test_logistic_growth():
    """Test logistic growth model."""
    print("Testing Logistic Growth Model...")
    
    r = 0.5
    K = 1000
    N0 = 50
    t = np.linspace(0, 30, 500)
    
    model = LogisticGrowth(growth_rate=r, carrying_capacity=K)
    
    # Test analytical solution
    N_analytical = model.analytical_solution(N0, t)
    assert N_analytical[0] == N0, "Initial population should match N0"
    assert N_analytical[-1] < K * 1.01, "Population should approach carrying capacity"
    assert N_analytical[-1] > K * 0.99, "Population should reach near carrying capacity"
    
    # Test numerical solution
    N_numerical = model.solve(N0, t, method='rk4')
    assert len(N_numerical) == len(t), "Solution length should match time array"
    assert N_numerical[0] == N0, "Initial population should match N0"
    
    # Check that population doesn't exceed carrying capacity significantly
    assert all(N_numerical <= K * 1.01), "Population should not exceed K significantly"
    
    print("  ✓ Logistic growth model passed all tests")


def test_predator_prey():
    """Test predator-prey model."""
    print("Testing Predator-Prey Model...")
    
    alpha = 0.1
    beta = 0.02
    delta = 0.01
    gamma = 0.1
    N0 = 40
    P0 = 9
    t = np.linspace(0, 100, 1000)
    
    model = PredatorPreyModel(alpha, beta, delta, gamma)
    prey, predator = model.solve(N0, P0, t)
    
    # Check solution properties
    assert len(prey) == len(t), "Prey solution length should match time array"
    assert len(predator) == len(t), "Predator solution length should match time array"
    assert prey[0] == N0, "Initial prey population should match N0"
    assert predator[0] == P0, "Initial predator population should match P0"
    
    # Check that populations remain positive
    assert all(prey >= 0), "Prey population should remain non-negative"
    assert all(predator >= 0), "Predator population should remain non-negative"
    
    # Check for oscillatory behavior (max should be after initial condition)
    assert prey.max() > N0, "Prey should show growth at some point"
    assert predator.max() > P0, "Predator should show growth at some point"
    
    print("  ✓ Predator-prey model passed all tests")


def test_numerical_methods():
    """Test numerical solution methods."""
    print("Testing Numerical Methods...")
    
    # Simple ODE: dy/dt = y, analytical solution: y = y0 * e^t
    def f(y, t):
        return y
    
    y0 = 1.0
    t = np.linspace(0, 2, 100)
    analytical = y0 * np.exp(t)
    
    solver = NumericalSolver()
    
    # Test Euler method
    y_euler = solver.euler_method(f, y0, t)
    assert len(y_euler) == len(t), "Euler solution length should match time array"
    assert y_euler[0] == y0, "Initial condition should match"
    
    # Test RK4 method
    y_rk4 = solver.rk4_method(f, y0, t)
    assert len(y_rk4) == len(t), "RK4 solution length should match time array"
    assert y_rk4[0] == y0, "Initial condition should match"
    
    # RK4 should be more accurate than Euler
    euler_error = np.abs(y_euler - analytical).max()
    rk4_error = np.abs(y_rk4 - analytical).max()
    assert rk4_error < euler_error, "RK4 should be more accurate than Euler"
    assert rk4_error < 0.001, f"RK4 error too large: {rk4_error}"
    
    print("  ✓ Numerical methods passed all tests")


def test_parameter_validation():
    """Test that models handle parameters correctly."""
    print("Testing Parameter Validation...")
    
    # Test exponential growth with different growth rates
    for r in [0.01, 0.1, 0.5, 1.0]:
        model = ExponentialGrowth(growth_rate=r)
        assert model.r == r, f"Growth rate should be {r}"
    
    # Test logistic growth with different parameters
    for r in [0.1, 0.5]:
        for K in [100, 1000]:
            model = LogisticGrowth(growth_rate=r, carrying_capacity=K)
            assert model.r == r, f"Growth rate should be {r}"
            assert model.K == K, f"Carrying capacity should be {K}"
    
    # Test predator-prey with different parameters
    model = PredatorPreyModel(0.1, 0.02, 0.01, 0.1)
    assert model.alpha == 0.1
    assert model.beta == 0.02
    assert model.delta == 0.01
    assert model.gamma == 0.1
    
    print("  ✓ Parameter validation passed all tests")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING POPULATION MODELS TESTS")
    print("="*60 + "\n")
    
    test_exponential_growth()
    test_logistic_growth()
    test_predator_prey()
    test_numerical_methods()
    test_parameter_validation()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
