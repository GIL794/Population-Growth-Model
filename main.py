#!/usr/bin/env python3
"""
Population Growth Models - Main Example Script

This script demonstrates various population growth models including:
1. Exponential Growth
2. Logistic Growth
3. Predator-Prey Dynamics (Lotka-Volterra)

It uses numerical methods to solve differential equations and creates
visualizations showing how populations change over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from population_models import (
    ExponentialGrowth,
    LogisticGrowth,
    PredatorPreyModel,
    compare_numerical_methods
)
from visualization import (
    plot_exponential_growth,
    plot_logistic_growth,
    plot_predator_prey,
    plot_method_comparison,
    plot_multiple_scenarios,
    create_comprehensive_report
)


def example_exponential_growth():
    """Demonstrate exponential growth model."""
    print("\n" + "="*60)
    print("EXPONENTIAL GROWTH MODEL")
    print("="*60)
    print("\nModel: dN/dt = r * N")
    print("Where r is the growth rate and N is the population size")
    
    # Parameters
    r = 0.1  # Growth rate (10% per time unit)
    N0 = 100  # Initial population
    t_max = 50  # Time span
    t = np.linspace(0, t_max, 500)
    
    print(f"\nParameters:")
    print(f"  Growth rate (r): {r}")
    print(f"  Initial population (N0): {N0}")
    print(f"  Time span: 0 to {t_max}")
    
    # Create and solve model
    model = ExponentialGrowth(growth_rate=r)
    N = model.solve(N0, t)
    
    print(f"\nResults:")
    print(f"  Initial population: {N[0]:.2f}")
    print(f"  Final population: {N[-1]:.2f}")
    print(f"  Growth factor: {N[-1]/N[0]:.2f}x")
    
    # Plot
    fig = plot_exponential_growth(t, N)
    plt.savefig('exponential_growth.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'exponential_growth.png'")
    plt.close()
    
    return t, N


def example_logistic_growth():
    """Demonstrate logistic growth model."""
    print("\n" + "="*60)
    print("LOGISTIC GROWTH MODEL")
    print("="*60)
    print("\nModel: dN/dt = r * N * (1 - N/K)")
    print("Where r is the growth rate, K is carrying capacity")
    
    # Parameters
    r = 0.5  # Growth rate
    K = 1000  # Carrying capacity
    N0 = 50  # Initial population
    t_max = 30  # Time span
    t = np.linspace(0, t_max, 500)
    
    print(f"\nParameters:")
    print(f"  Growth rate (r): {r}")
    print(f"  Carrying capacity (K): {K}")
    print(f"  Initial population (N0): {N0}")
    print(f"  Time span: 0 to {t_max}")
    
    # Create and solve model
    model = LogisticGrowth(growth_rate=r, carrying_capacity=K)
    N = model.solve(N0, t)
    
    print(f"\nResults:")
    print(f"  Initial population: {N[0]:.2f}")
    print(f"  Final population: {N[-1]:.2f}")
    print(f"  Percentage of carrying capacity: {N[-1]/K*100:.1f}%")
    
    # Plot
    fig = plot_logistic_growth(t, N, K)
    plt.savefig('logistic_growth.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'logistic_growth.png'")
    plt.close()
    
    return t, N, K


def example_predator_prey():
    """Demonstrate predator-prey dynamics."""
    print("\n" + "="*60)
    print("PREDATOR-PREY MODEL (LOTKA-VOLTERRA)")
    print("="*60)
    print("\nModel equations:")
    print("  dN/dt = α*N - β*N*P  (Prey)")
    print("  dP/dt = δ*N*P - γ*P  (Predator)")
    
    # Parameters
    alpha = 0.1   # Prey growth rate
    beta = 0.02   # Predation rate
    delta = 0.01  # Predator efficiency
    gamma = 0.1   # Predator death rate
    
    N0 = 40  # Initial prey population
    P0 = 9   # Initial predator population
    t_max = 200
    t = np.linspace(0, t_max, 2000)
    
    print(f"\nParameters:")
    print(f"  Prey growth rate (α): {alpha}")
    print(f"  Predation rate (β): {beta}")
    print(f"  Predator efficiency (δ): {delta}")
    print(f"  Predator death rate (γ): {gamma}")
    print(f"  Initial prey (N0): {N0}")
    print(f"  Initial predator (P0): {P0}")
    print(f"  Time span: 0 to {t_max}")
    
    # Create and solve model
    model = PredatorPreyModel(alpha, beta, delta, gamma)
    prey, predator = model.solve(N0, P0, t)
    
    print(f"\nResults:")
    print(f"  Prey - Min: {prey.min():.2f}, Max: {prey.max():.2f}, Avg: {prey.mean():.2f}")
    print(f"  Predator - Min: {predator.min():.2f}, Max: {predator.max():.2f}, Avg: {predator.mean():.2f}")
    
    # Plot
    fig = plot_predator_prey(t, prey, predator)
    plt.savefig('predator_prey.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'predator_prey.png'")
    plt.close()
    
    return t, prey, predator


def example_numerical_methods():
    """Compare different numerical methods."""
    print("\n" + "="*60)
    print("NUMERICAL METHODS COMPARISON")
    print("="*60)
    print("\nComparing Euler's method and RK4 with analytical solution")
    
    # Use exponential growth for comparison (has analytical solution)
    r = 0.2
    N0 = 100
    t = np.linspace(0, 20, 100)
    
    print(f"\nUsing Exponential Growth Model")
    print(f"  Growth rate (r): {r}")
    print(f"  Initial population (N0): {N0}")
    
    model = ExponentialGrowth(growth_rate=r)
    analytical, euler, rk4 = compare_numerical_methods(model, N0, t)
    
    # Calculate errors
    euler_error = np.abs(euler - analytical)
    rk4_error = np.abs(rk4 - analytical)
    
    print(f"\nNumerical Accuracy:")
    print(f"  Euler method - Max error: {euler_error.max():.6f}, Mean error: {euler_error.mean():.6f}")
    print(f"  RK4 method - Max error: {rk4_error.max():.6f}, Mean error: {rk4_error.mean():.6f}")
    print(f"  RK4 is {euler_error.max()/rk4_error.max():.1f}x more accurate than Euler")
    
    # Plot
    fig = plot_method_comparison(t, analytical, euler, rk4)
    plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'method_comparison.png'")
    plt.close()


def example_multiple_scenarios():
    """Compare different parameter values."""
    print("\n" + "="*60)
    print("MULTIPLE SCENARIOS COMPARISON")
    print("="*60)
    print("\nComparing logistic growth with different growth rates")
    
    t = np.linspace(0, 30, 500)
    K = 1000
    N0 = 50
    
    growth_rates = [0.1, 0.3, 0.5, 0.7]
    populations = []
    labels = []
    
    for r in growth_rates:
        model = LogisticGrowth(growth_rate=r, carrying_capacity=K)
        N = model.solve(N0, t)
        populations.append(N)
        labels.append(f'r = {r}')
    
    print(f"\nParameters:")
    print(f"  Carrying capacity (K): {K}")
    print(f"  Initial population (N0): {N0}")
    print(f"  Growth rates tested: {growth_rates}")
    
    # Plot
    fig = plot_multiple_scenarios(t, populations, labels,
                                  title="Effect of Growth Rate on Logistic Growth")
    plt.savefig('multiple_scenarios.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'multiple_scenarios.png'")
    plt.close()


def create_summary_report():
    """Create a comprehensive summary report."""
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE SUMMARY REPORT")
    print("="*60)
    
    # Exponential growth data
    t_exp = np.linspace(0, 30, 300)
    exp_model = ExponentialGrowth(growth_rate=0.1)
    N_exp = exp_model.solve(100, t_exp)
    
    # Logistic growth data
    t_log = np.linspace(0, 30, 300)
    log_model = LogisticGrowth(growth_rate=0.5, carrying_capacity=1000)
    N_log = log_model.solve(50, t_log)
    
    # Predator-prey data
    t_pp = np.linspace(0, 200, 1000)
    pp_model = PredatorPreyModel(0.1, 0.02, 0.01, 0.1)
    prey, predator = pp_model.solve(40, 9, t_pp)
    
    # Create comprehensive report
    fig = create_comprehensive_report(
        exponential_data={'t': t_exp, 'N': N_exp},
        logistic_data={'t': t_log, 'N': N_log, 'K': 1000},
        predator_prey_data={'t': t_pp, 'prey': prey, 'predator': predator}
    )
    plt.savefig('comprehensive_report.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comprehensive report saved as 'comprehensive_report.png'")
    plt.close()


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("POPULATION GROWTH MODELS - DEMONSTRATION")
    print("="*60)
    print("\nThis script demonstrates various population growth models")
    print("using numerical methods to solve differential equations.")
    
    # Run all examples
    example_exponential_growth()
    example_logistic_growth()
    example_predator_prey()
    example_numerical_methods()
    example_multiple_scenarios()
    create_summary_report()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. exponential_growth.png - Exponential growth visualization")
    print("  2. logistic_growth.png - Logistic growth with carrying capacity")
    print("  3. predator_prey.png - Predator-prey dynamics and phase space")
    print("  4. method_comparison.png - Numerical methods comparison")
    print("  5. multiple_scenarios.png - Multiple parameter scenarios")
    print("  6. comprehensive_report.png - Complete overview of all models")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
