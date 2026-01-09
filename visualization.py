"""
Visualization functions for population growth models.

This module provides functions to create plots showing how populations
change over time under different growth models and conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from matplotlib.figure import Figure


def plot_exponential_growth(t: np.ndarray, N: np.ndarray, 
                           title: str = "Exponential Growth Model",
                           save_path: Optional[str] = None) -> Figure:
    """
    Plot exponential growth over time.
    
    Args:
        t: Time array
        N: Population sizes
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, N, 'b-', linewidth=2, label='Population')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Population Size', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_logistic_growth(t: np.ndarray, N: np.ndarray, K: float,
                        title: str = "Logistic Growth Model",
                        save_path: Optional[str] = None) -> Figure:
    """
    Plot logistic growth over time with carrying capacity.
    
    Args:
        t: Time array
        N: Population sizes
        K: Carrying capacity
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, N, 'g-', linewidth=2, label='Population')
    ax.axhline(y=K, color='r', linestyle='--', linewidth=1.5, 
               label=f'Carrying Capacity (K={K})')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Population Size', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_predator_prey(t: np.ndarray, prey: np.ndarray, predator: np.ndarray,
                      title: str = "Predator-Prey Dynamics (Lotka-Volterra)",
                      save_path: Optional[str] = None) -> Figure:
    """
    Plot predator and prey populations over time.
    
    Args:
        t: Time array
        prey: Prey population sizes
        predator: Predator population sizes
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Time series plot
    ax1.plot(t, prey, 'b-', linewidth=2, label='Prey')
    ax1.plot(t, predator, 'r-', linewidth=2, label='Predator')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Population Size', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Phase space plot
    ax2.plot(prey, predator, 'purple', linewidth=1.5, alpha=0.7)
    ax2.plot(prey[0], predator[0], 'go', markersize=10, label='Start')
    ax2.plot(prey[-1], predator[-1], 'ro', markersize=10, label='End')
    ax2.set_xlabel('Prey Population', fontsize=12)
    ax2.set_ylabel('Predator Population', fontsize=12)
    ax2.set_title('Phase Space (Predator vs Prey)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_method_comparison(t: np.ndarray, analytical: np.ndarray,
                          euler: np.ndarray, rk4: np.ndarray,
                          title: str = "Comparison of Numerical Methods",
                          save_path: Optional[str] = None) -> Figure:
    """
    Compare different numerical methods with analytical solution.
    
    Args:
        t: Time array
        analytical: Analytical solution
        euler: Euler method solution
        rk4: RK4 method solution
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Solutions plot
    ax1.plot(t, analytical, 'k-', linewidth=2, label='Analytical', alpha=0.8)
    ax1.plot(t, euler, 'r--', linewidth=1.5, label='Euler Method', alpha=0.7)
    ax1.plot(t, rk4, 'b:', linewidth=2, label='RK4 Method', alpha=0.7)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Population Size', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Error plot
    euler_error = np.abs(euler - analytical)
    rk4_error = np.abs(rk4 - analytical)
    
    ax2.semilogy(t, euler_error, 'r-', linewidth=2, label='Euler Error')
    ax2.semilogy(t, rk4_error, 'b-', linewidth=2, label='RK4 Error')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Absolute Error (log scale)', fontsize=12)
    ax2.set_title('Numerical Error Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multiple_scenarios(t: np.ndarray, populations: list, labels: list,
                           title: str = "Multiple Growth Scenarios",
                           save_path: Optional[str] = None) -> Figure:
    """
    Plot multiple population scenarios on the same graph.
    
    Args:
        t: Time array
        populations: List of population arrays
        labels: List of labels for each scenario
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(populations)))
    
    for i, (pop, label) in enumerate(zip(populations, labels)):
        ax.plot(t, pop, linewidth=2, label=label, color=colors[i])
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Population Size', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comprehensive_report(exponential_data: dict, logistic_data: dict,
                               predator_prey_data: dict,
                               save_path: Optional[str] = None) -> Figure:
    """
    Create a comprehensive figure with all three models.
    
    Args:
        exponential_data: Dict with 't' and 'N' keys
        logistic_data: Dict with 't', 'N', and 'K' keys
        predator_prey_data: Dict with 't', 'prey', and 'predator' keys
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Exponential growth
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(exponential_data['t'], exponential_data['N'], 'b-', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Population')
    ax1.set_title('Exponential Growth', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Logistic growth
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(logistic_data['t'], logistic_data['N'], 'g-', linewidth=2, label='Population')
    if 'K' in logistic_data:
        ax2.axhline(y=logistic_data['K'], color='r', linestyle='--', 
                   label=f"K={logistic_data['K']}")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Population')
    ax2.set_title('Logistic Growth', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Predator-prey time series
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(predator_prey_data['t'], predator_prey_data['prey'], 
            'b-', linewidth=2, label='Prey')
    ax3.plot(predator_prey_data['t'], predator_prey_data['predator'], 
            'r-', linewidth=2, label='Predator')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Population')
    ax3.set_title('Predator-Prey Dynamics', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Predator-prey phase space
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(predator_prey_data['prey'], predator_prey_data['predator'], 
            'purple', linewidth=1.5)
    ax4.set_xlabel('Prey Population')
    ax4.set_ylabel('Predator Population')
    ax4.set_title('Phase Space', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Population Growth Models Overview', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
