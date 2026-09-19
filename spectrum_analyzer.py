#!/usr/bin/env python3
"""
Reward Spectrum Analyzer
========================
Input: joint reward table R(s, a) — shape (a_1, a_2, ..., a_n)
Output: energy share of each interaction order (zero, first, second, third, ...)

This implements the orthogonal ANOVA/Möbius decomposition described in:
    "Orthogonal Decomposition of Credit for Cooperative Multi-Agent Reinforcement Learning"

Usage:
    python spectrum_analyzer.py --reward my_reward.npy
    python spectrum_analyzer.py --demo  # run built-in demo
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import anova_decomposition, component_energies, energy_shares, total_variance


def analyze_reward(reward_table: np.ndarray, action_sizes: list = None):
    """
    Analyze the interaction spectrum of a joint reward function.

    Parameters
    ----------
    reward_table : np.ndarray
        Shape = (a_1, a_2, ..., a_n)
    action_sizes : list of int, optional
        Action sizes for each agent. If None, inferred from reward_table.shape.

    Returns
    -------
    energies : dict
        energies[k] = total energy of order k
    shares : dict
        shares[k] = fraction of total variance in order k
    """
    if action_sizes is None:
        action_sizes = list(reward_table.shape)

    f0, components = anova_decomposition(reward_table, action_sizes)
    energies = component_energies(f0, components, action_sizes)
    shares = energy_shares(energies)
    var = total_variance(energies)

    print("=" * 60)
    print("REWARD SPECTRUM ANALYSIS")
    print("=" * 60)
    print(f"Number of agents:       {len(action_sizes)}")
    print(f"Action sizes per agent: {action_sizes}")
    print(f"Total variance:         {var:.4f}")
    print()
    print(f"{'Order':<10} {'Energy':<15} {'Share (%)':<15}")
    print("-" * 40)
    for k in sorted(shares.keys()):
        print(f"{k:<10} {energies[k]:<15.4f} {shares[k]*100:<15.1f}")
    print()

    # Interpretation
    print("Interpretation:")
    max_order = max(shares.keys())
    dominant_order = max(shares, key=shares.get)
    print(f"  - Dominant interaction order: {dominant_order} ({shares[dominant_order]*100:.1f}% of variance)")
    if dominant_order == 1:
        print("  → Main effects dominate. First-order credit will be nearly perfect.")
    elif dominant_order == 2:
        print("  → Pairwise interactions dominate. Second-order credit needed; first-order will be poor.")
    elif dominant_order >= 3:
        print("  → Higher-order interactions present. Second-order model will have residual error.")

    return energies, shares


def demo():
    """Built-in demo: 3-agent Stag-Hunt style reward."""
    print("Running built-in demo: 3-agent Stag-Hunt style reward")
    print()
    np.random.seed(42)
    n, a = 3, 3
    action_sizes = [a, a, a]

    # Build reward: main effect + pairwise interaction
    main_effect = np.zeros((a, a, a))
    for i in range(n):
        vals = np.random.randn(a)
        shape = [1] * n
        shape[i] = a
        main_effect += vals.reshape(shape)

    pairwise = np.zeros((a, a, a))
    for i, j in __import__('itertools').combinations(range(n), 2):
        mat = np.random.randn(a, a)
        shape = [1] * n
        shape[i] = a
        shape[j] = a
        pairwise += mat.reshape(shape)

    reward_table = main_effect + 2.0 * pairwise
    analyze_reward(reward_table, action_sizes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward Spectrum Analyzer")
    parser.add_argument("--reward", type=str, help="Path to .npy reward table")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo")
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.reward:
        reward_table = np.load(args.reward)
        analyze_reward(reward_table)
    else:
        parser.print_help()
        print("\nNo input specified. Running demo...\n")
        demo()
