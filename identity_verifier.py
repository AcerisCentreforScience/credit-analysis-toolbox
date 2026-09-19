#!/usr/bin/env python3
"""
Identity Verification Script
=============================
Verifies the core identity from the paper:
    A_i^cf(u) = sum_{S ∋ i} f_S(u_S)

i.e., the counterfactual advantage of an agent equals the sum of all
ANOVA/Möbius components that involve that agent.

This should hold to numerical precision (~1e-15) on any reward table.

Usage:
    python identity_verifier.py --demo
    python identity_verifier.py --reward my_reward.npy
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import identity_check, anova_decomposition, counterfactual_advantage


def verify_identity(reward_table: np.ndarray, action_sizes: list = None):
    """
    Verify the counterfactual advantage identity for all agents.

    Returns the maximum absolute error across all agents.
    """
    if action_sizes is None:
        action_sizes = list(reward_table.shape)

    n = len(action_sizes)

    print("=" * 60)
    print("IDENTITY VERIFICATION")
    print("=" * 60)
    print(f"Number of agents:       {n}")
    print(f"Action sizes per agent: {action_sizes}")
    print()
    print("Verifying: A_i^cf(u) = sum_{S ∋ i} f_S(u_S)")
    print()

    max_err = 0.0
    for i in range(n):
        err = identity_check(reward_table, action_sizes, i)
        max_err = max(max_err, err)
        print(f"  Agent {i}: max absolute error = {err:.2e}")

    print()
    print("=" * 60)
    if max_err < 1e-10:
        print(f"✓ Identity verified! Max error = {max_err:.2e} (machine precision)")
    else:
        print(f"✗ Identity FAILED. Max error = {max_err:.2e}")
    print("=" * 60)

    return max_err


def demo():
    """Built-in demo: random reward table."""
    print("Running built-in demo: random 3-agent, 3-action reward table")
    print()
    np.random.seed(42)
    n, a = 3, 3
    action_sizes = [a, a, a]
    reward_table = np.random.randn(a, a, a)
    verify_identity(reward_table, action_sizes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identity Verification Script")
    parser.add_argument("--reward", type=str, help="Path to .npy reward table")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo")
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.reward:
        reward_table = np.load(args.reward)
        verify_identity(reward_table)
    else:
        parser.print_help()
        print("\nNo input specified. Running demo...\n")
        demo()
