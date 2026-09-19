#!/usr/bin/env python3
"""
Truncation Error Predictor
==========================
Given the interaction spectrum of a reward, predict the theoretical truncation MSE
of any order-truncated credit estimator.

For an exact order-truncated estimator that keeps components up to order k,
the truncation error = total energy of all orders > k.

This is the pre-training prediction protocol from the paper:
    "Orthogonal Decomposition of Credit for Cooperative Multi-Agent Reinforcement Learning"

Usage:
    python truncation_predictor.py --demo
    python truncation_predictor.py --energies 0.5 0.3 0.2  # order 1, 2, 3, ...
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import predict_truncation_error, energy_shares


def predict(energies: dict, max_order: int = None):
    """
    Predict truncation MSE for estimators of various orders.

    Parameters
    ----------
    energies : dict
        energies[k] = total energy of order k (k >= 1)
    max_order : int, optional
        Maximum order to report. If None, use max key in energies.
    """
    if max_order is None:
        max_order = max(energies.keys())

    total = sum(energies.values())
    shares = {k: v / total for k, v in energies.items()}

    print("=" * 60)
    print("TRUNCATION ERROR PREDICTION")
    print("=" * 60)
    print(f"Total interaction variance: {total:.4f}")
    print()
    print("Spectrum:")
    for k in sorted(energies.keys()):
        print(f"  Order {k}: energy = {energies[k]:.4f} ({shares[k]*100:.1f}%)")
    print()
    print("Predicted truncation MSE for each estimator:")
    print(f"{'Estimator':<25} {'Truncation MSE':<15} {'As % of total':<15}")
    print("-" * 55)
    for k in range(0, max_order + 1):
        mse = predict_truncation_error(energies, k)
        pct = mse / total * 100 if total > 0 else 0
        label = f"Keep up to order {k}"
        print(f"{label:<25} {mse:<15.4f} {pct:<15.1f}%")
    print()

    # Recommendation
    best_k = min(range(0, max_order + 1), key=lambda k: predict_truncation_error(energies, k))
    print("Recommendation:")
    print(f"  - The combined credit (keep up to order {max_order}) has zero truncation error.")
    if energies.get(3, 0) < 0.01 * total:
        print("  - Third-order+ energy is negligible (< 1%), so second-order approximation is sufficient.")
    else:
        print(f"  - Third-order+ energy is {energies.get(3, 0)/total*100:.1f}%, so second-order model will have non-trivial error.")


def demo():
    """Built-in demo."""
    print("Running built-in demo:")
    print("  Main effect: E1 = 1.0")
    print("  Pairwise interaction: E2 = 0.8")
    print("  Third-order: E3 = 0.2")
    print()
    energies = {1: 1.0, 2: 0.8, 3: 0.2}
    predict(energies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truncation Error Predictor")
    parser.add_argument("--energies", nargs="+", type=float, help="E1 E2 E3 ... (energies of orders 1, 2, 3, ...)")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo")
    args = parser.parse_args()

    if args.demo or not args.energies:
        if not args.demo:
            print("No energies specified. Running demo...\n")
        demo()
    else:
        energies = {i + 1: e for i, e in enumerate(args.energies)}
        predict(energies)
