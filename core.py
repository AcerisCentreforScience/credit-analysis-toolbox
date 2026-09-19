"""
ANOVA/Mobius decomposition utilities for MARL credit analysis.
TODO: add sampling-based version for large n
"""

import itertools
import numpy as np
from typing import Dict, List, Tuple


def anova_decomposition(reward_table, action_sizes):
    """
    Exact functional ANOVA decomposition of v(u) under uniform product measure.

    Returns f0 and dict of components[S] = f_S.
    """
    n = len(action_sizes)
    assert reward_table.ndim == n

    f0 = float(np.mean(reward_table))
    components = {}

    # nested conditional expectations, nothing fancy
    for order in range(1, n + 1):
        for S in itertools.combinations(range(n), order):
            other_axes = tuple(i for i in range(n) if i not in S)
            cond_exp = np.mean(reward_table, axis=other_axes)

            f_S = cond_exp.copy()
            for k in range(order):
                for T in itertools.combinations(S, k):
                    if len(T) == 0:
                        f_T = np.array([f0])
                    else:
                        f_T = components[T]
                    new_shape = [1] * order
                    for i, agent in enumerate(S):
                        if agent in T:
                            new_shape[i] = action_sizes[agent]
                    f_S = f_S - f_T.reshape(new_shape)
            components[S] = f_S

    return f0, components


def component_energies(f0, components, action_sizes):
    """E_k = sum of mean(f_S^2) for all |S|=k"""
    energies = {0: f0 ** 2}
    for S, f_S in components.items():
        k = len(S)
        e = float(np.mean(f_S ** 2))
        energies[k] = energies.get(k, 0.0) + e
    return energies


def total_variance(energies):
    return sum(v for k, v in energies.items() if k >= 1)


def energy_shares(energies):
    var = total_variance(energies)
    if var < 1e-15:
        return {k: 0.0 for k in energies if k >= 1}
    return {k: v / var for k, v in energies.items() if k >= 1}


def counterfactual_advantage(reward_table, action_sizes, agent_i):
    """A_i^cf(u) = v(u) - E[v | u^{-i}]"""
    marginalized = np.mean(reward_table, axis=agent_i, keepdims=True)
    return reward_table - marginalized


def identity_check(reward_table, action_sizes, agent_i):
    """Verify A_i^cf = sum_{S ∋ i} f_S. Returns max error."""
    f0, components = anova_decomposition(reward_table, action_sizes)
    A_i = counterfactual_advantage(reward_table, action_sizes, agent_i)

    reconstructed = np.zeros_like(reward_table)
    for S, f_S in components.items():
        if agent_i not in S:
            continue
        shape = [1] * len(action_sizes)
        for idx, agent in enumerate(S):
            shape[agent] = action_sizes[agent]
        reconstructed += f_S.reshape(shape)

    return float(np.max(np.abs(A_i - reconstructed)))


def predict_truncation_error(energies, truncation_order):
    """Predicted MSE = energy of all orders > truncation_order"""
    return sum(v for k, v in energies.items() if k > truncation_order)


if __name__ == "__main__":
    # quick sanity check
    np.random.seed(42)
    n, a = 3, 3
    action_sizes = [a, a, a]

    main_effect = np.zeros((a, a, a))
    for i in range(n):
        vals = np.random.randn(a)
        shape = [1, 1, 1]
        shape[i] = a
        main_effect += vals.reshape(shape)

    pairwise = np.zeros((a, a, a))
    for i, j in itertools.combinations(range(n), 2):
        mat = np.random.randn(a, a)
        shape = [1, 1, 1]
        shape[i] = a
        shape[j] = a
        pairwise += mat.reshape(shape)

    reward_table = main_effect + 2.0 * pairwise

    f0, components = anova_decomposition(reward_table, action_sizes)
    energies = component_energies(f0, components, action_sizes)
    shares = energy_shares(energies)

    print("Order 1 share: %.1f%%" % (shares[1] * 100))
    print("Order 2 share: %.1f%%" % (shares[2] * 100))
    print("First-order truncation MSE: %.4f" % predict_truncation_error(energies, 1))

    for i in range(n):
        err = identity_check(reward_table, action_sizes, i)
        print("Agent %d identity error: %.2e" % (i, err))
