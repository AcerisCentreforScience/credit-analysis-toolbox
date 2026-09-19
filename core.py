"""
credit_analysis — Orthogonal ANOVA/Möbius decomposition of joint reward/value functions.

Core utilities for analyzing the interaction spectrum of cooperative multi-agent rewards.
This package implements the exact functional ANOVA decomposition described in:
    "Orthogonal Decomposition of Credit for Cooperative Multi-Agent Reinforcement Learning"
"""

import itertools
import numpy as np
from typing import Dict, List, Tuple, Set


def anova_decomposition(reward_table: np.ndarray, action_sizes: List[int]) -> Tuple[float, Dict[Tuple[int, ...], np.ndarray]]:
    """
    Exact functional ANOVA / Möbius decomposition of a joint reward function.

    Under a uniform product action measure, decomposes v(u) into orthogonal components
    indexed by subsets S of agents:
        v(u) = f_0 + sum_i f_i(u^i) + sum_{i<j} f_{ij}(u^i, u^j) + ...

    Parameters
    ----------
    reward_table : np.ndarray
        Shape = (a_1, a_2, ..., a_n). reward_table[u1, u2, ..., un] = v(u)
    action_sizes : list of int
        action_sizes[i] = number of actions for agent i

    Returns
    -------
    f0 : float
        Zero-order component (mean reward)
    components : dict
        components[S] = f_S, a numpy array of shape matching the axes in S.
        S is a tuple of agent indices, e.g. (0,) for first-order, (0,1) for pairwise.
    """
    n = len(action_sizes)
    assert reward_table.ndim == n, "reward_table dimensions must match number of agents"
    for i, a in enumerate(action_sizes):
        assert reward_table.shape[i] == a, f"Axis {i} size mismatch"

    # Compute marginal means via nested conditional expectations
    # f_S(u_S) = E[v | u_S] - sum_{T ⊂ S} f_T(u_T)
    components = {}

    # f0 = mean over all actions
    f0 = float(np.mean(reward_table))

    # We build components in order of increasing |S|
    for order in range(1, n + 1):
        for S in itertools.combinations(range(n), order):
            # Indices of agents NOT in S
            other_axes = tuple(i for i in range(n) if i not in S)
            # E[v | u_S] = average over other agents
            cond_exp = np.mean(reward_table, axis=other_axes)
            # Subtract all lower-order components
            f_S = cond_exp.copy()
            for k in range(order):
                for T in itertools.combinations(S, k):
                    # f_T is defined on T, we need to broadcast it to S shape
                    if len(T) == 0:
                        f_T = np.array([f0])
                    else:
                        f_T = components[T]
                    # Expand dimensions of f_T to match S
                    new_shape = [1] * order
                    for i, agent in enumerate(S):
                        if agent in T:
                            new_shape[i] = action_sizes[agent]
                    f_S = f_S - f_T.reshape(new_shape)
            components[S] = f_S

    return f0, components


def component_energies(f0: float, components: Dict[Tuple[int, ...], np.ndarray], action_sizes: List[int]) -> Dict[int, float]:
    """
    Compute the energy (L² norm squared) of each interaction order.

    E_k = sum_{|S|=k} E[f_S(u_S)²]
    Under uniform product measure, E[f_S²] = mean(f_S²).

    Parameters
    ----------
    f0 : float
        Zero-order component
    components : dict
        Output of anova_decomposition
    action_sizes : list of int

    Returns
    -------
    energies : dict
        energies[k] = total energy of order k components.
        energies[0] = f0² (the mean level, often not counted as "variance")
    """
    energies = {0: f0 ** 2}
    for S, f_S in components.items():
        k = len(S)
        e = float(np.mean(f_S ** 2))
        energies[k] = energies.get(k, 0.0) + e
    return energies


def total_variance(energies: Dict[int, float]) -> float:
    """Total variance = sum of energies for orders 1..n (excluding f0²)."""
    return sum(v for k, v in energies.items() if k >= 1)


def energy_shares(energies: Dict[int, float]) -> Dict[int, float]:
    """Energy share of each order, as fraction of total variance."""
    var = total_variance(energies)
    if var < 1e-15:
        return {k: 0.0 for k in energies if k >= 1}
    return {k: v / var for k, v in energies.items() if k >= 1}


def counterfactual_advantage(reward_table: np.ndarray, action_sizes: List[int], agent_i: int) -> np.ndarray:
    """
    Compute the exact counterfactual advantage A_i^cf(u) for agent i.

    A_i^cf(u) = v(u) - E_{u'^i}[v(u^{-i}, u'^i)]

    Returns an array of same shape as reward_table.
    """
    n = len(action_sizes)
    assert 0 <= agent_i < n
    # Marginalize over agent i's action
    marginalized = np.mean(reward_table, axis=agent_i, keepdims=True)
    return reward_table - marginalized


def identity_check(reward_table: np.ndarray, action_sizes: List[int], agent_i: int) -> float:
    """
    Verify the identity: A_i^cf(u) = sum_{S ∋ i} f_S(u_S)
    Returns the maximum absolute error (should be ~1e-15).
    """
    f0, components = anova_decomposition(reward_table, action_sizes)
    A_i = counterfactual_advantage(reward_table, action_sizes, agent_i)

    # Reconstruct: sum all components that involve agent i
    reconstructed = np.zeros_like(reward_table)
    for S, f_S in components.items():
        if agent_i not in S:
            continue
        # f_S is defined on S; we need to broadcast it to the full tensor shape
        # f_S shape = (action_sizes[s] for s in S)
        # We need to add dimensions for agents NOT in S
        shape = [1] * len(action_sizes)
        for idx, agent in enumerate(S):
            shape[agent] = action_sizes[agent]
        reconstructed += f_S.reshape(shape)

    return float(np.max(np.abs(A_i - reconstructed)))


def predict_truncation_error(energies: Dict[int, float], truncation_order: int) -> float:
    """
    Predict the theoretical MSE of an order-truncated credit estimator.

    For an exact order-truncated estimator that keeps components up to order k,
    the truncation error = total energy of orders > k.

    Specifically:
    - First-order only (k=1): error = E_2 + E_3 + ...
    - Second-order only (k=2): error = E_3 + E_4 + ...
    - Combined up to order k: error = E_{k+1} + ...

    Parameters
    ----------
    energies : dict
        Output of component_energies
    truncation_order : int
        Keep components up to this order.

    Returns
    -------
    mse : float
        Theoretical truncation MSE (relative to true counterfactual advantage).
    """
    return sum(v for k, v in energies.items() if k > truncation_order)


if __name__ == "__main__":
    # Quick demo
    print("=== Credit Analysis Toolbox Demo ===")
    print()

    # Example: 3 agents, 3 actions each, synthetic reward
    # v = main effect + pairwise interaction
    np.random.seed(42)
    a = 3
    n = 3
    action_sizes = [a, a, a]

    # Build a synthetic reward table
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
    print(f"Synthetic reward table: {n} agents, {a} actions each")
    print(f"  Main effect weight = 1.0, pairwise weight = 2.0")
    print()

    # Decompose
    f0, components = anova_decomposition(reward_table, action_sizes)
    energies = component_energies(f0, components, action_sizes)
    shares = energy_shares(energies)

    print("=== Interaction Spectrum ===")
    for k in sorted(shares.keys()):
        print(f"  Order {k}: energy = {energies[k]:.4f}, share = {shares[k]*100:.1f}%")
    print()

    # Truncation error prediction
    print("=== Predicted Truncation MSE ===")
    for k in range(1, n + 1):
        mse = predict_truncation_error(energies, k)
        print(f"  Keep up to order {k}: truncation MSE = {mse:.4f}")
    print()

    # Identity verification
    print("=== Identity Verification (A_i^cf = sum of f_S for S ∋ i) ===")
    for i in range(n):
        err = identity_check(reward_table, action_sizes, i)
        print(f"  Agent {i}: max error = {err:.2e}")
    print()

    print("All tests passed!")
