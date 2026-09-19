# Credit Analysis Toolbox

Orthogonal ANOVA/Möbius decomposition of joint reward functions for cooperative multi-agent reinforcement learning.

This is the companion code for:
> **Orthogonal Decomposition of Credit for Cooperative Multi-Agent Reinforcement Learning**

---

## What this does

Given a joint reward table \(R(s, a)\) (shape = \(a_1 \times a_2 \times \dots \times a_n\)), this toolbox computes:

1. **The interaction spectrum** — how much of the reward's variation comes from individual agent effects, pairwise interactions, third-order group effects, etc.
2. **Truncation error predictions** — the theoretical MSE of any order-truncated credit estimator, *before any training*.
3. **Identity verification** — confirms the core mathematical identity: counterfactual advantage = sum of all components involving that agent.

---

## Quick start (30 seconds)

No installation needed beyond `numpy` (and `matplotlib` for plotting).

```bash
# Run the built-in demo for all three tools
python spectrum_analyzer.py --demo
python truncation_predictor.py --demo
python identity_verifier.py --demo
```

---

## Tool 1: Reward Spectrum Analyzer

**Input**: joint reward table \(R(a_1, a_2, \dots, a_n)\)
**Output**: energy share of each interaction order (0th, 1st, 2nd, 3rd, ...)

```bash
# Demo
python spectrum_analyzer.py --demo

# From a saved .npy file
python spectrum_analyzer.py --reward my_reward_table.npy
```

### What it shows

```
Order 1: energy = 6.48 (53.6%)   ← individual main effects
Order 2: energy = 5.61 (46.4%)   ← pairwise interactions
Order 3: energy = 0.00 (0.0%)    ← third-order interactions
```

This is the **interaction spectrum** of the reward — the central quantity that determines which credit estimator will be accurate.

---

## Tool 2: Truncation Error Predictor

**Input**: interaction energies \(E_1, E_2, E_3, \dots\)
**Output**: theoretical truncation MSE for any order-limited estimator

```bash
# Demo (E1=1.0, E2=0.8, E3=0.2)
python truncation_predictor.py --demo

# Custom energies
python truncation_predictor.py --energies 1.0 0.8 0.2
```

### What it shows

| Estimator | Truncation MSE | As % of total |
|-----------|---------------|---------------|
| Keep up to order 1 | 1.00 | 50% |
| Keep up to order 2 | 0.20 | 10% |
| Keep up to order 3 | 0.00 | 0% |

This is the **pre-training prediction protocol** from the paper: you can read off, before any learning, how accurate each order-truncated credit estimator will be.

---

## Tool 3: Identity Verifier

**Input**: joint reward table
**Output**: verification that \(A_i^{cf}(u) = \sum_{S \ni i} f_S(u_S)\) to machine precision

```bash
# Demo
python identity_verifier.py --demo

# From a saved .npy file
python identity_verifier.py --reward my_reward_table.npy
```

### What it shows

```
Agent 0: max absolute error = 2.22e-16
Agent 1: max absolute error = 2.22e-16
Agent 2: max absolute error = 2.22e-16

✓ Identity verified! Max error = 2.22e-16 (machine precision)
```

This confirms the **core theoretical identity** from the paper: the counterfactual advantage of an agent equals the sum of all ANOVA components that involve it.

---

## Core library

For use in your own code:

```python
import numpy as np
from core import anova_decomposition, component_energies, energy_shares, predict_truncation_error

# 3 agents, 3 actions each
n, a = 3, 3
reward_table = np.random.randn(a, a, a)
action_sizes = [a, a, a]

# Decompose
f0, components = anova_decomposition(reward_table, action_sizes)
energies = component_energies(f0, components, action_sizes)
shares = energy_shares(energies)

# Predict truncation error
mse_first_order = predict_truncation_error(energies, truncation_order=1)
print(f"First-order credit MSE: {mse_first_order:.4f}")
```

---

## File structure

```
credit_analysis/
├── core.py                    # Core decomposition library
├── spectrum_analyzer.py       # Tool 1: reward spectrum analysis
├── truncation_predictor.py    # Tool 2: truncation error prediction
├── identity_verifier.py       # Tool 3: identity verification
└── README.md                  # This file
```

---

## Theoretical background

The decomposition is the **functional ANOVA / Möbius inversion** of the joint action value \(v(u)\):

$$v(u) = f_0 + \sum_i f_i(u^i) + \sum_{i<j} f_{ij}(u^i, u^j) + \sum_{i<j<k} f_{ijk}(\cdot) + \cdots$$

Under a uniform product action measure, the components are mutually orthogonal, so the total variance decomposes as:

$$\text{Var}(v) = E_1 + E_2 + E_3 + \cdots$$

where \(E_k\) is the total energy of all \(k\)-order interaction components.

The counterfactual advantage of agent \(i\) is exactly:

$$A_i^{cf}(u) = \sum_{S \ni i} f_S(u_S)$$

This identity is verified to machine precision by `identity_verifier.py`.

---

## Requirements

- Python 3.7+
- numpy

That's it. No deep learning framework needed for the analysis tools.
