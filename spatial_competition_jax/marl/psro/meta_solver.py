"""Meta-game solvers for PSRO.

Provides Projected Replicator Dynamics (PRD) for finding a symmetric
Nash equilibrium of the empirical normal-form game, plus utilities to
measure exploitability.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Projected Replicator Dynamics
# ---------------------------------------------------------------------------


def projected_replicator_dynamics(
    payoff_matrix: np.ndarray,
    *,
    num_iters: int = 50_000,
    dt: float = 0.1,
    tol: float = 1e-8,
) -> np.ndarray:
    """Find an approximate symmetric Nash equilibrium via PRD.

    For a symmetric 2-player game with payoff matrix *U* (row player's
    perspective), finds a mixed strategy ``sigma`` such that
    ``(sigma, sigma)`` is an approximate Nash equilibrium of
    ``(U, U^T)``.

    The update rule is the discrete-time replicator dynamic::

        sigma_i ← sigma_i * (1 + dt * (e_i^T U sigma - sigma^T U sigma))

    followed by projection onto the probability simplex (clamp + renormalise).

    Args:
        payoff_matrix: ``(K, K)`` payoff matrix where ``U[i, j]`` is the
            row player's payoff when playing pure strategy *i* against
            column player's pure strategy *j*.
        num_iters: Maximum number of iterations.
        dt: Step size for the replicator update.
        tol: Convergence tolerance on the strategy change.

    Returns:
        ``(K,)`` mixed-strategy vector (probability simplex).
    """
    K = payoff_matrix.shape[0]
    if K == 1:
        return np.ones(1, dtype=np.float64)

    U = payoff_matrix.astype(np.float64)
    sigma = np.ones(K, dtype=np.float64) / K  # uniform initialisation

    for _ in range(num_iters):
        # Expected payoff for each pure strategy against current mixture
        expected = U @ sigma  # (K,)

        # Average payoff under current mixture
        avg = sigma @ expected  # scalar

        # Replicator update
        new_sigma = sigma * (1.0 + dt * (expected - avg))

        # Project onto probability simplex
        new_sigma = np.maximum(new_sigma, 0.0)
        total = new_sigma.sum()
        if total < 1e-12:
            # Degenerate case – fall back to uniform
            new_sigma = np.ones(K, dtype=np.float64) / K
        else:
            new_sigma /= total

        # Check convergence
        if np.max(np.abs(new_sigma - sigma)) < tol:
            sigma = new_sigma
            break

        sigma = new_sigma

    return sigma


# ---------------------------------------------------------------------------
# Exploitability
# ---------------------------------------------------------------------------


def compute_exploitability(
    payoff_matrix: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Compute the exploitability of a symmetric strategy profile.

    Exploitability measures how much a player could gain by deviating
    from the meta-strategy *sigma* to the best pure-strategy response,
    **within the current policy population**.

    ::

        exploitability = max_i (e_i^T U sigma) - sigma^T U sigma

    A value of zero means *sigma* is a Nash equilibrium of the
    empirical meta-game.

    Args:
        payoff_matrix: ``(K, K)`` payoff matrix.
        sigma: ``(K,)`` mixed strategy.

    Returns:
        Non-negative scalar exploitability.
    """
    U = payoff_matrix.astype(np.float64)
    s = sigma.astype(np.float64)

    expected = U @ s  # (K,)
    best_response_value = expected.max()
    mixture_value = s @ expected

    return float(max(best_response_value - mixture_value, 0.0))
