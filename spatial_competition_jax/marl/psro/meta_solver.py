"""Meta-game solvers for PSRO.

Provides solvers for finding Nash equilibria of the empirical
normal-form game, plus utilities to measure exploitability.

**Symmetric solvers** (single population):

* **Logit dynamics** — maintains logits and does additive gradient
  updates followed by softmax.
* **LP Nash** — exact via the antisymmetric game ``A = U − Uᵀ``.

**Asymmetric solver** (two populations):

* **Bimatrix logit dynamics** — alternating logit best-response
  updates for each player.  Finds an approximate Nash of the
  general-sum bimatrix game ``(U0, U1)``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Logit Dynamics  (default solver)
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax."""
    e: np.ndarray = np.exp(x - x.max())
    out: np.ndarray = e / e.sum()
    return out


def projected_replicator_dynamics(
    payoff_matrix: np.ndarray,
    *,
    num_iters: int = 100_000,
    dt: float = 0.05,
    tol: float = 1e-9,
) -> np.ndarray:
    """Find an approximate symmetric Nash equilibrium via logit dynamics.

    Maintains a vector of **logits** (unconstrained reals) and performs
    additive gradient ascent::

        logit_i  +=  dt * ( (U σ)_i  −  σᵀ U σ )
        σ = softmax(logits)

    This is equivalent to mirror descent with negative-entropy
    regularisation and guarantees that *every* strategy retains
    positive probability throughout optimisation — unlike
    multiplicative-weights / standard replicator dynamics, which
    irreversibly kills strategies once their weight approaches zero.

    Args:
        payoff_matrix: ``(K, K)`` payoff matrix where ``U[i, j]`` is
            the row player's payoff when playing pure strategy *i*
            against column player's pure strategy *j*.
        num_iters: Maximum number of iterations.
        dt: Base learning rate for the logit update.
        tol: Convergence tolerance on the strategy change (L∞).

    Returns:
        ``(K,)`` mixed-strategy vector (probability simplex).
    """
    K = payoff_matrix.shape[0]
    if K == 1:
        return np.ones(1, dtype=np.float64)

    U = payoff_matrix.astype(np.float64)
    logits = np.zeros(K, dtype=np.float64)  # uniform start
    sigma = _softmax(logits)

    for _ in range(num_iters):
        expected = U @ sigma                    # (K,)
        avg = sigma @ expected                  # scalar
        advantages = expected - avg             # (K,)

        # Adaptive step size to avoid overshooting
        max_adv = np.max(np.abs(advantages))
        dt_eff = min(dt, 1.0 / (max_adv + 1e-12))

        logits += dt_eff * advantages
        # Re-centre for numerical stability (doesn't change softmax)
        logits -= logits.mean()

        new_sigma = _softmax(logits)

        if np.max(np.abs(new_sigma - sigma)) < tol:
            sigma = new_sigma
            break

        sigma = new_sigma

    return sigma


# ---------------------------------------------------------------------------
# LP Nash (exact, for verification / small games)
# ---------------------------------------------------------------------------


def lp_nash_symmetric(
    payoff_matrix: np.ndarray,
) -> np.ndarray:
    """Find a symmetric Nash equilibrium via linear programming.

    For a 2-player symmetric game with payoff matrix *U*, the
    symmetric Nash is the **maximin strategy** of the antisymmetric
    game ``A = U − Uᵀ``.  We solve::

        max  v
        s.t. A σ  ≥  v·1
             σ ≥ 0,  Σ σ_i = 1

    which is a standard LP.

    Args:
        payoff_matrix: ``(K, K)`` payoff matrix.

    Returns:
        ``(K,)`` mixed-strategy vector.
    """
    K = payoff_matrix.shape[0]
    if K == 1:
        return np.ones(1, dtype=np.float64)

    A = (payoff_matrix - payoff_matrix.T).astype(np.float64)

    # Decision variables: [σ_0, σ_1, …, σ_{K-1}, v]
    # Objective: minimise −v  (i.e. maximise v)
    c = np.zeros(K + 1, dtype=np.float64)
    c[-1] = -1.0  # minimise −v

    # Inequality constraints:  v·1 − A σ ≤ 0
    #   i.e.  −A σ + v ≤ 0   for each row i
    A_ub = np.zeros((K, K + 1), dtype=np.float64)
    A_ub[:, :K] = -A          # −A σ
    A_ub[:, K] = 1.0          # + v
    b_ub = np.zeros(K, dtype=np.float64)

    # Equality constraint:  Σ σ_i = 1
    A_eq = np.zeros((1, K + 1), dtype=np.float64)
    A_eq[0, :K] = 1.0
    b_eq = np.ones(1, dtype=np.float64)

    # Bounds:  σ_i ≥ 0,  v is free
    bounds = [(0.0, None)] * K + [(None, None)]

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )

    if result.success:
        sigma: np.ndarray = result.x[:K].astype(np.float64)
        sigma = np.maximum(sigma, 0.0)
        sigma /= sigma.sum()
        return sigma

    # Fallback to logit dynamics if LP fails
    return projected_replicator_dynamics(payoff_matrix)


# ---------------------------------------------------------------------------
# Combined solver — pick best of logit dynamics and LP
# ---------------------------------------------------------------------------


def solve_meta_game(
    payoff_matrix: np.ndarray,
) -> np.ndarray:
    """Find the best symmetric Nash approximation.

    Runs both logit dynamics and the LP solver, then returns
    whichever strategy has **lower exploitability** in the original
    (general-sum) game.

    This is robust to both:
    - Games with cycling structure (logit dynamics shines)
    - General-sum games (LP on antisymmetric game provides a
      complementary solution)

    Args:
        payoff_matrix: ``(K, K)`` payoff matrix.

    Returns:
        ``(K,)`` mixed-strategy vector.
    """
    K = payoff_matrix.shape[0]
    if K == 1:
        return np.ones(1, dtype=np.float64)

    sigma_logit = projected_replicator_dynamics(payoff_matrix)
    exploit_logit = compute_exploitability(payoff_matrix, sigma_logit)

    sigma_lp = lp_nash_symmetric(payoff_matrix)
    exploit_lp = compute_exploitability(payoff_matrix, sigma_lp)

    if exploit_logit <= exploit_lp:
        return sigma_logit
    return sigma_lp


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


# ---------------------------------------------------------------------------
# Bimatrix solvers (asymmetric / two-population PSRO)
# ---------------------------------------------------------------------------


def solve_bimatrix_game(
    U0: np.ndarray,
    U1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find an approximate Nash equilibrium of a bimatrix game.

    Uses alternating logit dynamics: each player updates its logits
    via gradient ascent against the opponent's current mixed strategy.

    Args:
        U0: ``(K0, K1)`` payoff matrix for player 0.
            ``U0[i, j]`` is player 0's payoff when player 0 plays
            pure strategy *i* and player 1 plays pure strategy *j*.
        U1: ``(K0, K1)`` payoff matrix for player 1.
            ``U1[i, j]`` is player 1's payoff in the same matchup.

    Returns:
        ``(sigma0, sigma1)`` — mixed strategies for each player.
    """
    K0, K1 = U0.shape
    if K0 == 1 and K1 == 1:
        return np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64)

    U0 = U0.astype(np.float64)
    U1 = U1.astype(np.float64)

    logits0 = np.zeros(K0, dtype=np.float64)
    logits1 = np.zeros(K1, dtype=np.float64)
    sigma0 = _softmax(logits0)
    sigma1 = _softmax(logits1)

    dt = 0.05
    tol = 1e-9

    for _ in range(100_000):
        # Player 0 best-responds to sigma1
        expected0 = U0 @ sigma1  # (K0,)
        avg0 = sigma0 @ expected0
        adv0 = expected0 - avg0
        dt0 = min(dt, 1.0 / (np.max(np.abs(adv0)) + 1e-12))
        logits0 += dt0 * adv0
        logits0 -= logits0.mean()

        # Player 1 best-responds to sigma0
        expected1 = U1.T @ sigma0  # (K1,)
        avg1 = sigma1 @ expected1
        adv1 = expected1 - avg1
        dt1 = min(dt, 1.0 / (np.max(np.abs(adv1)) + 1e-12))
        logits1 += dt1 * adv1
        logits1 -= logits1.mean()

        new_sigma0 = _softmax(logits0)
        new_sigma1 = _softmax(logits1)

        if (np.max(np.abs(new_sigma0 - sigma0)) < tol
                and np.max(np.abs(new_sigma1 - sigma1)) < tol):
            sigma0 = new_sigma0
            sigma1 = new_sigma1
            break

        sigma0 = new_sigma0
        sigma1 = new_sigma1

    return sigma0, sigma1


def compute_exploitability_bimatrix(
    U0: np.ndarray,
    U1: np.ndarray,
    sigma0: np.ndarray,
    sigma1: np.ndarray,
) -> float:
    """Compute summed exploitability of a bimatrix strategy profile.

    ::

        exploit = (max_i e_i^T U0 σ1 - σ0^T U0 σ1)
                + (max_j e_j^T U1^T σ0 - σ1^T U1^T σ0)

    Zero means ``(σ0, σ1)`` is a Nash equilibrium.

    Args:
        U0: ``(K0, K1)`` player 0 payoff matrix.
        U1: ``(K0, K1)`` player 1 payoff matrix.
        sigma0: ``(K0,)`` player 0 mixed strategy.
        sigma1: ``(K1,)`` player 1 mixed strategy.

    Returns:
        Non-negative scalar exploitability (sum of both players).
    """
    U0 = U0.astype(np.float64)
    U1 = U1.astype(np.float64)
    s0 = sigma0.astype(np.float64)
    s1 = sigma1.astype(np.float64)

    # Player 0's incentive to deviate
    exp0 = U0 @ s1
    exploit0 = max(exp0.max() - s0 @ exp0, 0.0)

    # Player 1's incentive to deviate
    exp1 = U1.T @ s0
    exploit1 = max(exp1.max() - s1 @ exp1, 0.0)

    return float(exploit0 + exploit1)
