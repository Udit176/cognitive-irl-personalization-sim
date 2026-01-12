"""Shared utilities for learning methods."""

import numpy as np
from typing import Tuple


# Constants
DT = 0.1
GOAL = np.array([1.0, 1.0])
W_GOAL = 1.0
W_SMOOTH = 0.02
K_F = 3.0
SIGMA_U = 0.03
ALPHA_F = 0.03
BETA_F = 0.01


def compute_optimal_action(
    state: np.ndarray,
    assist: np.ndarray,
    w_effort_eff: float,
    u_prev: np.ndarray,
) -> np.ndarray:
    """
    Compute optimal action given effective effort weight.

    This is the same closed-form solution used in the patient model.

    Args:
        state: Current state [x, y, vx, vy].
        assist: Robot assistance.
        w_effort_eff: Effective effort weight.
        u_prev: Previous action.

    Returns:
        Optimal action (before clipping and noise).
    """
    position = state[:2]
    velocity = state[2:]

    # Predicted position error without patient action
    delta = position + DT * velocity + (DT ** 2) * assist - GOAL

    # Closed-form solution
    numerator = W_SMOOTH * u_prev - W_GOAL * (DT ** 2) * delta
    denominator = W_GOAL * (DT ** 4) + w_effort_eff + W_SMOOTH

    u_optimal = numerator / denominator

    return u_optimal


def compute_action_log_likelihood(
    u_observed: np.ndarray,
    u_optimal: np.ndarray,
    sigma: float = SIGMA_U,
) -> float:
    """
    Compute log-likelihood of observed action under Gaussian noise model.

    Args:
        u_observed: Observed action.
        u_optimal: Predicted optimal action.
        sigma: Standard deviation of action noise.

    Returns:
        Log-likelihood value.
    """
    diff = u_observed - u_optimal
    log_likelihood = -0.5 * np.sum(diff ** 2) / (sigma ** 2)
    log_likelihood -= np.log(2 * np.pi * sigma ** 2)  # Normalization constant
    return log_likelihood


def simulate_fatigue_forward(
    u_sequence: np.ndarray,
    f_0: float,
) -> np.ndarray:
    """
    Simulate fatigue dynamics forward given action sequence.

    Args:
        u_sequence: Array of actions, shape (T, 2).
        f_0: Initial fatigue.

    Returns:
        Array of fatigue values, shape (T,).
    """
    T = len(u_sequence)
    fatigue = np.zeros(T)
    f_t = f_0

    for t in range(T):
        fatigue[t] = f_t
        f_next = f_t + ALPHA_F * np.dot(u_sequence[t], u_sequence[t]) - BETA_F
        f_t = np.clip(f_next, 0.0, 1.0)

    return fatigue


def compute_episode_log_likelihood(
    states: np.ndarray,
    actions_observed: np.ndarray,
    actions_robot: np.ndarray,
    w_effort: float,
    fatigues: np.ndarray,
) -> float:
    """
    Compute total log-likelihood for an episode.

    Args:
        states: State sequence, shape (T, 4).
        actions_observed: Observed patient actions, shape (T, 2).
        actions_robot: Robot assistance sequence, shape (T, 2).
        w_effort: Patient effort weight.
        fatigues: Fatigue sequence, shape (T,).

    Returns:
        Total log-likelihood for the episode.
    """
    T = len(states)
    total_ll = 0.0
    u_prev = np.zeros(2)

    for t in range(T):
        # Compute effective effort weight
        w_effort_eff = w_effort * (1.0 + K_F * fatigues[t])

        # Compute optimal action
        u_optimal = compute_optimal_action(states[t], actions_robot[t], w_effort_eff, u_prev)

        # Compute log-likelihood
        ll = compute_action_log_likelihood(actions_observed[t], u_optimal)
        total_ll += ll

        u_prev = actions_observed[t]

    return total_ll


def compute_episode_log_likelihood_static(
    states: np.ndarray,
    actions_observed: np.ndarray,
    actions_robot: np.ndarray,
    w_effort: float,
) -> float:
    """
    Compute log-likelihood assuming static effort weight (no fatigue).

    Args:
        states: State sequence, shape (T, 4).
        actions_observed: Observed patient actions, shape (T, 2).
        actions_robot: Robot assistance sequence, shape (T, 2).
        w_effort: Patient effort weight (constant).

    Returns:
        Total log-likelihood for the episode.
    """
    T = len(states)
    total_ll = 0.0
    u_prev = np.zeros(2)

    for t in range(T):
        # Compute optimal action with constant effort weight
        u_optimal = compute_optimal_action(states[t], actions_robot[t], w_effort, u_prev)

        # Compute log-likelihood
        ll = compute_action_log_likelihood(actions_observed[t], u_optimal)
        total_ll += ll

        u_prev = actions_observed[t]

    return total_ll
