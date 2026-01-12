"""Baseline static IRL method that ignores fatigue dynamics."""

import numpy as np
from scipy.optimize import minimize_scalar
from typing import List, Dict
from utils import compute_episode_log_likelihood_static


def fit_static_model(episodes: List[Dict[str, np.ndarray]]) -> float:
    """
    Fit static effort weight by maximum likelihood.

    Assumes constant effort weight across all episodes, ignoring fatigue.

    Args:
        episodes: List of episode dictionaries with 'states', 'actions_patient',
                  'actions_robot'.

    Returns:
        Estimated w_effort.
    """

    def negative_log_likelihood(w_effort: float) -> float:
        """Compute negative log-likelihood across all episodes."""
        total_ll = 0.0
        for episode in episodes:
            ll = compute_episode_log_likelihood_static(
                states=episode['states'],
                actions_observed=episode['actions_patient'],
                actions_robot=episode['actions_robot'],
                w_effort=w_effort,
            )
            total_ll += ll
        return -total_ll

    # Optimize w_effort in [0.01, 0.5]
    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(0.01, 0.5),
        method='bounded',
        options={'xatol': 1e-3, 'maxiter': 20},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return result.x


def evaluate_static_model(
    episodes: List[Dict[str, np.ndarray]],
    w_effort: float,
) -> float:
    """
    Evaluate static model on episodes.

    Args:
        episodes: List of episode dictionaries.
        w_effort: Fitted effort weight.

    Returns:
        Mean negative log-likelihood per step.
    """
    total_nll = 0.0
    total_steps = 0

    for episode in episodes:
        ll = compute_episode_log_likelihood_static(
            states=episode['states'],
            actions_observed=episode['actions_patient'],
            actions_robot=episode['actions_robot'],
            w_effort=w_effort,
        )
        total_nll -= ll  # Convert to negative log-likelihood
        total_steps += len(episode['states'])

    return total_nll / total_steps
