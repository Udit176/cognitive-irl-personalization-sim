"""Cognitive IRL with EM algorithm for latent fatigue inference."""

import numpy as np
from scipy.optimize import minimize_scalar
from typing import List, Dict, Tuple
from utils import (
    compute_episode_log_likelihood,
    simulate_fatigue_forward,
)


def e_step_episode(
    episode: Dict[str, np.ndarray],
    w_effort: float,
) -> Tuple[float, np.ndarray]:
    """
    E-step: Infer initial fatigue f_0 for an episode.

    Given current w_effort, find f_0 that maximizes episode log-likelihood.

    Args:
        episode: Episode dictionary with 'states', 'actions_patient', 'actions_robot'.
        w_effort: Current estimate of effort weight.

    Returns:
        Tuple of (inferred_f_0, inferred_fatigue_trajectory).
    """
    states = episode['states']
    actions_patient = episode['actions_patient']
    actions_robot = episode['actions_robot']

    def negative_log_likelihood(f_0: float) -> float:
        """Compute negative log-likelihood for given f_0."""
        # Simulate fatigue forward
        fatigues = simulate_fatigue_forward(actions_patient, f_0)

        # Compute episode log-likelihood
        ll = compute_episode_log_likelihood(
            states=states,
            actions_observed=actions_patient,
            actions_robot=actions_robot,
            w_effort=w_effort,
            fatigues=fatigues,
        )
        return -ll

    # Optimize f_0 in [0, 1]
    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(0.0, 1.0),
        method='bounded',
        options={'xatol': 1e-2, 'maxiter': 20},
    )

    if not result.success:
        raise RuntimeError(f"E-step optimization failed: {result.message}")

    f_0_hat = result.x
    fatigue_trajectory = simulate_fatigue_forward(actions_patient, f_0_hat)

    return f_0_hat, fatigue_trajectory


def m_step(
    episodes: List[Dict[str, np.ndarray]],
    inferred_fatigues: List[np.ndarray],
) -> float:
    """
    M-step: Update w_effort given inferred fatigue trajectories.

    Args:
        episodes: List of episode dictionaries.
        inferred_fatigues: List of inferred fatigue trajectories for each episode.

    Returns:
        Updated w_effort estimate.
    """

    def negative_log_likelihood(w_effort: float) -> float:
        """Compute negative total log-likelihood."""
        total_ll = 0.0
        for episode, fatigues in zip(episodes, inferred_fatigues):
            ll = compute_episode_log_likelihood(
                states=episode['states'],
                actions_observed=episode['actions_patient'],
                actions_robot=episode['actions_robot'],
                w_effort=w_effort,
                fatigues=fatigues,
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
        raise RuntimeError(f"M-step optimization failed: {result.message}")

    return result.x


def fit_em_model(
    episodes: List[Dict[str, np.ndarray]],
    n_iterations: int = 6,
    w_effort_init: float = 0.15,
) -> Tuple[float, List[float]]:
    """
    Fit cognitive IRL model using EM algorithm.

    Args:
        episodes: List of episode dictionaries.
        n_iterations: Number of EM iterations.
        w_effort_init: Initial value for w_effort.

    Returns:
        Tuple of (fitted_w_effort, list_of_inferred_f_0_per_episode).
    """
    w_effort = w_effort_init
    inferred_f_0s = []

    for iteration in range(n_iterations):
        # E-step: infer fatigue for each episode
        inferred_fatigues = []
        inferred_f_0s = []

        for episode in episodes:
            f_0_hat, fatigue_traj = e_step_episode(episode, w_effort)
            inferred_f_0s.append(f_0_hat)
            inferred_fatigues.append(fatigue_traj)

        # M-step: update w_effort
        w_effort = m_step(episodes, inferred_fatigues)

    return w_effort, inferred_f_0s


def evaluate_em_model(
    episodes: List[Dict[str, np.ndarray]],
    w_effort: float,
) -> float:
    """
    Evaluate EM model on episodes.

    For each episode, infer optimal f_0 and compute log-likelihood.

    Args:
        episodes: List of episode dictionaries.
        w_effort: Fitted effort weight.

    Returns:
        Mean negative log-likelihood per step.
    """
    total_nll = 0.0
    total_steps = 0

    for episode in episodes:
        # Infer f_0 for this episode
        f_0_hat, fatigue_traj = e_step_episode(episode, w_effort)

        # Compute log-likelihood
        ll = compute_episode_log_likelihood(
            states=episode['states'],
            actions_observed=episode['actions_patient'],
            actions_robot=episode['actions_robot'],
            w_effort=w_effort,
            fatigues=fatigue_traj,
        )

        total_nll -= ll  # Convert to negative log-likelihood
        total_steps += len(episode['states'])

    return total_nll / total_steps
