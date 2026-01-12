"""Fixed robot assistance policy using PD controller."""

import numpy as np


def compute_assist(state: np.ndarray, lambda_episode: float) -> np.ndarray:
    """
    Compute robot assistance using a PD controller.

    The nominal controller computes:
        b_t = Kp * (g - p_t) - Kd * v_t

    The actual assistance is scaled and clipped:
        a_t = clip(lambda_episode * b_t, [-1, 1])

    Args:
        state: Current state [x, y, vx, vy].
        lambda_episode: Assistance level scalar in [0, 1].

    Returns:
        Assistance command a_t in R^2.
    """
    Kp = 2.0
    Kd = 0.5
    goal = np.array([1.0, 1.0])

    position = state[:2]
    velocity = state[2:]

    # Nominal PD control
    b_t = Kp * (goal - position) - Kd * velocity

    # Scale and clip
    a_t = np.clip(lambda_episode * b_t, -1.0, 1.0)

    return a_t


def get_lambda_schedule(episode_idx: int) -> float:
    """
    Return the assistance level for a given episode.

    Args:
        episode_idx: Episode number (0-indexed).

    Returns:
        Lambda value for this episode.
    """
    # Episodes 0-99: lambda = 0.7
    return 0.7
