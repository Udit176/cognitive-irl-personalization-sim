"""Patient generative model with fatigue dynamics."""

import numpy as np
from typing import Tuple


class Patient:
    """
    Patient that generates motor commands based on a quadratic cost function.

    The patient minimizes a one-step cost:
        C_t(u_t) = w_goal * ||p_{t+1}(u_t) - g||^2
                 + w_effort_eff(t) * ||u_t||^2
                 + w_smooth * ||u_t - u_{t-1}||^2

    where w_effort_eff(t) = w_effort * (1 + k_f * f_t).
    """

    def __init__(self, w_goal: float, w_effort: float, w_smooth: float) -> None:
        """
        Initialize patient with objective parameters.

        Args:
            w_goal: Weight on goal-reaching cost.
            w_effort: Weight on effort cost.
            w_smooth: Weight on smoothness cost.
        """
        self.w_goal = w_goal
        self.w_effort = w_effort
        self.w_smooth = w_smooth
        self.dt = 0.1
        self.goal = np.array([1.0, 1.0])
        self.k_f = 3.0  # Fatigue scaling factor
        self.sigma_u = 0.03  # Action noise

    def act(
        self,
        state: np.ndarray,
        assist: np.ndarray,
        fatigue: float,
        u_prev: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate patient motor command.

        Args:
            state: Current state [x, y, vx, vy].
            assist: Robot assistance a_t.
            fatigue: Current fatigue level f_t in [0, 1].
            u_prev: Previous patient command (u_{t-1}).
            rng: Random generator for action noise.

        Returns:
            Tuple of (u_t_final, u_t_optimal) where u_t_final includes noise
            and clipping, u_t_optimal is the noiseless solution before clipping.
        """
        # Compute effective effort weight
        w_effort_eff = self.w_effort * (1.0 + self.k_f * fatigue)

        # Compute optimal action (closed-form)
        u_t_optimal = self._compute_optimal_action(state, assist, w_effort_eff, u_prev)

        # Add noise
        noise = rng.normal(0, self.sigma_u, size=2)
        u_t_noisy = u_t_optimal + noise

        # Clip
        u_t_final = np.clip(u_t_noisy, -1.0, 1.0)

        return u_t_final, u_t_optimal

    def _compute_optimal_action(
        self,
        state: np.ndarray,
        assist: np.ndarray,
        w_effort_eff: float,
        u_prev: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the closed-form optimal action that minimizes the quadratic cost.

        The cost is:
            C(u) = w_goal * ||p_{t+1}(u) - g||^2
                 + w_effort_eff * ||u||^2
                 + w_smooth * ||u - u_prev||^2

        where p_{t+1}(u) = p_t + dt * v_t + dt^2 * (u + a).

        Taking the derivative and setting to zero:
            dC/du = 2 * w_goal * dt^2 * (delta + dt^2 * u)
                  + 2 * w_effort_eff * u
                  + 2 * w_smooth * (u - u_prev) = 0

        where delta = p_t + dt * v_t + dt^2 * a - g.

        Solving for u:
            u = (w_smooth * u_prev - w_goal * dt^2 * delta) /
                (w_goal * dt^4 + w_effort_eff + w_smooth)

        Args:
            state: Current state [x, y, vx, vy].
            assist: Robot assistance a_t.
            w_effort_eff: Effective effort weight.
            u_prev: Previous action.

        Returns:
            Optimal action u_t (before clipping).
        """
        position = state[:2]
        velocity = state[2:]

        # Predicted position without patient action
        delta = position + self.dt * velocity + (self.dt ** 2) * assist - self.goal

        # Coefficients
        numerator = self.w_smooth * u_prev - self.w_goal * (self.dt ** 2) * delta
        denominator = self.w_goal * (self.dt ** 4) + w_effort_eff + self.w_smooth

        u_optimal = numerator / denominator

        return u_optimal


def update_fatigue(fatigue: float, u: np.ndarray, alpha_f: float = 0.03, beta_f: float = 0.01) -> float:
    """
    Update fatigue level based on motor command.

    f_{t+1} = clip(f_t + alpha_f * ||u_t||^2 - beta_f, [0, 1])

    Args:
        fatigue: Current fatigue level.
        u: Motor command.
        alpha_f: Fatigue accumulation rate.
        beta_f: Fatigue recovery rate.

    Returns:
        Updated fatigue level.
    """
    f_next = fatigue + alpha_f * np.dot(u, u) - beta_f
    return np.clip(f_next, 0.0, 1.0)


def get_initial_fatigue(episode_idx: int) -> float:
    """
    Return initial fatigue for an episode.

    Args:
        episode_idx: Episode number (0-indexed).

    Returns:
        Initial fatigue f_0.
    """
    if episode_idx < 50:
        return 0.0
    else:
        return 0.6


def sample_patient_theta(rng: np.random.Generator) -> Tuple[float, float, float]:
    """
    Sample patient objective parameters.

    Args:
        rng: Random generator.

    Returns:
        Tuple of (w_goal, w_effort, w_smooth).
    """
    w_goal = 1.0  # Fixed
    w_effort = rng.uniform(0.05, 0.25)
    w_smooth = 0.02  # Fixed
    return w_goal, w_effort, w_smooth
