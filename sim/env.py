"""2D point-mass reaching environment with robot assistance."""

import numpy as np
from typing import Tuple


class ReachingEnv:
    """
    2D point-mass reaching task with discrete-time dynamics.

    State: s = [x, y, vx, vy] in R^4
    Actions: patient u in R^2, robot assistance a in R^2
    Goal: fixed at [1.0, 1.0]
    """

    def __init__(self) -> None:
        self.dt: float = 0.1
        self.sigma_v: float = 0.02
        self.goal: np.ndarray = np.array([1.0, 1.0])
        self.success_threshold: float = 0.05
        self.max_steps: int = 80

        self.state: np.ndarray = np.zeros(4)
        self.t: int = 0

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        """
        Initialize episode with random start position and zero velocity.

        Args:
            rng: Numpy random generator for reproducibility.

        Returns:
            Initial state [x, y, vx, vy].
        """
        x0 = rng.uniform(-1.0, -0.5)
        y0 = rng.uniform(-1.0, -0.5)
        self.state = np.array([x0, y0, 0.0, 0.0])
        self.t = 0
        return self.state.copy()

    def step(self, u: np.ndarray, a: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, bool, bool, float]:
        """
        Execute one step of the dynamics.

        Args:
            u: Patient motor command (2D).
            a: Robot assistance (2D).
            rng: Random generator for process noise.

        Returns:
            Tuple of (next_state, done, success, distance_to_goal).
        """
        # Clip actions
        u_clipped = np.clip(u, -1.0, 1.0)
        a_clipped = np.clip(a, -1.0, 1.0)

        # Current state
        x, y, vx, vy = self.state

        # Process noise
        eps_vx = rng.normal(0, self.sigma_v)
        eps_vy = rng.normal(0, self.sigma_v)

        # Update dynamics
        x_next = x + self.dt * vx
        y_next = y + self.dt * vy
        vx_next = vx + self.dt * (u_clipped[0] + a_clipped[0]) + eps_vx
        vy_next = vy + self.dt * (u_clipped[1] + a_clipped[1]) + eps_vy

        self.state = np.array([x_next, y_next, vx_next, vy_next])
        self.t += 1

        # Check termination
        distance = np.linalg.norm(self.state[:2] - self.goal)
        success = distance <= self.success_threshold
        timeout = self.t >= self.max_steps
        done = success or timeout

        return self.state.copy(), done, success, distance

    def get_position(self) -> np.ndarray:
        """Return current 2D position."""
        return self.state[:2].copy()

    def get_velocity(self) -> np.ndarray:
        """Return current 2D velocity."""
        return self.state[2:].copy()
