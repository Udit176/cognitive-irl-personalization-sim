"""Dataset generation and storage utilities."""

import numpy as np
from typing import List, Dict, Any
import os


class TrajectoryBuffer:
    """Buffer to store trajectories for multiple patients and episodes."""

    def __init__(self) -> None:
        self.patient_ids: List[int] = []
        self.episode_ids: List[int] = []
        self.states: List[np.ndarray] = []
        self.actions_patient: List[np.ndarray] = []
        self.actions_robot: List[np.ndarray] = []
        self.fatigues: List[float] = []
        self.distances: List[float] = []
        self.successes: List[bool] = []
        self.steps_to_success: List[int] = []
        self.patient_thetas: Dict[int, np.ndarray] = {}

    def add_step(
        self,
        patient_id: int,
        episode_id: int,
        state: np.ndarray,
        u: np.ndarray,
        a: np.ndarray,
        fatigue: float,
        distance: float,
    ) -> None:
        """Add a single timestep to the buffer."""
        self.patient_ids.append(patient_id)
        self.episode_ids.append(episode_id)
        self.states.append(state.copy())
        self.actions_patient.append(u.copy())
        self.actions_robot.append(a.copy())
        self.fatigues.append(fatigue)
        self.distances.append(distance)

    def add_episode_outcome(self, success: bool, steps: int) -> None:
        """Add episode-level outcome information."""
        self.successes.append(success)
        self.steps_to_success.append(steps)

    def set_patient_theta(self, patient_id: int, theta: np.ndarray) -> None:
        """Store ground-truth theta for a patient."""
        self.patient_thetas[patient_id] = theta.copy()

    def save(self, filepath: str) -> None:
        """
        Save all collected data to an npz file.

        Args:
            filepath: Path to save the dataset.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert lists to arrays
        data = {
            'patient_ids': np.array(self.patient_ids, dtype=np.int32),
            'episode_ids': np.array(self.episode_ids, dtype=np.int32),
            'states': np.array(self.states, dtype=np.float32),
            'actions_patient': np.array(self.actions_patient, dtype=np.float32),
            'actions_robot': np.array(self.actions_robot, dtype=np.float32),
            'fatigues': np.array(self.fatigues, dtype=np.float32),
            'distances': np.array(self.distances, dtype=np.float32),
            'successes': np.array(self.successes, dtype=bool),
            'steps_to_success': np.array(self.steps_to_success, dtype=np.int32),
        }

        # Add patient thetas
        n_patients = len(self.patient_thetas)
        theta_array = np.zeros((n_patients, 3), dtype=np.float32)
        for pid in range(n_patients):
            theta_array[pid] = self.patient_thetas[pid]
        data['patient_thetas'] = theta_array

        np.savez_compressed(filepath, **data)
        print(f"Dataset saved to {filepath}")


def load_dataset(filepath: str) -> Dict[str, Any]:
    """
    Load dataset from npz file.

    Args:
        filepath: Path to the dataset file.

    Returns:
        Dictionary containing all dataset arrays.
    """
    data = np.load(filepath)
    return {key: data[key] for key in data.keys()}


def get_patient_episodes(
    dataset: Dict[str, Any],
    patient_id: int,
    episode_indices: List[int],
) -> Dict[str, Any]:
    """
    Extract data for specific episodes of a patient.

    Args:
        dataset: Full dataset dictionary.
        patient_id: Patient ID.
        episode_indices: List of episode indices to extract.

    Returns:
        Dictionary with filtered data for the specified episodes.
    """
    patient_mask = dataset['patient_ids'] == patient_id

    # Get unique episodes for this patient
    patient_data_idx = np.where(patient_mask)[0]
    patient_episodes = dataset['episode_ids'][patient_data_idx]

    # Create mask for selected episodes
    episode_mask = np.isin(patient_episodes, episode_indices)
    selected_idx = patient_data_idx[episode_mask]

    return {
        'states': dataset['states'][selected_idx],
        'actions_patient': dataset['actions_patient'][selected_idx],
        'actions_robot': dataset['actions_robot'][selected_idx],
        'fatigues': dataset['fatigues'][selected_idx],
        'episode_ids': dataset['episode_ids'][selected_idx],
    }


def split_by_episode(
    dataset: Dict[str, Any],
    patient_id: int,
) -> List[Dict[str, np.ndarray]]:
    """
    Split patient data into individual episodes.

    Args:
        dataset: Full dataset dictionary.
        patient_id: Patient ID.

    Returns:
        List of episode dictionaries, one per episode.
    """
    patient_mask = dataset['patient_ids'] == patient_id
    patient_idx = np.where(patient_mask)[0]

    episode_ids = dataset['episode_ids'][patient_idx]
    unique_episodes = np.unique(episode_ids)

    episodes = []
    for ep_id in unique_episodes:
        ep_mask = episode_ids == ep_id
        ep_idx = patient_idx[ep_mask]

        episodes.append({
            'episode_id': ep_id,
            'states': dataset['states'][ep_idx],
            'actions_patient': dataset['actions_patient'][ep_idx],
            'actions_robot': dataset['actions_robot'][ep_idx],
            'fatigues': dataset['fatigues'][ep_idx],
        })

    return episodes
