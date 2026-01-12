"""Generate dataset of patient-robot interaction trajectories."""

import argparse
import numpy as np
from env import ReachingEnv
from patient import Patient, sample_patient_theta, get_initial_fatigue, update_fatigue
from robot import compute_assist, get_lambda_schedule
from data import TrajectoryBuffer


def generate_episode(
    env: ReachingEnv,
    patient: Patient,
    episode_idx: int,
    rng: np.random.Generator,
) -> tuple:
    """
    Run a single episode of patient-robot interaction.

    Args:
        env: Reaching environment.
        patient: Patient model.
        episode_idx: Episode number.
        rng: Random generator.

    Returns:
        Tuple of (trajectory_data, success, steps).
    """
    state = env.reset(rng)
    lambda_ep = get_lambda_schedule(episode_idx)
    fatigue = get_initial_fatigue(episode_idx)
    u_prev = np.zeros(2)

    trajectory = {
        'states': [],
        'actions_patient': [],
        'actions_robot': [],
        'fatigues': [],
        'distances': [],
    }

    done = False
    success = False
    step_count = 0

    while not done:
        # Robot computes assistance
        a_t = compute_assist(state, lambda_ep)

        # Patient generates action
        u_t, _ = patient.act(state, a_t, fatigue, u_prev, rng)

        # Log before stepping
        trajectory['states'].append(state.copy())
        trajectory['actions_patient'].append(u_t.copy())
        trajectory['actions_robot'].append(a_t.copy())
        trajectory['fatigues'].append(fatigue)

        # Step environment
        state, done, success, distance = env.step(u_t, a_t, rng)
        trajectory['distances'].append(distance)

        # Update fatigue
        fatigue = update_fatigue(fatigue, u_t)
        u_prev = u_t
        step_count += 1

    steps_to_success = step_count if success else env.max_steps

    return trajectory, success, steps_to_success


def generate_dataset(seed: int, n_patients: int = 30, n_episodes: int = 100) -> None:
    """
    Generate full dataset for multiple patients.

    Args:
        seed: Random seed.
        n_patients: Number of patients to simulate.
        n_episodes: Number of episodes per patient.
    """
    rng = np.random.default_rng(seed)
    buffer = TrajectoryBuffer()
    env = ReachingEnv()

    print(f"Generating dataset with seed {seed}")
    print(f"Patients: {n_patients}, Episodes per patient: {n_episodes}")

    for patient_id in range(n_patients):
        # Sample patient parameters
        w_goal, w_effort, w_smooth = sample_patient_theta(rng)
        patient = Patient(w_goal, w_effort, w_smooth)
        buffer.set_patient_theta(patient_id, np.array([w_goal, w_effort, w_smooth]))

        if patient_id % 10 == 0:
            print(f"  Patient {patient_id}/{n_patients}")

        for episode_idx in range(n_episodes):
            trajectory, success, steps = generate_episode(env, patient, episode_idx, rng)

            # Add trajectory to buffer
            for t in range(len(trajectory['states'])):
                buffer.add_step(
                    patient_id=patient_id,
                    episode_id=episode_idx,
                    state=trajectory['states'][t],
                    u=trajectory['actions_patient'][t],
                    a=trajectory['actions_robot'][t],
                    fatigue=trajectory['fatigues'][t],
                    distance=trajectory['distances'][t],
                )

            buffer.add_episode_outcome(success, steps)

    # Save dataset
    output_path = f"sim/data/generated_dataset_seed{seed}.npz"
    buffer.save(output_path)

    # Print statistics
    success_rate = np.mean(buffer.successes) * 100
    mean_steps = np.mean(buffer.steps_to_success)
    print(f"\nDataset statistics:")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Mean steps to success: {mean_steps:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulation dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_patients", type=int, default=30, help="Number of patients")
    parser.add_argument("--n_episodes", type=int, default=100, help="Episodes per patient")
    args = parser.parse_args()

    generate_dataset(args.seed, args.n_patients, args.n_episodes)


if __name__ == "__main__":
    main()
