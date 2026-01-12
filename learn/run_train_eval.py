"""Train and evaluate both learning methods."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sim'))

from data import load_dataset, split_by_episode
from baseline_static_irl import fit_static_model, evaluate_static_model
from cognitive_irl_em import fit_em_model, evaluate_em_model


def train_and_evaluate(dataset_path: str, seed: int) -> None:
    """
    Load dataset, train both models, and evaluate.

    Args:
        dataset_path: Path to the generated dataset.
        seed: Random seed used for dataset generation.
    """
    print(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)

    n_patients = len(dataset['patient_thetas'])
    print(f"Number of patients: {n_patients}")

    # Storage for results
    static_param_errors = []
    em_param_errors = []
    static_test_nlls = []
    em_test_nlls = []

    for patient_id in range(n_patients):
        if patient_id % 10 == 0:
            print(f"  Processing patient {patient_id}/{n_patients}")

        # Get all episodes for this patient
        episodes = split_by_episode(dataset, patient_id)

        # Split into train (0-79) and test (80-99)
        train_episodes = [ep for ep in episodes if ep['episode_id'] < 80]
        test_episodes = [ep for ep in episodes if ep['episode_id'] >= 80]

        # Ground truth
        w_effort_true = dataset['patient_thetas'][patient_id][1]

        # Fit static model
        w_effort_static = fit_static_model(train_episodes)
        param_error_static = abs(w_effort_static - w_effort_true)
        test_nll_static = evaluate_static_model(test_episodes, w_effort_static)

        static_param_errors.append(param_error_static)
        static_test_nlls.append(test_nll_static)

        # Fit EM model
        w_effort_em, _ = fit_em_model(train_episodes, n_iterations=6, w_effort_init=0.15)
        param_error_em = abs(w_effort_em - w_effort_true)
        test_nll_em = evaluate_em_model(test_episodes, w_effort_em)

        em_param_errors.append(param_error_em)
        em_test_nlls.append(test_nll_em)

    # Compute statistics
    static_param_mean = np.mean(static_param_errors)
    static_param_se = np.std(static_param_errors) / np.sqrt(n_patients)
    em_param_mean = np.mean(em_param_errors)
    em_param_se = np.std(em_param_errors) / np.sqrt(n_patients)

    static_nll_mean = np.mean(static_test_nlls)
    static_nll_se = np.std(static_test_nlls) / np.sqrt(n_patients)
    em_nll_mean = np.mean(em_test_nlls)
    em_nll_se = np.std(em_test_nlls) / np.sqrt(n_patients)

    print("\nResults:")
    print(f"Static model - Param error: {static_param_mean:.4f} ± {static_param_se:.4f}")
    print(f"EM model     - Param error: {em_param_mean:.4f} ± {em_param_se:.4f}")
    print(f"Static model - Test NLL: {static_nll_mean:.4f} ± {static_nll_se:.4f}")
    print(f"EM model     - Test NLL: {em_nll_mean:.4f} ± {em_nll_se:.4f}")

    # Save table
    save_table(
        seed,
        static_param_mean,
        static_param_se,
        em_param_mean,
        em_param_se,
        static_nll_mean,
        static_nll_se,
        em_nll_mean,
        em_nll_se,
    )

    # Generate plots
    plot_results(
        static_param_mean,
        static_param_se,
        em_param_mean,
        em_param_se,
        static_nll_mean,
        static_nll_se,
        em_nll_mean,
        em_nll_se,
    )


def save_table(
    seed: int,
    static_param_mean: float,
    static_param_se: float,
    em_param_mean: float,
    em_param_se: float,
    static_nll_mean: float,
    static_nll_se: float,
    em_nll_mean: float,
    em_nll_se: float,
) -> None:
    """Save results table to CSV."""
    output_path = f"results/tables/summary_seed{seed}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("method,mean_param_error,se_param_error,mean_test_nll,se_test_nll\n")
        f.write(f"static,{static_param_mean:.6f},{static_param_se:.6f},{static_nll_mean:.6f},{static_nll_se:.6f}\n")
        f.write(f"em,{em_param_mean:.6f},{em_param_se:.6f},{em_nll_mean:.6f},{em_nll_se:.6f}\n")

    print(f"\nTable saved to {output_path}")


def plot_results(
    static_param_mean: float,
    static_param_se: float,
    em_param_mean: float,
    em_param_se: float,
    static_nll_mean: float,
    static_nll_se: float,
    em_nll_mean: float,
    em_nll_se: float,
) -> None:
    """Generate and save result plots."""
    # Figure 1: Parameter error
    fig, ax = plt.subplots(figsize=(6, 5))
    methods = ['Static', 'EM']
    means = [static_param_mean, em_param_mean]
    errors = [static_param_se, em_param_se]

    x_pos = np.arange(len(methods))
    ax.bar(x_pos, means, yerr=errors, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Mean Parameter Error', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Parameter Estimation Error', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(['Static IRL', 'Cognitive EM'], loc='upper right')

    plt.tight_layout()
    output_path = "results/figures/fig1_param_error.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Figure saved to {output_path}")

    # Figure 2: Test NLL
    fig, ax = plt.subplots(figsize=(6, 5))
    means = [static_nll_mean, em_nll_mean]
    errors = [static_nll_se, em_nll_se]

    ax.bar(x_pos, means, yerr=errors, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Mean Test NLL per Step', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Test Set Negative Log-Likelihood', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(['Static IRL', 'Cognitive EM'], loc='upper right')

    plt.tight_layout()
    output_path = "results/figures/fig2_test_nll.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Figure saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate learning methods")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    dataset_path = f"sim/data/generated_dataset_seed{args.seed}.npz"

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please run: python sim/run_generate.py first")
        return

    train_and_evaluate(dataset_path, args.seed)


if __name__ == "__main__":
    main()
