# Cognitive IRL Personalization Simulation

This repository implements a simulation cognitive inverse reinforcement learning with personalization in human-robot interaction.

## What This Demonstrates

This simulation demonstrates how accounting for latent cognitive states (fatigue) improves personalized modeling of human motor behavior during robot-assisted rehabilitation. A patient interacts with a robot assistant in a 2D reaching task, where the patient's effort cost changes over time due to fatigue accumulation. The cognitive EM method infers this hidden fatigue trajectory and achieves better parameter estimation and predictive accuracy compared to a static baseline that ignores fatigue dynamics.

## Installation

Requires Python 3.11 or higher.

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Generate Dataset

Generate synthetic patient-robot interaction data:

```bash
python sim/run_generate.py --seed 42
```

This creates 30 patients with 100 episodes each and saves the dataset to `sim/data/generated_dataset_seed42.npz`.

Optional arguments:
- `--seed`: Random seed for reproducibility (default: 42)
- `--n_patients`: Number of patients (default: 30)
- `--n_episodes`: Episodes per patient (default: 100)

### Step 2: Train and Evaluate

Train both learning methods and generate evaluation results:

```bash
python learn/run_train_eval.py --seed 42
```

This fits a static IRL baseline and a cognitive EM model on training episodes (0-79), evaluates on test episodes (80-99), and produces results.

Optional arguments:
- `--seed`: Random seed matching the dataset (default: 42)

## Outputs

### Generated Files

1. **Dataset**: `sim/data/generated_dataset_seed{seed}.npz`
   - Contains trajectories for all patients and episodes
   - Includes ground-truth objective parameters

2. **Results Table**: `results/tables/summary_seed{seed}.csv`
   - Mean parameter error and standard error for both methods
   - Mean test negative log-likelihood per step for both methods

3. **Figures**:
   - `results/figures/fig1_param_error.png`: Parameter estimation error comparison
   - `results/figures/fig2_test_nll.png`: Test set predictive accuracy comparison

### Expected Results

The cognitive EM method should achieve lower parameter estimation error and better test log-likelihood compared to the static baseline, demonstrating the value of modeling latent fatigue dynamics.

## Project Structure

```
.
├── sim/                        # Simulation components
│   ├── env.py                  # 2D reaching environment
│   ├── patient.py              # Patient generative model with fatigue
│   ├── robot.py                # Fixed robot assistance policy
│   ├── data.py                 # Dataset utilities
│   └── run_generate.py         # Dataset generation script
├── learn/                      # Learning methods
│   ├── baseline_static_irl.py  # Static IRL baseline
│   ├── cognitive_irl_em.py     # Cognitive EM method
│   ├── utils.py                # Shared utilities
│   └── run_train_eval.py       # Training and evaluation script
├── results/                    # Output directory
│   ├── figures/                # Generated plots
│   └── tables/                 # Result tables
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Technical Details

- **Environment**: 2D point-mass reaching with discrete-time dynamics (dt=0.1)
- **Patient Model**: Quadratic cost optimization with fatigue-modulated effort weight
- **Robot Policy**: PD controller with fixed assistance schedule
- **Learning Methods**:
  - Static IRL: Maximum likelihood assuming constant effort weight
  - Cognitive EM: Expectation-maximization with latent fatigue inference
- **Evaluation**: Train/test split (episodes 0-79 / 80-99), parameter error and test NLL

## Performance

End-to-end runtime is under 2 minutes on CPU for default settings (30 patients, 100 episodes).
