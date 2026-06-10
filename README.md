# Pursuit-Evasion RL Research Codebase

This repository trains and evaluates reinforcement-learning policies for orbital
pursuit-evasion games. The main workflow is staged PPO training for defender and
attacker policies, followed by Monte Carlo evaluation over repeatable shell-grid
initial conditions. The current stack supports HCW, elliptic LTV, and nonlinear
two-body-style rollout paths, with optional UKF/EKF estimator variants and a
velocity-CBF safety filter.

The short version: train with `rl_loop.py`, roll policies out with
`game_runner.py`, evaluate them with `evaluate_policy.py`, and use the
`run_eval_*` scripts when you want the full sweep without typing the long command
every time.

## Repository Map

- `config_rl.py` is the source of default experiment configuration. Treat this
  file as the main knob board, but most day-to-day sweeps should use CLI
  overrides instead of editing it.
- `rl_loop.py` is the staged training entry point. It selects phases, applies
  runtime overrides, checks checkpoint prerequisites, and delegates the real work
  to `core/train_eval.py`.
- `core/` contains the environment, PPO implementation, models, rollout buffers,
  plotting helpers, safety filters, distillation helpers, and run logging.
- `game_runner.py` runs trained RL policies against trained RL opponents.
- `matchup_runner.py` runs trained policies against comparison opponents:
  `paper`, `game_theory`, `ipopt`, and `rule`.
- `evaluate_policy.py` is the Monte Carlo evaluation harness. It writes
  `results.json`, `trials.csv`, start plots, outcome plots, trajectory overlays,
  and optional estimator diagnostics.
- `run_eval_*` scripts are batch launchers for 20 m, 50 m, 100 m, advantage-shell,
  and KF-enabled evaluation suites.
- `run_train_*` scripts replay training jobs or selected KF reruns from saved run
  manifests.
- `docs/` and `Algorithms/` contain diagrams and algorithm writeups.
- `Training_Policy*`, `Legacy_Runs/`, `Distillation_Tester/`, and `eval_out/`
  are experiment output folders with checkpoints, metrics, plots, and manifests.
- `Archive/` contains older experiments. Treat it as history unless you are
  intentionally reviving an old path.

## Setup

Use Python 3.11+ if possible. The environment used for this repo includes
`torch`, `numpy`, `scipy`, `matplotlib`, `casadi`, `pyomo`, `tensorboard`,
`imageio`, and `imageio-ffmpeg`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` complains about local Conda build paths, install the core
packages directly in your Conda environment instead:

```bash
pip install torch numpy scipy matplotlib casadi pyomo tensorboard imageio imageio-ffmpeg
```

## Training

Run the full staged PPO sequence:

```bash
python rl_loop.py \
  --out_dir Training_Policy_my_run \
  --phases 0,1,2,3,4 \
  --device cuda \
  --vec_backend torch \
  --vmax 1.5 \
  --umax 0.5 \
  --steps_per_env 1024
```

The phase order is:

- Phase 0: train `def0_teacher.pt`
- Phase 1: train `att1_teacher.pt`
- Phase 2: train `def1_teacher.pt`
- Phase 3: train `att2_teacher.pt`
- Phase 4: train/distill `def2_teacher.pt`

For quick manual sweeps, use:

```bash
./run_train_sequential.sh
```

For KF reruns from existing manifests:

```bash
./run_train_selected_training_policies_zero_sum_kf.sh --dry-run
./run_train_selected_training_policies_zero_sum_kf.sh
```

Use `--dry-run` first. It prints the reconstructed command before launching the
long jobs.

## Evaluation

A direct policy-vs-policy evaluation looks like this:

```bash
python evaluate_policy.py \
  --run_dir Training_Policy_my_run \
  --def_ckpt_path Training_Policy_my_run/def1_teacher.pt \
  --att_ckpt_path Training_Policy_my_run/att1_teacher.pt \
  --out_dir Training_Policy_my_run/MC_eval_50m/def1_vs_att1 \
  --auto_shell_grid \
  --grid_mode cartesian \
  --arena_radius 50.0 \
  --shell_fracs 0.2,0.4,0.6,0.8 \
  --points_per_shell 40 \
  --steps 6000 \
  --device cuda
```

For the standard scripted sweeps:

```bash
./run_eval_sequence_all 20m
./run_eval_sequence_all 50m
./run_eval_sequence_all 100m
./run_eval_sequence_all 100m_advantage
./run_eval_20m_sequence_all
./run_eval_50m_sequence_all
./run_eval_100m_sequence_all
./run_eval_100m_advantage_sequence_all
./run_eval_kf_sequence_all
```

You can pass explicit run specs to the eval launchers:

```bash
./run_eval_sequence_all 50m "Training_Policy_2.0u_1vmax_1.0_icVmax:elliptic_ltv"
```

Each run spec accepts `run_dir:dynamics` or `run_dir|dynamics`. If dynamics is
omitted, the script falls back to the common/default dynamics behavior.

## Outputs

Training folders usually contain:

- `run_manifest.json`: resolved config, CLI args, and run metadata
- `*_teacher.pt`: phase checkpoints
- `*_teacher__best.pt`: best checkpoint snapshots when available
- `train_metrics_*.npz`: per-phase training curves
- `Plots/`: training curves, IC support plots, and comparisons

Evaluation folders usually contain:

- `results.json`: aggregate outcome statistics and timing
- `trials.csv`: one row per Monte Carlo trial
- `outcome_hist.png`: outcome breakdown
- `starts_xy.png`, `starts_xz.png`: initial-condition coverage
- `trajectory_overlay_xy.png`, `trajectory_overlay_xz.png`: rollout overlays
- `*_cases.csv`: timeout, clean non-hit, or rollout-error case lists when enabled

## Current vs. Historical Solver Paths

The current mainline does not rely on the old PATH/NEOS solver path. The active
workflow is PPO/KF training and evaluation through `rl_loop.py`,
`game_runner.py`, `matchup_runner.py`, and `evaluate_policy.py`.

Historical solver material is intentionally isolated:

- `Archive/neos_path_game.py` is old archive code.
- `mcp_baseline_runner.py` is retained as a historical MCP/PATH comparison
  runner and is not imported by the main workflow.
- `nash_ipopt_solver.py` is a standalone IPOPT/CasADi experiment and is not used
  by the main training loop.

## Tests

Run the focused regression tests with:

```bash
python -m unittest discover tests
```

For a faster syntax-only check across the active source files:

```bash
python -m py_compile rl_loop.py evaluate_policy.py game_runner.py core/env.py
```

That does not replace the tests, but it is a useful first pass after broad
documentation or refactor edits.
