#!/usr/bin/env bash
#
# Train a second 0.5u / 1.5vmax policy pair with lower initial-speed support.
#
# This launcher intentionally writes to new v2 folders so the current thesis
# folders are not overwritten:
#   Training_Policy_0.5u_1.5vmax_0.5_icVmax_v2
#   Training_Policy_0.5u_1.5vmax_0.5_icVmax_v2_KF_ON
#
# The training knobs mirror Training_Policy_0.5u_1.5vmax_1.0_icVmax except for
# train_ic_vmax, which is set to 0.5 m/s.
set -uo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
KF_OFF_DIR="${KF_OFF_DIR:-Training_Policy_0.5u_1.5vmax_0.5_icVmax_v2}"
KF_ON_DIR="${KF_ON_DIR:-${KF_OFF_DIR}_KF_ON}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--dry-run]"
  exit 2
fi

run_step() {
  local name="$1"
  local out_dir="$2"
  shift 2

  if [[ -e "${out_dir}" ]]; then
    echo "[run_train_0.5u_1.5vmax_0.5_icVmax_v2] Skipping ${name}: ${out_dir} already exists"
    return 0
  fi

  echo "[run_train_0.5u_1.5vmax_0.5_icVmax_v2] Starting ${name}"
  if (( DRY_RUN )); then
    printf '[run_train_0.5u_1.5vmax_0.5_icVmax_v2] Command:'
    printf ' %q' "$@"
    printf '\n'
    echo "[run_train_0.5u_1.5vmax_0.5_icVmax_v2] Dry-run complete for ${name}"
    return 0
  fi

  if "$@"; then
    echo "[run_train_0.5u_1.5vmax_0.5_icVmax_v2] Completed ${name}"
    return 0
  fi

  local status=$?
  if [[ "${status}" -ge 128 ]]; then
    echo "[run_train_0.5u_1.5vmax_0.5_icVmax_v2] ${name} was killed by signal $((status - 128)); continuing"
  else
    echo "[run_train_0.5u_1.5vmax_0.5_icVmax_v2] ${name} failed with exit code ${status}; continuing"
  fi
  return 0
}

common_args=(
  --phases 0,1,2,3,4
  --device cuda
  --seed 42
  --vmax 1.5
  --umax 0.5
  --k_pos 0.05
  --k_dock 0.0125
  --train_ic_vmax 0.5
  --num_envs 256
  --steps_per_env 1024
  --total_updates 1000
  --train_epochs 3
  --minibatch_size 4096
  --log_every 10
  --vec_backend torch
  --torch_fast_reset false
)

run_step "KF off ${KF_OFF_DIR}" "${KF_OFF_DIR}" \
  env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" rl_loop.py \
    --out_dir "${KF_OFF_DIR}" \
    "${common_args[@]}" \
    --use_kf false \
    --reward_type zero_sum \
    --tb_run_prefix "${KF_OFF_DIR}"

run_step "KF on ${KF_ON_DIR}" "${KF_ON_DIR}" \
  env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" rl_loop.py \
    --out_dir "${KF_ON_DIR}" \
    "${common_args[@]}" \
    --use_kf true \
    --reward_type zero_sum_kf \
    --estimator_kind ekf \
    --ekf_jacobian_mode exact \
    --tb_run_prefix "${KF_ON_DIR}"

echo "[run_train_0.5u_1.5vmax_0.5_icVmax_v2] Done"
