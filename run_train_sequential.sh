#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

run_step() {
  local name="$1"
  shift

  echo "[run_train_sequential] Starting ${name}"
  if "$@"; then
    echo "[run_train_sequential] Completed ${name}"
    return 0
  else
    local status=$?
    if [ "$status" -ge 128 ]; then
      echo "[run_train_sequential] ${name} was killed by signal $((status - 128)); continuing"
    else
      echo "[run_train_sequential] ${name} failed with exit code ${status}; continuing"
    fi
  fi
  return 0
}

run_step "Training_Policy_0.5u_1vmax" \
  env CUDA_VISIBLE_DEVICES=0 python rl_loop.py \
    --out_dir Training_Policy_0.5u_1vmax \
    --device cuda \
    --vec_backend torch \
    --vmax 1.0 \
    --umax 0.5 \
    --k_pos 0.05 \
    --k_dock 0.0125 \
    --train_ic_vmax 1.0 \
    --steps_per_env 1024

run_step "Training_Policy_0.5u_1.5vmax_1.5icvel" \
  env CUDA_VISIBLE_DEVICES=0 python rl_loop.py \
    --out_dir Training_Policy_0.5u_1.5vmax_1.5icvel \
    --device cuda \
    --vec_backend torch \
    --train_ic_vmax 1.50 \
    --vmax 1.5 \
    --umax 0.5 \
    --k_pos 0.05 \
    --k_dock 0.0125 \
    --steps_per_env 1024


# run_step "Training_Policy_2.0u_1.5vmax" \
#   env CUDA_VISIBLE_DEVICES=0 python rl_loop.py \
#     --out_dir Training_Policy_2.0u_1.5vmax \
#     --device cuda \
#     --vec_backend torch \
#     --vmax 1.5 \
#     --umax 2.0  \
#     --k_pos 0.1 \
#     --k_dock 0.025 \
#     --train_ic_vmax 1.50 \
#     --steps_per_env 512

# run_step "Training_Policy_0.1u_1.5vmax" \
#   env CUDA_VISIBLE_DEVICES=0 python rl_loop.py \
#     --out_dir Training_Policy_0.1u_1.5vmax \
#     --device cuda \
#     --vec_backend torch \
#     --vmax 1.5 \
#     --umax 0.1  \
#     --k_pos 0.025 \
#     --k_dock 0.00625 \
#     --train_ic_vmax 0.05 \
#     --steps_per_env 2048


echo "[run_train_sequential] Done"
