#!/usr/bin/env bash
#
# Replay selected Training_Policy folders as KF-enabled zero-sum runs.
#
# This is the safer, hand-curated version of run_train_all_training_policies_zero_sum_kf.sh.
# Uncomment entries in SELECTED_RUN_DIRS when you want to replay only a few parent
# runs, and use EXTRA_OVERRIDES for last-mile CLI changes without touching
# config_rl.py or the source manifest.
set -uo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TARGET_SUFFIX="${TARGET_SUFFIX:-_KF_ON}"
DRY_RUN=0

# Uncomment the parent run directories you want to replay as KF runs.
# These may be top-level runs or nested paths such as Legacy_Runs/... .
SELECTED_RUN_DIRS=(

  # "Training_Policy_0.5u_1.5vmax_0.25_icVmax"
  # "Training_Policy_0.5u_1.5vmax_1.0_icVmax"
  "Training_Policy_0.1u_1vmax_0.05_icVmax"



  #Failed
  # "Failed_Runs/Training_Policy_0.5u_1vmax_1.0_icVmax"
  # "Failed_Runs/Training_Policy_0.5u_1.5vmax_1.5_icVmax"


#Ignored

)

# These flags are appended after the manifest-derived command, so they win.
# The script already forces:
#   --use_kf true
#   --reward_type zero_sum_kf
#   --tb_run_prefix <target_basename>
#
# If you also add --use_kf or --reward_type here, the last occurrence wins.
EXTRA_OVERRIDES=(
  # "--estimator_kind" "ekf"
  # "--ekf_jacobian_mode" "exact"
  # "--device" "cuda"
  # "--num_envs" "256"
  # "--steps_per_env" "1024"
  # "--total_updates" "1000"
  # "--train_epochs" "3"
  # "--minibatch_size" "4096"
  # "--log_every" "10"
  # "--disable_tensorboard"
)

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--dry-run]"
  exit 2
fi

run_step() {
  # Print the command in dry-run mode, otherwise run it and continue on failure.
  local name="$1"
  shift

  echo "[run_train_selected_training_policies_zero_sum_kf] Starting ${name}"
  if (( DRY_RUN )); then
    printf '[run_train_selected_training_policies_zero_sum_kf] Command:'
    printf ' %q' "$@"
    printf '\n'
    echo "[run_train_selected_training_policies_zero_sum_kf] Dry-run complete for ${name}"
    return 0
  fi

  if "$@"; then
    echo "[run_train_selected_training_policies_zero_sum_kf] Completed ${name}"
    return 0
  else
    local status=$?
    if [ "$status" -ge 128 ]; then
      echo "[run_train_selected_training_policies_zero_sum_kf] ${name} was killed by signal $((status - 128)); continuing"
    else
      echo "[run_train_selected_training_policies_zero_sum_kf] ${name} failed with exit code ${status}; continuing"
    fi
  fi
  return 0
}

if (( ${#SELECTED_RUN_DIRS[@]} == 0 )); then
  echo "[run_train_selected_training_policies_zero_sum_kf] No parent runs selected."
  echo "[run_train_selected_training_policies_zero_sum_kf] Uncomment entries inside SELECTED_RUN_DIRS and rerun."
  exit 2
fi

echo "[run_train_selected_training_policies_zero_sum_kf] Selected ${#SELECTED_RUN_DIRS[@]} parent runs"
echo "[run_train_selected_training_policies_zero_sum_kf] Using CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"

skipped=0
for raw_run_dir in "${SELECTED_RUN_DIRS[@]}"; do
  run_dir="${raw_run_dir%/}"

  if [[ ! -d "${run_dir}" ]]; then
    echo "[run_train_selected_training_policies_zero_sum_kf] Skipping ${run_dir}: directory does not exist"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ "${run_dir}" == *"${TARGET_SUFFIX}" ]]; then
    echo "[run_train_selected_training_policies_zero_sum_kf] Skipping ${run_dir}: already matches target suffix"
    skipped=$((skipped + 1))
    continue
  fi

  manifest_path="${run_dir}/run_manifest.json"
  target_dir="${run_dir}${TARGET_SUFFIX}"

  if [[ -e "${target_dir}" ]]; then
    echo "[run_train_selected_training_policies_zero_sum_kf] Skipping ${run_dir}: target already exists at ${target_dir}"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ ! -f "${manifest_path}" ]]; then
    echo "[run_train_selected_training_policies_zero_sum_kf] Skipping ${run_dir}: missing ${manifest_path}"
    skipped=$((skipped + 1))
    continue
  fi

  mapfile -d '' -t cmd < <(
    "${PYTHON_BIN}" - "${manifest_path}" "${target_dir}" "${PYTHON_BIN}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
python_bin = sys.argv[3]

data = json.loads(manifest_path.read_text())
configs = data.get("configs", {}) or {}
cli_args = configs.get("cli_args", {}) or {}
train_cfg = configs.get("training config used", {}) or {}
safety_filter = train_cfg.get("safety_filter", {}) or {}
ukf_cfg = train_cfg.get("ukf", {}) or {}

cmd = [
    python_bin,
    "rl_loop.py",
    "--out_dir",
    str(target_dir),
]


def add(flag, value):
    if value is None:
        return
    cmd.extend([flag, str(value)])


add("--phases", cli_args.get("phases", "0,1,2,3,4"))
add("--device", train_cfg.get("device", cli_args.get("device")))
add("--seed", train_cfg.get("seed"))
add("--umax", train_cfg.get("umax"))
add("--vmax", safety_filter.get("vmax", train_cfg.get("vmax")))
add("--k_pos", train_cfg.get("k_pos"))
add("--k_dock", train_cfg.get("k_dock"))
add("--train_ic_vmax", train_cfg.get("train_ic_vmax"))
add("--estimator_kind", train_cfg.get("estimator_kind"))
add("--num_envs", train_cfg.get("num_envs"))
add("--steps_per_env", train_cfg.get("steps_per_env"))
add("--total_updates", train_cfg.get("total_updates"))
add("--train_epochs", train_cfg.get("train_epochs"))
add("--minibatch_size", train_cfg.get("minibatch_size"))
add("--log_every", train_cfg.get("log_every"))
add("--vec_backend", train_cfg.get("vec_backend", cli_args.get("vec_backend")))
add("--vec_workers", train_cfg.get("vec_workers"))
add("--mp_start_method", train_cfg.get("mp_start_method"))
add("--ekf_jacobian_mode", ukf_cfg.get("ekf_jacobian_mode"))

disable_tb = bool(cli_args.get("disable_tensorboard", False)) or not bool(train_cfg.get("use_tensorboard", True))
if disable_tb:
    cmd.append("--disable_tensorboard")
else:
    add("--tb_logdir", cli_args.get("tb_logdir"))

cmd.extend([
    "--use_kf",
    "true",
    "--reward_type",
    "zero_sum_kf",
    "--tb_run_prefix",
    target_dir.name,
])

for item in cmd:
    sys.stdout.buffer.write(item.encode("utf-8"))
    sys.stdout.buffer.write(b"\0")
PY
  )

  if (( ${#EXTRA_OVERRIDES[@]} > 0 )); then
    cmd+=("${EXTRA_OVERRIDES[@]}")
  fi

  run_step "${run_dir} -> ${target_dir}" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${cmd[@]}"
done

echo "[run_train_selected_training_policies_zero_sum_kf] Done. skipped=${skipped} total=${#SELECTED_RUN_DIRS[@]}"
