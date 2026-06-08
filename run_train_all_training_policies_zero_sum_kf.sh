#!/usr/bin/env bash
#
# Replay every top-level Training_Policy* run as a KF-enabled zero-sum run.
#
# The script reads each parent run_manifest.json, reconstructs the original
# rl_loop.py command, then appends the KF/reward overrides needed for the
# *_KF_ON rerun. It skips runs that already have the target suffix, runs without
# manifests, and target directories that already exist.
set -uo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TARGET_SUFFIX="${TARGET_SUFFIX:-_KF_ON}"
DRY_RUN=0

# The only accepted CLI flag is --dry-run. Everything else is controlled through
# environment variables so this script stays easy to repeat from shell history.
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

  echo "[run_train_all_training_policies_zero_sum_kf] Starting ${name}"
  if (( DRY_RUN )); then
    printf '[run_train_all_training_policies_zero_sum_kf] Command:'
    printf ' %q' "$@"
    printf '\n'
    echo "[run_train_all_training_policies_zero_sum_kf] Dry-run complete for ${name}"
    return 0
  fi

  if "$@"; then
    echo "[run_train_all_training_policies_zero_sum_kf] Completed ${name}"
    return 0
  else
    local status=$?
    if [ "$status" -ge 128 ]; then
      echo "[run_train_all_training_policies_zero_sum_kf] ${name} was killed by signal $((status - 128)); continuing"
    else
      echo "[run_train_all_training_policies_zero_sum_kf] ${name} failed with exit code ${status}; continuing"
    fi
  fi
  return 0
}

mapfile -t RUN_DIRS < <(
  # Only top-level policy folders are replayed here. Nested/legacy runs belong in
  # the selected-run launcher where the list is intentionally hand-curated.
  find . -maxdepth 1 -mindepth 1 -type d -name 'Training_Policy*' -printf '%f\n' | sort
)

echo "[run_train_all_training_policies_zero_sum_kf] Found ${#RUN_DIRS[@]} top-level Training_Policy directories"
echo "[run_train_all_training_policies_zero_sum_kf] Using CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"

skipped=0
for run_dir in "${RUN_DIRS[@]}"; do
  # Never overwrite a completed or partially completed KF rerun. If the target
  # folder exists, leave it for manual inspection.
  if [[ "${run_dir}" == *"${TARGET_SUFFIX}" ]]; then
    echo "[run_train_all_training_policies_zero_sum_kf] Skipping ${run_dir}: already matches target suffix"
    skipped=$((skipped + 1))
    continue
  fi

  manifest_path="${run_dir}/run_manifest.json"
  target_dir="${run_dir}${TARGET_SUFFIX}"

  if [[ -e "${target_dir}" ]]; then
    echo "[run_train_all_training_policies_zero_sum_kf] Skipping ${run_dir}: target already exists at ${target_dir}"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ ! -f "${manifest_path}" ]]; then
    echo "[run_train_all_training_policies_zero_sum_kf] Skipping ${run_dir}: missing ${manifest_path}"
    skipped=$((skipped + 1))
    continue
  fi

  mapfile -d '' -t cmd < <(
    # The inline Python reconstructs the rl_loop.py invocation from the saved
    # manifest. It emits NUL-delimited argv items so bash preserves spaces safely.
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

cmd = [
    python_bin,
    "rl_loop.py",
    "--out_dir",
    target_dir.name,
    "--phases",
    str(cli_args.get("phases", "0,1,2,3,4")),
    "--use_kf",
    "true",
    "--reward_type",
    "zero_sum_kf",
    "--tb_run_prefix",
    target_dir.name,
]


def add(flag, value):
    # Missing manifest fields stay missing; the training defaults can fill them.
    if value is None:
        return
    cmd.extend([flag, str(value)])


add("--device", train_cfg.get("device", cli_args.get("device")))
add("--seed", train_cfg.get("seed"))
add("--umax", train_cfg.get("umax"))
add("--vmax", safety_filter.get("vmax", train_cfg.get("vmax")))
add("--k_pos", train_cfg.get("k_pos"))
add("--k_dock", train_cfg.get("k_dock"))
add("--train_ic_vmax", train_cfg.get("train_ic_vmax"))
add("--num_envs", train_cfg.get("num_envs"))
add("--steps_per_env", train_cfg.get("steps_per_env"))
add("--total_updates", train_cfg.get("total_updates"))
add("--train_epochs", train_cfg.get("train_epochs"))
add("--minibatch_size", train_cfg.get("minibatch_size"))
add("--log_every", train_cfg.get("log_every"))
add("--vec_backend", train_cfg.get("vec_backend", cli_args.get("vec_backend")))
add("--vec_workers", train_cfg.get("vec_workers"))
add("--mp_start_method", train_cfg.get("mp_start_method"))

disable_tb = bool(cli_args.get("disable_tensorboard", False)) or not bool(train_cfg.get("use_tensorboard", True))
if disable_tb:
    cmd.append("--disable_tensorboard")
else:
    add("--tb_logdir", cli_args.get("tb_logdir"))

# NUL delimiters let mapfile rebuild argv without shell re-splitting.
for item in cmd:
    sys.stdout.buffer.write(item.encode("utf-8"))
    sys.stdout.buffer.write(b"\0")
PY
  )

  run_step "${run_dir} -> ${target_dir}" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${cmd[@]}"
done

echo "[run_train_all_training_policies_zero_sum_kf] Done. skipped=${skipped} total=${#RUN_DIRS[@]}"
